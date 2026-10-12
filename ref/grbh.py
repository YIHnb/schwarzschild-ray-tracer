# -*- coding: utf-8 -*-
"""
grbh.py -- General-relativistic black-hole ray tracer (CPU reference implementation).

Physics
-------
Schwarzschild spacetime, geometrized units G = c = 1, M = 1  =>  r_s = 2M.

Null geodesics are planar.  In the orbital plane, with u = 1/r and phi the
position angle measured from the camera's radial direction, the exact geodesic
equation is

        d^2u/dphi^2  =  -u + (3/2) r_s u^2                                  (1)

which we integrate with RK4 and an adaptive step in phi.  The first integral is

        (du/dphi)^2  =  1/b^2 - u^2 (1 - r_s u)                             (2)

so the impact parameter b is fixed by the initial conditions.

Everything downstream (disk crossing, Doppler/gravitational shift, blackbody
radiance, beaming) follows from b and the local trajectory.

This file is also the blueprint for the GLSL/WebGL2 port: every function here
has a 1:1 counterpart in the shader.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from PIL import Image

# =====================================================================
#  Constants
# =====================================================================

M = 1.0                     # black-hole mass
RS = 2.0 * M                # Schwarzschild radius
R_PHOTON = 3.0 * M          # photon sphere
B_CRIT = 3.0 * math.sqrt(3.0) * M   # critical impact parameter  ~ 5.19615 M
R_ISCO = 6.0 * M            # innermost stable circular orbit


# =====================================================================
#  Blackbody chromaticity  (Planck x CIE 1931, Wyman et al. multi-lobe fit)
# =====================================================================

def _gauss_piecewise(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def _cie_xyz_bar(lam_nm):
    """Analytic approximation of the CIE 1931 2-deg colour matching functions."""
    x = (1.056 * _gauss_piecewise(lam_nm, 599.8, 37.9, 31.0)
         + 0.362 * _gauss_piecewise(lam_nm, 442.0, 16.0, 26.7)
         - 0.065 * _gauss_piecewise(lam_nm, 501.1, 20.4, 26.2))
    y = (0.821 * _gauss_piecewise(lam_nm, 568.8, 46.9, 40.5)
         + 0.286 * _gauss_piecewise(lam_nm, 530.9, 16.3, 31.1))
    z = (1.217 * _gauss_piecewise(lam_nm, 437.0, 11.8, 36.0)
         + 0.681 * _gauss_piecewise(lam_nm, 459.0, 26.0, 13.8))
    return x, y, z


_XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])


def make_blackbody_lut(n=1024, t_min=500.0, t_max=120000.0):
    """Blackbody LUT, log-spaced in T.

    Returns (chroma, lum):
      chroma[i] : linear sRGB *chromaticity* of a blackbody at T[i], normalised
                  so that its own luminance is 1 (colour only).
      lum[i]    : the *visible* luminance  int B_lambda(T) ybar(lambda) dlambda,
                  in arbitrary but consistent units.

    Keeping these separate matters.  A cool blackbody radiates most of its
    power in the infrared, so at equal bolometric output it is far dimmer in
    the visible than a hot one.  Folding that into the LUT is what gives the
    outer disk its correct dim, red falloff instead of a uniform white sheet.
    """
    lam = np.linspace(360.0, 830.0, 236)
    xb, yb, zb = _cie_xyz_bar(lam)
    h = 6.62607015e-34
    c = 2.99792458e8
    kB = 1.380649e-23
    lam_m = lam * 1e-9
    Ts = np.exp(np.linspace(math.log(t_min), math.log(t_max), n))
    # Planck spectral radiance (per unit wavelength), constants dropped
    spec = 1.0 / (lam_m ** 5)[None, :] / np.expm1((h * c / kB) / (lam_m[None, :] * Ts[:, None]))
    X = spec @ xb
    Y = spec @ yb
    Z = spec @ zb
    xyz = np.stack([X / Y, np.ones_like(Y), Z / Y], axis=-1)   # chromaticity, Y == 1
    rgb = xyz @ _XYZ_TO_RGB.T
    rgb = np.clip(rgb, 0.0, None)
    # renormalise so the clamped colour still carries unit luminance
    lum_rgb = rgb @ np.array([0.2126, 0.7152, 0.0722])
    rgb = rgb / np.maximum(lum_rgb, 1e-6)[:, None]
    return (np.ascontiguousarray(rgb, dtype=np.float32),
            np.ascontiguousarray(Y, dtype=np.float64),
            math.log(t_min), math.log(t_max))


_LUT, _LUM, _LUT_LO, _LUT_HI = make_blackbody_lut()


def _lut_index(T):
    T = np.asarray(T, dtype=np.float64)
    f = (np.log(np.clip(T, 1.0, None)) - _LUT_LO) / (_LUT_HI - _LUT_LO)
    f = np.clip(f, 0.0, 1.0) * (len(_LUT) - 1)
    i0 = np.floor(f).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(_LUT) - 1)
    return i0, i1, f - i0


def blackbody_rgb(T):
    """Chromaticity (linear sRGB, unit luminance) of a blackbody at T."""
    i0, i1, w = _lut_index(T)
    w = w[..., None]
    return _LUT[i0] * (1.0 - w) + _LUT[i1] * w


def blackbody_lum(T):
    """Visible luminance of a blackbody at T (arbitrary units)."""
    i0, i1, w = _lut_index(T)
    return _LUM[i0] * (1.0 - w) + _LUM[i1] * w


# =====================================================================
#  Procedural sky : hash noise -> fbm nebula + star layer
# =====================================================================

def _hash31(p, seed):
    n = p[..., 0] * 127.1 + p[..., 1] * 311.7 + p[..., 2] * 74.7 + seed * 13.13
    s = np.sin(n) * 43758.5453123
    return s - np.floor(s)


def vnoise3(p, seed=0):
    """Trilinear value noise on a unit lattice."""
    i = np.floor(p)
    f = p - i
    w = f * f * (3.0 - 2.0 * f)
    acc = np.zeros(p.shape[:-1], dtype=np.float64)
    for dz in (0.0, 1.0):
        for dy in (0.0, 1.0):
            for dx in (0.0, 1.0):
                h = _hash31(i + np.array([dx, dy, dz]), seed)
                wx = w[..., 0] if dx else 1.0 - w[..., 0]
                wy = w[..., 1] if dy else 1.0 - w[..., 1]
                wz = w[..., 2] if dz else 1.0 - w[..., 2]
                acc += h * wx * wy * wz
    return acc


def fbm3(p, octaves=5, seed=0, lac=2.03, gain=0.5):
    acc = np.zeros(p.shape[:-1], dtype=np.float64)
    amp, norm = 1.0, 0.0
    q = p.copy()
    for o in range(octaves):
        acc += amp * vnoise3(q, seed + o * 17)
        norm += amp
        amp *= gain
        q = q * lac
    return acc / norm


def _hash2i(ix, iy, seed):
    n = ix * 127.1 + iy * 311.7 + seed * 74.7
    s = np.sin(n) * 43758.5453123
    return s - np.floor(s)


STAR_CELL = math.radians(0.55)      # angular size of one star cell
STAR_DENSITY = 0.16                 # fraction of cells that actually get a star


def star_layer(d, seed=7, boost=1.0):
    """Sparse point-star field.

    The celestial sphere is diced into cells of ~STAR_CELL radians; a cell holds
    a star with probability STAR_DENSITY.  Stars are drawn as tight Airy-like
    cores (sigma << cell) so that the 3x3 neighbour search stays well separated
    and the field reads as discrete points rather than overlapping blobs.
    Magnitudes follow a power law, so a few stars dominate.
    """
    lat = np.arcsin(np.clip(d[..., 2], -1.0, 1.0))
    lon = np.arctan2(d[..., 1], d[..., 0])
    cl = np.maximum(np.cos(lat), 0.05)
    # Longitude cells must wrap: atan2 jumps from +pi to -pi, so without
    # wrapping the 3x3 neighbourhood misses every star along that meridian and
    # a visible one-cell seam appears in the sky.  Rounding the cell count to an
    # integer makes the sphere close exactly.
    nx = np.maximum(1.0, np.floor(2.0 * math.pi / (STAR_CELL / cl) + 0.5))
    cs_lon = 2.0 * math.pi / nx
    gx, gy = lon / cs_lon, lat / STAR_CELL
    ix, iy = np.floor(gx), np.floor(gy)
    out = np.zeros(d.shape[:-1] + (3,), dtype=np.float64)
    for oy in (-1.0, 0.0, 1.0):
        for ox in (-1.0, 0.0, 1.0):
            cx, cy = np.mod(ix + ox, nx), iy + oy
            h0 = _hash2i(cx, cy, seed)
            present = h0 < STAR_DENSITY
            if not np.any(present):
                continue
            h1 = _hash2i(cx, cy, seed + 11.0)
            h2 = _hash2i(cx, cy, seed + 23.0)
            h3 = _hash2i(cx, cy, seed + 37.0)
            h4 = _hash2i(cx, cy, seed + 53.0)
            slon = (cx + 0.08 + 0.84 * h1) * cs_lon
            slat = (cy + 0.08 + 0.84 * h2) * STAR_CELL
            dlon = (lon - slon + math.pi) % (2.0 * math.pi) - math.pi
            dlat = lat - slat
            ang = np.sqrt((dlon * cl) ** 2 + dlat ** 2)
            # power-law brightness: most stars faint, rare ones brilliant
            mag = h3 ** 4.5
            # tight core, widened slightly for the brightest stars
            sigma = STAR_CELL * (0.045 + 0.085 * mag)
            prof = np.exp(-0.5 * (ang / sigma) ** 2)
            # soft halo for the brightest few
            halo = np.exp(-0.5 * (ang / (sigma * 3.2)) ** 2) * 0.10 * mag
            amp = (0.03 + 14.0 * mag ** 2) * boost
            prof = np.where(present, prof * amp + halo * amp, 0.0)
            temp = 2400.0 + 14000.0 * (h4 ** 2.0)
            col = blackbody_rgb(np.array(temp))
            out += prof[..., None] * col
    return out


def milky_way(d, seed=101):
    """A soft galactic band + nebula, aligned to a tilted great circle."""
    axis = np.array([0.32, -0.86, 0.40])
    axis = axis / np.linalg.norm(axis)
    z = d @ axis
    band = np.exp(-0.5 * (z / 0.19) ** 2)
    n = fbm3(d * 2.6 + 11.0, 6, seed)
    n2 = fbm3(d * 7.5 + 3.0, 5, seed + 91)
    neb = np.clip(n * 1.35 - 0.30, 0.0, None) ** 1.7
    neb *= (0.35 + 0.9 * np.clip(n2 * 1.6 - 0.4, 0.0, None))
    # dusty dark lanes
    dust = np.clip(fbm3(d * 5.0 - 21.0, 5, seed + 5) * 1.8 - 0.55, 0.0, None)
    glow = band * (0.55 + 2.2 * neb) * (1.0 - 0.75 * dust)
    warm = np.array([1.00, 0.80, 0.62])
    cool = np.array([0.42, 0.55, 1.00])
    mix = np.clip(fbm3(d * 1.9 + 40.0, 3, seed + 3) * 1.6 - 0.25, 0.0, 1.0)
    col = (warm[None, :] * (1.0 - mix[..., None]) + cool[None, :] * mix[..., None])
    faint = 0.030 * np.clip(fbm3(d * 3.4 + 77.0, 5, seed + 9) * 1.7 - 0.45, 0.0, None) ** 1.4
    return glow[..., None] * col * 0.075 + faint[..., None] * np.array([0.35, 0.42, 0.75])


def sky_color(d, star_boost=1.0, sky_gain=1.0):
    return (star_layer(d, boost=star_boost) + milky_way(d)) * sky_gain


# =====================================================================
#  Accretion disk
# =====================================================================

T_SCALE = 19000.0        # peak disk temperature scale [K]
# T(r) peaks at r = (49/36) r_in with T = 0.4884 T_scale.  Radiance is referred
# to the *visible* luminance of that peak, so a bright ring pixel sits near unit
# exposure while the cool outer disk correctly falls off in the infrared.
T_REF = 0.4884 * T_SCALE
# Visible-luminance normalisation: the reference blackbody's own luminance.
REF_LUM = float(blackbody_lum(np.array(T_REF)))


def disk_emissivity(r, phi, t, r_in, r_out, seed=3, t_scale=None):
    """Shakura-Sunyaev temperature profile + sheared turbulence.

        T(r) = t_scale * (r_in/r)^(3/4) * (1 - sqrt(r_in/r))^(1/4)

    which peaks at r = (49/36) r_in with T = 0.4884 t_scale.  The turbulent
    pattern is advected with the local Keplerian rate Omega ~ r^(-3/2), so the
    disk visibly shears as time advances.
    """
    if t_scale is None:
        t_scale = T_SCALE
    x = np.clip(r_in / r, 0.0, 1.0)
    T = t_scale * (x ** 0.75) * np.clip(1.0 - np.sqrt(x), 0.0, None) ** 0.25
    om = (r / r_in) ** -1.5
    a = phi - om * t
    # Filamentary turbulence.  The noise domain must span many cells in both
    # directions or the disk renders as a featureless sheet: across the annulus
    # log(r/r_in) runs 0..~1.1 and the azimuth 0..2pi, so the multipliers below
    # give ~9 radial cells and ~14 azimuthal ones, stretched so features are
    # long in azimuth (shear) and short in radius (ring-like).
    p = np.stack([np.log(r / r_in) * 8.0, a * 2.2,
                  np.log(r / r_in) * 3.1 + a * 0.55], -1)
    n = fbm3(p + 17.0, 5, seed)
    turb = 0.18 + 2.60 * np.clip(n * 1.55 - 0.34, 0.0, None) ** 0.85
    edge = np.clip((r_out - r) / (0.22 * r_out), 0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)
    return T, turb * edge


# =====================================================================
#  Geodesic integration
# =====================================================================

def _hermite_u(s, u1, up1, u2, up2, h):
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * u1 + h10 * h * up1 + h01 * u2 + h11 * h * up2


def _rhs(u, up):
    return up, -u + 1.5 * RS * u * u


def render(W=800, H=600, r0=28.0, incl_deg=80.0, fov_deg=40.0,
           r_in=R_ISCO, r_out=18.0, exposure=1.0, h0=0.012, max_steps=6000,
           star_boost=1.0, sky_gain=1.0, disk_gain=1.0, time_s=0.0,
           r_esc_factor=900.0, verbose=True, debug=False, t_scale=None):
    """Trace the full image.  Returns a linear-light float32 RGB array (H, W, 3).

    With debug=True also returns a dict of per-pixel diagnostics:
      kind  : 0 captured, 1 escaped to sky, 2 terminated on the disk
      rhit  : radius of the disk crossing (NaN where not applicable)
      ncr   : number of equatorial-plane crossings before termination
    """
    t_start = time.time()
    N = W * H
    if t_scale is None:
        t_scale = T_SCALE
    # Radiance reference: the visible luminance of the disk's peak temperature,
    # so a bright ring pixel lands near unit exposure.
    ref_lum = float(blackbody_lum(np.array(0.4884 * t_scale)))

    # ---- camera -----------------------------------------------------
    inc = math.radians(incl_deg)
    cam = r0 * np.array([math.sin(inc), 0.0, math.cos(inc)])
    cam_hat = cam / r0
    fwd = -cam_hat
    # Reference the image's vertical axis to the disk's rotation axis so the disk
    # projects to the conventional flattened ellipse (rotation axis vertical)
    # instead of a vertical band.
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(fwd, ref))) > 0.9995:
        ref = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, ref)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    tan_h = math.tan(0.5 * math.radians(fov_deg))
    aspect = W / H
    px = (2.0 * (np.arange(W) + 0.5) / W - 1.0) * aspect * tan_h
    py = (1.0 - 2.0 * (np.arange(H) + 0.5) / H) * tan_h
    PX, PY = np.meshgrid(px, py)
    dirs = (fwd[None, None, :] + PX[..., None] * right[None, None, :]
            + PY[..., None] * up[None, None, :])
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    d = dirs.reshape(N, 3)

    # ---- local orbital frame per ray --------------------------------
    # The camera is a *static* observer.  Its orthonormal frame is
    #     e_t = (1/sqrt(1-x)) d_t,  e_r = sqrt(1-x) d_r,  e_phi = (1/r) d_phi
    # with x = r_s/r0.  Decomposing the photon as p = E_loc (e_t + c e_r + s e_phi)
    # gives p^r = E c and p^phi = E s / (r sqrt(1-x)), hence
    #     b = L/E = r0 s / sqrt(1-x),      du/dphi|0 = -c sqrt(1-x) / (r0 s).
    # Dropping that sqrt factor tilts every ray and inflates the shadow by up
    # to ~7% at r0 = 15 M -- small enough to look plausible, large enough to be
    # wrong.  The end-to-end shadow test in validate_shadow() pins it down.
    c = d @ cam_hat                                  # cos(psi), psi from outward radial
    s = np.sqrt(np.clip(1.0 - c * c, 1e-14, None))
    t_hat = (d - c[:, None] * cam_hat[None, :]) / s[:, None]
    u = np.full(N, 1.0 / r0)
    sqrt_f = math.sqrt(max(1.0 - RS / r0, 1e-12))
    up = -c * sqrt_f / (r0 * s)
    phi = np.zeros(N)

    b = 1.0 / np.sqrt(up * up + u * u * (1.0 - RS * u))
    # Signed axial angular momentum of the *physical* photon, which travels the
    # reverse of the traced path: b_z = -b (n_hat . z_hat).  Getting this sign
    # wrong mirrors the Doppler map, putting the bright beamed side on the
    # receding limb.
    n_hat = np.cross(np.broadcast_to(cam_hat, (N, 3)), t_hat)
    bz = -b * (n_hat @ np.array([0.0, 0.0, 1.0]))
    ez_r = cam_hat[2]
    ez_t = t_hat[:, 2]

    # ---- state ------------------------------------------------------
    u_esc = 1.0 / (r_esc_factor * r0)
    u_h = 1.0 / RS
    out = np.zeros((N, 3), dtype=np.float64)
    idx = np.arange(N, dtype=np.int64)
    if debug:
        kind = np.zeros(N, dtype=np.int32)      # 0 captured (default)
        rhit = np.full(N, np.nan)
        ncr = np.zeros(N, dtype=np.int32)
        dT = np.zeros(N)        # emitted disk temperature at the crossing
        dg = np.zeros(N)        # relativistic shift factor g = nu_obs/nu_emit
        dturb = np.zeros(N)     # turbulence multiplier

    ez_r_a = np.broadcast_to(ez_r, (N,)).copy()
    bz_a = bz
    er3 = np.broadcast_to(cam_hat, (N, 3))

    nstep = 0
    for nstep in range(max_steps):
        if idx.size == 0:
            break
        uu = u[idx]
        pp = up[idx]
        ph = phi[idx]

        h = h0 / (1.0 + 6.0 * (M * uu) ** 2)

        k1u, k1p = _rhs(uu, pp)
        k2u, k2p = _rhs(uu + 0.5 * h * k1u, pp + 0.5 * h * k1p)
        k3u, k3p = _rhs(uu + 0.5 * h * k2u, pp + 0.5 * h * k2p)
        k4u, k4p = _rhs(uu + h * k3u, pp + h * k3p)
        u_new = uu + (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        p_new = pp + (h / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        ph_new = ph + h

        # ---- equatorial-plane crossing test -------------------------
        ezr = ez_r_a[idx]
        ezt = ez_t[idx]
        f_old = (ezr * np.cos(ph) + ezt * np.sin(ph)) / uu
        f_new = (ezr * np.cos(ph_new) + ezt * np.sin(ph_new)) / u_new
        captured = u_new >= u_h
        escaped = u_new <= u_esc
        cross = (f_old * f_new < 0.0) & ~captured & ~escaped
        if debug:
            ncr[idx[cross]] += 1

        hit_sel = np.zeros(idx.size, dtype=bool)
        hit_r = np.zeros(idx.size)
        hit_phi = np.zeros(idx.size)

        if np.any(cross):
            ci = np.nonzero(cross)[0]
            lo = np.zeros(ci.size)
            hi = np.ones(ci.size)
            hh = h[ci]
            u1, up1 = uu[ci], pp[ci]
            u2, up2 = u_new[ci], p_new[ci]
            p1 = ph[ci]
            f1 = f_old[ci]
            for _ in range(26):
                mid = 0.5 * (lo + hi)
                um = _hermite_u(mid, u1, up1, u2, up2, hh)
                pm = p1 + mid * hh
                fm = (ezr[ci] * np.cos(pm) + ezt[ci] * np.sin(pm)) / um
                same = (fm * f1) > 0.0
                lo = np.where(same, mid, lo)
                hi = np.where(same, hi, mid)
            sm = 0.5 * (lo + hi)
            um = _hermite_u(sm, u1, up1, u2, up2, hh)
            pm = p1 + sm * hh
            rh = 1.0 / um
            inside = (rh >= r_in) & (rh <= r_out)
            if np.any(inside):
                sel = ci[inside]
                hit_sel[sel] = True
                hit_r[sel] = rh[inside]
                hit_phi[sel] = pm[inside]

        # ---- write terminated rays ----------------------------------
        done = captured | escaped | hit_sel
        if np.any(done):
            di = np.nonzero(done)[0]
            gidx = idx[di]
            # escaped -> sky
            esc_m = escaped[di]
            if np.any(esc_m):
                e_idx = gidx[esc_m]
                if debug:
                    kind[e_idx] = 1
                ue = u_new[di][esc_m]
                pe = p_new[di][esc_m]
                phe = ph_new[di][esc_m]
                drdp = -pe / (ue * ue)
                rr = 1.0 / ue
                v_r = drdp * np.cos(phe) - rr * np.sin(phe)
                v_t = drdp * np.sin(phe) + rr * np.cos(phe)
                dd = v_r[:, None] * er3[e_idx] + v_t[:, None] * t_hat[e_idx]
                dd /= np.linalg.norm(dd, axis=-1, keepdims=True)
                out[e_idx] = sky_color(dd, star_boost, sky_gain)
            # disk -> emission
            hm = hit_sel[di]
            if np.any(hm):
                h_idx = gidx[hm]
                rh = hit_r[di][hm]
                phh = hit_phi[di][hm]
                if debug:
                    kind[h_idx] = 2
                    rhit[h_idx] = rh
                T, turb = disk_emissivity(rh, phh, time_s, r_in, r_out,
                                          t_scale=t_scale)
                om = np.sqrt(M / rh ** 3)
                ut = 1.0 / np.sqrt(np.clip(1.0 - 3.0 * M / rh, 1e-6, None))
                g = 1.0 / (ut * (1.0 - bz_a[h_idx] * om))
                g = np.clip(g, 1e-4, None)
                if debug:
                    dT[h_idx] = T
                    dg[h_idx] = g
                    dturb[h_idx] = turb
                # The observed spectrum is a blackbody at the shifted
                # temperature T_obs = g * T.  Its colour is that blackbody's
                # chromaticity and its brightness is that blackbody's *visible*
                # luminance, which already carries the g^4 beaming factor and
                # the infrared loss of cool gas.
                T_obs = g * T
                out[h_idx] = (blackbody_rgb(T_obs)
                              * (blackbody_lum(T_obs) / ref_lum * turb * disk_gain)[:, None])
            # captured stays black
            keep = ~done
            idx = idx[keep]
            u[idx] = u_new[keep]
            up[idx] = p_new[keep]
            phi[idx] = ph_new[keep]
        else:
            u[idx] = u_new
            up[idx] = p_new
            phi[idx] = ph_new

    if idx.size:
        # rays still winding after the step budget are on the critical orbit:
        # they fall in.  Leave them black.
        pass

    if verbose:
        print(f"  traced in {time.time() - t_start:6.2f}s   "
              f"steps used {nstep + 1}   unabsorbed {idx.size}")

    img = out.reshape(H, W, 3).astype(np.float32)
    if debug:
        info = {"kind": kind.reshape(H, W), "rhit": rhit.reshape(H, W),
                "ncr": ncr.reshape(H, W), "stuck": idx.size,
                "dT": dT.reshape(H, W), "dg": dg.reshape(H, W),
                "dturb": dturb.reshape(H, W)}
        return img, info
    return img


# =====================================================================
#  Tone mapping
# =====================================================================

def aces_approx(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def tonemap(img, exposure=1.0, bloom=0.0):
    x = np.clip(img * exposure, 0.0, None)
    if bloom > 0.0:
        blur = x.copy()
        for _ in range(4):
            blur = 0.25 * (np.roll(blur, 1, 0) + np.roll(blur, -1, 0)
                           + np.roll(blur, 1, 1) + np.roll(blur, -1, 1))
        bright = np.clip(x - 1.0, 0.0, None)
        b2 = bright.copy()
        for _ in range(10):
            b2 = 0.25 * (np.roll(b2, 1, 0) + np.roll(b2, -1, 0)
                         + np.roll(b2, 1, 1) + np.roll(b2, -1, 1))
        x = x + bloom * b2
    return aces_approx(x) ** (1.0 / 2.2)


def save_png(img_linear, path, exposure=1.0, bloom=0.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    srgb = tonemap(img_linear, exposure, bloom)
    arr = (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    return path


# =====================================================================
#  Physics validation
# =====================================================================

def _sweep_phi(b, r0, h0=0.02, max_steps=4_000_000, bisect_turn=True):
    """Deflection angle of a null geodesic with impact parameter b, launched
    inward from radius r0.

    The orbit is symmetric about its perihelion, so we integrate only as far as
    the turning point (du/dphi = 0) and double the swept angle:

        alpha = 2 * phi_turn - (pi - 2 asin(b/r0))

    The last parenthesis is the angle a *straight* line at the same impact
    parameter sweeps between the same two radii.  Locating the turning point by
    interpolation inside the final step is what makes the scheme converge at
    RK4's proper fourth order -- stopping at a step boundary instead leaves a
    first-order error that swamps everything else.
    """
    u = 1.0 / r0
    disc = 1.0 / (b * b) - u * u * (1.0 - RS * u)
    up = math.sqrt(max(disc, 0.0))            # inward => du/dphi > 0
    phi = 0.0
    u_h = 1.0 / RS

    for _ in range(max_steps):
        h = h0 / (1.0 + 2.0 * M * u)
        k1u, k1p = _rhs(u, up)
        k2u, k2p = _rhs(u + 0.5 * h * k1u, up + 0.5 * h * k1p)
        k3u, k3p = _rhs(u + 0.5 * h * k2u, up + 0.5 * h * k2p)
        k4u, k4p = _rhs(u + h * k3u, up + h * k3p)
        du = (h / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        dp = (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        u_n, up_n = u + du, up + dp

        if u_n >= u_h:                        # fell through the horizon
            return phi, "captured"

        if up_n <= 0.0:                       # turning point inside this step
            s = up / (up - up_n) if up != up_n else 1.0
            if bisect_turn:
                lo, hi = 0.0, 1.0
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    um = _hermite_u(mid, u, up, u_n, up_n, h)
                    upm = _hermite_du(mid, u, up, u_n, up_n, h)
                    if upm > 0.0:
                        lo = mid
                    else:
                        hi = mid
                s = 0.5 * (lo + hi)
            phi_turn = phi + s * h
            straight = math.pi - 2.0 * math.asin(min(b / r0, 1.0))
            return 2.0 * phi_turn - straight, "escaped"

        u, up, phi = u_n, up_n, phi + h

    return phi, "budget"


def _hermite_du(s, u1, up1, u2, up2, h):
    """d/ds of the cubic Hermite interpolant used for root finding."""
    s2 = s * s
    return ((6.0 * s2 - 6.0 * s) * u1 + (3.0 * s2 - 4.0 * s + 1.0) * h * up1
            + (-6.0 * s2 + 6.0 * s) * u2 + (3.0 * s2 - 2.0 * s) * h * up2)


def exact_deflection_quad(b, r0=None, nodes=600):
    """Exact deflection from the first integral, by Gauss-Legendre quadrature:

        alpha = 2 * integral_{u0}^{u1} du / sqrt(1/b^2 - u^2 + 2M u^3)
                - (pi - 2 asin(b/r0))

    with u1 the perihelion root.  Substituting u = u1 (1 - s^2) removes the
    integrable 1/sqrt(u1-u) endpoint singularity, after which the quadrature is
    spectrally accurate.  Entirely independent of the RK4 scheme above.
    """
    coeffs = [2.0 * M, -1.0, 0.0, 1.0 / (b * b)]
    rts = np.roots(coeffs)
    real = np.sort(np.real(rts[np.abs(np.imag(rts)) < 1e-9 * max(1.0, b)]))
    pos = real[real > 0.0]
    u1 = pos[0]
    u0 = 0.0 if r0 is None else 1.0 / r0

    x, w = np.polynomial.legendre.leggauss(nodes)
    s = 0.5 * (x + 1.0)
    ws = 0.5 * w
    # Restrict the quadrature domain to u >= u0 *exactly* by rescaling the
    # substitution, so no grid mask is needed and spectral accuracy survives:
    #   u = u1 * (1 - s^2),  s in [0, s_cut]  ->  s = s_cut * sigma, sigma in [0,1]
    s_cut = math.sqrt(max(0.0, 1.0 - u0 / u1)) if r0 is not None else 1.0
    s = s * s_cut
    ws = ws * s_cut
    u = u1 * (1.0 - s * s)
    g = 1.0 / (b * b) - u * u + 2.0 * M * u ** 3
    g = np.clip(g, 0.0, None)
    integrand = 2.0 * u1 * s / np.sqrt(g + 1e-300)
    total = float(np.sum(ws * integrand))

    if r0 is None:
        return 2.0 * total - math.pi
    straight = math.pi - 2.0 * math.asin(min(b / r0, 1.0))
    return 2.0 * total - straight


def pn_series(b):
    """Post-Newtonian / strong-deflection series in x = M/b (Keeton & Petters)."""
    x = M / b
    return (4 * x + 15 * math.pi / 4 * x ** 2 + 128.0 / 3 * x ** 3
            + 3465 * math.pi / 64 * x ** 4 + 3584.0 / 5 * x ** 5
            + 315 * math.pi * x ** 6)


def validate_shadow(verbose=True):
    """End-to-end test of the actual renderer.

    A static observer at r0 sees the shadow with angular radius

        sin(theta_sh) = b_crit * sqrt(1 - r_s/r0) / r0 ,   b_crit = 3 sqrt(3) M

    We measure the largest polar angle among captured pixels and compare.  This
    exercises the camera frame, the initial conditions, the integrator and the
    capture test together -- the one test that catches a wrong sqrt(1-r_s/r0)
    or a mis-signed radial momentum.
    """
    rows = []
    W = H = 360
    fov = 36.0
    tanh = math.tan(0.5 * math.radians(fov))
    py = (1.0 - 2.0 * (np.arange(H) + 0.5) / H) * tanh
    px = (2.0 * (np.arange(W) + 0.5) / W - 1.0) * tanh
    PX, PY = np.meshgrid(px, py)
    ang = np.arctan(np.sqrt(PX ** 2 + PY ** 2))
    for r0 in (12.0, 20.0, 40.0, 100.0):
        img, info = render(W=W, H=H, r0=r0, incl_deg=80.0, fov_deg=fov,
                           r_in=R_ISCO, r_out=R_ISCO, verbose=False, debug=True)
        cap = info["kind"] == 0
        meas = float(ang[cap].max())
        ana = math.asin(min(B_CRIT * math.sqrt(max(1.0 - RS / r0, 0.0)) / r0, 1.0))
        rows.append((r0, meas, ana, abs(meas - ana) / ana))
    if verbose:
        print("  shadow angular radius: renderer vs exact formula")
        print("       r0/M     measured       analytic       rel.err")
        for r0, m, a, e in rows:
            print(f"  {r0:9.1f}  {math.degrees(m):11.6f}d  {math.degrees(a):11.6f}d"
                  f"   {e:.2e}")
    return rows


def validate_deflection(verbose=True):
    """Three independent methods must agree on the light-bending angle."""
    r0 = 1.0e6
    rows = []
    for bM in (5.5, 8.0, 20.0, 100.0, 1000.0):
        b = bM * M
        a_ode, how = _sweep_phi(b, r0)
        a_quad = exact_deflection_quad(b, r0=r0)
        a_ser = pn_series(b)
        rows.append((bM, a_ode, a_quad, a_ser, abs(a_ode - a_quad)))
    if verbose:
        print("  light deflection  alpha  [rad]")
        print("       b/M   RK4 geodesic    exact quadrature     PN series    |dRK4|")
        for bM, ao, aq, asr, d in rows:
            print(f"  {bM:8.1f}   {ao:12.9f}     {aq:12.9f}   {asr:12.9f}   {d:.2e}")
    return rows


def validate_convergence(verbose=True):
    """RK4 vs the exact quadrature: report the agreement floor and confirm the
    residual is round-off, not truncation (refining the step must not move it)."""
    r0 = 1.0e6
    out = []
    for bM in (8.0, 100.0, 1000.0):
        b = bM * M
        aq = exact_deflection_quad(b, r0=r0, nodes=800)
        errs = []
        for h0 in (0.02, 0.01, 0.005):
            a, _ = _sweep_phi(b, r0, h0=h0)
            errs.append(abs(a - aq))
        if verbose:
            print(f"  b = {bM:6.1f} M   |RK4 - quadrature| = "
                  + "  ".join(f"{e:.2e}" for e in errs)
                  + "   (flat => round-off floor, not truncation)")
        out.append((bM, errs))
    return out


def validate_critical_b(verbose=True):
    """b_crit is where the first integral has a double root:
       max_u [ u^2 - 2M u^3 ] = 1/b^2  at  u = 1/(3M)   =>   b = 3 sqrt(3) M.
       Check the analytic maximum numerically, then bisect the RK4 capture
       threshold against it."""
    us = np.linspace(1e-6, 1.0 / RS, 4_000_001)
    hh = us * us - 2.0 * M * us ** 3
    i = int(np.argmax(hh))
    b_from_max = 1.0 / math.sqrt(hh[i])
    u_at_max = us[i]

    lo, hi = 3.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        _, how = _sweep_phi(mid * M, 200.0, h0=0.004, max_steps=2_000_000)
        if how in ("captured", "budget"):
            lo = mid
        else:
            hi = mid
    bc = 0.5 * (lo + hi) * M
    if verbose:
        print("  photon sphere / critical impact parameter")
        print(f"    argmax of u^2-2Mu^3 at u = {u_at_max:.9f}   (exact 1/(3M) = {1/3:.9f})")
        print(f"    b_crit from that maximum : {b_from_max:.9f} M")
        print(f"    b_crit by bisecting RK4  : {bc:.9f} M")
        print(f"    exact 3*sqrt(3) M        : {B_CRIT:.9f} M")
        print(f"    rel.err (RK4 vs exact)   : {abs(bc - B_CRIT) / B_CRIT:.3e}")
    return bc


def validate_photon_sphere(verbose=True):
    """A circular photon orbit must sit exactly at r = 3M.
       Circular orbit: u' = 0 and u'' = 0  =>  1/b^2 = u^2(1-rs u), u = 1.5 rs u^2."""
    u_c = 1.0 / (1.5 * RS)
    r_c = 1.0 / u_c
    if verbose:
        print(f"  circular photon orbit from u''=0 : r = {r_c:.9f} M   (exact 3.0)")
    return r_c


# =====================================================================
#  main
# =====================================================================

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "out")
    argv = sys.argv[1:]

    if any(a.startswith("--validate") for a in argv) or not argv:
        print("=" * 68)
        print(" physics validation")
        print("=" * 68)
        validate_deflection()
        print()
        validate_convergence()
        print()
        validate_critical_b()
        validate_photon_sphere()
        print()
        validate_shadow()
        print()

    if "--validate-only" in argv:
        return

    W = int(next((a.split("=")[1] for a in argv if a.startswith("--w=")), 800))
    H = int(next((a.split("=")[1] for a in argv if a.startswith("--h=")), 600))
    expo = float(next((a.split("=")[1] for a in argv if a.startswith("--expo=")), 1.0))
    tag = next((a.split("=")[1] for a in argv if a.startswith("--tag=")), "ref")

    print(f"rendering {W}x{H} ...")
    img = render(W=W, H=H, exposure=expo)
    p = save_png(img, os.path.join(outdir, f"bh_{tag}.png"), exposure=expo, bloom=0.5)
    print("wrote", os.path.abspath(p))


if __name__ == "__main__":
    main()
