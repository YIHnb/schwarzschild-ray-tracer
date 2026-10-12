# -*- coding: utf-8 -*-
"""Per-pixel ray-fate diagnostics.

Renders the same frame four ways so the geometry can be read off directly:
  top-left     tonemapped image
  top-right    ray fate  (black = captured, blue = escaped to sky, fire = disk)
  bottom-left  disk-crossing radius (viridis-ish ramp, NaN = never hit)
  bottom-right number of equatorial crossings (shows the higher-order images)
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grbh

W, H = 440, 330
KW = dict(r0=48.0, incl_deg=78.0, fov_deg=30.0, r_out=16.0)


def ramp(t):
    """Simple perceptually-reasonable ramp (dark blue -> cyan -> yellow -> white)."""
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.6 * t - 0.35, 0, 1)
    g = np.clip(1.5 * t - 0.10, 0, 1)
    b = np.clip(1.9 * t + 0.45 - 2.2 * t ** 2, 0, 1)
    return np.stack([r, g, b], -1)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "out")
    img, info = grbh.render(W=W, H=H, debug=True, **KW)
    kind, rhit, ncr = info["kind"], info["rhit"], info["ncr"]

    print("ray fate census:")
    tot = kind.size
    for k, name in ((0, "captured (shadow)"), (1, "escaped to sky"), (2, "hit disk")):
        n = int((kind == k).sum())
        print(f"   {name:20s} {n:8d}  {100.0*n/tot:6.2f}%")
    print(f"   still winding       {info['stuck']:8d}  {100.0*info['stuck']/tot:6.2f}%")

    hit = kind == 2
    if hit.any():
        rr = rhit[hit]
        print(f"disk-hit radius: min {rr.min():.3f}  max {rr.max():.3f}  "
              f"mean {rr.mean():.3f}   r_in=6 r_out=16")
    print("equatorial crossings among disk hits:",
          {int(k): int(v) for k, v in zip(*np.unique(ncr[hit], return_counts=True))})
    print("equatorial crossings among sky rays:",
          {int(k): int(v) for k, v in zip(*np.unique(ncr[kind == 1], return_counts=True))})

    # --- panel 1: image
    p1 = (np.clip(grbh.tonemap(img, exposure=0.35, bloom=0.6), 0, 1) * 255).astype(np.uint8)

    # --- panel 2: fate
    fate = np.zeros((H, W, 3))
    fate[kind == 1] = (0.15, 0.35, 0.95)
    fate[hit] = (1.0, 0.45, 0.05)
    p2 = (fate * 255).astype(np.uint8)

    # --- panel 3: crossing radius
    rr = np.where(hit, (rhit - 6.0) / 10.0, 0.0)
    p3 = (ramp(rr) * 255).astype(np.uint8)

    # --- panel 4: crossing count
    p4 = (ramp(ncr / 3.0) * 255).astype(np.uint8)

    sheet = np.concatenate([
        np.concatenate([p1, p2], axis=1),
        np.concatenate([p3, p4], axis=1)], axis=0)
    p = os.path.join(outdir, "diag.png")
    Image.fromarray(sheet, "RGB").save(p)
    print("wrote", os.path.abspath(p))


if __name__ == "__main__":
    main()
