# -*- coding: utf-8 -*-
"""Parameter sweep -> single contact sheet, so framing/colour can be judged at a glance."""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grbh

VARIANTS = [
    # tag,       r0,  incl, fov, r_out, expo, t_scale
    ("A_19k",    40.0, 80.0, 38.0, 16.0, 1.0, 19000.0),
    ("B_11k",    40.0, 80.0, 38.0, 16.0, 1.0, 11000.0),
    ("C_7k",     40.0, 80.0, 38.0, 16.0, 1.0,  7000.0),
    ("D_11k_w",  60.0, 76.0, 34.0, 26.0, 1.0, 11000.0),
]

W, H = 420, 315


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "out")
    tiles = []
    for tag, r0, inc, fov, rout, expo, ts in VARIANTS:
        print(f"[{tag}] r0={r0} incl={inc} fov={fov} r_out={rout} t_scale={ts}")
        img = grbh.render(W=W, H=H, r0=r0, incl_deg=inc, fov_deg=fov,
                          r_out=rout, exposure=expo, t_scale=ts,
                          star_boost=3.0, sky_gain=2.5)
        srgb = grbh.tonemap(img, exposure=expo, bloom=0.25)
        arr = (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        arr[:14, :, :] = 40
        tiles.append(arr)

    sheet = np.concatenate([np.concatenate(tiles[:2], axis=1),
                            np.concatenate(tiles[2:], axis=1)], axis=0)
    p = os.path.join(outdir, "sweep.png")
    Image.fromarray(sheet, "RGB").save(p)
    print("wrote", os.path.abspath(p))
    for i, v in enumerate(VARIANTS):
        r, c = divmod(i, 2)
        print(f"  tile ({r},{c}) = {v[0]}")


if __name__ == "__main__":
    main()
