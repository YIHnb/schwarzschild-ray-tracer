# -*- coding: utf-8 -*-
"""Render the hero stills: high-resolution, physically validated frames."""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grbh

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")

SHOTS = [
    # name,        W,    H,   r0,  incl, fov, r_out, expo, t_scale
    ("hero",      1000,  750, 40.0, 80.0, 38.0, 16.0, 1.00, 11000.0),
    ("edge",       820,  615, 55.0, 88.5, 30.0, 20.0, 1.00, 11000.0),
    ("closeup",    820,  615, 16.0, 74.0, 55.0, 14.0, 1.00, 11000.0),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, W, H, r0, inc, fov, rout, expo, ts in SHOTS:
        print(f"[{name}] {W}x{H} r0={r0} incl={inc} fov={fov}")
        img = grbh.render(W=W, H=H, r0=r0, incl_deg=inc, fov_deg=fov,
                          r_out=rout, exposure=expo, t_scale=ts,
                          star_boost=3.0, sky_gain=2.5)
        p = os.path.join(OUT, f"{name}.png")
        grbh.save_png(img, p, exposure=expo, bloom=0.25)
        print("   wrote", os.path.abspath(p))


if __name__ == "__main__":
    main()
