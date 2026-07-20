#!/usr/bin/env python3
"""Dump Interlace() output from the VapourSynth frontend for test_interlace_avs.

Generates the same synthetic clip the AviSynth+ test builds, runs it through
the VapourSynth Interlace(), and writes each output plane as raw rows:

    <outdir>/<tag>_f<NNN>p<P>.raw

Usage:
    python3 test/dump_vs_reference.py build/interlace.so /tmp/vsref
"""

import os
import sys

import numpy as np
import vapoursynth as vs

# (tag, VS format, AviSynth+ pixel_type) — must match test_interlace_avs.c
FORMATS = [
    ("yv12", vs.YUV420P8, "YV12"),
    ("yv24", vs.YUV444P8, "YV24"),
    ("yuv420p16", vs.YUV420P16, "YUV420P16"),
]

WIDTH, HEIGHT, LENGTH = 96, 32, 12

# per-plane (xmul, ymul, nmul), matching the Expr strings in the AVS test
COEFS = [(7, 13, 29), (3, 5, 11), (17, 19, 23)]


def make_clip(core, fmt):
    """blank clip -> per-plane ramp, identical to the AviSynth+ Expr."""
    blank = core.std.BlankClip(format=fmt, width=WIDTH, height=HEIGHT,
                               length=LENGTH, keep=True)

    def fill(n, f):
        out = f.copy()
        for p in range(out.format.num_planes):
            xm, ym, nm = COEFS[p]
            a = np.asarray(out[p])
            h, w = a.shape
            x = np.arange(w, dtype=np.int64)[None, :]
            y = np.arange(h, dtype=np.int64)[:, None]
            a[:] = ((x * xm + y * ym + n * nm) % 256).astype(a.dtype)
        return out

    return core.std.ModifyFrame(blank, blank, fill)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    plugin, outdir = sys.argv[1], sys.argv[2]

    core = vs.core
    core.std.LoadPlugin(path=os.path.abspath(plugin))
    os.makedirs(outdir, exist_ok=True)

    for tag, fmt, _avs in FORMATS:
        clip = make_clip(core, fmt)
        out = core.interlace.Interlace(clip, tff=1, filter=1)
        for n, frame in enumerate(out.frames()):
            for p in range(frame.format.num_planes):
                a = np.asarray(frame[p])
                path = os.path.join(outdir, f"{tag}_f{n:03d}p{p}.raw")
                with open(path, "wb") as fh:
                    fh.write(a.tobytes())
        print(f"{tag}: {out.num_frames} frames, "
              f"{out.format.num_planes} planes -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
