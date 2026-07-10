#!/usr/bin/env python3
# Rank zimg downscalers for 2160p -> 1080i: pass-band MTF and alias
# suppression, per axis. In a 2:1 downscale, source content at f x 1080p
# Nyquist (1 < f < 2) aliases to (2 - f); vertically the interlacing filter
# then attenuates whatever lands above its cutoff, horizontally nothing does.
# Synthesizes banded 2160p gratings, measures RMS amplitude after each path.
#
# usage: resizer_alias.py interlace.so

import sys

import numpy as np
import vapoursynth as vs

core = vs.core
core.std.LoadPlugin(sys.argv[1])

W, H = 3840, 2160
AMP, MID = 360, 512
PASS_F = [0.50, 0.70, 0.85, 0.95]                  # x 1080p Nyquist
ALIAS_F = [1.05, 1.20, 1.35, 1.50, 1.65, 1.80, 1.95]
FREQS = PASS_F + ALIAS_F

RESIZERS = [
    ('Bilinear',  lambda c: core.resize.Bilinear(c, 1920, 1080)),
    ('Bicubic b=0 c=.5', lambda c: core.resize.Bicubic(c, 1920, 1080)),
    ('Mitchell',  lambda c: core.resize.Bicubic(c, 1920, 1080,
                                                filter_param_a=1/3, filter_param_b=1/3)),
    ('Spline16',  lambda c: core.resize.Spline16(c, 1920, 1080)),
    ('Spline36',  lambda c: core.resize.Spline36(c, 1920, 1080)),
    ('Spline64',  lambda c: core.resize.Spline64(c, 1920, 1080)),
    ('Lanczos3',  lambda c: core.resize.Lanczos(c, 1920, 1080, filter_param_a=3)),
    ('Lanczos4',  lambda c: core.resize.Lanczos(c, 1920, 1080, filter_param_a=4)),
]


def bands_v():
    # horizontal-line gratings stacked in row bands: vertical frequencies
    img = np.full((H, W), MID, np.float64)
    bh = H // len(FREQS)
    y = np.arange(H)
    for i, f in enumerate(FREQS):
        vf = f * 0.25                                # c/line at 2160p
        rows = slice(i * bh, (i + 1) * bh)
        img[rows] = MID + AMP * np.sin(2 * np.pi * vf * y[rows])[:, None]
    return np.clip(img, 0, 1023).astype(np.uint16), bh


def bands_h():
    # vertical-bar gratings in column bands: horizontal frequencies
    img = np.full((H, W), MID, np.float64)
    bw = W // len(FREQS)
    x = np.arange(W)
    for i, f in enumerate(FREQS):
        hf = f * 0.25
        cols = slice(i * bw, (i + 1) * bw)
        img[:, cols] = MID + AMP * np.sin(2 * np.pi * hf * x[cols])[None, :]
    return np.clip(img, 0, 1023).astype(np.uint16), bw


def clip_of(img):
    src = core.std.BlankClip(format=vs.GRAY10, width=W, height=H, length=6,
                             fpsnum=50, fpsden=1)
    def fill(n, f):
        fo = f.copy()
        np.copyto(np.asarray(fo[0]), img)
        return fo
    return core.std.ModifyFrame(src, src, fill)


def db(x):
    return 20 * np.log10(max(abs(x) / AMP, 1e-6))


def amp(region):
    return np.sqrt(2.0) * region.astype(np.float64).std()


vimg, bh = bands_v()
himg, bw = bands_h()
vclip, hclip = clip_of(vimg), clip_of(himg)

print('response in dB relative to the source grating; pass bands want 0,')
print('alias bands want -inf. frequencies are multiples of 1080p Nyquist.')
hdr = ' '.join(f'{f:5.2f}' for f in FREQS)
for name, fn in RESIZERS:
    rows = {}
    hres = np.asarray(fn(hclip).get_frame(0)[0])          # resizer only
    vres = np.asarray(fn(vclip).get_frame(0)[0])
    vcmp = np.asarray(core.interlace.Interlace(fn(vclip)).get_frame(1)[0])
    obh, obw = bh // 2, bw // 2
    rows['H resize'] = [db(amp(hres[200:880, i * obw + 40:(i + 1) * obw - 40]))
                        for i in range(len(FREQS))]
    rows['V resize'] = [db(amp(vres[i * obh + 30:(i + 1) * obh - 30, 200:1700]))
                        for i in range(len(FREQS))]
    rows['V + VT  '] = [db(amp(vcmp[i * obh + 30:(i + 1) * obh - 30, 200:1700]))
                        for i in range(len(FREQS))]
    print(f'\n{name}:  freq {hdr}')
    for k, v in rows.items():
        print(f'  {k}   ' + ' '.join(f'{x:5.1f}' for x in v))
