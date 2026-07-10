#!/usr/bin/env python3
# EBU TECH 3384 measurements (sec. 2.2) through the Interlace plugin.
#
# Default: patterns synthesized from the published construction (10-bit R103
# levels, sinusoidal gratings at 13.5/26.9/40.4/53.9/67.3/80.8% of Nyquist,
# vertical-temporal zone plate, 0/512/940 step crosses, colour squares) —
# the official files require an EBU login.
# --official: measure the EBU "Interlace and Scaling Test 1080p" file itself
# (v210 .mov, decoded via ffmpeg). Analysis cells' (V,T) frequencies are
# measured from the input, so no assumptions about the pattern's frequency
# ramps enter the comparison.
#
# usage: tech3384_validate.py interlace.so [--template template.png]
#                             [--official file.mov] [--ffmpeg ffmpeg]

import pathlib
import re
import subprocess
import sys

import numpy as np
import vapoursynth as vs

core = vs.core

args = sys.argv[1:]
TEMPLATE, OFFICIAL, FFMPEG = None, None, 'ffmpeg'
if '--template' in args:
    from PIL import Image
    i = args.index('--template')
    TEMPLATE = np.asarray(Image.open(args[i + 1]).convert('L'))
    del args[i:i + 2]
if '--ffmpeg' in args:
    i = args.index('--ffmpeg'); FFMPEG = args[i + 1]; del args[i:i + 2]
if '--official' in args:
    i = args.index('--official'); OFFICIAL = args[i + 1]; del args[i:i + 2]
core.std.LoadPlugin(args[0])

GRATING_PCT = [13.5, 26.9, 40.4, 53.9, 67.3, 80.8]

_hdr = (pathlib.Path(__file__).parent / '../src/vt_coeffs.h').read_text()
_body = _hdr.split('#define VT_COEFS_F32')[1].split('\n\n')[0].replace('\\\n', ' ')
CF = np.array([float(x.strip().rstrip('f'))
               for x in _body.split(',') if x.strip()][:95]).reshape(5, 19)


def resp(v, t):
    h = 0.0
    for n in range(5):
        for m in range(19):
            h += CF[n][m] * np.cos(2 * np.pi * (m - 9) * v) * np.cos(2 * np.pi * (n - 2) * t)
    return h


def db(x):
    return 20 * np.log10(max(abs(x), 1e-9))


def fpeak(sig, pad=8):
    sig = sig - sig.mean()
    f = np.abs(np.fft.rfft(sig * np.hanning(len(sig)), len(sig) * pad))
    return np.argmax(f) / (len(sig) * pad)


def amp(a):
    return np.sqrt(2.0) * a.astype(np.float64).std()


def clip_from(frames, fmt):
    n = len(frames)
    h, w = frames[0][0].shape
    src = core.std.BlankClip(format=fmt, width=w, height=h, length=n, fpsnum=50, fpsden=1)
    def fill(n, f, frames=frames):
        fo = f.copy()
        for p in range(len(frames[n])):
            np.copyto(np.asarray(fo[p]), frames[n][p])
        return fo
    return core.std.ModifyFrame(src, src, fill)


def template_boundary(t):
    col = min(int(round(t / 0.5 * 255)), 255)
    return 0.5 * (255 - np.where(TEMPLATE[:, col] == 255)[0][0]) / 255


def synthetic():
    def fields(clip):
        out = []
        for n in range(clip.num_frames):
            a = np.asarray(clip.get_frame(n)[0]).copy()
            out.append((a[0::2], a[1::2]))
        return out

    H, NFRAMES = 1080, 12
    print('== 2.2.1 frequency gratings (static, V and H) ==')
    print('grating  freq%%   %-10s %-10s %s' % ('measured', 'designed', 'H-grating identity'))
    for i, pct in enumerate(GRATING_PCT):
        vfreq = 0.5 * pct / 100
        y = np.arange(H)
        vg = np.clip(502 + 438 * np.sin(2 * np.pi * vfreq * y), 0, 1023).astype(np.uint16)
        frames = [[np.tile(vg[:, None], (1, 64))]] * NFRAMES
        out = fields(core.interlace.Interlace(clip_from(frames, vs.GRAY10)))
        meas = amp(out[3][0][40:500]) / amp(frames[0][0][0::2][40:500])

        x = np.arange(256)
        hg = np.clip(502 + 438 * np.sin(2 * np.pi * vfreq * x), 0, 1023).astype(np.uint16)
        hframes = [[np.tile(hg[None, :], (64, 1))]] * NFRAMES
        hout = core.interlace.Interlace(clip_from(hframes, vs.GRAY10)).get_frame(2)
        hdiff = np.abs(np.asarray(hout[0]).astype(int) - hframes[0][0].astype(int)).max()
        print('   %d     %4.1f   %6.2f dB  %6.2f dB   max diff %d code' %
              (i + 1, pct, db(meas), db(resp(vfreq, 0.0)), hdiff))

    print()
    print('== 2.2.2 vertical-temporal zone plate ==')
    W, NZ = 480, 48
    row = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    yb = (H - 1 - row).astype(np.float64)
    vphase = 0.5 * yb * yb / (2 * H)
    tfreq = 0.5 * x / W
    zp = [np.clip(512 + 400 * np.sin(2 * np.pi * (vphase + tfreq * t)), 0, 1023).astype(np.uint16)
          for t in range(NZ)]
    zclip = clip_from([[f] for f in zp], vs.GRAY10)
    flt = fields(core.interlace.Interlace(zclip))
    ref = fields(core.interlace.Interlace(zclip, filter=0))

    tops_f = np.stack([f[0] for f in flt[4:20]]).astype(np.float64)
    tops_r = np.stack([f[0] for f in ref[4:20]]).astype(np.float64)
    print('T(c/frame)  measured -6dB V cutoff' + ('   template boundary' if TEMPLATE is not None else ''))
    worst = 0.0
    for xc in range(8, W - 8, 16):
        att = []
        for y0 in range(0, 540 - 8, 8):
            a_f = tops_f[:, y0:y0 + 8, xc - 8:xc + 8].std()
            a_r = tops_r[:, y0:y0 + 8, xc - 8:xc + 8].std()
            att.append(a_f / a_r if a_r > 1 else 1.0)
        att = np.array(att)
        yb_line = H - 1 - (np.arange(0, 540 - 8, 8) * 2 + 4)
        vf = 0.5 * yb_line / H
        below = np.where(att < 0.5)[0]
        v50 = vf[below].min() if len(below) else 0.5
        t = 0.5 * xc / W
        line = f'  {t:.3f}       {v50:.3f}'
        if TEMPLATE is not None:
            vb = template_boundary(t)
            worst = max(worst, abs(v50 - vb))
            line += f'              {vb:.3f}'
        print(line)
    if TEMPLATE is not None:
        print(f'max |measured - template| = {worst:.3f} (transition band spans ~0.05,'
              ' 8-row measurement window ~0.007)')

    print()
    print('== 2.2.3 step ringing (vertical steps; horizontal steps must be untouched) ==')
    for level, name in ((0, '512->0'), (940, '512->940')):
        img = np.full((H, 64), 512, np.uint16)
        img[500:580] = level
        out = fields(core.interlace.Interlace(clip_from([[img]] * NFRAMES, vs.GRAY10)))
        woven = np.empty(H)
        woven[0::2], woven[1::2] = out[3][0][:, 32], out[3][1][:, 32]
        lo, hi = min(512, level), max(512, level)
        over = int(woven.max() - hi)
        under = int(lo - woven.min())
        print(f'  {name}: overshoot +{over}, undershoot -{under} codes'
              f' ({100 * over / (hi - lo):.1f}% / {100 * under / (hi - lo):.1f}% of step)')

    vimg = np.full((64, 256), 512, np.uint16)
    vimg[:, 100:140] = 940
    vout = core.interlace.Interlace(clip_from([[vimg]] * NFRAMES, vs.GRAY10)).get_frame(3)
    print('  horizontal step (vertical arm): max diff',
          np.abs(np.asarray(vout[0]).astype(int) - vimg.astype(int)).max(), 'code')

    print()
    print('== 2.2.4 chroma ringing and timing (YUV422P10) ==')
    y = np.full((H, 64), 940, np.uint16)
    u = np.full((H, 32), 512, np.uint16)
    v = u.copy()
    y[540:], u[540:], v[540:] = 250, 409, 960
    out = core.interlace.Interlace(clip_from([[y, u, v]] * NFRAMES, vs.YUV422P10)).get_frame(3)
    shifts = []
    for p, ref_plane in ((0, y), (1, u), (2, v)):
        got = np.asarray(out[p])
        woven = np.empty(H)
        woven[0::2], woven[1::2] = got[0::2, 8], got[1::2, 8]
        shifts.append(edge_center(woven) - edge_center(ref_plane[:, 8]))
    print('  edge centre shift (lines): Y %+.3f  U %+.3f  V %+.3f' % tuple(shifts))
    print('  chroma-luma timing offset: %.3f lines' % max(abs(shifts[1] - shifts[0]),
                                                          abs(shifts[2] - shifts[0])))


def edge_center(prof, flat=20):
    p = prof.astype(np.float64)
    mid = (p[:flat].mean() + p[-flat:].mean()) / 2
    i = np.where((p[:-1] - mid) * (p[1:] - mid) <= 0)[0][0]
    return i + (mid - p[i]) / (p[i + 1] - p[i]) if p[i + 1] != p[i] else i


def official(path):
    # geometry of the EBU 1080p pattern (fig 2): six 150-row grating blocks
    # from y=50, V gratings x1505-1645, H gratings x1300-1450; zone plate
    # y700-999 x100-1249; crosses at x100-350/x400-650; red square x1000-1250
    W, H, N = 1920, 1080, 64
    raw = subprocess.run(
        [FFMPEG, '-v', 'error', '-i', path, '-frames:v', str(N),
         '-pix_fmt', 'yuv422p10le', '-f', 'rawvideo', '-'],
        check=True, stdout=subprocess.PIPE).stdout
    fsz = W * H + 2 * (W // 2) * H
    raw = np.frombuffer(raw, np.uint16).reshape(-1, fsz)
    N = raw.shape[0]
    assert N >= 56, 'need at least 56 frames'
    Ys = raw[:, :W * H].reshape(N, H, W)
    Us = raw[:, W * H:W * H + (W // 2) * H].reshape(N, H, W // 2)
    Vs = raw[:, W * H + (W // 2) * H:].reshape(N, H, W // 2)
    assert abs(int(Ys[0, 0, 0]) - 512) <= 2, 'not the 1080p pattern (grey bg expected)'

    frames = [[Ys[n], Us[n], Vs[n]] for n in range(N)]
    out = core.interlace.Interlace(clip_from(frames, vs.YUV422P10))
    outp = [[np.asarray(out.get_frame(k)[p]).copy() for p in range(3)]
            for k in range(out.num_frames)]
    wov = outp[8][0]

    print('== official 2.2.1: vertical gratings ==')
    print('block  nominal%%  measured   designed')
    for i, pct in enumerate(GRATING_PCT):
        y0, y1 = 50 + i * 150 + 12, 50 + (i + 1) * 150 - 12
        a_in = amp(Ys[16, y0:y1, 1505:1645])
        a_out = amp(wov[y0:y1, 1505:1645])
        print(f'  {i + 1}     {pct:5.1f}   {db(a_out / a_in):7.2f} dB {db(resp(0.5 * pct / 100, 0)):7.2f} dB')

    worst = 0
    for i in range(6):
        y0, y1 = 50 + i * 150 + 12, 50 + (i + 1) * 150 - 12
        worst = max(worst, np.abs(wov[y0:y1, 1300:1450].astype(int) -
                                  Ys[16, y0:y1, 1300:1450].astype(int)).max())
    print(f'  H gratings, block interiors: max |out-in| = {worst} code'
          ' (block boundaries are real vertical edges and are filtered)')

    print()
    print('== official 2.2.2: zone plate, measured vs designed over (V,T) ==')
    ZR0, ZR1, ZC0, ZC1 = 700, 999, 100, 1249
    in_f = np.stack([Ys[t][t % 2::2].astype(np.float64) for t in range(8, 56)])
    out_f = np.stack([outp[t // 2][0][t % 2::2].astype(np.float64) for t in range(8, 56)])
    cells = []
    for xc in range(ZC0 + 40, ZC1 - 40, 40):
        tf = fpeak(Ys[8:56, (ZR0 + ZR1) // 2, xc].astype(np.float64))
        for yc in range(ZR0 + 30, ZR1 - 30, 12):
            vf = fpeak(Ys[16, yc - 24:yc + 24, xc].astype(np.float64))
            fr0, fr1 = (yc - 12) // 2, (yc + 12) // 2
            a_in = in_f[:, fr0:fr1, xc - 16:xc + 16].std()
            a_out = out_f[:, fr0:fr1, xc - 16:xc + 16].std()
            if a_in > 20:
                cells.append((vf, tf, a_out / a_in, abs(resp(vf, tf))))
    cells = np.array(cells)
    meas_db = 20 * np.log10(np.maximum(cells[:, 2], 1e-9))
    des_db = 20 * np.log10(np.maximum(cells[:, 3], 1e-9))
    strong = des_db > -24
    diff = np.abs(meas_db[strong] - des_db[strong])
    print(f'  {len(cells)} cells; V {cells[:, 0].min():.3f}-{cells[:, 0].max():.3f},'
          f' T {cells[:, 1].min():.3f}-{cells[:, 1].max():.3f}')
    print(f'  designed > -24 dB ({strong.sum()} cells):'
          f' |measured-designed| mean {diff.mean():.2f} dB, max {diff.max():.2f} dB')
    if (~strong).any():
        print(f'  designed <= -24 dB ({(~strong).sum()} cells):'
              f' measured mean {meas_db[~strong].mean():.1f} dB, max {meas_db[~strong].max():.1f} dB')
    if TEMPLATE is not None:
        print('  -6 dB boundary where the plate covers it:')
        for tf in np.unique(cells[:, 1]):
            sub = cells[cells[:, 1] == tf]
            below = sub[sub[:, 2] < 0.5]
            if len(below):
                v50 = below[:, 0].min()
                vb = template_boundary(tf)
                print(f'    T={tf:.3f}: measured V50={v50:.3f}, template={vb:.3f},'
                      f' diff {abs(v50 - vb):.3f}')

    print()
    print('== official 2.2.3: cross ringing ==')
    for x0, x1, name in ((100, 350, 'black cross'), (400, 650, 'white cross')):
        col = wov[:, (x0 + x1) // 2].astype(int)[60:400]
        icol = Ys[16][:, (x0 + x1) // 2].astype(int)[60:400]
        lo, hi = icol.min(), icol.max()
        over, under = max(col.max() - hi, 0), max(lo - col.min(), 0)
        print(f'  {name}: levels {lo}/{hi}, overshoot +{over}, undershoot -{under} codes'
              f' ({100 * over / (hi - lo):.1f}% / {100 * under / (hi - lo):.1f}%)')
    d = np.abs(wov[130, 60:700].astype(int) - Ys[16, 130, 60:700].astype(int)).max()
    print(f'  horizontal steps (row through vertical arms): max |out-in| = {d} code')

    print()
    print('== official 2.2.4: chroma/luma timing (red-square top edge) ==')
    shifts = []
    for plane_in, p, xc in ((Ys, 0, 1120), (Us, 1, 560), (Vs, 2, 560)):
        shifts.append(edge_center(outp[8][p][60:160, xc]) -
                      edge_center(plane_in[16, 60:160, xc]))
    print('  edge centre shift (lines): Y %+.3f U %+.3f V %+.3f' % tuple(shifts))
    print('  chroma-luma timing offset: %.3f lines' % max(abs(shifts[1] - shifts[0]),
                                                          abs(shifts[2] - shifts[0])))


if OFFICIAL:
    official(OFFICIAL)
else:
    synthetic()
