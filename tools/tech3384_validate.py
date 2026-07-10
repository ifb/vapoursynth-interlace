#!/usr/bin/env python3
# EBU TECH 3384 measurements against synthesized test patterns (sec. 2.2).
# The official test files require an EBU login; the patterns are synthesized
# here from the published construction: 10-bit R103 levels, sinusoidal
# gratings at 13.5/26.9/40.4/53.9/67.3/80.8% of Nyquist, vertical-temporal
# zone plate, 0/512/940 step crosses, colour squares.
#
# usage: tech3384_validate.py interlace.so [--template template.png]
#        (template.png from whp315_extract.py adds a measured-vs-template
#         cutoff contour comparison)

import sys

import numpy as np
import vapoursynth as vs

core = vs.core
core.std.LoadPlugin(sys.argv[1])
TEMPLATE = None
if '--template' in sys.argv:
    from PIL import Image
    TEMPLATE = np.asarray(Image.open(sys.argv[sys.argv.index('--template') + 1]).convert('L'))

GRATING_PCT = [13.5, 26.9, 40.4, 53.9, 67.3, 80.8]


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


def fields(clip):
    # woven output -> list per output frame of (top, bottom) row-decimated arrays
    out = []
    for n in range(clip.num_frames):
        fr = clip.get_frame(n)
        a = np.asarray(fr[0]).copy()
        out.append((a[0::2], a[1::2]))
    return out


def amp(a):
    return np.sqrt(2.0) * a.astype(np.float64).std()


def vt_response(v, t):
    # designed response from the shipped float table
    import pathlib, re
    hdr = (pathlib.Path(__file__).parent / '../src/vt_coeffs.h').read_text()
    rows = re.findall(r'\{\s*([^{}]+?)\s*\}', hdr.split('vt_coefs_f32')[1].split(';')[0])
    c = np.array([[float(x.strip().rstrip('f')) for x in r.split(',')] for r in rows])
    h = 0.0
    for n in range(5):
        for m in range(19):
            h += c[n][m] * np.cos(2 * np.pi * (m - 9) * v) * np.cos(2 * np.pi * (n - 2) * t)
    return h


def db(x):
    return 20 * np.log10(max(x, 1e-9))


H, NFRAMES = 1080, 12
print('== 2.2.1 frequency gratings (static, V and H) ==')
print('grating  freq%%   %-10s %-10s %s' % ('measured', 'designed', 'H-grating identity'))
for i, pct in enumerate(GRATING_PCT):
    vfreq = 0.5 * pct / 100                      # cycles/line
    y = np.arange(H)
    vg = np.clip(502 + 438 * np.sin(2 * np.pi * vfreq * y), 0, 1023).astype(np.uint16)
    frames = [[np.tile(vg[:, None], (1, 64))]] * NFRAMES
    out = fields(core.interlace.Interlace(clip_from(frames, vs.GRAY10)))
    meas = amp(out[3][0][40:500]) / amp(frames[0][0][0::2][40:500])
    design = abs(vt_response(vfreq, 0.0))

    x = np.arange(256)
    hg = np.clip(502 + 438 * np.sin(2 * np.pi * vfreq * x), 0, 1023).astype(np.uint16)
    hframes = [[np.tile(hg[None, :], (64, 1))]] * NFRAMES
    hout = core.interlace.Interlace(clip_from(hframes, vs.GRAY10)).get_frame(2)
    hdiff = np.abs(np.asarray(hout[0]).astype(int) - hframes[0][0].astype(int)).max()
    print('   %d     %4.1f   %6.2f dB  %6.2f dB   max diff %d code' %
          (i + 1, pct, db(meas), db(design), hdiff))

print()
print('== 2.2.2 vertical-temporal zone plate ==')
W, NZ = 480, 48
row = np.arange(H)[:, None]
x = np.arange(W)[None, :]
yb = (H - 1 - row).astype(np.float64)            # DC at bottom-left
vphase = 0.5 * yb * yb / (2 * H)                 # local vfreq = 0.5*yb/H c/line
tfreq = 0.5 * x / W                              # c/frame
zp = [np.clip(512 + 400 * np.sin(2 * np.pi * (vphase + tfreq * t)), 0, 1023).astype(np.uint16)
      for t in range(NZ)]
zclip = clip_from([[f] for f in zp], vs.GRAY10)
flt = fields(core.interlace.Interlace(zclip))
ref = fields(core.interlace.Interlace(zclip, filter=0))


def amap(fieldseq):
    # amplitude over time and a 16-row window, per (row-window, column)
    stack = np.stack([np.concatenate(f) for f in fieldseq[4:20]])  # [t][540*2? no: top+bot stacked]
    return stack


def cutoff_contour(flt, ref):
    # per column band: attenuation vs progressive line, find -6 dB height
    tops_f = np.stack([f[0] for f in flt[4:20]]).astype(np.float64)   # [16][540][W]
    tops_r = np.stack([f[0] for f in ref[4:20]]).astype(np.float64)
    vs_ = []
    for xc in range(8, W - 8, 16):
        att = []
        for y0 in range(0, 540 - 8, 8):
            a_f = tops_f[:, y0:y0 + 8, xc - 8:xc + 8].std()
            a_r = tops_r[:, y0:y0 + 8, xc - 8:xc + 8].std()
            att.append(a_f / a_r if a_r > 1 else 1.0)
        att = np.array(att)
        yb_line = H - 1 - (np.arange(0, 540 - 8, 8) * 2 + 4)          # progressive line from bottom
        vf = 0.5 * yb_line / H                                        # descending with index
        below = np.where(att < 0.5)[0]
        v50 = vf[below].min() if len(below) else 0.5                  # lowest attenuated V
        vs_.append((0.5 * xc / W, v50))
    return vs_


contour = cutoff_contour(flt, ref)
print('T(c/frame)  measured -6dB V cutoff' + ('   template boundary' if TEMPLATE is not None else ''))
worst = 0.0
for t, v in contour:
    line = f'  {t:.3f}       {v:.3f}'
    if TEMPLATE is not None:
        col = min(int(round(t / 0.5 * 255)), 255)
        r = np.where(TEMPLATE[:, col] == 255)[0][0]
        vb = 0.5 * (255 - r) / 255
        worst = max(worst, abs(v - vb))
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
    frames = [[img]] * NFRAMES
    out = fields(core.interlace.Interlace(clip_from(frames, vs.GRAY10)))
    prof = np.concatenate([out[3][0], out[3][1]])[:, 32]  # any column
    woven = np.empty(H); woven[0::2] = out[3][0][:, 32]; woven[1::2] = out[3][1][:, 32]
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
y = np.full((H, 64), 940, np.uint16); u = np.full((H, 32), 512, np.uint16); v = u.copy()
y[540:], u[540:], v[540:] = 250, 409, 960          # white over red-ish
frames = [[y, u, v]] * NFRAMES
out = core.interlace.Interlace(clip_from(frames, vs.YUV422P10)).get_frame(3)


def edge_center(prof):
    p = prof.astype(np.float64)
    mid = (p[:400].mean() + p[-400:].mean()) / 2
    i = np.where((p[:-1] - mid) * (p[1:] - mid) <= 0)[0][0]
    return i + (mid - p[i]) / (p[i + 1] - p[i])


shifts = []
for p, ref_plane in ((0, y), (1, u), (2, v)):
    got = np.asarray(out[p])
    woven = np.empty(H)
    woven[0::2], woven[1::2] = got[0::2, 8], got[1::2, 8]
    shifts.append(edge_center(woven) - edge_center(ref_plane[:, 8]))
print('  edge centre shift (lines): Y %+.3f  U %+.3f  V %+.3f' % tuple(shifts))
print('  chroma-luma timing offset: %.3f lines' % max(abs(shifts[1] - shifts[0]),
                                                      abs(shifts[2] - shifts[0])))
