#!/usr/bin/env python3
# VT prefilter path vs an independent reference implementation.
# Integer formats must match bit-exactly; float within accumulation-order noise.
# usage: test_vt_filter.py path/to/interlace.so

import pathlib
import re
import sys

import numpy as np
import vapoursynth as vs

core = vs.core
core.std.LoadPlugin(sys.argv[1])

hdr = (pathlib.Path(__file__).parent / '../src/vt_coeffs.h').read_text()
BITS = int(re.search(r'VT_COEF_BITS (\d+)', hdr).group(1))


def table(name, conv):
    body = hdr.split('#define ' + name)[1].split('\n\n')[0].replace('\\\n', ' ')
    vals = [conv(x.strip().rstrip('f')) for x in body.split(',') if x.strip()]
    return np.array(vals[:95]).reshape(5, 19)


CI = table('VT_COEFS_I16', int)        # [5][19]
CF = table('VT_COEFS_F32', float)
assert CI.shape == CF.shape == (5, 19) and CI.sum() == 1 << BITS
# fractional-delay variants for interlaced chroma siting
CI_S = {s: table(f'VT_COEFS_{s}_I16', int) for s in ('M375', 'M125', 'P125', 'P375')}
CF_S = {s: table(f'VT_COEFS_{s}_F32', float) for s in ('M375', 'M125', 'P125', 'P375')}


def ref(frames, tff, integer, peak, ctab=None):
    # ctab: (top, bottom) coefficient tables by output row parity
    n_out = len(frames) // 2
    h, w = frames[0].shape
    out = []
    for n in range(n_out):
        dst = np.empty((h, w), np.int64 if integer else np.float64)
        for r in range(h):
            coefs = ctab[r % 2] if ctab else (CI if integer else CF)
            o = (0 if tff else 1) if r % 2 == 0 else (1 if tff else 0)
            acc = np.zeros(w, dst.dtype)
            for t in range(5):
                q = min(max(2 * n + o + t - 2, 0), len(frames) - 1)
                for v in range(19):
                    line = min(max(r + v - 9, 0), h - 1)
                    acc += coefs[t][v] * frames[q][line].astype(dst.dtype)
            dst[r] = np.clip((acc + (1 << BITS - 1)) >> BITS, 0, peak) if integer else acc
        out.append(dst)
    return out


def planes(clip, n):
    fr = clip.get_frame(n)
    return [np.asarray(fr[p]).copy() for p in range(fr.format.num_planes)]


rng = np.random.default_rng(315)

for fmt, dtype, peak in [(vs.GRAY8, np.uint8, 255), (vs.GRAY16, np.uint16, 65535),
                         (vs.YUV420P10, np.uint16, 1023)]:
    nplanes = 3 if fmt == vs.YUV420P10 else 1
    frames = [[rng.integers(0, peak + 1, (12 >> (p and 1), 24 >> (p and 1)), dtype=dtype)
               for p in range(nplanes)] for _ in range(9)]
    src = core.std.BlankClip(format=fmt, width=24, height=12, length=9, fpsnum=50, fpsden=1)
    src = core.std.ModifyFrame(src, src, lambda n, f, frames=frames: (
        lambda fo: ([np.copyto(np.asarray(fo[p]), frames[n][p]) for p in range(len(frames[n]))], fo)[1]
    )(f.copy()))
    for tff in (True, False):
        out = core.interlace.Interlace(src, tff=tff)
        assert out.num_frames == 4
        for n in range(4):
            got = planes(out, n)
            for p in range(nplanes):
                # 420 chroma is moved onto the interlaced siting grid
                ctab = (CI_S['M125'], CI_S['P125']) if p and fmt == vs.YUV420P10 else None
                want = ref([fr[p] for fr in frames], tff, True, peak, ctab)[n]
                assert np.array_equal(got[p], want), (fmt, tff, n, p)
    print(f'{src.format.name}: bit-exact vs reference (both field orders)')

frames = [[rng.random((12, 24), np.float32)] for _ in range(9)]
src = core.std.BlankClip(format=vs.GRAYS, width=24, height=12, length=9, fpsnum=50, fpsden=1)
src = core.std.ModifyFrame(src, src, lambda n, f: (
    lambda fo: (np.copyto(np.asarray(fo[0]), frames[n][0]), fo)[1])(f.copy()))
out = core.interlace.Interlace(src)
for n in range(4):
    want = ref([fr[0] for fr in frames], True, False, None)[n]
    assert np.allclose(planes(out, n)[0], want, rtol=0, atol=1e-5), n
print('GrayS: matches reference within 1e-5')

# unit DC gain: constant input is preserved exactly (integer)
src = core.std.BlankClip(format=vs.GRAY8, width=24, height=12, length=8, color=137)
fr = core.interlace.Interlace(src).get_frame(1)
assert {fr[0][r, c] for r in range(12) for c in range(24)} == {137}
print('DC preservation exact')

# half float: valid output within f16 precision of a float64 reference
hfr = [[rng.random((12, 24)).astype(np.float16)] for _ in range(9)]
hsrc = core.std.BlankClip(format=vs.GRAYH, width=24, height=12, length=9, fpsnum=50, fpsden=1)
hsrc = core.std.ModifyFrame(hsrc, hsrc, lambda n, f: (
    lambda fo: (np.copyto(np.asarray(fo[0]), hfr[n][0]), fo)[1])(f.copy()))
hout = core.interlace.Interlace(hsrc)
for n in range(4):
    want = ref([fr[0].astype(np.float64) for fr in hfr], True, False, None)[n]
    got = planes(hout, n)[0].astype(np.float64)
    assert np.allclose(got, want, rtol=0, atol=2e-3), n
print('GrayH matches reference within f16 precision')

# chroma siting: a ramp whose values equal their vertical position reads out
# each sample's position after resampling. Progressive 4:2:0 chroma row c sits
# at luma line 2c+voff; interlaced siting puts top-field chroma at 4j+0.25 and
# bottom-field at 4j+2.75.
H2 = 96
for cl, voff in ((0, 0.5), (1, 0.5), (2, 0.0), (4, 1.0)):
    uram = np.tile((2 * np.arange(H2 // 2) + voff)[:, None] * 256, (1, 8)).astype(np.uint16)
    yram = np.full((H2, 16), 512, np.uint16)
    ssrc = core.std.BlankClip(format=vs.YUV420P16, width=16, height=H2, length=8,
                              fpsnum=50, fpsden=1)
    ssrc = core.std.ModifyFrame(ssrc, ssrc, lambda n, f: (
        lambda fo: ([np.copyto(np.asarray(fo[0]), yram),
                     np.copyto(np.asarray(fo[1]), uram),
                     np.copyto(np.asarray(fo[2]), uram)], fo)[1])(f.copy()))
    ssrc = core.std.SetFrameProps(ssrc, _ChromaLocation=cl)
    ofr = core.interlace.Interlace(ssrc).get_frame(1)
    u = np.asarray(ofr[1]).astype(np.float64) / 256
    for j in range(6, H2 // 4 - 6):
        assert abs(u[2 * j, 4] - (4 * j + 0.25)) < 0.02, (cl, 'top', j, u[2 * j, 4])
        assert abs(u[2 * j + 1, 4] - (4 * j + 2.75)) < 0.02, (cl, 'bot', j, u[2 * j + 1, 4])
    assert ofr.props['_ChromaLocation'] == (1 if cl == 1 else 0)
print('interlaced chroma siting exact for left/center/topleft/bottomleft input')

# cpu levels: integer output identical everywhere; float within FMA tolerance
frames = [[rng.integers(0, 1024, (12, 24), dtype=np.uint16)] for _ in range(9)]
src = core.std.BlankClip(format=vs.GRAY10, width=24, height=12, length=9, fpsnum=50, fpsden=1)
src = core.std.ModifyFrame(src, src, lambda n, f: (
    lambda fo: (np.copyto(np.asarray(fo[0]), frames[n][0]), fo)[1])(f.copy()))
fsrc = core.std.BlankClip(format=vs.GRAYS, width=24, height=12, length=9, fpsnum=50, fpsden=1)
ffr = [[rng.random((12, 24), np.float32)] for _ in range(9)]
fsrc = core.std.ModifyFrame(fsrc, fsrc, lambda n, f: (
    lambda fo: (np.copyto(np.asarray(fo[0]), ffr[n][0]), fo)[1])(f.copy()))
ref_i = planes(core.interlace.Interlace(src, cpu='none'), 1)[0]
ref_f = planes(core.interlace.Interlace(fsrc, cpu='none'), 1)[0]
for level in ('sse2', 'avx2', 'avx512'):
    got = planes(core.interlace.Interlace(src, cpu=level), 1)[0]
    assert np.array_equal(got, ref_i), level
    gotf = planes(core.interlace.Interlace(fsrc, cpu=level), 1)[0]
    assert np.allclose(gotf, ref_f, rtol=0, atol=1e-5), level
try:
    core.interlace.Interlace(src, cpu='mmx')
    sys.exit('accepted bad cpu')
except vs.Error as e:
    assert 'cpu' in str(e)
print('cpu levels consistent (none/sse2/avx2/avx512)')

print('all VT filter tests pass')
