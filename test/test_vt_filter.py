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


def ref(frames, tff, integer, peak):
    n_out = len(frames) // 2
    h, w = frames[0].shape
    coefs = CI if integer else CF
    out = []
    for n in range(n_out):
        dst = np.empty((h, w), np.int64 if integer else np.float64)
        for r in range(h):
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
                want = ref([fr[p] for fr in frames], tff, True, peak)[n]
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
