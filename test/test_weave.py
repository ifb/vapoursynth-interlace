#!/usr/bin/env python3
# Weave correctness: parity, temporal order, props, rates, formats, errors.
# usage: test_weave.py path/to/interlace.so

import sys

import vapoursynth as vs

core = vs.core
core.std.LoadPlugin(sys.argv[1])


def graded(fmt, w, h, n, fpsnum, fpsden, mul=1):
    # frame i = constant value i*mul
    return core.std.Splice([
        core.std.BlankClip(format=fmt, width=w, height=h, length=1, fpsnum=fpsnum,
                           fpsden=fpsden, color=[i * m for m in mul] if isinstance(mul, list) else i * mul)
        for i in range(n)])


src = graded(vs.GRAY8, 32, 8, 6, 50, 1)
for tff in (True, False):
    out = core.interlace.Interlace(src, tff=tff, filter=0)
    assert out.num_frames == 3 and (out.fps.numerator, out.fps.denominator) == (25, 1)
    for n in range(3):
        fr = out.get_frame(n)
        top, bot = (2 * n, 2 * n + 1) if tff else (2 * n + 1, 2 * n)
        assert [fr[0][r, 0] for r in range(8)] == [top, bot] * 4
        assert fr.props['_FieldBased'] == (2 if tff else 1)
        assert (fr.props['_DurationNum'], fr.props['_DurationDen']) == (1, 25)

    # std.SeparateFields must restore temporal order
    fields = core.std.SeparateFields(core.interlace.Interlace(src, tff=tff, filter=0))
    for n in range(6):
        fr = fields.get_frame(n)
        assert {fr[0][r, c] for r in range(fr.height) for c in range(fr.width)} == {n}

src = graded(vs.YUV420P16, 32, 8, 4, 50000, 1001, mul=[1000, 500, 250])
out = core.interlace.Interlace(src, filter=0)
assert (out.fps.numerator, out.fps.denominator) == (25000, 1001)
fr = out.get_frame(1)
for p, m in ((0, 1000), (1, 500), (2, 250)):
    rows = [fr[p][r, 0] for r in range(fr[p].shape[0])]
    assert rows == [2 * m, 3 * m] * (len(rows) // 2)

fr = core.interlace.Interlace(graded(vs.GRAYS, 16, 4, 4, 60, 1), filter=0).get_frame(0)
assert [fr[0][r, 0] for r in range(4)] == [0.0, 1.0, 0.0, 1.0]

for clip, msg in [
    (core.std.BlankClip(format=vs.YUV420P8, width=16, height=6, length=4), 'even'),
    (core.std.BlankClip(format=vs.GRAY8, width=16, height=4, length=1), 'short'),
]:
    try:
        core.interlace.Interlace(clip, filter=0)
        sys.exit('expected error: ' + msg)
    except vs.Error as e:
        assert msg in str(e)

print('all weave tests pass')
