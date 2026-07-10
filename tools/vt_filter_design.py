#!/usr/bin/env python3
# Design the vertical-temporal interlacing prefilter of BBC WHP 315:
# 19 lines x 5 frames, zero phase, symmetric in each axis, weighted least
# squares (1/delta) against the fig 7 specification bands from
# whp315_extract.py. Also designs fractionally delayed variants (same
# magnitude spec, linear phase e^{-j2pi*v*shift}) used to move subsampled
# chroma onto the interlaced siting grid; y[r] samples x[r + shift].
# Emits unit-DC-gain float and fixed-point tables.
#
# usage: vt_filter_design.py bands.png vt_coeffs.h [--bits N] [--ripple-db X]
#                            [--response out.png]

SHIFTS = [-0.375, -0.125, 0.125, 0.375]   # chroma rows

import sys

import numpy as np
from PIL import Image

NV, NT = 19, 5          # taps
KV, KT = NV // 2, NT // 2

BANDS = {               # grey level: (desired, attenuation dB for reporting)
    255: (1.0, None),
    125: (0.0, 12.0),
    60:  (0.0, 40.0),
    0:   (0.0, 50.0),
}


def response(a, v, t):
    # a[k,l] on cos(2*pi*k*v)cos(2*pi*l*t); v,t broadcastable
    h = 0.0
    for k in range(KV + 1):
        for l in range(KT + 1):
            h = h + a[k, l] * np.cos(2 * np.pi * k * v) * np.cos(2 * np.pi * l * t)
    return h


def response_taps(h, v, t):
    # complex response of a full tap grid h[NV][NT]
    H = 0.0
    for m in range(NV):
        for n in range(NT):
            H = H + h[m, n] * np.exp(-2j * np.pi * v * (m - KV)) * np.cos(2 * np.pi * t * (n - KT))
    return H


def design_shifted(vv, tt, desired, w, shift):
    # vertical basis loses its symmetry to carry the linear phase;
    # temporal symmetry is kept. Unit DC gain and the vertical first moment
    # (= ramp shift / group delay at DC) are enforced exactly via KKT.
    cols = []
    for m in range(-KV, KV + 1):
        for l in range(KT + 1):
            cols.append(np.exp(-2j * np.pi * vv * m) * np.cos(2 * np.pi * l * tt)
                        * (1 if l == 0 else 2))
    A = np.stack(cols, axis=1)
    b = desired * np.exp(-2j * np.pi * vv * shift)
    Ar = np.concatenate([A.real, A.imag]) * np.concatenate([w, w])[:, None]
    br = np.concatenate([b.real, b.imag]) * np.concatenate([w, w])
    tapsum = np.array([1.0, 2.0, 2.0])            # x[m,:] -> sum of taps at m
    C = np.zeros((2, NV * (KT + 1)))
    for m in range(NV):
        C[0, m * (KT + 1):(m + 1) * (KT + 1)] = tapsum
        C[1, m * (KT + 1):(m + 1) * (KT + 1)] = tapsum * (m - KV)
    d = np.array([1.0, shift])                    # y[r] samples x[r + shift]
    n = C.shape[1]
    kkt = np.block([[Ar.T @ Ar, C.T], [C, np.zeros((2, 2))]])
    rhs = np.concatenate([Ar.T @ br, d])
    x = np.linalg.solve(kkt, rhs)[:n].reshape(NV, KT + 1)
    h = np.zeros((NV, NT))
    for m in range(NV):
        h[m, KT] = x[m, 0]
        for l in range(1, KT + 1):
            h[m, KT - l] = h[m, KT + l] = x[m, l]
    return h


def quantize(h, bits, symmetric):
    scale = 1 << bits
    q = np.round(h * scale).astype(np.int64)
    q[np.unravel_index(np.argmax(np.abs(q)), q.shape)] += scale - q.sum()
    assert q.sum() == scale and np.all(np.abs(q) < 32768)
    if symmetric:
        assert np.array_equal(q, q[::-1]) and np.array_equal(q, q[:, ::-1])
    return q


def main():
    args = sys.argv[1:]
    bits, ripple_db, resp_png = 13, 0.05, None
    if '--bits' in args:
        i = args.index('--bits'); bits = int(args[i + 1]); del args[i:i + 2]
    if '--ripple-db' in args:
        i = args.index('--ripple-db'); ripple_db = float(args[i + 1]); del args[i:i + 2]
    if '--response' in args:
        i = args.index('--response'); resp_png = args[i + 1]; del args[i:i + 2]
    bands_png, out_h = args

    g = np.asarray(Image.open(bands_png).convert('L'))
    r, c = np.mgrid[0:256, 0:256]
    v = 0.5 * (255 - r) / 255
    t = 0.5 * c / 255

    # fit a[k,l] by WLS over all non-transition points
    mask = g != 190
    vv, tt, gg = v[mask], t[mask], g[mask]
    desired = np.array([BANDS[x][0] for x in gg])
    delta = np.where(desired == 1.0, 10 ** (ripple_db / 20) - 1,
                     [10 ** (-(BANDS[x][1] or 0) / 20) for x in gg])
    w = np.sqrt(1.0 / delta)

    cols = []
    for k in range(KV + 1):
        for l in range(KT + 1):
            cols.append(np.cos(2 * np.pi * k * vv) * np.cos(2 * np.pi * l * tt))
    A = np.stack(cols, axis=1)
    x, *_ = np.linalg.lstsq(A * w[:, None], desired * w, rcond=None)
    a = x.reshape(KV + 1, KT + 1)
    a /= a.sum()                      # exact unit DC gain

    # expand to full tap grid
    h = np.zeros((NV, NT))
    for k in range(KV + 1):
        for l in range(KT + 1):
            q = a[k, l] / ((2 if k else 1) * (2 if l else 1))
            for sm in ({0} if k == 0 else {-k, k}):
                for sn in ({0} if l == 0 else {-l, l}):
                    h[KV + sm, KT + sn] = q

    def report(name, H, q):
        hdb = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
        m = g == 255
        print(f'{name}: pass-band ripple max {np.abs(hdb[m]).max():.3f} dB, '
              f'mean {np.abs(hdb[m]).mean():.3f} dB')
        for lvl, (_, att) in BANDS.items():
            if att is None:
                continue
            m = g == lvl
            print(f'  stop {att:>4.0f} dB band: worst {-hdb[m].max():6.2f} dB, '
                  f'mean {-hdb[m].mean():6.2f} dB')
        acc = int(np.abs(q).sum()) * 65535
        print(f'  Q{bits}: sum|q| = {int(np.abs(q).sum())}, worst 16-bit accumulation '
              f'{acc} ({"fits" if acc < 2**31 else "OVERFLOWS"} int32)')
        return hdb

    print(f'{NV}x{NT} taps, ripple spec +/-{ripple_db} dB, weights 1/delta')
    q = quantize(h, bits, symmetric=True)
    hdb = report('centered', response(a, v, t), q)

    shifted = []
    for s in SHIFTS:
        hs = design_shifted(vv, tt, desired, w, s)
        qs = quantize(hs, bits, symmetric=False)
        Hs = response_taps(hs, v, t)
        report(f'shift {s:+.3f}', Hs, qs)
        # measured delay: phase slope over the low pass-band at t=0
        vprobe = np.linspace(0.01, 0.15, 20)
        ph = np.unwrap(np.angle(response_taps(hs, vprobe, np.zeros(20))))
        meas = -np.polyfit(2 * np.pi * vprobe, ph, 1)[0]
        print(f'  measured delay {meas:+.4f} rows (target {s:+.3f})')
        shifted.append((s, hs, qs))

    if resp_png:
        img = np.clip((hdb + 60) / 60, 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(resp_png)

    with open(out_h, 'w') as f:
        f.write('/* Generated by tools/vt_filter_design.py. Do not edit.\n'
                ' * Vertical-temporal interlacing prefilter (19 lines x 5 frames)\n'
                ' * designed per BBC R&D WHP 315 / WHP 230.\n'
                ' * Flat [t][v] initializers; i16 is padded with a 0 to 96 entries\n'
                ' * so SIMD can consume coefficients as pmaddwd pairs. */\n\n'
                '#ifndef VT_COEFS_H\n#define VT_COEFS_H\n\n'
                f'#define VT_TAPS_V {NV}\n'
                f'#define VT_TAPS_T {NT}\n'
                f'#define VT_COEF_BITS {bits}\n\n')
        def emit(name, hh, qq):
            rows = [', '.join(f'{int(qq[m, n]):6d}' for m in range(NV)) for n in range(NT)]
            f.write(f'#define {name}_I16 \\\n    ' + ', \\\n    '.join(rows) + ', 0\n\n')
            rows = [', '.join(f'{hh[m, n]:.8e}f' for m in range(NV)) for n in range(NT)]
            f.write(f'#define {name}_F32 \\\n    ' + ', \\\n    '.join(rows) + ', 0\n\n')

        emit('VT_COEFS', h, q)
        f.write('/* fractional vertical delays for interlaced chroma siting,\n'
                '   in chroma rows; y[r] samples x[r + shift] */\n\n')
        for s, hs, qs in shifted:
            tag = ('M' if s < 0 else 'P') + f'{abs(s):.3f}'.replace('0.', '')
            emit(f'VT_COEFS_{tag}', hs, qs)
        f.write('#endif\n')
    print('wrote', out_h)


if __name__ == '__main__':
    main()
