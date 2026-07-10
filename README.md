# vapoursynth-interlace

Progressive to interlaced conversion for VapourSynth.

The filter implements the perceptually optimal interlacing filter of BBC R&D
White Paper WHP 315 (K.C. Noland, "Optimal Interlacing using Human Sensitivity
Measurements", 2016): a 19-line by 5-frame linear vertical-temporal low-pass
applied to the progressive sequence before discarding alternate lines, with a
pass-band shaped by the measured human sensitivity to interlacing aliases
(WHP 230). Output behavior has been validated against the EBU TECH 3384
test patterns and measurement methodology.

## Usage

```python
clip = core.interlace.Interlace(clip, tff=True, filter=True)
```

2N progressive frames become N woven frames at half rate. `_FieldBased`
is set and frame durations doubled.

- `clip` — constant-format progressive clip; 8–16-bit integer, 16-bit or
  32-bit float, any subsampling with even plane heights. Half precision is
  a first-class citizen for memory-bound pipelines: it filters faster than
  single precision at half the frame memory (F16C conversion on load, f32
  accumulation). Spatial resizing is out of scope: for e.g. 2160p→1080i,
  scale to 1080p with `resize` first.
- `tff` — field order of the output; the temporally first frame supplies the
  top field when true (default).
- `filter` — apply the WHP 315 vertical-temporal prefilter (default). With
  `filter=False` fields are woven with no filtering, which aliases. Only
  useful as a worst-case reference.
- `cpu` — cap the instruction set used, for testing and debugging:
  `"none"` (plain C), `"sse2"`, `"avx2"` or `"avx512"`. Default: everything
  the host supports. Levels the hardware lacks are simply not used.

Same filter on all planes (per WHP 230); zero phase, so no chroma/luma
timing shift. SSE2/AVX2/AVX-512 selected at runtime. Integer output is
bit-identical to the C implementation on every path. Float paths use FMA
on AVX2 and above, so float output can differ from C by a few ulp
(~1e-6 relative). The fused sum skips one rounding per tap and is the more
accurate of the two.

## 4:2:0 and chroma siting

Progressive 4:2:0 sites chroma relative to luma per `_ChromaLocation`.
Interlaced 4:2:0 sites each field's chroma at the ¼/¾ positions between the
field's luma lines. The filter honors the input `_ChromaLocation`
(left/center/topleft/top/bottomleft/bottom; assumed left when absent) and
**resamples vertically subsampled chroma onto the interlaced siting grid**
inside the same vertical-temporal convolution - fractionally delayed
variants of the same kernel, at no extra cost.

Chroma bandwidth note: subsampled chroma gets the same filter at its own
lattice (the cutoff sits at the same fraction of the halved chroma Nyquist).
The WHP 315 / TECH 3384 references only cover 4:2:2; per-plane application
is the natural reading of WHP 230's "same filter on all components".

## Recommended workflows

Do all spatial scaling, color conversion and tonemapping progressive;
interlace **last**. Never resize after interlacing.

```python
# 1080p50 -> 1080i25, staying 4:2:0
out = core.interlace.Interlace(clip)      # chroma siting handled
# encode with an interlaced-aware encoder (e.g. x264 --tff)

# 1080p50 4:2:0 -> 1080i25 4:2:2
c = core.resize.Bicubic(clip, format=vs.YUV422P10)
out = core.interlace.Interlace(c)

# UHD 2160p50 -> 1080i25 (down-convert then interlace)
c = core.resize.Lanczos(clip, 1920, 1080, format=vs.YUV422P10, filter_param_a=4)
out = core.interlace.Interlace(c)

# HDR UHD -> SDR 1080i: tonemap/convert color first, progressive
c = your_hdr_to_sdr(clip)                 # e.g. resize/placebo chain
c = core.resize.Lanczos(c, 1920, 1080, format=vs.YUV422P10, filter_param_a=4)
out = core.interlace.Interlace(c)

# 1080p50 -> 576i25 (PAL SD, 16:9 anamorphic). The active picture is
# 702 wide (52us at 13.5MHz). Fudge the image to 704 (for mod 16 and even
# chroma) instead or don't fudge and crop the width 5.5 first.
c = core.resize.Lanczos(c, 704, 576, format=vs.YUV420P8, filter_param_a=4)
c = core.std.AddBorders(c, left=8, right=8, color=[16, 128, 128])
out = core.interlace.Interlace(c)

# 1080p59.94 -> 480i29.97 (NTSC SD, 16:9 anamorphic). SMPTE RP 187 puts the
# 525-line active picture at 708 samples; resize yields SAR 640/531.
c = core.std.SetFrameProps(clip, _SARNum=1, _SARDen=1)
c = core.resize.Lanczos(c, 708, 480, format=vs.YUV422P10, filter_param_a=4)
c = core.std.AddBorders(c, left=6, right=6, color=[64, 512, 512])
out = core.interlace.Interlace(c)
```

## Choosing a downscaler for 2160p→1080i

EBU TECH 3384 ships a companion 2160p test pattern for exactly this chain.
As shipped, its gratings sit at the same absolute frequencies as the 1080p
pattern (13.5–80.8% of 1080p Nyquist after the 2:1 downscale) and its zone
plate tops out at 1080p Nyquist — worst-case oversampled 1080p content. The
pattern therefore measures how faithfully a 2160p→1080i chain reproduces
the native 1080i response. It contains nothing above 1080p Nyquist, so it
does not rank alias suppression.

Every zimg kernel, measured through `resize` + `Interlace` against the
official file: horizontal grating MTF (the interlacer leaves horizontal
untouched, so this is the resizer alone), worst deviation of the composed
vertical gratings from the native-1080p chain over gratings 1–4 (the
pass/knee region of the interlacing filter), and white-cross ringing
(native chain: −11.8%).

| kernel | g1 | g2 | g3 | g4 | g5 | g6 (dB) | V dev | ring |
|---|---|---|---|---|---|---|---|---|
| Bilinear | −0.15 | −0.59 | −1.32 | −2.42 | −3.81 | −5.67 | 2.42 dB | −2.8% |
| Bicubic b=0 c=.5 | −0.00 | −0.06 | −0.25 | −0.76 | −1.63 | −3.12 | 0.76 | −5.4% |
| Bicubic Mitchell | −0.09 | −0.38 | −0.89 | −1.78 | −3.02 | −4.86 | 1.78 | −3.7% |
| Spline16 | +0.03 | +0.03 | −0.10 | −0.62 | −1.60 | −3.31 | 0.62 | −5.8% |
| Spline36 | −0.00 | +0.00 | +0.04 | −0.12 | −0.71 | −2.24 | 0.12 | −5.8% |
| Spline64 | +0.00 | −0.02 | −0.04 | −0.17 | −0.57 | −1.91 | 0.17 | −5.6% |
| Lanczos3 | +0.00 | +0.04 | +0.13 | +0.04 | −0.43 | −1.83 | 0.13 | −6.1% |
| Lanczos4 | −0.00 | −0.04 | −0.02 | +0.05 | −0.00 | −1.10 | 0.05 | −5.8% |

Gratings 1–6 are 13.5/26.9/40.4/53.9/67.3/80.8% of 1080p Nyquist. Spline36,
Spline64, Lanczos3 and Lanczos4 are one quality class: within 0.2 dB of the
native chain where it matters, identical ringing. Lanczos4 retains the most
detail near Nyquist; Bilinear, Mitchell and Spline16 measurably dull the
chain. Composed ringing is lower than native because the down-conversion
band-limits the edges. The full composed chain's vertical-temporal cutoff
tracks the WHP 315 template within 0.021 normalized frequency over the
entire temporal range.

### Alias suppression

The official material carries nothing above 1080p Nyquist, so alias
rejection is scored separately (`tools/resizer_alias.py`): synthetic 2160p
gratings at 1.05–1.95× 1080p Nyquist, whose downscale residue lands at
2−f. Horizontal is the resizer alone; vertically the interlacing filter's
stop band absorbs whatever lands above its cutoff — aliases from
1.05–1.35× Nyquist land at 0.95–0.65× and come out ≤ −37 dB **for every
kernel**, so the vertical axis only discriminates at 1.5× and beyond.
Response in dB relative to the source grating; −54 dB is the 10-bit
measurement floor:

| kernel | H 1.05 | H 1.2 | H 1.35 | H 1.5 | H 1.65–1.95 | V worst, after chain |
|---|---|---|---|---|---|---|
| Bilinear | −9.9 | −13.8 | −18.7 | −25.1 | −34…−54 | −25.4 (@1.5) |
| Bicubic b=0 c=.5 | −7.2 | −11.3 | −16.6 | −23.7 | −33…−54 | −24.0 (@1.5) |
| Mitchell | −9.5 | −13.9 | −19.4 | −26.6 | −36…−54 | −26.6 (@1.5) |
| Spline16 | −8.2 | −13.3 | −20.0 | −29.0 | −41…−54 | −29.3 (@1.5) |
| Spline36 | −7.9 | −14.7 | −24.7 | −41.2 | −54 | −41.1 (@1.5) |
| Spline64 | −8.0 | −15.6 | −26.7 | −41.5 | −54 | −41.7 (@1.5) |
| Lanczos3 | −7.5 | −14.9 | −28.5 | −39.5 | −44…−50 | −39.5 (@1.5) |
| Lanczos4 | −8.2 | −19.6 | −44.9 | −47.1 | −45…−54 | −45.1 (@1.65) |

At 1.05× every kernel sits at −7…−10 dB — the transition width is
inherent, and the soft kernels buy their ~1.5 dB there with 4–6 dB of
passband loss. Lanczos4 leads from 1.2× on both axes and has the best
composed-vertical worst case; its only gap versus the splines is side-lobe
leakage around 1.5–1.65× at ≈ −45 dB, below every alternative's worst
case. Lanczos3's side lobes never reach the floor. Taken together with the
scorecard above: **Lanczos4 is the measured pick for 2160p→1080i**, with
Spline36/Spline64 as slightly softer, marginally cleaner-far-field
alternatives.

## Building

```sh
meson setup build
`meson test -C build` runs the checkasm verification (fetched as a meson subproject)
```

Requires VapourSynth (V4 API), meson, and nasm on x86-64. `meson test -C
build` runs the checkasm kernel verification (fetched as a meson subproject;
`-Dcheckasm=disabled` to skip). Or `pip install vapoursynth-interlace`.

## License

GPL-2.0-or-later.
