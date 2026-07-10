# vapoursynth-interlace

Progressive to interlaced conversion for VapourSynth.

The filter implements the perceptually optimal interlacing filter of BBC R&D
White Paper WHP 315 (K.C. Noland, "Optimal Interlacing using Human Sensitivity
Measurements", 2016): a 19-line by 5-frame linear vertical-temporal low-pass
applied to the progressive sequence before discarding alternate lines, with a
pass-band shaped by the measured human sensitivity to interlacing aliases
(WHP 230). Output behaviour has been validated against the EBU TECH 3384
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

## 4:2:0 workflows

Each plane is filtered on its own sampling lattice and chroma rows are woven
by field parity, consistent with `SeparateFields`/`DoubleWeave`. Two things
to know for 1080p→1080i in 4:2:0:

- **Chroma siting.** Progressive 4:2:0 sites chroma between luma line pairs;
  interlaced 4:2:0 sites each field's chroma at ¼/¾ positions. This filter
  does not resample chroma to interlaced siting (that is a `resize` job), so
  staying 4:2:0 end to end carries a half-luma-line chroma siting offset
  versus strict interlaced-4:2:0 convention. Many real chains carry exactly
  this; if you need it strictly correct, work in 4:2:2 through the
  interlace step (`resize` up, `Interlace`, then let the interlaced-aware
  encoder subsample) — production interlaced video is 4:2:2 anyway.
- **Chroma bandwidth.** Subsampled chroma gets the same filter at its own
  lattice, i.e. the cutoff sits at the same fraction of the (halved) chroma
  Nyquist. The WHP 315 / TECH 3384 references only cover 4:2:2; per-plane
  application is the natural reading of WHP 230's "same filter on all
  components".

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
