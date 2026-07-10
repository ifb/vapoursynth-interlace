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

- `clip` — constant-format progressive clip; 8–16-bit integer or 32-bit
  float, any subsampling with even plane heights. For 16-bit float, convert
  to single precision first. Spatial resizing is out of scope: for e.g.
  2160p→1080i, scale to 1080p with `resize` first.
- `tff` — field order of the output; the temporally first frame supplies the
  top field when true (default).
- `filter` — apply the WHP 315 vertical-temporal prefilter (default). With
  `filter=False` fields are woven with no filtering, which aliases. Only
  useful as a worst-case reference.
- `cpu` — cap the instruction set used, for testing and debugging:
  `"none"` (plain C), `"sse2"`, `"avx2"` or `"avx512"`. Default: everything
  the host supports. Levels the hardware lacks are simply not used.

Same filter on all planes (per WHP 230); zero phase, so no chroma/luma
timing shift. SSE2/AVX2/AVX-512 kernels are selected at runtime and produce
bit-identical output to the C implementation, floats included.


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
