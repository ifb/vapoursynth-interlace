#ifndef P2I_CPU_H
#define P2I_CPU_H

#define P2I_CPU_SSE2      (1 << 0)
#define P2I_CPU_AVX2      (1 << 1)
#define P2I_CPU_AVXVNNI   (1 << 2)  /* VEX-encoded VNNI, with AVX2 */
#define P2I_CPU_AVX512    (1 << 3)
#define P2I_CPU_AVX512ICL (1 << 4)  /* AVX512 plus VNNI/IFMA/VBMI/VBMI2/GFNI/... */
/* Zen 4: a zmm is two 256-bit halves. Not a tier: kernels that measure
 * faster there in a narrower form test it alongside P2I_CPU_AVX512. */
#define P2I_CPU_SLOW_AVX512 (1 << 5)

unsigned p2i_cpu_detect(void);

/* The tiers the compiler target already guarantees, as dav1d's trim_dsp
 * does. The dispatch getters OR this into their flags, so in a per-CPU
 * build the branches below the guaranteed tier are dead and the C
 * references they would return drop out of the module. The baseline
 * guarantees nothing here: checkasm, linked against it, gets its C
 * reference by passing no flags. */
#if defined(__AVX2__) && defined(__FMA__) && defined(__BMI2__) && defined(__F16C__)
#define P2I_CPU_BUILTIN_AVX2 (P2I_CPU_AVX2 | P2I_CPU_SSE2)
#else
#define P2I_CPU_BUILTIN_AVX2 0
#endif

#if P2I_CPU_BUILTIN_AVX2 && defined(__AVXVNNI__)
#define P2I_CPU_BUILTIN_AVXVNNI P2I_CPU_AVXVNNI
#else
#define P2I_CPU_BUILTIN_AVXVNNI 0
#endif

#if P2I_CPU_BUILTIN_AVX2 && defined(__AVX512F__) && defined(__AVX512CD__) && \
    defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
#define P2I_CPU_BUILTIN_AVX512 P2I_CPU_AVX512
#else
#define P2I_CPU_BUILTIN_AVX512 0
#endif

#if P2I_CPU_BUILTIN_AVX512 && defined(__AVX512IFMA__) && \
    defined(__AVX512VBMI__) && defined(__AVX512VBMI2__) && \
    defined(__AVX512VNNI__) && defined(__AVX512BITALG__) && \
    defined(__AVX512VPOPCNTDQ__) && defined(__GFNI__) && defined(__VAES__) && \
    defined(__VPCLMULQDQ__)
#define P2I_CPU_BUILTIN_AVX512ICL P2I_CPU_AVX512ICL
#else
#define P2I_CPU_BUILTIN_AVX512ICL 0
#endif

#define P2I_CPU_BUILTIN (P2I_CPU_BUILTIN_AVX2 | P2I_CPU_BUILTIN_AVXVNNI | \
                         P2I_CPU_BUILTIN_AVX512 | P2I_CPU_BUILTIN_AVX512ICL)

#endif
