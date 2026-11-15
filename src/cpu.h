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

#endif
