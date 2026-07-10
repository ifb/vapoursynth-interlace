#ifndef P2I_CPU_H
#define P2I_CPU_H

#define P2I_CPU_SSE2   (1 << 0)
#define P2I_CPU_AVX2   (1 << 1)
#define P2I_CPU_AVX512 (1 << 2)

unsigned p2i_cpu_detect(void);

#endif
