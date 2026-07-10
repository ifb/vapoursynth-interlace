/* x86 flag parsing after dav1d src/x86/cpu.c */

#include <stdint.h>

#include "cpu.h"

#if defined(__x86_64__)

typedef struct {
    uint32_t eax, ebx, ecx, edx;
} p2i_cpuid_regs;

void p2i_cpu_cpuid(uint32_t regs[4], unsigned leaf, unsigned subleaf);
uint64_t p2i_cpu_xgetbv(unsigned xcr);

#define X(reg, mask) (((reg) & (mask)) == (mask))

unsigned p2i_cpu_detect(void)
{
    unsigned flags = 0;
    uint32_t r[4];

    p2i_cpu_cpuid(r, 0, 0);
    const unsigned max_leaf = r[0];
    if (max_leaf < 1)
        return 0;

    p2i_cpu_cpuid(r, 1, 0);
    if (X(r[3], 0x06008000)) /* CMOV/SSE/SSE2 */
        flags |= P2I_CPU_SSE2;

    if (X(r[2], 0x18000000) && max_leaf >= 7) { /* OSXSAVE/AVX */
        const uint64_t xcr0 = p2i_cpu_xgetbv(0);
        if (X(xcr0, 0x00000006)) { /* XMM/YMM state */
            p2i_cpu_cpuid(r, 7, 0);
            if (X(r[1], 0x00000128)) /* BMI1/BMI2/AVX2 */
                flags |= P2I_CPU_AVX2;
            if (X(xcr0, 0x000000e0) && /* opmask/ZMM state */
                X(r[1], 0xd0030000)) /* F/DQ/CD/BW/VL */
                flags |= P2I_CPU_AVX512;
        }
    }
    return flags;
}

#else

unsigned p2i_cpu_detect(void)
{
    return 0;
}

#endif
