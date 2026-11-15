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
    const int amd = r[1] == 0x68747541 && r[3] == 0x69746e65 && r[2] == 0x444d4163; /* AuthenticAMD */

    p2i_cpu_cpuid(r, 1, 0);
    unsigned family = (r[0] >> 8) & 0xf;
    if (family == 0xf)
        family += (r[0] >> 20) & 0xff;
    if (X(r[3], 0x06008000)) /* CMOV/SSE/SSE2 */
        flags |= P2I_CPU_SSE2;

    if (X(r[2], 0x18000000) && max_leaf >= 7) { /* OSXSAVE/AVX */
        const uint64_t xcr0 = p2i_cpu_xgetbv(0);
        if (X(xcr0, 0x00000006)) { /* XMM/YMM state */
            p2i_cpu_cpuid(r, 7, 0);
            const unsigned max_subleaf = r[0];
            if (X(r[1], 0x00000128)) /* BMI1/BMI2/AVX2 */
                flags |= P2I_CPU_AVX2;
            if (X(xcr0, 0x000000e0) && /* opmask/ZMM state */
                X(r[1], 0xd0030000)) { /* F/DQ/CD/BW/VL */
                flags |= P2I_CPU_AVX512;
                if (amd && family == 0x19) /* Zen 4 */
                    flags |= P2I_CPU_SLOW_AVX512;
                if (X(r[1], 0x00200000) && /* IFMA */
                    X(r[2], 0x00005f42)) /* VBMI/VBMI2/GFNI/VAES/VPCLMULQDQ/VNNI/BITALG/VPOPCNTDQ */
                    flags |= P2I_CPU_AVX512ICL;
            }
            if ((flags & P2I_CPU_AVX2) && max_subleaf >= 1) {
                p2i_cpu_cpuid(r, 7, 1);
                if (X(r[0], 0x00000010)) /* AVX-VNNI */
                    flags |= P2I_CPU_AVXVNNI;
            }
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
