#include <checkasm/checkasm.h>

#include "cpu.h"

void checkasm_test_vt_filter(void);

static const CheckasmCpuInfo cpu_flags[] = {
    { "SSE2", "sse2", P2I_CPU_SSE2 },
    { "AVX2", "avx2", P2I_CPU_AVX2 },
    { "AVX512", "avx512", P2I_CPU_AVX512 },
    {0}
};

static const CheckasmTest tests[] = {
    { "vt_filter", checkasm_test_vt_filter },
    {0}
};

int main(int argc, const char *argv[])
{
    CheckasmConfig config = {
        .cpu_flags = cpu_flags,
        .tests = tests,
        .cpu = p2i_cpu_detect(),
    };
    return checkasm_main(&config, argc, argv);
}
