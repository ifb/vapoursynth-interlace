#include <stdint.h>
#include <string.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "vt_filter.h"

#define MAXW 1920
#define ROWB ((MAXW + 128) * 4)   /* bytes per source row, any pixel type */

static CHECKASM_ALIGN(uint8_t src_buf[VT_SRC_PTRS][ROWB]);
static CHECKASM_ALIGN(uint8_t dst_c[(MAXW + 64) * 4]);
static CHECKASM_ALIGN(uint8_t dst_a[(MAXW + 64) * 4]);

static const int widths[] = {
    1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 255, 719, 960, 1279, 1919, 1920,
};
#define NWIDTHS (int)(sizeof(widths) / sizeof(widths[0]))

/* deterministic element-aligned misalignment per row and run */
static void setup_ptrs(const void *ptrs[VT_SRC_PTRS], int run, int elem)
{
    for (int i = 0; i < VT_SRC_PTRS; i++)
        ptrs[i] = src_buf[i] + ((i * 7 + run * 13) % 32) * elem;
}

static void test_u8(void)
{
    const void *ptrs[VT_SRC_PTRS];
    checkasm_declare(void, void *, const void *const *, int, int);
    if (checkasm_check_func(p2i_get_vt_filter_fn(P2I_KIND_U8, checkasm_get_cpu_flags()),
                            "vt_filter_line_u8")) {
        INITIALIZE_BUF(src_buf);
        for (int i = 0; i < NWIDTHS; i++) {
            setup_ptrs(ptrs, i, 1);
            memset(dst_c, 0xaa, sizeof(dst_c));
            memset(dst_a, 0x55, sizeof(dst_a));
            checkasm_call_ref(dst_c, ptrs, widths[i], 255);
            checkasm_call_new(dst_a, ptrs, widths[i], 255);
            checkasm_check1d(uint8_t, dst_c, dst_a, widths[i], "dst");
        }
        setup_ptrs(ptrs, 0, 1);
        checkasm_bench_new(dst_a, ptrs, 1920, 255);
    }
    checkasm_report("u8");
}

static void test_u16(void)
{
    static const int peaks[] = { 1023, 4095, 65535 };
    const void *ptrs[VT_SRC_PTRS];
    checkasm_declare(void, void *, const void *const *, int, int);
    if (checkasm_check_func(p2i_get_vt_filter_fn(P2I_KIND_U16, checkasm_get_cpu_flags()),
                            "vt_filter_line_u16")) {
        INITIALIZE_BUF(src_buf);
        /* peak-respecting inputs matter less than clamp coverage: leave the
           buffer full-range, all peaks must clamp identically to scalar */
        for (int i = 0; i < NWIDTHS; i++) {
            setup_ptrs(ptrs, i, 2);
            memset(dst_c, 0xaa, sizeof(dst_c));
            memset(dst_a, 0x55, sizeof(dst_a));
            const int peak = peaks[i % 3];
            checkasm_call_ref(dst_c, ptrs, widths[i], peak);
            checkasm_call_new(dst_a, ptrs, widths[i], peak);
            checkasm_check1d(uint16_t, (uint16_t *)dst_c, (uint16_t *)dst_a, widths[i], "dst");
        }
        setup_ptrs(ptrs, 0, 2);
        checkasm_bench_new(dst_a, ptrs, 1920, 65535);
    }
    checkasm_report("u16");
}

static void test_f32(void)
{
    const void *ptrs[VT_SRC_PTRS];
    checkasm_declare(void, void *, const void *const *, int, int);
    if (checkasm_check_func(p2i_get_vt_filter_fn(P2I_KIND_F32, checkasm_get_cpu_flags()),
                            "vt_filter_line_f32")) {
        INITIALIZE_BUF(src_buf);
        /* force finite floats: random sign/mantissa, exponent pinned to [0.5,1) */
        uint32_t *bits = (uint32_t *)src_buf;
        for (size_t i = 0; i < sizeof(src_buf) / 4; i++)
            bits[i] = (bits[i] & 0x807fffffu) | 0x3f000000u;
        for (int i = 0; i < NWIDTHS; i++) {
            setup_ptrs(ptrs, i, 4);
            memset(dst_c, 0xaa, sizeof(dst_c));
            memset(dst_a, 0x55, sizeof(dst_a));
            checkasm_call_ref(dst_c, ptrs, widths[i], 0);
            checkasm_call_new(dst_a, ptrs, widths[i], 0);
            /* AVX2 up accumulates with FMA: one rounding fewer per tap than
               the C sum, so compare with a small ULP budget */
            const float *fc = (const float *)dst_c, *fa = (const float *)dst_a;
            for (int j = 0; j < widths[i]; j++)
                if (!checkasm_float_near_abs_eps_ulp(fc[j], fa[j], 2e-6f, 64)) {
                    checkasm_fail_func("dst[%d]: %.9g vs %.9g", j, fc[j], fa[j]);
                    break;
                }
        }
        setup_ptrs(ptrs, 0, 4);
        checkasm_bench_new(dst_a, ptrs, 1920, 0);
    }
    checkasm_report("f32");
}

void checkasm_test_vt_filter(void)
{
    test_u8();
    test_u16();
    test_f32();
}
