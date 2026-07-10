#include <math.h>
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
    checkasm_declare(void, void *, const void *const *, int, int, const void *);
    if (checkasm_check_func(p2i_get_vt_filter_fn(P2I_KIND_U8, checkasm_get_cpu_flags()),
                            "vt_filter_line_u8")) {
        INITIALIZE_BUF(src_buf);
        for (int i = 0; i < NWIDTHS; i++) {
            setup_ptrs(ptrs, i, 1);
            memset(dst_c, 0xaa, sizeof(dst_c));
            memset(dst_a, 0x55, sizeof(dst_a));
            const void *coefs = p2i_vt_coefs(P2I_KIND_U8, i % 5);
            checkasm_call_ref(dst_c, ptrs, widths[i], 255, coefs);
            checkasm_call_new(dst_a, ptrs, widths[i], 255, coefs);
            checkasm_check1d(uint8_t, dst_c, dst_a, widths[i], "dst");
        }
        setup_ptrs(ptrs, 0, 1);
        checkasm_bench_new(dst_a, ptrs, 1920, 255, p2i_vt_coefs(P2I_KIND_U8, P2I_SHIFT_CENTER));
    }
    checkasm_report("u8");
}

static void test_u16(void)
{
    static const int peaks[] = { 1023, 4095, 65535 };
    const void *ptrs[VT_SRC_PTRS];
    checkasm_declare(void, void *, const void *const *, int, int, const void *);
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
            const void *coefs = p2i_vt_coefs(P2I_KIND_U16, i % 5);
            checkasm_call_ref(dst_c, ptrs, widths[i], peak, coefs);
            checkasm_call_new(dst_a, ptrs, widths[i], peak, coefs);
            checkasm_check1d(uint16_t, (uint16_t *)dst_c, (uint16_t *)dst_a, widths[i], "dst");
        }
        setup_ptrs(ptrs, 0, 2);
        checkasm_bench_new(dst_a, ptrs, 1920, 65535, p2i_vt_coefs(P2I_KIND_U16, P2I_SHIFT_CENTER));
    }
    checkasm_report("u16");
}

static void test_f32(void)
{
    const void *ptrs[VT_SRC_PTRS];
    checkasm_declare(void, void *, const void *const *, int, int, const void *);
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
            const void *coefs = p2i_vt_coefs(P2I_KIND_F32, i % 5);
            checkasm_call_ref(dst_c, ptrs, widths[i], 0, coefs);
            checkasm_call_new(dst_a, ptrs, widths[i], 0, coefs);
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
        checkasm_bench_new(dst_a, ptrs, 1920, 0, p2i_vt_coefs(P2I_KIND_F32, P2I_SHIFT_CENTER));
    }
    checkasm_report("f32");
}

/* test-local copy of the kernel's soft conversion */
static float f16_value(uint16_t h)
{
    int e = (h >> 10) & 0x1f;
    float m = (float)((h & 0x3ff) | (e ? 0x400 : 0));
    float v = ldexpf(m, (e ? e : 1) - 25);
    return h & 0x8000 ? -v : v;
}

/* FMA accumulation vs the unfused C sum differs by up to one f32 rounding
   per tap of the running sum, which cancellation can magnify well past the
   final value's f16 ulp; accept one f16 step or a difference within the
   worst-case fused/unfused bound for the input range */
static void check_f16(const uint16_t *hc, const uint16_t *ha, int w, float eps)
{
    for (int j = 0; j < w; j++) {
        int ka = ha[j] & 0x8000 ? -(ha[j] & 0x7fff) : ha[j];
        int kc = hc[j] & 0x8000 ? -(hc[j] & 0x7fff) : hc[j];
        int dk = ka - kc;
        float dv = f16_value(ha[j]) - f16_value(hc[j]);
        if ((dk > 1 || dk < -1) && (dv > eps || dv < -eps)) {
            checkasm_fail_func("dst[%d]: 0x%04x vs 0x%04x", j, hc[j], ha[j]);
            return;
        }
    }
}

static void test_f16(void)
{
    const void *ptrs[VT_SRC_PTRS];
    checkasm_declare(void, void *, const void *const *, int, int, const void *);
    if (checkasm_check_func(p2i_get_vt_filter_fn(P2I_KIND_F16, checkasm_get_cpu_flags()),
                            "vt_filter_line_f16")) {
        uint16_t *h = (uint16_t *)src_buf;
        for (int phase = 0; phase < 2; phase++) {
            INITIALIZE_BUF(src_buf);
            for (size_t i = 0; i < sizeof(src_buf) / 2; i++) {
                if (phase) /* +/-[0.5,1): tight fused/unfused bound */
                    h[i] = (h[i] & 0x83ff) | 0x3800;
                else if ((h[i] & 0x7c00) == 0x7c00) /* finite full range */
                    h[i] ^= 0x0400;
            }
            for (int i = 0; i < NWIDTHS; i++) {
                setup_ptrs(ptrs, i, 2);
                memset(dst_c, 0xaa, sizeof(dst_c));
                memset(dst_a, 0x55, sizeof(dst_a));
                const void *coefs = p2i_vt_coefs(P2I_KIND_F16, i % 5);
                checkasm_call_ref(dst_c, ptrs, widths[i], 0, coefs);
                checkasm_call_new(dst_a, ptrs, widths[i], 0, coefs);
                check_f16((const uint16_t *)dst_c, (const uint16_t *)dst_a,
                          widths[i], phase ? 3e-5f : 2.0f);
            }
        }
        setup_ptrs(ptrs, 0, 2);
        checkasm_bench_new(dst_a, ptrs, 1920, 0, p2i_vt_coefs(P2I_KIND_F16, P2I_SHIFT_CENTER));
    }
    checkasm_report("f16");
}

void checkasm_test_vt_filter(void)
{
    test_u8();
    test_u16();
    test_f32();
    test_f16();
}
