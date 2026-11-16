#include <math.h>
#include <stdint.h>
#include <string.h>

#include <checkasm/checkasm.h>
#include <checkasm/test.h>
#include <checkasm/utils.h>

#include "cpu.h"
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

#define BSTRIDE ((MAXW + 64) * 2)
static CHECKASM_ALIGN(uint8_t blk_c[2 * VT_BLOCK_MAX * BSTRIDE]);
static CHECKASM_ALIGN(uint8_t blk_a[2 * VT_BLOCK_MAX * BSTRIDE]);
static CHECKASM_ALIGN(uint8_t blk_tmp[VT_BLOCK_TMP]);

static void setup_block_ptrs(const void *ptrs[5 * VT_BLOCK_ROWS], int run, int elem)
{
    for (int i = 0; i < 5 * VT_BLOCK_ROWS; i++)
        ptrs[i] = src_buf[(i * 37 + run * 11) % VT_SRC_PTRS] + ((i * 7 + run * 13) % 32) * elem;
}

/* slow: the kernels picked where P2I_CPU_SLOW_AVX512 is set, on any AVX-512 host */
static void test_block(int kind, int depth, int slow)
{
    const int elem = kind == P2I_KIND_U8 ? 1 : 2;
    const int peak = (1 << depth) - 1;
    const void *ptrs[5 * VT_BLOCK_ROWS];
    int min_width;
    checkasm_declare(void, void *, ptrdiff_t, const void *const *, void *, int, int, int,
                     const void *);
    unsigned cpu = checkasm_get_cpu_flags();
    if (slow) {
        if (!(cpu & P2I_CPU_AVX512))
            return;
        cpu |= P2I_CPU_SLOW_AVX512;
    }
    if (checkasm_check_func(p2i_get_vt_filter_block_fn(kind, depth, cpu, &min_width),
                            "vt_filter_block_u%d%s", depth, slow ? "n" : "")) {
        INITIALIZE_BUF(src_buf);
        /* the kernels require samples <= peak */
        if (kind == P2I_KIND_U16) {
            uint16_t *s = (uint16_t *)src_buf;
            for (size_t i = 0; i < sizeof(src_buf) / 2; i++)
                s[i] &= peak;
        }
        for (int i = 0; i < NWIDTHS; i++) {
            const int w = widths[i];
            const void *coefs = p2i_vt_coefs(kind, i % 5);
            if (w < min_width)
                continue;
            for (int nb = 1 + i % 3; nb <= VT_BLOCK_MAX; nb += 1 + i % 5) {
                setup_block_ptrs(ptrs, i * VT_BLOCK_MAX + nb, elem);
                memset(blk_c, 0xaa, sizeof(blk_c));
                memset(blk_a, 0xaa, sizeof(blk_a));
                checkasm_call_ref(blk_c, 2 * BSTRIDE, ptrs, blk_tmp, w, nb, peak, coefs);
                checkasm_call_new(blk_a, 2 * BSTRIDE, ptrs, blk_tmp, w, nb, peak, coefs);
                for (int k = 0; k < nb; k++) {
                    const uint8_t *c = blk_c + k * 2 * BSTRIDE, *a = blk_a + k * 2 * BSTRIDE;
                    if (kind == P2I_KIND_U8)
                        checkasm_check1d(uint8_t, c, a, w, "dst");
                    else
                        checkasm_check1d(uint16_t, (const uint16_t *)c, (const uint16_t *)a, w, "dst");
                }
                /* nothing else may be written */
                if (memcmp(blk_c, blk_a, sizeof(blk_c))) {
                    checkasm_fail_func("write outside the lines, width %d, %d lines", w, nb);
                    break;
                }
            }
        }
        setup_block_ptrs(ptrs, 0, elem);
        checkasm_bench_new(blk_a, 2 * BSTRIDE, ptrs, blk_tmp, MAXW, VT_BLOCK_MAX, peak,
                           p2i_vt_coefs(kind, P2I_SHIFT_CENTER));
    }
    checkasm_report("block_u%d%s", depth, slow ? "n" : "");
}

static void test_block_flt(int kind)
{
    const int f32 = kind == P2I_KIND_F32;
    const int elem = f32 ? 4 : 2;
    const void *ptrs[5 * VT_BLOCK_ROWS];
    int min_width;
    checkasm_declare(void, void *, ptrdiff_t, const void *const *, void *, int, int, int,
                     const void *);
    if (checkasm_check_func(p2i_get_vt_filter_block_fn(kind, 0, checkasm_get_cpu_flags(),
                                                       &min_width),
                            "vt_filter_block_%s", f32 ? "f32" : "f16")) {
        INITIALIZE_BUF(src_buf);
        /* +/-[0.5,1), as the line tests */
        if (f32) {
            uint32_t *bits = (uint32_t *)src_buf;
            for (size_t i = 0; i < sizeof(src_buf) / 4; i++)
                bits[i] = (bits[i] & 0x807fffffu) | 0x3f000000u;
        } else {
            uint16_t *h = (uint16_t *)src_buf;
            for (size_t i = 0; i < sizeof(src_buf) / 2; i++)
                h[i] = (h[i] & 0x83ff) | 0x3800;
        }
        for (int i = 0; i < NWIDTHS; i++) {
            const int w = widths[i];
            const void *coefs = p2i_vt_coefs(kind, i % 5);
            if (w < min_width)
                continue;
            for (int nb = 1 + i % 3; nb <= VT_BLOCK_MAX; nb += 1 + i % 5) {
                setup_block_ptrs(ptrs, i * VT_BLOCK_MAX + nb, elem);
                memset(blk_c, 0xaa, sizeof(blk_c));
                memset(blk_a, 0xaa, sizeof(blk_a));
                checkasm_call_ref(blk_c, 2 * BSTRIDE, ptrs, blk_tmp, w, nb, 0, coefs);
                checkasm_call_new(blk_a, 2 * BSTRIDE, ptrs, blk_tmp, w, nb, 0, coefs);
                for (int k = 0; k < VT_BLOCK_MAX; k++) {
                    const uint8_t *c = blk_c + k * 2 * BSTRIDE, *a = blk_a + k * 2 * BSTRIDE;
                    const int used = k < nb ? w * elem : 0;
                    /* the sums are in another order than the line kernel's */
                    if (!used)
                        ;
                    else if (f32) {
                        const float *fc = (const float *)c, *fa = (const float *)a;
                        for (int j = 0; j < w; j++)
                            if (!checkasm_float_near_abs_eps_ulp(fc[j], fa[j], 2e-6f, 64)) {
                                checkasm_fail_func("line %d dst[%d]: %.9g vs %.9g", k, j, fc[j], fa[j]);
                                break;
                            }
                    } else
                        check_f16((const uint16_t *)c, (const uint16_t *)a, w, 3e-5f);
                    /* nothing else may be written */
                    for (int j = used; j < 2 * BSTRIDE; j++)
                        if (a[j] != 0xaa) {
                            checkasm_fail_func("write outside the lines, width %d, %d lines", w, nb);
                            break;
                        }
                }
            }
        }
        setup_block_ptrs(ptrs, 0, elem);
        checkasm_bench_new(blk_a, 2 * BSTRIDE, ptrs, blk_tmp, MAXW, VT_BLOCK_MAX, 0,
                           p2i_vt_coefs(kind, P2I_SHIFT_CENTER));
    }
    checkasm_report("block_%s", f32 ? "f32" : "f16");
}

void checkasm_test_vt_filter(void)
{
    test_u8();
    test_u16();
    test_f32();
    test_f16();
    test_block(P2I_KIND_U8, 8, 0);
    for (int depth = 10; depth <= 14; depth += 2) {
        test_block(P2I_KIND_U16, depth, 0);
        test_block(P2I_KIND_U16, depth, 1);
    }
    test_block_flt(P2I_KIND_F32);
    test_block_flt(P2I_KIND_F16);
}
