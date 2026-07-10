/*
 * Vertical-temporal prefilter kernels (BBC WHP 315).
 */

#include <stdint.h>
#include <string.h>

#include "cpu.h"
#include "vt_coeffs.h"
#include "vt_filter.h"

/* [shift][taps], ordered as the P2I_SHIFT_* enum */
static const int16_t coefs_i16[5][96] = {
    { VT_COEFS_M375_I16 }, { VT_COEFS_M125_I16 }, { VT_COEFS_I16 },
    { VT_COEFS_P125_I16 }, { VT_COEFS_P375_I16 },
};
static const float coefs_f32[5][96] = {
    { VT_COEFS_M375_F32 }, { VT_COEFS_M125_F32 }, { VT_COEFS_F32 },
    { VT_COEFS_P125_F32 }, { VT_COEFS_P375_F32 },
};

const void *p2i_vt_coefs(int kind, int shift)
{
    return kind == P2I_KIND_U8 || kind == P2I_KIND_U16 ? (const void *)coefs_i16[shift]
                                                       : (const void *)coefs_f32[shift];
}

#define pixel uint8_t
#define acc_t int32_t
#define coef_t int16_t
#define FUNC(name) name##_u8
#define VT_INT 1
#include "vt_filter_tmpl.h"
#undef pixel
#undef FUNC

#define pixel uint16_t
#define FUNC(name) name##_u16
#include "vt_filter_tmpl.h"
#undef pixel
#undef acc_t
#undef coef_t
#undef FUNC
#undef VT_INT

#define pixel float
#define acc_t float
#define coef_t float
#define FUNC(name) name##_f32
#define VT_INT 0
#include "vt_filter_tmpl.h"
#undef pixel
#undef acc_t
#undef coef_t
#undef FUNC
#undef VT_INT

/* soft binary16 conversion, round-to-nearest-even, matching F16C exactly */
static float f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t em = h & 0x7fff;
    uint32_t bits;
    float f;

    if (em >= 0x7c00)      /* inf/nan */
        bits = sign | 0x7f800000 | ((uint32_t)(em & 0x3ff) << 13);
    else if (em >= 0x400)  /* normal */
        bits = sign | ((em + 0x1c000) << 13);
    else if (em) {         /* subnormal */
        uint32_t m = em;
        int shift = 0;
        while (!(m & 0x400)) {
            m <<= 1;
            shift++;
        }
        bits = sign | ((uint32_t)(113 - shift) << 23) | ((m & 0x3ff) << 13);
    } else
        bits = sign;
    memcpy(&f, &bits, 4);
    return f;
}

static uint16_t f32_to_f16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t em = bits & 0x7fffffff;

    if (em >= 0x47800000)  /* overflow -> inf, nan stays nan */
        return sign | (em > 0x7f800000 ? 0x7e00 : 0x7c00);
    if (em >= 0x38800000) { /* normal */
        uint32_t r = em - 0x38000000;
        uint32_t h = r >> 13;
        uint32_t rem = r & 0x1fff;
        h += rem > 0x1000 || (rem == 0x1000 && (h & 1));
        return sign | h;
    }
    if (em >= 0x33000000) { /* subnormal */
        uint32_t m = (em & 0x7fffff) | 0x800000;
        unsigned s = 126 - (em >> 23);
        return sign | ((m + ((1u << (s - 1)) - 1) + ((m >> s) & 1)) >> s);
    }
    return sign;
}

static void vt_filter_line_f16(void *dstv, const void *const *srcv, int width, int peak,
                               const void *coefsv)
{
    uint16_t *dst = dstv;
    const uint16_t *const *src = (const uint16_t *const *)srcv;
    const float *coefs = coefsv;
    (void)peak;

    for (int x = 0; x < width; x++) {
        float sum = 0;
        for (int i = 0; i < VT_TAPS_T * VT_TAPS_V; i++)
            sum += coefs[i] * f16_to_f32(src[i][x]);
        dst[x] = f32_to_f16(sum);
    }
}

#if defined(__x86_64__)
#define HAVE_X86_ASM 1

#define DECL_KERNELS(isa) \
    void p2i_vt_filter_line_u8_##isa(void *, const void *const *, int, int, const void *); \
    void p2i_vt_filter_line_u16_##isa(void *, const void *const *, int, int, const void *); \
    void p2i_vt_filter_line_f32_##isa(void *, const void *const *, int, int, const void *);
DECL_KERNELS(sse2)
DECL_KERNELS(avx2)
DECL_KERNELS(avx512)
void p2i_vt_filter_line_f16_avx2(void *, const void *const *, int, int, const void *);
void p2i_vt_filter_line_f16_avx512(void *, const void *const *, int, int, const void *);
#endif

vt_filter_line_fn p2i_get_vt_filter_fn(int kind, unsigned cpu)
{
    static const vt_filter_line_fn fns_c[4] = {
        vt_filter_line_u8, vt_filter_line_u16, vt_filter_line_f32, vt_filter_line_f16,
    };
    vt_filter_line_fn fn = fns_c[kind];
#if HAVE_X86_ASM
    /* no sse2 f16 kernel (F16C implies AVX); C covers it */
    static const vt_filter_line_fn fns_sse2[4] = {
        p2i_vt_filter_line_u8_sse2, p2i_vt_filter_line_u16_sse2,
        p2i_vt_filter_line_f32_sse2, vt_filter_line_f16,
    };
    static const vt_filter_line_fn fns_avx2[4] = {
        p2i_vt_filter_line_u8_avx2, p2i_vt_filter_line_u16_avx2,
        p2i_vt_filter_line_f32_avx2, p2i_vt_filter_line_f16_avx2,
    };
    static const vt_filter_line_fn fns_avx512[4] = {
        p2i_vt_filter_line_u8_avx512, p2i_vt_filter_line_u16_avx512,
        p2i_vt_filter_line_f32_avx512, p2i_vt_filter_line_f16_avx512,
    };
    if (cpu & P2I_CPU_SSE2)
        fn = fns_sse2[kind];
    if (cpu & P2I_CPU_AVX2)
        fn = fns_avx2[kind];
    if (cpu & P2I_CPU_AVX512)
        fn = fns_avx512[kind];
#else
    (void)cpu;
#endif
    return fn;
}
