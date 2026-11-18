/*
 * Vertical-temporal prefilter kernels (BBC WHP 315).
 */

#include <stdint.h>
#include <string.h>

#include "cpu.h"
#include "vt_coeffs.h"
#include "vt_filter.h"

/* [shift][taps], ordered as the P2I_SHIFT_* enum */
static const int16_t coefs_i16[P2I_NB_SHIFTS][96] = {
    { VT_COEFS_M250_I16 }, { VT_COEFS_I16 }, { VT_COEFS_P250_I16 },
};
static const float coefs_f32[P2I_NB_SHIFTS][96] = {
    { VT_COEFS_M250_F32 }, { VT_COEFS_F32 }, { VT_COEFS_P250_F32 },
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

#define VT_INT 1
#define coef_t int16_t
#define acc_t int32_t
#define LOAD(x) (x)

#define pixel uint8_t
#define row_t int16_t
#define VT_BX 64
#define FUNC(name) name##_u8
#include "vt_block_tmpl.h"
#undef pixel
#undef row_t
#undef VT_BX
#undef FUNC

#define pixel uint16_t
#define row_t int32_t
#define VT_BX 32
#define FUNC(name) name##_u16
#include "vt_block_tmpl.h"
#undef pixel
#undef row_t
#undef FUNC

#undef VT_INT
#undef coef_t
#undef acc_t
#define VT_INT 0
#define coef_t float
#define acc_t float
#define row_t float

#define pixel float
#define STORE(x) (x)
#define FUNC(name) name##_f32
#include "vt_block_tmpl.h"
#undef pixel
#undef LOAD
#undef STORE
#undef FUNC

#if defined(__aarch64__)
#define pixel __fp16
#define LOAD(x) (float)(x)
#define STORE(x) (__fp16)(x)
#else
#define pixel uint16_t
#define LOAD(x) f16_to_f32(x)
#define STORE(x) f32_to_f16(x)
#endif
#define FUNC(name) name##_f16
#include "vt_block_tmpl.h"
#undef pixel
#undef LOAD
#undef STORE
#undef FUNC
#undef VT_INT
#undef coef_t
#undef acc_t
#undef row_t
#undef VT_BX

#if defined(__x86_64__)
#define HAVE_X86_ASM 1

#define DECL_LINE(type, isa) \
    void p2i_vt_filter_line_##type##_##isa(void *, const void *const *, int, int, const void *);
#define DECL_BLOCK(isa) \
    void p2i_vt_filter_block_u8_##isa(void *, ptrdiff_t, const void *const *, void *, int, \
                                      int, int, const void *);                             \
    void p2i_vt_filter_block_u16_##isa(void *, ptrdiff_t, const void *const *, void *, int, \
                                       int, int, const void *);                             \
    void p2i_vt_filter_block_u16h_##isa(void *, ptrdiff_t, const void *const *, void *, int, \
                                        int, int, const void *);
DECL_LINE(u16, sse2)
DECL_LINE(f32, sse2)
DECL_LINE(u16, avx2)
DECL_LINE(f32, avx2)
DECL_LINE(f16, avx2)
DECL_LINE(u16, avx512)
DECL_LINE(f32, avx512)
DECL_LINE(f16, avx512)
DECL_BLOCK(sse2)
DECL_BLOCK(avx2)
DECL_BLOCK(avxvnni)
DECL_BLOCK(avx512)
DECL_BLOCK(avx512icl)
#define DECL_BLOCK_FLT(isa) \
    void p2i_vt_filter_block_f32_##isa(void *, ptrdiff_t, const void *const *, void *, int, \
                                       int, int, const void *);                             \
    void p2i_vt_filter_block_f16_##isa(void *, ptrdiff_t, const void *const *, void *, int, \
                                       int, int, const void *);
DECL_BLOCK_FLT(avx2)
DECL_BLOCK_FLT(avx512)
void p2i_vt_filter_block_u16n_avx512(void *, ptrdiff_t, const void *const *, void *, int, int, int,
                                     const void *);
void p2i_vt_filter_block_u16n_avx512icl(void *, ptrdiff_t, const void *const *, void *, int, int,
                                        int, const void *);
void p2i_vt_filter_block_u16hn_avx512(void *, ptrdiff_t, const void *const *, void *, int, int, int,
                                      const void *);
void p2i_vt_filter_block_u16hn_avx512icl(void *, ptrdiff_t, const void *const *, void *, int, int,
                                         int, const void *);
#endif

vt_filter_line_fn p2i_get_vt_filter_fn(int kind, unsigned cpu)
{
    static const vt_filter_line_fn fns_c[4] = {
        vt_filter_line_u8, vt_filter_line_u16, vt_filter_line_f32, vt_filter_line_f16,
    };
    vt_filter_line_fn fn = fns_c[kind];
    cpu |= P2I_CPU_BUILTIN;
#if HAVE_X86_ASM
    /* u8 has block kernels only, and f16 none without F16C; C covers both */
    static const vt_filter_line_fn fns_sse2[4] = {
        vt_filter_line_u8, p2i_vt_filter_line_u16_sse2,
        p2i_vt_filter_line_f32_sse2, vt_filter_line_f16,
    };
    static const vt_filter_line_fn fns_avx2[4] = {
        vt_filter_line_u8, p2i_vt_filter_line_u16_avx2,
        p2i_vt_filter_line_f32_avx2, p2i_vt_filter_line_f16_avx2,
    };
    static const vt_filter_line_fn fns_avx512[4] = {
        vt_filter_line_u8, p2i_vt_filter_line_u16_avx512,
        p2i_vt_filter_line_f32_avx512, p2i_vt_filter_line_f16_avx512,
    };
    if (cpu & P2I_CPU_SSE2)
        fn = fns_sse2[kind];
    if (cpu & (kind >= P2I_KIND_F32 ? P2I_CPU_FMA : P2I_CPU_AVX2))
        fn = fns_avx2[kind];
    if (cpu & P2I_CPU_AVX512)
        fn = fns_avx512[kind];
#else
    (void)cpu;
#endif
    return fn;
}

vt_filter_block_fn p2i_get_vt_filter_block_fn(int kind, int depth, unsigned cpu, int *min_width)
{
    vt_filter_block_fn fn;
    *min_width = 1;
    cpu |= P2I_CPU_BUILTIN;
    if (kind == P2I_KIND_F32 || kind == P2I_KIND_F16) {
        const int f32 = kind == P2I_KIND_F32;
        fn = f32 ? vt_filter_block_f32 : vt_filter_block_f16;
#if HAVE_X86_ASM
        /* one x block: 64 bytes of samples */
        if (cpu & P2I_CPU_FMA) {
            fn = f32 ? p2i_vt_filter_block_f32_avx2 : p2i_vt_filter_block_f16_avx2;
            *min_width = f32 ? 16 : 32;
        }
        if (cpu & P2I_CPU_AVX512)
            fn = f32 ? p2i_vt_filter_block_f32_avx512 : p2i_vt_filter_block_f16_avx512;
#endif
        return fn;
    }
    const int u8 = kind == P2I_KIND_U8;
    fn = u8 ? vt_filter_block_u8 : vt_filter_block_u16;
#if HAVE_X86_ASM
    /* the u16 kernels add sample pairs in int16 */
    const int hi = !u8 && depth > 14;
#define SET_BLOCK(isa, width_u8, width_u16) do {                                     \
        fn = u8 ? p2i_vt_filter_block_u8_##isa : hi ? p2i_vt_filter_block_u16h_##isa \
                                                    : p2i_vt_filter_block_u16_##isa; \
        *min_width = u8 ? width_u8 : width_u16;                                      \
    } while (0)
    if (cpu & P2I_CPU_SSE2)
        SET_BLOCK(sse2, 16, 16);
    if (cpu & P2I_CPU_AVX2)
        SET_BLOCK(avx2, 32, 32);
    if (cpu & P2I_CPU_AVXVNNI)
        SET_BLOCK(avxvnni, 32, 32);
    if (cpu & P2I_CPU_AVX512)
        SET_BLOCK(avx512, 64, 64);
    if (cpu & P2I_CPU_AVX512ICL)
        SET_BLOCK(avx512icl, 64, 64);
    if (!u8 && (cpu & P2I_CPU_SLOW_AVX512)) {
        if (cpu & P2I_CPU_AVX512) {
            fn = hi ? p2i_vt_filter_block_u16hn_avx512 : p2i_vt_filter_block_u16n_avx512;
            *min_width = 32;
        }
        if (cpu & P2I_CPU_AVX512ICL)
            fn = hi ? p2i_vt_filter_block_u16hn_avx512icl : p2i_vt_filter_block_u16n_avx512icl;
    }
#undef SET_BLOCK
#else
    (void)cpu;
    (void)depth;
#endif
    return fn;
}

void p2i_vt_filter_init(vt_filter_t *f, int kind, int depth, unsigned cpu)
{
    f->bytes = kind == P2I_KIND_U8 ? 1 : kind == P2I_KIND_F32 ? 4 : 2;
    f->line = p2i_get_vt_filter_fn(kind, cpu);
    f->block = p2i_get_vt_filter_block_fn(kind, depth, cpu, &f->block_min_width);
}

/* bytes of each input line per tile, and the band size from which to tile */
#define VT_TILE_BYTES 512
#define VT_TILE_BAND  (512 << 10)

static int clip_line(int line, int rows)
{
    return line < 0 ? 0 : line >= rows ? rows - 1 : line;
}

/* output line r = progressive line r of the frame supplying that field,
   VT-filtered; line edges repeated */
void p2i_vt_filter_plane(const vt_filter_t *f, uint8_t *dst, ptrdiff_t dst_stride,
                         const uint8_t *const *base, const ptrdiff_t *stride,
                         int width, int rows, int tff, int peak, const void *coefs)
{
    const int top_off = !tff;

    if (f->block_min_width && width >= f->block_min_width) {
        uint8_t tmp_buf[VT_BLOCK_TMP + 63];
        void *tmp = (void *)(((uintptr_t)tmp_buf + 63) & ~(uintptr_t)63);
        /* A band of block calls reads VT_BLOCK_ROWS lines of 6 frames.
           Where that is more than a small L2 holds, go through it in x
           tiles, so that the second parity finds the 4 frames it shares
           with the first in cache. */
        int tile = width;
        if ((VT_TAPS_T + 1) * VT_BLOCK_ROWS * width * f->bytes > VT_TILE_BAND) {
            tile = VT_TILE_BYTES / f->bytes;
            if (tile < f->block_min_width)
                tile = f->block_min_width;
        }

        for (int r0 = 0; r0 < rows; r0 += 2 * VT_BLOCK_MAX) {
            const uint8_t *lines[2][VT_TAPS_T * VT_BLOCK_ROWS];
            int n[2] = { 0, 0 };
            for (int p = 0; p < 2 && r0 + p < rows; p++) {
                const int r = r0 + p;
                const int left = (rows - r + 1) / 2;
                const int o = (r & 1) ^ top_off;
                n[p] = left < VT_BLOCK_MAX ? left : VT_BLOCK_MAX;
                for (int t = 0; t < VT_TAPS_T; t++)
                    for (int j = 0; j < VT_BLOCK_ROWS; j++)
                        lines[p][t * VT_BLOCK_ROWS + j] = base[o + t] +
                            clip_line(r + j - VT_TAPS_V / 2, rows) * stride[o + t];
            }
            for (int x0 = 0; x0 < width;) {
                const int w = width - x0 >= tile + f->block_min_width ? tile : width - x0;
                for (int p = 0; p < 2 && n[p]; p++) {
                    const void *ptrs[VT_TAPS_T * VT_BLOCK_ROWS];
                    for (int i = 0; i < VT_TAPS_T * VT_BLOCK_ROWS; i++)
                        ptrs[i] = lines[p][i] + x0 * f->bytes;
                    f->block(dst + (r0 + p) * dst_stride + x0 * f->bytes, 2 * dst_stride, ptrs,
                             tmp, w, n[p], peak, coefs);
                }
                x0 += w;
            }
        }
        return;
    }

    for (int r = 0; r < rows; r++) {
        const int o = (r & 1) ^ top_off;
        const void *ptrs[VT_SRC_PTRS];
        for (int t = 0; t < VT_TAPS_T; t++)
            for (int v = 0; v < VT_TAPS_V; v++)
                ptrs[t * VT_TAPS_V + v] = base[o + t] +
                    clip_line(r + v - VT_TAPS_V / 2, rows) * stride[o + t];
        ptrs[VT_SRC_PTRS - 1] = ptrs[VT_SRC_PTRS - 2];
        f->line(dst + r * dst_stride, ptrs, width, peak, coefs);
    }
}
