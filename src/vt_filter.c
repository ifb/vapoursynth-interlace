/*
 * Vertical-temporal prefilter kernels (BBC WHP 315).
 */

#include <stdint.h>

#include "cpu.h"
#include "vt_coeffs.h"
#include "vt_filter.h"

/* external linkage: the asm kernels read these */
const int16_t p2i_vt_coefs_i16[96] = { VT_COEFS_I16 };
const float p2i_vt_coefs_f32[95] = { VT_COEFS_F32 };

#define pixel uint8_t
#define acc_t int32_t
#define FUNC(name) name##_u8
#define COEFS p2i_vt_coefs_i16
#define VT_INT 1
#include "vt_filter_tmpl.h"
#undef pixel
#undef FUNC

#define pixel uint16_t
#define FUNC(name) name##_u16
#include "vt_filter_tmpl.h"
#undef pixel
#undef acc_t
#undef FUNC
#undef COEFS
#undef VT_INT

#define pixel float
#define acc_t float
#define FUNC(name) name##_f32
#define COEFS p2i_vt_coefs_f32
#define VT_INT 0
#include "vt_filter_tmpl.h"
#undef pixel
#undef acc_t
#undef FUNC
#undef COEFS
#undef VT_INT

#if defined(__x86_64__)
#define HAVE_X86_ASM 1

#define DECL_KERNELS(isa) \
    void p2i_vt_filter_line_u8_##isa(void *, const void *const *, int, int); \
    void p2i_vt_filter_line_u16_##isa(void *, const void *const *, int, int); \
    void p2i_vt_filter_line_f32_##isa(void *, const void *const *, int, int);
DECL_KERNELS(sse2)
DECL_KERNELS(avx2)
DECL_KERNELS(avx512)
#endif

vt_filter_line_fn p2i_get_vt_filter_fn(int kind, unsigned cpu)
{
    static const vt_filter_line_fn fns_c[3] = {
        vt_filter_line_u8, vt_filter_line_u16, vt_filter_line_f32,
    };
    vt_filter_line_fn fn = fns_c[kind];
#if HAVE_X86_ASM
    static const vt_filter_line_fn fns_sse2[3] = {
        p2i_vt_filter_line_u8_sse2, p2i_vt_filter_line_u16_sse2, p2i_vt_filter_line_f32_sse2,
    };
    static const vt_filter_line_fn fns_avx2[3] = {
        p2i_vt_filter_line_u8_avx2, p2i_vt_filter_line_u16_avx2, p2i_vt_filter_line_f32_avx2,
    };
    static const vt_filter_line_fn fns_avx512[3] = {
        p2i_vt_filter_line_u8_avx512, p2i_vt_filter_line_u16_avx512,
        p2i_vt_filter_line_f32_avx512,
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
