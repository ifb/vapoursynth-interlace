#ifndef P2I_VT_FILTER_H
#define P2I_VT_FILTER_H

enum {
    P2I_KIND_U8,
    P2I_KIND_U16,
    P2I_KIND_F32,
    P2I_KIND_F16,
};

/* src: VT_TAPS_T * VT_TAPS_V line pointers, [t][v] flattened, edges
 * pre-clamped, plus one extra (any valid line) so SIMD can process
 * coefficients in pairs; its coefficient is 0. */
#define VT_SRC_PTRS (5 * 19 + 1)

typedef void (*vt_filter_line_fn)(void *dst, const void *const *src, int width, int peak);

vt_filter_line_fn p2i_get_vt_filter_fn(int kind, unsigned cpu);

#endif
