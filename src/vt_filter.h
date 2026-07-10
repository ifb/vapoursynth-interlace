#ifndef P2I_VT_FILTER_H
#define P2I_VT_FILTER_H

#include <VapourSynth4.h>

/* src: VT_TAPS_T * VT_TAPS_V line pointers, [t][v] flattened, edges pre-clamped */
typedef void (*vt_filter_line_fn)(void *dst, const void *const *src, int width, int peak);

vt_filter_line_fn p2i_get_vt_filter_fn(const VSVideoFormat *fmt);

#endif
