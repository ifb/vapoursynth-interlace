/*
 * Vertical-temporal prefilter kernels (BBC WHP 315), scalar.
 */

#include <stdint.h>

#include "vt_coeffs.h"
#include "vt_filter.h"

#define pixel uint8_t
#define acc_t int32_t
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
#undef FUNC
#undef VT_INT

#define pixel float
#define acc_t float
#define FUNC(name) name##_f32
#define VT_INT 0
#include "vt_filter_tmpl.h"
#undef pixel
#undef acc_t
#undef FUNC
#undef VT_INT

vt_filter_line_fn p2i_get_vt_filter_fn(const VSVideoFormat *fmt)
{
    if (fmt->sampleType == stFloat)
        return fmt->bytesPerSample == 4 ? vt_filter_line_f32 : NULL;
    return fmt->bytesPerSample == 1 ? vt_filter_line_u8 : vt_filter_line_u16;
}
