#ifndef P2I_VT_FILTER_H
#define P2I_VT_FILTER_H

#include <stddef.h>
#include <stdint.h>

enum {
    P2I_KIND_U8,
    P2I_KIND_U16,
    P2I_KIND_F32,
    P2I_KIND_F16,
};

/* fractional vertical delay of the coefficient set, in chroma rows;
   moves subsampled chroma onto the interlaced siting grid */
enum {
    P2I_SHIFT_M250,
    P2I_SHIFT_CENTER,
    P2I_SHIFT_P250,
    P2I_NB_SHIFTS,
};

/* src: VT_TAPS_T * VT_TAPS_V line pointers, [t][v] flattened, edges
 * pre-clamped, plus one extra (any valid line) so SIMD can process
 * coefficients in pairs; its coefficient is 0. */
#define VT_SRC_PTRS (5 * 19 + 1)

typedef void (*vt_filter_line_fn)(void *dst, const void *const *src, int width, int peak,
                                  const void *coefs);

/* output lines per block call at most */
#define VT_BLOCK_MAX  12
/* line pointers per frame for a block call */
#define VT_BLOCK_ROWS (2 * VT_BLOCK_MAX + 18)
/* block scratch size in bytes: the integer kernels' tables, their prepared
   rows at up to 512 bytes each and pairs at 256; the float kernels' three
   row sets at 128 bytes a row fit in it */
#define VT_BLOCK_TMP  (192 + (2 * VT_BLOCK_ROWS + VT_BLOCK_MAX + 9) * 4 * 64)

/* Filter nb_rows (1..VT_BLOCK_MAX) output lines r, r + 2, ..., which use
 * the same frames and coefficients; line k goes to dst + k * dst_stride.
 * src: 5 * VT_BLOCK_ROWS line pointers, [t][j] flattened, [t][j] = line
 * r - 9 + j of frame t, edges pre-clamped; all must be valid, even those
 * past 2 * nb_rows + 18. tmp: VT_BLOCK_TMP bytes, 64-byte aligned.
 * At 14 bits and below, samples must not exceed peak. width must be at
 * least the getter's min_width. Nothing outside the nb_rows x width output is
 * written. Float kernels sum in another order than the line kernels. */
typedef void (*vt_filter_block_fn)(void *dst, ptrdiff_t dst_stride, const void *const *src,
                                   void *tmp, int width, int nb_rows, int peak,
                                   const void *coefs);

typedef struct vt_filter_t {
    vt_filter_line_fn line;
    vt_filter_block_fn block;
    int block_min_width;        /* 0: no block kernel, filter line by line */
    int bytes;                  /* per sample */
} vt_filter_t;

vt_filter_line_fn p2i_get_vt_filter_fn(int kind, unsigned cpu);
vt_filter_block_fn p2i_get_vt_filter_block_fn(int kind, int depth, unsigned cpu, int *min_width);
void p2i_vt_filter_init(vt_filter_t *f, int kind, int depth, unsigned cpu);

/* Filter one plane of an output frame. base/stride: the plane in input
 * frames 2n-2 .. 2n+3. */
void p2i_vt_filter_plane(const vt_filter_t *f, uint8_t *dst, ptrdiff_t dst_stride,
                         const uint8_t *const *base, const ptrdiff_t *stride,
                         int width, int rows, int tff, int peak, const void *coefs);
const void *p2i_vt_coefs(int kind, int shift);

#endif
