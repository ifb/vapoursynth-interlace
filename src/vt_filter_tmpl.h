/* Instantiated per pixel type: pixel, acc_t, coef_t, FUNC, VT_INT defined by includer. */

#define VT_TILE 512

static void FUNC(vt_filter_line)(void *dstv, const void *const *srcv, int width, int peak,
                                 const void *coefsv)
{
    pixel *restrict dst = dstv;
    const pixel *const *restrict src = (const pixel *const *)srcv;
    const coef_t *restrict coefs = coefsv;

    for (int x0 = 0; x0 < width; x0 += VT_TILE) {
        int n = width - x0 < VT_TILE ? width - x0 : VT_TILE;
        acc_t acc[VT_TILE];

        for (int x = 0; x < n; x++)
            acc[x] = 0;
        for (int i = 0; i < VT_TAPS_T * VT_TAPS_V; i++) {
            const pixel *restrict s = src[i] + x0;
            coef_t c = coefs[i];
            for (int x = 0; x < n; x++)
                acc[x] += c * (acc_t)s[x];
        }
#if VT_INT
        for (int x = 0; x < n; x++) {
            acc_t v = (acc[x] + (1 << (VT_COEF_BITS - 1))) >> VT_COEF_BITS;
            dst[x0 + x] = v < 0 ? 0 : v > peak ? peak : v;
        }
#else
        (void)peak;
        for (int x = 0; x < n; x++)
            dst[x0 + x] = acc[x];
#endif
    }
}

#undef VT_TILE
