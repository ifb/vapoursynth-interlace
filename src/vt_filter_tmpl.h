/* Instantiated per pixel type: pixel, acc_t, coef_t, FUNC, VT_INT defined by includer. */

static void FUNC(vt_filter_line)(void *dstv, const void *const *srcv, int width, int peak,
                                 const void *coefsv)
{
    pixel *dst = dstv;
    const pixel *const *src = (const pixel *const *)srcv;
    const coef_t *coefs = coefsv;

    for (int x = 0; x < width; x++) {
        acc_t sum = 0;
        for (int i = 0; i < VT_TAPS_T * VT_TAPS_V; i++)
            sum += coefs[i] * (acc_t)src[i][x];
#if VT_INT
        sum = (sum + (1 << (VT_COEF_BITS - 1))) >> VT_COEF_BITS;
        dst[x] = sum < 0 ? 0 : sum > peak ? peak : sum;
#else
        (void)peak;
        dst[x] = sum;
#endif
    }
}
