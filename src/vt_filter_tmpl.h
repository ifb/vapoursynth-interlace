/* Instantiated per pixel type: pixel, acc_t, FUNC, VT_INT defined by includer. */

static void FUNC(vt_filter_line)(void *dstv, const void *const *srcv, int width, int peak)
{
    pixel *dst = dstv;
    const pixel *const *src = (const pixel *const *)srcv;

    for (int x = 0; x < width; x++) {
        acc_t sum = 0;
        for (int t = 0; t < VT_TAPS_T; t++)
            for (int v = 0; v < VT_TAPS_V; v++)
#if VT_INT
                sum += vt_coefs_i16[t][v] * (acc_t)src[t * VT_TAPS_V + v][x];
        sum = (sum + (1 << (VT_COEF_BITS - 1))) >> VT_COEF_BITS;
        dst[x] = sum < 0 ? 0 : sum > peak ? peak : sum;
#else
                sum += vt_coefs_f32[t][v] * src[t * VT_TAPS_V + v][x];
        (void)peak;
        dst[x] = sum;
#endif
    }
}
