/* Instantiated per pixel type: pixel, row_t, acc_t, coef_t, FUNC, VT_INT, VT_BX, LOAD and
 * STORE defined by includer. VT_BX is the x block in samples: 3 * VT_BLOCK_ROWS rows of it
 * must fit VT_BLOCK_TMP. Output is summed VT_XV samples at a time, a multiple of which
 * VT_BX must be. */

#define VT_XV 32

static void FUNC(vt_filter_block)(void *dstv, ptrdiff_t dst_stride, const void *const *srcv,
                                  void *tmpv, int width, int nb_rows, int peak,
                                  const void *coefsv)
{
    const pixel *const *src = (const pixel *const *)srcv;
    const coef_t *coefs = coefsv;
    row_t (*rows)[VT_BLOCK_ROWS][VT_BX] = tmpv;
    const int nb_in = 2 * nb_rows + VT_TAPS_V - 2;
    (void)peak;

    for (int x0 = 0; x0 < width; x0 += VT_BX) {
        const int n = width - x0 < VT_BX ? width - x0 : VT_BX;

        for (int j = 0; j < nb_in; j++) {
            const pixel *restrict s0 = src[0 * VT_BLOCK_ROWS + j] + x0;
            const pixel *restrict s1 = src[1 * VT_BLOCK_ROWS + j] + x0;
            const pixel *restrict s2 = src[2 * VT_BLOCK_ROWS + j] + x0;
            const pixel *restrict s3 = src[3 * VT_BLOCK_ROWS + j] + x0;
            const pixel *restrict s4 = src[4 * VT_BLOCK_ROWS + j] + x0;
            row_t *restrict a = rows[0][j], *restrict b = rows[1][j], *restrict c = rows[2][j];
            for (int x = 0; x < n; x++) {
                a[x] = LOAD(s0[x]) + LOAD(s4[x]);
                b[x] = LOAD(s1[x]) + LOAD(s3[x]);
                c[x] = LOAD(s2[x]);
            }
            for (int x = n; x < VT_BX; x++)
                a[x] = b[x] = c[x] = 0;
        }

        for (int k = 0; k < nb_rows; k++) {
            pixel *restrict d = (pixel *)((uint8_t *)dstv + k * dst_stride) + x0;
            for (int xb = 0; xb < n; xb += VT_XV) {
                const int m = n - xb < VT_XV ? n - xb : VT_XV;
                acc_t acc[VT_XV];
                for (int x = 0; x < VT_XV; x++)
                    acc[x] = VT_INT ? 1 << (VT_COEF_BITS - 1) : 0;
                /* unrolled fully, gcc keeps acc in memory */
#pragma GCC unroll 1
                for (int v = 0; v < VT_TAPS_V; v++) {
                    const coef_t ca = coefs[v];
                    const coef_t cb = coefs[VT_TAPS_V + v];
                    const coef_t cc = coefs[2 * VT_TAPS_V + v];
                    const row_t *restrict a = rows[0][2 * k + v] + xb;
                    const row_t *restrict b = rows[1][2 * k + v] + xb;
                    const row_t *restrict c = rows[2][2 * k + v] + xb;
                    for (int x = 0; x < VT_XV; x++)
                        acc[x] += ca * (acc_t)a[x] + cb * (acc_t)b[x] + cc * (acc_t)c[x];
                }
#if VT_INT
                for (int x = 0; x < m; x++) {
                    const acc_t v = acc[x] >> VT_COEF_BITS;
                    d[xb + x] = v < 0 ? 0 : v > peak ? peak : v;
                }
#else
                for (int x = 0; x < m; x++)
                    d[xb + x] = STORE(acc[x]);
#endif
            }
        }
    }
}

#undef VT_XV
