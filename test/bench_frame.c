/*
 * Whole-frame benchmark for the vertical-temporal filter.
 *
 *   bench_frame [reps]
 *
 * checkasm --bench times one line with its 96 source pointers already hot,
 * which answers "how fast is the kernel" but not "how fast is the filter at
 * a real frame size". This drives the same per-row loop src/interlace.c
 * runs — 96 line pointers rebuilt per output row, edges clamped, all planes —
 * over HD and UHD geometry, so the cost includes the memory behaviour of
 * walking a real frame.
 *
 * Reports the minimum over `reps` passes (default 9). Minimums, not means:
 * the distribution is one-sided, so the fastest pass is the one least
 * disturbed by the scheduler.
 *
 * Self-contained — no VapourSynth, no Python. Build it static and copy it
 * to whatever machine you want the numbers from.
 */

/* clock_gettime under -std=c99 */
#define _POSIX_C_SOURCE 200112L

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cpu.h"
#include "vt_coeffs.h"
#include "vt_filter.h"

#define WINDOW 6

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

/* MinGW-UCRT lacks posix_memalign; over-allocate and stash the base pointer so
 * plain free() works on every host. */
static void *aligned_alloc64(size_t sz)
{
    void *base = malloc(sz + 64 + sizeof(void *));
    if (!base)
        return NULL;
    uintptr_t p = (uintptr_t)base + sizeof(void *);
    p = (p + 63) & ~(uintptr_t)63;
    ((void **)p)[-1] = base;
    return (void *)p;
}

static void aligned_free(void *p)
{
    if (p)
        free(((void **)p)[-1]);
}

struct geom {
    const char *name;
    int w, h;
};

struct fmt {
    const char *name;
    int kind;
    int bytes;
    int peak;
    int sub_w, sub_h;   /* chroma subsampling shifts */
};

/* One plane, one output frame: the row loop from src/interlace.c. */
static void filter_plane(vt_filter_line_fn fn, uint8_t *dst, ptrdiff_t dst_stride,
                         uint8_t *const *base, ptrdiff_t stride,
                         int rows, int width, int peak, const void *coefs, int tff)
{
    const int top_off = tff ? 0 : 1;
    for (int r = 0; r < rows; r++) {
        const int o = (r & 1) ? !top_off : top_off;
        const void *ptrs[VT_SRC_PTRS];
        for (int t = 0; t < VT_TAPS_T; t++)
            for (int v = 0; v < VT_TAPS_V; v++) {
                int line = r + v - VT_TAPS_V / 2;
                line = line < 0 ? 0 : (line >= rows ? rows - 1 : line);
                ptrs[t * VT_TAPS_V + v] = base[o + t] + (ptrdiff_t)line * stride;
            }
        ptrs[VT_TAPS_T * VT_TAPS_V] = ptrs[VT_TAPS_T * VT_TAPS_V - 1];
        fn(dst + (ptrdiff_t)r * dst_stride, ptrs, width, peak, coefs);
    }
}

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 9;
    const unsigned cpu = p2i_cpu_detect();

    static const struct geom geoms[] = {
        { "HD  1920x1080", 1920, 1080 },
        { "UHD 3840x2160", 3840, 2160 },
    };
    static const struct fmt fmts[] = {
        { "YUV420P8 ", P2I_KIND_U8,  1,   255, 1, 1 },
        { "YUV422P10", P2I_KIND_U16, 2,  1023, 1, 0 },
        { "YUV420P16", P2I_KIND_U16, 2, 65535, 1, 1 },
        { "YUV420PS ", P2I_KIND_F32, 4,     0, 1, 1 },
        { "YUV420PH ", P2I_KIND_F16, 2,     0, 1, 1 },
    };

    printf("cpu: %s%s%s\n",
           cpu & P2I_CPU_SSE2 ? "sse2 " : "",
           cpu & P2I_CPU_AVX2 ? "avx2 " : "",
           cpu & P2I_CPU_AVX512 ? "avx512" : "");
    printf("min of %d passes, per luma pixel of the output frame\n", reps);
    printf("take the minimum across several invocations: the large-working-set\n"
           "rows (UHD f32 reads 158 MB) move with allocation placement\n\n");
    printf("%-14s %-10s %10s %10s %9s\n",
           "geometry", "format", "ms/frame", "Mpx/s", "ns/luma-px");

    for (size_t g = 0; g < sizeof(geoms) / sizeof(*geoms); g++) {
        for (size_t f = 0; f < sizeof(fmts) / sizeof(*fmts); f++) {
            const struct geom *G = &geoms[g];
            const struct fmt *F = &fmts[f];
            vt_filter_line_fn fn = p2i_get_vt_filter_fn(F->kind, cpu);
            const void *coefs = p2i_vt_coefs(F->kind, P2I_SHIFT_CENTER);

            /* one luma plane + two chroma planes per input frame */
            uint8_t *frames[WINDOW] = { NULL };
            const ptrdiff_t ystride = (ptrdiff_t)G->w * F->bytes;
            const ptrdiff_t cstride = ystride >> F->sub_w;
            const int crows = G->h >> F->sub_h;
            const size_t fsz = (size_t)ystride * G->h + 2 * (size_t)cstride * crows;
            for (int k = 0; k < WINDOW; k++) {
                frames[k] = aligned_alloc64(fsz);
                if (!frames[k]) {
                    fprintf(stderr, "out of memory\n");
                    return 1;
                }
                for (size_t i = 0; i < fsz; i++)
                    frames[k][i] = (uint8_t)(i * 7 + k * 31);
                /* Random bytes reinterpreted as floats give wild exponents,
                 * and the subnormals among them cost hundreds of cycles
                 * apiece — that would time the FPU's denormal handling
                 * rather than the filter. Pin the exponent to [0.5,1) the
                 * way the checkasm test does. */
                if (F->kind == P2I_KIND_F32) {
                    uint32_t *p = (uint32_t *)frames[k];
                    for (size_t i = 0; i < fsz / 4; i++)
                        p[i] = (p[i] & 0x807fffffu) | 0x3f000000u;
                } else if (F->kind == P2I_KIND_F16) {
                    uint16_t *p = (uint16_t *)frames[k];
                    for (size_t i = 0; i < fsz / 2; i++)
                        p[i] = (uint16_t)((p[i] & 0x83ffu) | 0x3800u);
                }
            }
            uint8_t *dst = aligned_alloc64(fsz);
            if (!dst) {
                fprintf(stderr, "out of memory\n");
                return 1;
            }

            uint8_t *ybase[WINDOW], *ubase[WINDOW], *vbase[WINDOW];
            for (int k = 0; k < WINDOW; k++) {
                ybase[k] = frames[k];
                ubase[k] = frames[k] + (size_t)ystride * G->h;
                vbase[k] = ubase[k] + (size_t)cstride * crows;
            }
            uint8_t *dy = dst;
            uint8_t *du = dst + (size_t)ystride * G->h;
            uint8_t *dv = du + (size_t)cstride * crows;

            uint64_t best = ~0ull;
            for (int r = 0; r < reps; r++) {
                const uint64_t t0 = now_ns();
                filter_plane(fn, dy, ystride, ybase, ystride, G->h, G->w,
                             F->peak, coefs, 1);
                filter_plane(fn, du, cstride, ubase, cstride, crows,
                             G->w >> F->sub_w, F->peak, coefs, 1);
                filter_plane(fn, dv, cstride, vbase, cstride, crows,
                             G->w >> F->sub_w, F->peak, coefs, 1);
                const uint64_t dt = now_ns() - t0;
                if (dt < best)
                    best = dt;
            }

            const double ms = best / 1e6;
            const double luma = (double)G->w * G->h;
            printf("%-14s %-10s %10.2f %10.1f %9.3f\n",
                   G->name, F->name, ms, luma / (best / 1e3), best / luma);

            for (int k = 0; k < WINDOW; k++)
                aligned_free(frames[k]);
            aligned_free(dst);
        }
        printf("\n");
    }
    return 0;
}
