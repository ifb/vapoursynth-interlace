/*
 * Progressive to interlaced conversion for AviSynth+.
 *
 * The AviSynth+ C-interface half of a dual-host module: the same binary
 * also exports VapourSynthPluginInit2 (interlace.c), and each host looks
 * up only the entry point it knows. The avs_* API resolves by ordinary
 * dynamic linking against the host.
 *
 * Frame n of the output weaves input frames 2n and 2n+1, so the clip
 * halves in length and frame rate. No spatial scaling happens here.
 */

#include <stdlib.h>
#include <string.h>

#include "avisynth_c.h"

#include "cpu.h"
#include "vt_coeffs.h"
#include "vt_filter.h"

#define WINDOW 6

#define P2I_MIN(a, b) ((a) < (b) ? (a) : (b))
#define P2I_MAX(a, b) ((a) > (b) ? (a) : (b))

/* _ChromaLocation values, matching VapourSynth's VSChromaLocation */
enum {
    P2I_CLOC_LEFT,
    P2I_CLOC_CENTER,
    P2I_CLOC_TOPLEFT,
    P2I_CLOC_TOP,
    P2I_CLOC_BOTTOMLEFT,
    P2I_CLOC_BOTTOM,
};

/* see interlace.c: the shift from the progressive chroma position depends
   on the vertical siting class of _ChromaLocation */
static const int chroma_shift[3][2] = { /* [vclass][bottom] */
    { P2I_SHIFT_M125, P2I_SHIFT_P125 },
    { P2I_SHIFT_P125, P2I_SHIFT_P375 },
    { P2I_SHIFT_M375, P2I_SHIFT_M125 },
};

typedef struct p2i_avs_t p2i_avs_t;

struct p2i_avs_t {
    int in_frames;
    int tff;
    int filter;
    int peak;
    int kind;
    int chroma_sited;
    int planes[3];      /* AVS_PLANAR_* for each plane, in order */
    int num_planes;
    int bytes_per_sample;
    int subsampling_h;
    vt_filter_line_fn vt_filter_line;
};

static AVS_VideoFrame *AVSC_CC p2i_avs_get_frame(AVS_FilterInfo *fi, int n)
{
    p2i_avs_t *f = fi->user_data;
    AVS_ScriptEnvironment *env = fi->env;
    const int last = f->in_frames - 1;

    /* src[k] = input frame 2n+k-2, frame edges repeated */
    AVS_VideoFrame *src[WINDOW] = { NULL };
    const int lo = f->filter ? 0 : 2;
    const int hi = f->filter ? WINDOW : 4;
    for (int k = lo; k < hi; k++) {
        int q = P2I_MIN(P2I_MAX(n * 2 + k - 2, 0), last);
        src[k] = avs_get_frame(fi->child, q);
        if (!src[k]) {
            for (int j = lo; j < k; j++)
                avs_release_video_frame(src[j]);
            return NULL;
        }
    }

    AVS_VideoFrame *first = src[2];
    AVS_VideoFrame *second = src[3];
    AVS_VideoFrame *top = f->tff ? first : second;
    AVS_VideoFrame *bottom = f->tff ? second : first;

    int64_t cloc = P2I_CLOC_LEFT;
    const void *chroma_coefs[2] = { NULL, NULL };
    if (f->chroma_sited) {
        const AVS_Map *props = avs_get_frame_props_ro(env, first);
        int err = 0;
        cloc = avs_prop_get_int(env, props, "_ChromaLocation", 0, &err);
        if (err)
            cloc = P2I_CLOC_LEFT;
        if (cloc < 0 || cloc > P2I_CLOC_BOTTOM) {
            for (int k = lo; k < hi; k++)
                avs_release_video_frame(src[k]);
            fi->error = "Interlace: unsupported _ChromaLocation";
            return NULL;
        }
        for (int b = 0; b < 2; b++)
            chroma_coefs[b] = p2i_vt_coefs(f->kind, chroma_shift[cloc >> 1][b]);
    }

    AVS_VideoFrame *dst = avs_new_video_frame_p(env, &fi->vi, first);

    for (int p = 0; p < f->num_planes; p++) {
        const int plane = f->planes[p];
        const int rows = avs_get_height_p(dst, plane);
        const int width = avs_get_row_size_p(dst, plane) / f->bytes_per_sample;
        const ptrdiff_t dst_stride = avs_get_pitch_p(dst, plane);
        uint8_t *dstp = avs_get_write_ptr_p(dst, plane);

        if (!f->filter) {
            const ptrdiff_t top_stride = avs_get_pitch_p(top, plane);
            const ptrdiff_t bot_stride = avs_get_pitch_p(bottom, plane);
            const int row_bytes = width * f->bytes_per_sample;
            avs_bit_blt(env, dstp, dst_stride * 2,
                        avs_get_read_ptr_p(top, plane), top_stride * 2,
                        row_bytes, rows / 2);
            avs_bit_blt(env, dstp + dst_stride, dst_stride * 2,
                        avs_get_read_ptr_p(bottom, plane) + bot_stride, bot_stride * 2,
                        row_bytes, rows / 2);
            continue;
        }

        const uint8_t *base[WINDOW];
        ptrdiff_t stride[WINDOW];
        for (int k = 0; k < WINDOW; k++) {
            base[k] = avs_get_read_ptr_p(src[k], plane);
            stride[k] = avs_get_pitch_p(src[k], plane);
        }

        const void *center = p2i_vt_coefs(f->kind, P2I_SHIFT_CENTER);
        const int sited = p > 0 && f->chroma_sited;
        const int top_off = f->tff ? 0 : 1;
        for (int r = 0; r < rows; r++) {
            const int o = (r & 1) ? !top_off : top_off;
            const void *ptrs[VT_SRC_PTRS];
            for (int t = 0; t < VT_TAPS_T; t++)
                for (int v = 0; v < VT_TAPS_V; v++) {
                    int line = P2I_MIN(P2I_MAX(r + v - VT_TAPS_V / 2, 0), rows - 1);
                    ptrs[t * VT_TAPS_V + v] = base[o + t] + line * stride[o + t];
                }
            ptrs[VT_TAPS_T * VT_TAPS_V] = ptrs[VT_TAPS_T * VT_TAPS_V - 1];
            f->vt_filter_line(dstp + r * dst_stride, ptrs, width, f->peak,
                              sited ? chroma_coefs[r & 1] : center);
        }
    }

    AVS_Map *props = avs_get_frame_props_rw(env, dst);
    avs_prop_set_int(env, props, "_FieldBased", f->tff ? 2 : 1, 0);
    if (f->chroma_sited)
        avs_prop_set_int(env, props, "_ChromaLocation",
                         (cloc & 1) ? P2I_CLOC_CENTER : P2I_CLOC_LEFT, 0);

    int err = 0;
    int64_t dur_num = avs_prop_get_int(env, props, "_DurationNum", 0, &err);
    if (!err) {
        int64_t dur_den = avs_prop_get_int(env, props, "_DurationDen", 0, &err);
        if (!err && dur_den > 0) {
            /* halve the rate: double the duration, reduced by gcd */
            int64_t a = dur_num * 2, b = dur_den;
            while (b) {
                int64_t t = a % b;
                a = b;
                b = t;
            }
            avs_prop_set_int(env, props, "_DurationNum", dur_num * 2 / (a ? a : 1), 0);
            avs_prop_set_int(env, props, "_DurationDen", dur_den / (a ? a : 1), 0);
        }
    }

    for (int k = lo; k < hi; k++)
        avs_release_video_frame(src[k]);
    return dst;
}

static int AVSC_CC p2i_avs_set_cache_hints(AVS_FilterInfo *fi, int cachehints,
                                           int frame_range)
{
    (void)fi; (void)frame_range;
    /* filter state is read-only after init and each call writes only its
     * own output frame */
    return cachehints == AVS_CACHE_GET_MTMODE ? AVS_MT_NICE_FILTER : 0;
}

static void AVSC_CC p2i_avs_free(AVS_FilterInfo *fi)
{
    free(fi->user_data);
}

static AVS_Value AVSC_CC p2i_avs_create(AVS_ScriptEnvironment *env,
                                        AVS_Value args, void *user_data)
{
    (void)user_data;

    /* args: clip, tff, filter, cpu */
    AVS_Value clip_v = avs_array_elt(args, 0);
    const int tff = avs_defined(avs_array_elt(args, 1))
                    ? !!avs_as_bool(avs_array_elt(args, 1)) : 1;
    const int filter = avs_defined(avs_array_elt(args, 2))
                       ? !!avs_as_bool(avs_array_elt(args, 2)) : 1;

    unsigned cpu = p2i_cpu_detect();
    if (avs_defined(avs_array_elt(args, 3))) {
        static const struct { const char *name; unsigned mask; } levels[] = {
            { "none",   0 },
            { "sse2",   P2I_CPU_SSE2 },
            { "avx2",   P2I_CPU_SSE2 | P2I_CPU_AVX2 },
            { "avx512", P2I_CPU_SSE2 | P2I_CPU_AVX2 | P2I_CPU_AVX512 },
        };
        const char *cpu_req = avs_as_string(avs_array_elt(args, 3));
        int i;
        for (i = 0; i < 4; i++)
            if (!strcmp(cpu_req, levels[i].name)) {
                cpu &= levels[i].mask;
                break;
            }
        if (i == 4)
            return avs_new_value_error("Interlace: cpu must be one of none/sse2/avx2/avx512");
    }

    AVS_Clip *src_clip = avs_take_clip(clip_v, env);
    const AVS_VideoInfo *svi = avs_get_video_info(src_clip);
    const int sub_h = avs_is_y(svi) ? 0 : avs_get_plane_height_subsampling(svi, AVS_PLANAR_U);
    const int height = svi->height;
    const int num_frames = svi->num_frames;
    const int is_planar = avs_is_planar(svi);
    const int is_rgb = avs_is_rgb(svi);
    const int comp_size = avs_component_size(svi);
    /* every 32-bit AviSynth+ sample format is float (no int32 planar type) */
    const int is_float = comp_size == 4;
    const int bits = avs_bits_per_component(svi);
    avs_release_clip(src_clip);

    if (!is_planar || is_rgb)
        return avs_new_value_error("Interlace: clip must be planar YUV or Y");
    if ((height >> sub_h) & 1)
        return avs_new_value_error("Interlace: all plane heights must be even");
    if (num_frames < 2)
        return avs_new_value_error("Interlace: clip too short");
    if (filter && sub_h > 1)
        return avs_new_value_error("Interlace: vertical chroma subsampling greater than 2x is not supported");

    AVS_FilterInfo *fi;
    AVS_Clip *out = avs_new_c_filter(env, &fi, clip_v, 1);
    if (!out)
        return avs_new_value_error("Interlace: failed to create filter");

    p2i_avs_t *f = calloc(1, sizeof(*f));
    if (!f) {
        avs_release_clip(out);
        return avs_new_value_error("Interlace: out of memory");
    }

    f->in_frames = num_frames;
    f->tff = tff;
    f->filter = filter;
    f->bytes_per_sample = comp_size;
    f->subsampling_h = sub_h;

    if (avs_is_y(&fi->vi)) {
        f->planes[0] = AVS_PLANAR_Y;
        f->num_planes = 1;
    } else {
        f->planes[0] = AVS_PLANAR_Y;
        f->planes[1] = AVS_PLANAR_U;
        f->planes[2] = AVS_PLANAR_V;
        f->num_planes = 3;
    }

    if (filter) {
        /* AviSynth+ has no half-float format, so f32 is the only float kind */
        f->kind = is_float ? P2I_KIND_F32
                           : (comp_size == 1 ? P2I_KIND_U8 : P2I_KIND_U16);
        f->vt_filter_line = p2i_get_vt_filter_fn(f->kind, cpu);
        f->peak = is_float ? 0 : (1 << bits) - 1;
        f->chroma_sited = sub_h > 0;
    }

    fi->vi.num_frames = num_frames / 2;
    fi->vi.fps_denominator *= 2;
    fi->vi.image_type |= AVS_IT_FIELDBASED | (tff ? AVS_IT_TFF : AVS_IT_BFF);
    fi->vi.image_type &= ~(tff ? AVS_IT_BFF : AVS_IT_TFF);

    fi->get_frame = p2i_avs_get_frame;
    fi->set_cache_hints = p2i_avs_set_cache_hints;
    fi->free_filter = p2i_avs_free;
    fi->user_data = f;

    AVS_Value ret;
    avs_set_to_clip(&ret, out);
    avs_release_clip(out);
    return ret;
}

__attribute__((visibility("default")))
AVSC_EXPORT const char *AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment *env)
{
    avs_add_function(env, "Interlace", "c[tff]b[filter]b[cpu]s",
                     p2i_avs_create, NULL);
    return "interlace: progressive to interlaced converter";
}
