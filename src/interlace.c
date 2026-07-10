/*
 * Progressive to interlaced conversion for VapourSynth.
 */

#include <stdlib.h>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>

#include "vt_coeffs.h"
#include "vt_filter.h"

/* input frames needed per output frame: dt -2..+2 around both field centers */
#define WINDOW 6

typedef struct p2i_filter_t p2i_filter_t;

struct p2i_filter_t {
    VSNode *node;
    VSVideoInfo vi;
    int in_frames;
    int tff;
    int filter;
    int peak;
    vt_filter_line_fn vt_filter_line;
};

static const VSFrame *VS_CC p2i_get_frame(int n, int activation_reason, void *instance_data,
                                          void **frame_data, VSFrameContext *frame_ctx,
                                          VSCore *core, const VSAPI *vsapi)
{
    p2i_filter_t *f = instance_data;
    (void)frame_data;

    if (activation_reason == arInitial) {
        if (f->filter) {
            for (int k = 0; k < WINDOW; k++) {
                int q = VSMIN(VSMAX(n * 2 + k - 2, 0), f->in_frames - 1);
                if (k == 0 || q != VSMIN(VSMAX(n * 2 + k - 3, 0), f->in_frames - 1))
                    vsapi->requestFrameFilter(q, f->node, frame_ctx);
            }
        } else {
            vsapi->requestFrameFilter(n * 2,     f->node, frame_ctx);
            vsapi->requestFrameFilter(n * 2 + 1, f->node, frame_ctx);
        }
        return NULL;
    }
    if (activation_reason != arAllFramesReady)
        return NULL;

    /* src[k] = input frame 2n+k-2, frame edges repeated */
    const VSFrame *src[WINDOW];
    if (f->filter) {
        for (int k = 0; k < WINDOW; k++)
            src[k] = vsapi->getFrameFilter(VSMIN(VSMAX(n * 2 + k - 2, 0), f->in_frames - 1),
                                           f->node, frame_ctx);
    } else {
        src[2] = vsapi->getFrameFilter(n * 2,     f->node, frame_ctx);
        src[3] = vsapi->getFrameFilter(n * 2 + 1, f->node, frame_ctx);
    }
    const VSFrame *first  = src[2];
    const VSFrame *second = src[3];
    const VSFrame *top    = f->tff ? first : second;
    const VSFrame *bottom = f->tff ? second : first;

    VSFrame *dst = vsapi->newVideoFrame(&f->vi.format, f->vi.width, f->vi.height, first, core);

    for (int plane = 0; plane < f->vi.format.numPlanes; plane++) {
        int rows = vsapi->getFrameHeight(dst, plane);
        int width = vsapi->getFrameWidth(dst, plane);
        ptrdiff_t dst_stride = vsapi->getStride(dst, plane);
        uint8_t *dstp = vsapi->getWritePtr(dst, plane);

        if (!f->filter) {
            ptrdiff_t top_stride = vsapi->getStride(top, plane);
            ptrdiff_t bot_stride = vsapi->getStride(bottom, plane);
            vsh_bitblt(dstp, dst_stride * 2,
                       vsapi->getReadPtr(top, plane), top_stride * 2,
                       width * f->vi.format.bytesPerSample, rows / 2);
            vsh_bitblt(dstp + dst_stride, dst_stride * 2,
                       vsapi->getReadPtr(bottom, plane) + bot_stride, bot_stride * 2,
                       width * f->vi.format.bytesPerSample, rows / 2);
            continue;
        }

        const uint8_t *base[WINDOW];
        ptrdiff_t stride[WINDOW];
        for (int k = 0; k < WINDOW; k++) {
            base[k] = vsapi->getReadPtr(src[k], plane);
            stride[k] = vsapi->getStride(src[k], plane);
        }

        /* output line r = progressive line r of the frame supplying that
           field, VT-filtered; line edges repeated */
        int top_off = f->tff ? 0 : 1;
        for (int r = 0; r < rows; r++) {
            int o = (r & 1) ? !top_off : top_off;
            const void *ptrs[VT_TAPS_T * VT_TAPS_V];
            for (int t = 0; t < VT_TAPS_T; t++)
                for (int v = 0; v < VT_TAPS_V; v++) {
                    int line = VSMIN(VSMAX(r + v - VT_TAPS_V / 2, 0), rows - 1);
                    ptrs[t * VT_TAPS_V + v] = base[o + t] + line * stride[o + t];
                }
            f->vt_filter_line(dstp + r * dst_stride, ptrs, width, f->peak);
        }
    }

    VSMap *props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetInt(props, "_FieldBased", f->tff ? VSC_FIELD_TOP : VSC_FIELD_BOTTOM, maReplace);

    int err;
    int64_t dur_num = vsapi->mapGetInt(props, "_DurationNum", 0, &err);
    int64_t dur_den = err ? 0 : vsapi->mapGetInt(props, "_DurationDen", 0, &err);
    if (!err && dur_den > 0) {
        vsh_muldivRational(&dur_num, &dur_den, 2, 1);
        vsapi->mapSetInt(props, "_DurationNum", dur_num, maReplace);
        vsapi->mapSetInt(props, "_DurationDen", dur_den, maReplace);
    }

    if (f->filter)
        for (int k = 0; k < WINDOW; k++)
            vsapi->freeFrame(src[k]);
    else {
        vsapi->freeFrame(first);
        vsapi->freeFrame(second);
    }
    return dst;
}

static void VS_CC p2i_free(void *instance_data, VSCore *core, const VSAPI *vsapi)
{
    p2i_filter_t *f = instance_data;
    (void)core;
    vsapi->freeNode(f->node);
    free(f);
}

#define RETERROR(x) do { vsapi->mapSetError(out, "Interlace: " x); vsapi->freeNode(d.node); return; } while (0)

static void VS_CC p2i_create(const VSMap *in, VSMap *out, void *user_data, VSCore *core, const VSAPI *vsapi)
{
    p2i_filter_t d = {0};
    int err;
    (void)user_data;

    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);

    d.tff = vsapi->mapGetIntSaturated(in, "tff", 0, &err);
    if (err)
        d.tff = 1;
    d.tff = !!d.tff;

    d.filter = vsapi->mapGetIntSaturated(in, "filter", 0, &err);
    if (err)
        d.filter = 1;
    d.filter = !!d.filter;

    if (!vsh_isConstantVideoFormat(&d.vi))
        RETERROR("clip must have constant format and dimensions");
    if ((d.vi.height >> d.vi.format.subSamplingH) & 1)
        RETERROR("all plane heights must be even");
    if (d.vi.numFrames < 2)
        RETERROR("clip too short");
    if (d.filter) {
        d.vt_filter_line = p2i_get_vt_filter_fn(&d.vi.format);
        if (!d.vt_filter_line)
            RETERROR("unsupported sample type (use 8-16 bit integer or 32-bit float)");
        d.peak = d.vi.format.sampleType == stInteger ? (1 << d.vi.format.bitsPerSample) - 1 : 0;
    }

    d.in_frames = d.vi.numFrames;
    d.vi.numFrames /= 2;
    vsh_muldivRational(&d.vi.fpsNum, &d.vi.fpsDen, 1, 2);

    p2i_filter_t *data = malloc(sizeof(*data));
    *data = d;

    VSFilterDependency deps[] = {{ data->node, rpGeneral }};
    vsapi->createVideoFilter(out, "Interlace", &data->vi, p2i_get_frame, p2i_free,
                             fmParallel, deps, 1, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi)
{
    vspapi->configPlugin("com.ifb.interlace", "interlace",
                         "Progressive to interlaced converter",
                         VS_MAKE_VERSION(0, 1), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Interlace",
                             "clip:vnode;"
                             "tff:int:opt;"
                             "filter:int:opt;",
                             "clip:vnode;",
                             p2i_create, NULL, plugin);
}
