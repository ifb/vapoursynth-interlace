/*
 * Progressive to interlaced conversion for VapourSynth.
 */

#include <stdlib.h>

#include <VapourSynth4.h>
#include <VSConstants4.h>
#include <VSHelper4.h>

typedef struct p2i_filter_t p2i_filter_t;

struct p2i_filter_t {
    VSNode *node;
    VSVideoInfo vi;
    int tff;
};

static const VSFrame *VS_CC p2i_get_frame(int n, int activation_reason, void *instance_data,
                                          void **frame_data, VSFrameContext *frame_ctx,
                                          VSCore *core, const VSAPI *vsapi)
{
    p2i_filter_t *f = instance_data;
    (void)frame_data;

    if (activation_reason == arInitial) {
        vsapi->requestFrameFilter(n * 2,     f->node, frame_ctx);
        vsapi->requestFrameFilter(n * 2 + 1, f->node, frame_ctx);
        return NULL;
    }
    if (activation_reason != arAllFramesReady)
        return NULL;

    const VSFrame *first  = vsapi->getFrameFilter(n * 2,     f->node, frame_ctx);
    const VSFrame *second = vsapi->getFrameFilter(n * 2 + 1, f->node, frame_ctx);
    const VSFrame *top    = f->tff ? first : second;
    const VSFrame *bottom = f->tff ? second : first;

    VSFrame *dst = vsapi->newVideoFrame(&f->vi.format, f->vi.width, f->vi.height, first, core);

    for (int plane = 0; plane < f->vi.format.numPlanes; plane++) {
        int rows = vsapi->getFrameHeight(dst, plane);
        int row_bytes = vsapi->getFrameWidth(dst, plane) * f->vi.format.bytesPerSample;
        ptrdiff_t dst_stride = vsapi->getStride(dst, plane);
        ptrdiff_t top_stride = vsapi->getStride(top, plane);
        ptrdiff_t bot_stride = vsapi->getStride(bottom, plane);
        uint8_t *dstp = vsapi->getWritePtr(dst, plane);

        vsh_bitblt(dstp, dst_stride * 2,
                   vsapi->getReadPtr(top, plane), top_stride * 2, row_bytes, rows / 2);
        vsh_bitblt(dstp + dst_stride, dst_stride * 2,
                   vsapi->getReadPtr(bottom, plane) + bot_stride, bot_stride * 2, row_bytes, rows / 2);
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

    vsapi->freeFrame(first);
    vsapi->freeFrame(second);
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

    if (!vsh_isConstantVideoFormat(&d.vi))
        RETERROR("clip must have constant format and dimensions");
    if ((d.vi.height >> d.vi.format.subSamplingH) & 1)
        RETERROR("all plane heights must be even");
    if (d.vi.numFrames < 2)
        RETERROR("clip too short");

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
                             "tff:int:opt;",
                             "clip:vnode;",
                             p2i_create, NULL, plugin);
}
