/*
 * AviSynth+ end-to-end test for the interlace plugin.
 *
 *   test_interlace_avs <interlace.so> [reference_dir]
 *
 * Without reference_dir it self-checks the AviSynth+ frontend: clip length
 * and frame rate halve, _FieldBased is set, the vi parity flags are set,
 * the naive weave (filter=false) interleaves the right source lines, and
 * argument validation rejects what the VapourSynth frontend rejects.
 *
 * With reference_dir it additionally compares every output frame against
 * raw planes written by test/dump_vs_reference.py, which pushes the same
 * generated clip through the VapourSynth frontend. That cross-host
 * comparison is the check that matters: both frontends drive the identical
 * kernel, so any mismatch is a frame-orchestration bug (temporal indexing,
 * field parity, chroma siting, format mapping) that checkasm cannot see.
 *
 * Needs an AviSynth+ host: this links libavisynth to build (the repo does
 * not vendor one — set -Davisynth_lib_dir=/path/to/libavisynth).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avisynth_c.h"

static int failures = 0;
static AVS_ScriptEnvironment *env;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        failures = 1; \
    } \
} while (0)

/* A deterministic clip whose every pixel depends on frame number and
 * position, so a wrong temporal index or field parity cannot go unnoticed.
 * Must match dump_vs_reference.py exactly. */
static AVS_Value make_clip(int w, int h, const char *pixel_type, int length)
{
    AVS_Value bc[5] = { avs_new_value_int(w), avs_new_value_int(h),
                        avs_new_value_string(pixel_type),
                        avs_new_value_int(0), avs_new_value_int(length) };
    const char *bcn[5] = { "width", "height", "pixel_type", "color_yuv", "length" };
    AVS_Value blank = avs_invoke(env, "BlankClip", avs_new_value_array(bc, 5),
                                 (const char **)bcn);
    if (avs_is_error(blank))
        return blank;

    AVS_Value ea[4] = { blank,
        avs_new_value_string("sx 7 * sy 13 * + frameno 29 * + 256 %"),
        avs_new_value_string("sx 3 * sy 5 * + frameno 11 * + 256 %"),
        avs_new_value_string("sx 17 * sy 19 * + frameno 23 * + 256 %") };
    AVS_Value expr = avs_invoke(env, "Expr", avs_new_value_array(ea, 4), NULL);
    avs_release_value(blank);
    return expr;
}

/* Interlace(clip, tff, filter). Positional: avs_invoke rejects "" as the
 * name of an unnamed leading parameter. */
static AVS_Value interlace(AVS_Value clip, int tff, int filter)
{
    AVS_Value a[3] = { clip, avs_new_value_bool(tff), avs_new_value_bool(filter) };
    return avs_invoke(env, "Interlace", avs_new_value_array(a, 3), NULL);
}

static void test_vi(void)
{
    AVS_Value src = make_clip(64, 32, "YV12", 10);
    if (avs_is_error(src)) {
        CHECK(0, "source clip: %s", avs_as_error(src));
        return;
    }
    AVS_Value out = interlace(src, 1, 1);
    if (avs_is_error(out)) {
        CHECK(0, "Interlace: %s", avs_as_error(out));
        avs_release_value(src);
        return;
    }

    AVS_Clip *c = avs_take_clip(out, env);
    const AVS_VideoInfo *vi = avs_get_video_info(c);
    const AVS_Clip *sc = avs_take_clip(src, env);
    const AVS_VideoInfo *svi = avs_get_video_info((AVS_Clip *)sc);

    CHECK(vi->num_frames == 5, "num_frames %d, want 5", vi->num_frames);
    CHECK(vi->width == 64 && vi->height == 32, "size %dx%d, want 64x32",
          vi->width, vi->height);
    CHECK(vi->pixel_type == svi->pixel_type, "pixel_type changed");
    /* fps halves: same numerator, doubled denominator */
    CHECK((uint64_t)vi->fps_numerator * svi->fps_denominator * 2
          == (uint64_t)svi->fps_numerator * vi->fps_denominator,
          "fps not halved: %u/%u from %u/%u", vi->fps_numerator,
          vi->fps_denominator, svi->fps_numerator, svi->fps_denominator);
    CHECK(avs_is_field_based(vi), "vi not field based");
    CHECK(avs_is_tff(vi), "vi not TFF for tff=true");

    AVS_VideoFrame *f = avs_get_frame(c, 0);
    CHECK(f != NULL, "get_frame(0) failed");
    if (f) {
        const AVS_Map *props = avs_get_frame_props_ro(env, f);
        int err = 0;
        int64_t fb = avs_prop_get_int(env, props, "_FieldBased", 0, &err);
        CHECK(!err && fb == 2, "_FieldBased %lld (err %d), want 2",
              (long long)fb, err);
        avs_release_video_frame(f);
    }

    avs_release_clip((AVS_Clip *)sc);
    avs_release_clip(c);
    avs_release_value(out);
    avs_release_value(src);
}

static void test_bff(void)
{
    AVS_Value src = make_clip(64, 32, "YV12", 10);
    if (avs_is_error(src))
        return;
    AVS_Value out = interlace(src, 0, 1);
    if (avs_is_error(out)) {
        CHECK(0, "Interlace(tff=false): %s", avs_as_error(out));
        avs_release_value(src);
        return;
    }
    AVS_Clip *c = avs_take_clip(out, env);
    const AVS_VideoInfo *vi = avs_get_video_info(c);
    CHECK(avs_is_bff(vi), "vi not BFF for tff=false");

    AVS_VideoFrame *f = avs_get_frame(c, 0);
    if (f) {
        const AVS_Map *props = avs_get_frame_props_ro(env, f);
        int err = 0;
        int64_t fb = avs_prop_get_int(env, props, "_FieldBased", 0, &err);
        CHECK(!err && fb == 1, "_FieldBased %lld, want 1 (BFF)", (long long)fb);
        avs_release_video_frame(f);
    }
    avs_release_clip(c);
    avs_release_value(out);
    avs_release_value(src);
}

/* filter=false is a plain weave: output row r comes from source frame
 * (2n + parity) row r, so it is checkable without the kernel. */
static void test_weave(void)
{
    const int w = 64, h = 16;
    AVS_Value src = make_clip(w, h, "YV12", 8);
    if (avs_is_error(src))
        return;
    AVS_Value out = interlace(src, 1, 0);
    if (avs_is_error(out)) {
        CHECK(0, "Interlace(filter=false): %s", avs_as_error(out));
        avs_release_value(src);
        return;
    }

    AVS_Clip *oc = avs_take_clip(out, env);
    AVS_Clip *sc = avs_take_clip(src, env);
    AVS_VideoFrame *of = avs_get_frame(oc, 1);      /* weaves src 2 and 3 */
    AVS_VideoFrame *s0 = avs_get_frame(sc, 2);
    AVS_VideoFrame *s1 = avs_get_frame(sc, 3);

    if (of && s0 && s1) {
        const uint8_t *op = avs_get_read_ptr_p(of, AVS_PLANAR_Y);
        const uint8_t *a = avs_get_read_ptr_p(s0, AVS_PLANAR_Y);
        const uint8_t *b = avs_get_read_ptr_p(s1, AVS_PLANAR_Y);
        const int opitch = avs_get_pitch_p(of, AVS_PLANAR_Y);
        const int apitch = avs_get_pitch_p(s0, AVS_PLANAR_Y);
        const int bpitch = avs_get_pitch_p(s1, AVS_PLANAR_Y);
        const int row = avs_get_row_size_p(of, AVS_PLANAR_Y);
        for (int r = 0; r < h; r++) {
            const uint8_t *want = (r & 1) ? b + (ptrdiff_t)r * bpitch
                                          : a + (ptrdiff_t)r * apitch;
            if (memcmp(op + (ptrdiff_t)r * opitch, want, row)) {
                CHECK(0, "weave row %d differs (tff, %s field)",
                      r, (r & 1) ? "bottom" : "top");
                break;
            }
        }
    } else {
        CHECK(0, "weave: frame fetch failed");
    }

    avs_release_video_frame(of);
    avs_release_video_frame(s0);
    avs_release_video_frame(s1);
    avs_release_clip(oc);
    avs_release_clip(sc);
    avs_release_value(out);
    avs_release_value(src);
}

static void expect_error(const char *what, AVS_Value clip, int tff, int filter)
{
    AVS_Value out = interlace(clip, tff, filter);
    CHECK(avs_is_error(out), "%s: expected an error, got none", what);
    avs_release_value(out);
}

static void test_validation(void)
{
    /* odd chroma height: 4:2:0 with height 2 gives a 1-line chroma plane */
    AVS_Value odd = make_clip(64, 2, "YV12", 8);
    if (!avs_is_error(odd)) {
        AVS_Value out = interlace(odd, 1, 1);
        /* height 2 is even at luma and 1 at chroma -> must be rejected */
        CHECK(avs_is_error(out), "odd chroma height: expected an error");
        avs_release_value(out);
        avs_release_value(odd);
    }

    AVS_Value rgb = make_clip(64, 32, "RGBP8", 8);
    if (!avs_is_error(rgb)) {
        expect_error("planar RGB", rgb, 1, 1);
        avs_release_value(rgb);
    }

    AVS_Value shrt = make_clip(64, 32, "YV12", 1);
    if (!avs_is_error(shrt)) {
        expect_error("clip too short", shrt, 1, 1);
        avs_release_value(shrt);
    }

    AVS_Value ok = make_clip(64, 32, "YV12", 8);
    if (!avs_is_error(ok)) {
        /* clip, tff, filter, cpu — positional, as above */
        AVS_Value a[4] = { ok, avs_new_value_bool(1), avs_new_value_bool(1),
                           avs_new_value_string("bogus") };
        AVS_Value out = avs_invoke(env, "Interlace", avs_new_value_array(a, 4), NULL);
        CHECK(avs_is_error(out), "bad cpu string: expected an error");
        avs_release_value(out);
        avs_release_value(ok);
    }
}

/* Compare every output frame against planes dumped from the VapourSynth
 * frontend. Layout per frame: <dir>/f<NNN>p<P>.raw, plane-major raw rows. */
static void test_vs_reference(const char *dir)
{
    static const struct { const char *pix; const char *tag; } fmts[] = {
        { "YV12", "yv12" }, { "YV24", "yv24" }, { "YUV420P16", "yuv420p16" },
    };
    const int w = 96, h = 32, len = 12;

    for (size_t i = 0; i < sizeof(fmts) / sizeof(fmts[0]); i++) {
        AVS_Value src = make_clip(w, h, fmts[i].pix, len);
        if (avs_is_error(src)) {
            fprintf(stderr, "skip %s: %s\n", fmts[i].pix, avs_as_error(src));
            continue;
        }
        AVS_Value out = interlace(src, 1, 1);
        if (avs_is_error(out)) {
            CHECK(0, "%s: %s", fmts[i].pix, avs_as_error(out));
            avs_release_value(src);
            continue;
        }
        AVS_Clip *c = avs_take_clip(out, env);
        const AVS_VideoInfo *vi = avs_get_video_info(c);
        const int planes[3] = { AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V };
        const int nplanes = avs_is_y(vi) ? 1 : 3;

        for (int n = 0; n < vi->num_frames; n++) {
            AVS_VideoFrame *f = avs_get_frame(c, n);
            if (!f) {
                CHECK(0, "%s frame %d: fetch failed", fmts[i].pix, n);
                break;
            }
            for (int p = 0; p < nplanes; p++) {
                char path[512];
                snprintf(path, sizeof(path), "%s/%s_f%03dp%d.raw",
                         dir, fmts[i].tag, n, p);
                FILE *fp = fopen(path, "rb");
                if (!fp) {
                    fprintf(stderr, "skip %s (no reference)\n", path);
                    continue;
                }
                const uint8_t *ptr = avs_get_read_ptr_p(f, planes[p]);
                const int pitch = avs_get_pitch_p(f, planes[p]);
                const int rowsz = avs_get_row_size_p(f, planes[p]);
                const int rows = avs_get_height_p(f, planes[p]);
                uint8_t *ref = malloc((size_t)rowsz);
                for (int r = 0; r < rows; r++) {
                    if (fread(ref, 1, (size_t)rowsz, fp) != (size_t)rowsz) {
                        CHECK(0, "%s: reference truncated at row %d", path, r);
                        break;
                    }
                    if (memcmp(ptr + (ptrdiff_t)r * pitch, ref, (size_t)rowsz)) {
                        CHECK(0, "%s frame %d plane %d row %d differs from VapourSynth",
                              fmts[i].pix, n, p, r);
                        r = rows;
                    }
                }
                free(ref);
                fclose(fp);
            }
            avs_release_video_frame(f);
        }
        avs_release_clip(c);
        avs_release_value(out);
        avs_release_value(src);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <interlace.so> [reference_dir]\n", argv[0]);
        return 2;
    }

    env = avs_create_script_environment(AVISYNTH_INTERFACE_VERSION);
    if (!env) {
        fprintf(stderr, "could not create an AviSynth environment\n");
        return 2;
    }

    AVS_Value load[1] = { avs_new_value_string(argv[1]) };
    AVS_Value lr = avs_invoke(env, "LoadPlugin", avs_new_value_array(load, 1), NULL);
    if (avs_is_error(lr)) {
        fprintf(stderr, "LoadPlugin(%s): %s\n", argv[1], avs_as_error(lr));
        return 2;
    }
    avs_release_value(lr);

    test_vi();
    test_bff();
    test_weave();
    test_validation();
    if (argc > 2)
        test_vs_reference(argv[2]);
    else
        fprintf(stderr, "note: no reference_dir given, skipping the "
                        "cross-host comparison (the check that matters)\n");

    avs_delete_script_environment(env);
    printf("%s\n", failures ? "FAILED" : "all tests passed");
    return failures;
}
