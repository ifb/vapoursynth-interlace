;*****************************************************************************
;* Vertical-temporal prefilter kernels
;*****************************************************************************

%include "x86inc.asm"

SECTION_RODATA 64

pd_4096: times 16 dd 4096
pw_8000: times 32 dw 0x8000

SECTION .text

cextern vt_coefs_i16 ; int16_t[96], flat [t][v], padded with 0
cextern vt_coefs_f32 ; float[95]

; log2 of the vector byte width
%define LOG2M (4 + (mmsize == 32) + 2 * (mmsize == 64))

; broadcast one coefficient pmaddwd pair (dword) / one float
%macro COEF_PD 2
%if mmsize >= 32
    vpbroadcastd %1, %2
%else
    movd         %1, %2
    pshufd       %1, %1, 0
%endif
%endmacro

; zmm stores are masked so nothing past width is written; narrower kernels
; may write up to a full vector (dst stride padding absorbs it)
%macro TAIL_MASK 5 ; kxnor, kmov, gpr64, kmov_gpr, rem_bits
%if mmsize == 64
    mov        remd, wd
    and        remd, (1 << %5) - 1
    %1           k2, k2, k2
    test       remd, remd
    jz %%full
    mov          %3, -1
    bzhi         %3, %3, remq
    %2           k2, %4
%%full:
%endif
%endmacro

%macro STORE_MASKED 3 ; kxnor, store_op, mem
%if mmsize == 64
    %1           k1, k1, k1
    cmp          wd, 1
    jne %%whole
    kmovq        k1, k2
%%whole:
    %2           %3 {k1}, m0
%else
    movu         %3, m0
%endif
%endmacro

; void vt_filter_line_u8(void *dst, const void *const *src, int width, int peak)
; src has VT_SRC_PTRS (96) entries; coefficients consumed as 48 pmaddwd pairs.
; peak is implied (packuswb).
%macro VT_U8 0
cglobal vt_filter_line_u8, 3, 9, 12, dst, src, w, x, k, ptra, ptrb, coef, rem
    lea       coefq, [vt_coefs_i16]
    TAIL_MASK kxnorq, kmovq, ptraq, ptraq, LOG2M
    add          wd, mmsize - 1
    shr          wd, LOG2M
    pxor         m9, m9
    xor          xq, xq
.xloop:
    mova         m0, [pd_4096]
    mova         m1, m0
    mova         m2, m0
    mova         m3, m0
    xor          kq, kq
.kloop:
    mov       ptraq, [srcq + kq*8]
    mov       ptrbq, [srcq + kq*8 + 8]
    COEF_PD      m6, [coefq + kq*2]
    movu         m4, [ptraq + xq]
    movu         m5, [ptrbq + xq]
    punpckhbw    m7, m4, m9
    punpcklbw    m4, m9
    punpckhbw    m8, m5, m9
    punpcklbw    m5, m9
    punpckhwd   m10, m4, m5
    punpcklwd    m4, m5
    punpckhwd   m11, m7, m8
    punpcklwd    m7, m8
    pmaddwd      m4, m6
    pmaddwd     m10, m6
    pmaddwd      m7, m6
    pmaddwd     m11, m6
    paddd        m0, m4
    paddd        m1, m10
    paddd        m2, m7
    paddd        m3, m11
    add          kq, 2
    cmp          kq, 96
    jl .kloop
    psrad        m0, 13
    psrad        m1, 13
    psrad        m2, 13
    psrad        m3, 13
    packssdw     m0, m1
    packssdw     m2, m3
    packuswb     m0, m2
    STORE_MASKED kxnorq, vmovdqu8, [dstq + xq]
    add          xq, mmsize
    dec          wd
    jg .xloop
    RET
%endmacro

INIT_XMM sse2
VT_U8
INIT_YMM avx2
VT_U8
INIT_ZMM avx512
VT_U8

; void vt_filter_line_u16(void *dst, const void *const *src, int width, int peak)
; pixels are biased to i16 for pmaddwd; sum(coefs) = 1<<13 makes the bias
; correction cancel against the post-shift unbias, so only rounding remains
%macro VT_U16 0
cglobal vt_filter_line_u16, 4, 10, 9, dst, src, w, peak, x, k, ptra, ptrb, coef, rem
    lea       coefq, [vt_coefs_i16]
    mova         m6, [pw_8000]
    sub       peakd, 32768
    movd        xm8, peakd
%if mmsize >= 32
    vpbroadcastw m8, xm8
%else
    punpcklwd    m8, m8
    pshufd       m8, m8, 0
%endif
    TAIL_MASK kxnord, kmovd, ptraq, ptrad, LOG2M - 1
    add          wd, mmsize/2 - 1
    shr          wd, LOG2M - 1
    xor          xq, xq
.xloop:
    mova         m0, [pd_4096]
    mova         m1, m0
    xor          kq, kq
.kloop:
    mov       ptraq, [srcq + kq*8]
    mov       ptrbq, [srcq + kq*8 + 8]
    COEF_PD      m5, [coefq + kq*2]
    movu         m2, [ptraq + xq]
    movu         m3, [ptrbq + xq]
    pxor         m2, m6
    pxor         m3, m6
    punpckhwd    m4, m2, m3
    punpcklwd    m2, m3
    pmaddwd      m2, m5
    pmaddwd      m4, m5
    paddd        m0, m2
    paddd        m1, m4
    add          kq, 2
    cmp          kq, 96
    jl .kloop
    psrad        m0, 13
    psrad        m1, 13
    packssdw     m0, m1
    pminsw       m0, m8
    pxor         m0, m6
    STORE_MASKED kxnord, vmovdqu16, [dstq + xq]
    add          xq, mmsize
    dec          wd
    jg .xloop
    RET
%endmacro

INIT_XMM sse2
VT_U16
INIT_YMM avx2
VT_U16
INIT_ZMM avx512
VT_U16

; void vt_filter_line_f32(void *dst, const void *const *src, int width, int peak)
; ascending tap order; AVX2 up uses FMA (one rounding fewer per tap, so a few
; ulp from the C sum and slightly more accurate), SSE2 matches C bit-exactly
%macro VT_F32 0
cglobal vt_filter_line_f32, 3, 8, 3, dst, src, w, x, k, ptra, coef, rem
    lea       coefq, [vt_coefs_f32]
    TAIL_MASK kxnorw, kmovw, ptraq, ptrad, LOG2M - 2
    add          wd, mmsize/4 - 1
    shr          wd, LOG2M - 2
    xor          xq, xq
.xloop:
    xorps        m0, m0
    xor          kq, kq
.kloop:
    mov       ptraq, [srcq + kq*8]
%if mmsize >= 32
    vbroadcastss m1, [coefq + kq*4]
    vfmadd231ps  m0, m1, [ptraq + xq]
%else
    movss        m1, [coefq + kq*4]
    shufps       m1, m1, 0
    movups       m2, [ptraq + xq]
    mulps        m2, m1
    addps        m0, m2
%endif
    add          kq, 1
    cmp          kq, 95
    jl .kloop
    STORE_MASKED kxnorw, vmovups, [dstq + xq]
    add          xq, mmsize
    dec          wd
    jg .xloop
    RET
%endmacro

INIT_XMM sse2
VT_F32
INIT_YMM avx2
VT_F32
INIT_ZMM avx512
VT_F32

; void vt_filter_line_f16(void *dst, const void *const *src, int width, int peak)
; F16C convert on load, f32 FMA accumulation in scalar tap order,
; round-to-nearest-even store. AVX2 up only; the C fallback covers the rest.
%macro VT_F16 0
cglobal vt_filter_line_f16, 3, 8, 3, dst, src, w, x, k, ptra, coef, rem
    lea       coefq, [vt_coefs_f32]
    TAIL_MASK kxnorw, kmovw, ptraq, ptrad, LOG2M - 2
    add          wd, mmsize/4 - 1
    shr          wd, LOG2M - 2
    xor          xq, xq
.xloop:
    xorps        m0, m0
    xor          kq, kq
.kloop:
    mov       ptraq, [srcq + kq*8]
    vbroadcastss m1, [coefq + kq*4]
    vcvtph2ps    m2, [ptraq + xq]
    vfmadd231ps  m0, m1, m2
    add          kq, 1
    cmp          kq, 95
    jl .kloop
%if mmsize == 64
    kxnorw       k1, k1, k1
    cmp          wd, 1
    jne %%whole
    kmovw        k1, k2
%%whole:
    vcvtps2ph [dstq + xq]{k1}, m0, 4
%else
    vcvtps2ph [dstq + xq], m0, 4
%endif
    add          xq, mmsize/2
    dec          wd
    jg .xloop
    RET
%endmacro

INIT_YMM avx2
VT_F16
INIT_ZMM avx512
VT_F16
