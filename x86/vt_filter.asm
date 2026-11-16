;*****************************************************************************
;* Vertical-temporal prefilter kernels
;*****************************************************************************

%include "x86inc.asm"

SECTION_RODATA 64

pd_4096: times 16 dd 4096
pw_8000: times 32 dw 0x8000
pb_1:    times 64 db 1

; must match vt_filter.h
%define VT_BLOCK_MAX 12

SECTION .text

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

; void vt_filter_line_u16(void *dst, const void *const *src, int width, int peak,
;                        const void *coefs)
; src has VT_SRC_PTRS (96) entries; coefficients consumed as 48 pmaddwd pairs.
; pixels are biased to i16 for pmaddwd; sum(coefs) = 1<<13 makes the bias
; correction cancel against the post-shift unbias, so only rounding remains
%macro VT_U16 0
cglobal vt_filter_line_u16, 5, 11, 9, dst, src, w, peak, coef, x, k, ptra, ptrb, rem
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

; void vt_filter_line_f32(void *dst, const void *const *src, int width, int peak,
;                        const void *coefs)
; ascending tap order; AVX2 up uses FMA (one rounding fewer per tap, so a few
; ulp from the C sum and slightly more accurate), SSE2 matches C bit-exactly
%macro VT_F32 0
cglobal vt_filter_line_f32, 5, 9, 3, dst, src, w, peak, coef, x, k, ptra, rem
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

; void vt_filter_line_f16(void *dst, const void *const *src, int width, int peak,
;                        const void *coefs)
; F16C convert on load, f32 FMA accumulation in scalar tap order,
; round-to-nearest-even store. AVX2 up only; the C fallback covers the rest.
%macro VT_F16 0
cglobal vt_filter_line_f16, 5, 9, 3, dst, src, w, peak, coef, x, k, ptra, rem
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

; ---------------------------------------------------------------------------
; Block kernels: nb_rows output lines r, r + 2, ... of one field parity,
; which share their 5 frames and coefficient set.
;
; void vt_filter_block_<type>(void *dst, ptrdiff_t dst_stride,
;                             const void *const *src, void *tmp, int width,
;                             int nb_rows, int peak, const int16_t *coefs)
;
; The coefficient sets are symmetric in time (c[t] == c[4 - t]), so per
; input row the taps fold to A = f0 + f4, B = f1 + f3 and C = f2. Each x
; block is done in two passes over a scratch buffer:
;  1. every input row j of the window is prepared once: (A, B) as
;     interleaved words for pmaddwd, and C as (row j, row j + 1) word pairs
;     for even j;
;  2. every output line k sums 19 (A, B) rows and 10 C pairs, one pmaddwd or
;     vpdpwssd per pair, so 29 multiplies per 95 taps, and none of the
;     unpacking is repeated per output line.
; With the CENTER set, which is also vertically symmetric, rows v and
; 18 - v are added first: one paddw instead of a multiply and an add.
; The fold keeps u16 sums in int16 for peak < 1 << 14 (and the vertical
; pre-add for peak < 1 << 13), so u16 samples must not exceed peak.
;
; An x block is one vector per input row for u8 and two for u16. The u16n
; kernels are u16 with one vector per x block, for zmm on Zen 4: two zmm
; are two cache lines per input row, which costs it more in cache misses
; than the wider block saves. A prepared row holds the block as NQ vectors
; of dwords, its "quarters" (4, or 2 for u16n), in the order
; punpck{l,h}{bw,wd} leave them; the packs of the final store undo that
; order.
;
; Scratch buffer at tmp (64-byte aligned):
;   0               coefficient table, 29 dwords (BLOCK_COEF_TABLE)
;   SCRATCH_PEAK    peak as words, one vector (u16 only)
;   SCRATCH_AB      (A, B) rows j = 0 .. 2 * nb_rows + 17
;   SCRATCH_C       C pairs p = 0 .. nb_rows + 8
; A prepared row is PREP_ROW = NQ * mmsize bytes: (A, B) row j, quarter q
; is at SCRATCH_AB + (NQ * j + q) * mmsize, and C pairs likewise.
; ---------------------------------------------------------------------------

%define BLOCK_ROWS    (2 * VT_BLOCK_MAX + 18)    ; line pointers per frame
%define PREP_ROW      (NQ * mmsize)               ; bytes per prepared row
%define SCRATCH_PEAK  128
%define SCRATCH_AB    192
%define SCRATCH_C     (SCRATCH_AB + BLOCK_ROWS * PREP_ROW)

; coefficient pair of (A, B) row v and of C pair p in the table
%define coef_ab(v)    [tmpq + 4 * (v)]
%define coef_c(p)     [tmpq + 4 * (19 + (p))]

; line pointer of window row (row + offset) of frame t, in pass 1
%define src_line(t, offset) [srcq + rowq * 8 + ((t) * BLOCK_ROWS + (offset)) * 8]

; pass 1 constants (u8)
%define ONES          m6                          ; bytes of 1, for pmaddubsw
%define ZERO          m7

; Register names per phase. r0-r3 and r5 keep their meaning; width is turned
; into last_x in place; peak, coefs and the rest are reused once free.
%macro BLOCK_ARGS_SETUP 0
    DEFINE_ARGS dst, dst_stride, src, tmp, last_x, nb_rows, peak, spare, x, coef_pair
%endmacro

; pass 1: row = window row, line0..3 = input lines, ab_out and c_out = the
; next prepared (A, B) row and C pair
%macro BLOCK_ARGS_PREP 0
    DEFINE_ARGS dst, dst_stride, src, tmp, last_x, nb_rows, nb_prep, c_out, x, row, line0, line1, line2, line3, ab_out
%endmacro

; pass 2: line = output line, ab_in and c_in = its first (A, B) row and C
; pair, dst_line = its output
%macro BLOCK_ARGS_SUM 0
    DEFINE_ARGS dst, dst_stride, src, tmp, last_x, nb_rows, nb_prep, spare, x, line, ab_in, c_in, dst_line, next_line
%endmacro

; table[v]      = (c[0][v], c[1][v]), v = 0..18: the (A, B) pairs
; table[19 + p] = (c[2][2p], c[2][2p + 1]), p = 0..9, with c[2][19] = 0
%macro BLOCK_COEF_TABLE 0
%assign %%v 0
%rep 19
    movzx     paird, word [coefsq + 2 * %%v]            ; c[0][v]
    movzx  pair_hid, word [coefsq + 2 * (19 + %%v)]     ; c[1][v]
    shl    pair_hid, 16
    or        paird, pair_hid
    mov  coef_ab(%%v), paird
%assign %%v %%v + 1
%endrep
%assign %%p 0
%rep 9
    mov       paird, [coefsq + 2 * (38 + 2 * %%p)]      ; c[2][2p], c[2][2p + 1]
    mov  coef_c(%%p), paird
%assign %%p %%p + 1
%endrep
    movzx     paird, word [coefsq + 2 * 56]             ; c[2][18], 0
    mov  coef_c(9), paird
%endmacro

; Pass 1 walks some 200 line streams at once, more than the hardware
; prefetchers follow. Fetch the next x block of each into L2: prefetches
; into L1 evict each other and the scratch, and cost more than they save
; when the lines are in L2 already.
%macro PREFETCH_LINES 2-* ; bytes of x per block, lines
%assign %%bytes %1
%if %%bytes < 64
%assign %%bytes 64
%endif
%rep %0 - 1
%rotate 1
%assign %%offset 0
%rep %%bytes / 64
    prefetcht1   [%1 + xq + %%bytes + %%offset]
%assign %%offset %%offset + 64
%endrep
%endrep
%endmacro

; ---------------------------------------------------------------------------
; Pass 1: prepare window row j = row + %1 of the x block at x
;   (A, B)[j] = (f0[j] + f4[j], f1[j] + f3[j])     as word pairs per dword
;   C pair    = (f2[row], f2[row + 1])              as word pairs per dword
; Temporaries m0-m5.
; ---------------------------------------------------------------------------

%macro PREP_AB_U8 1 ; row offset
    mov      line0q, src_line(0, %1)
    mov      line1q, src_line(4, %1)
    mov      line2q, src_line(1, %1)
    mov      line3q, src_line(3, %1)
    PREFETCH_LINES mmsize, line0q, line1q, line2q, line3q
    movu         m0, [line0q + xq]                      ; f0
    movu         m1, [line1q + xq]                      ; f4
%if cpuflag(ssse3)
    punpckhbw    m2, m0, m1
    punpcklbw    m0, m1
    pmaddubsw    m0, ONES                               ; A, low and high half of each lane
    pmaddubsw    m2, ONES
    movu         m1, [line2q + xq]                      ; f1
    movu         m3, [line3q + xq]                      ; f3
    punpckhbw    m4, m1, m3
    punpcklbw    m1, m3
    pmaddubsw    m1, ONES                               ; B
    pmaddubsw    m4, ONES
%else
    punpckhbw    m2, m0, ZERO
    punpcklbw    m0, ZERO
    punpckhbw    m3, m1, ZERO
    punpcklbw    m1, ZERO
    paddw        m0, m1                                 ; A
    paddw        m2, m3
    movu         m1, [line2q + xq]                      ; f1
    movu         m3, [line3q + xq]                      ; f3
    punpckhbw    m4, m1, ZERO
    punpcklbw    m1, ZERO
    punpckhbw    m5, m3, ZERO
    punpcklbw    m3, ZERO
    paddw        m1, m3                                 ; B
    paddw        m4, m5
%endif
    punpcklwd    m3, m0, m1                             ; (A, B), quarters 0..3
    punpckhwd    m0, m1
    punpcklwd    m1, m2, m4
    punpckhwd    m2, m4
    mova [ab_outq + 0 * mmsize], m3
    mova [ab_outq + 1 * mmsize], m0
    mova [ab_outq + 2 * mmsize], m1
    mova [ab_outq + 3 * mmsize], m2
    add     ab_outq, PREP_ROW
%endmacro

%macro PREP_C_U8 0
    mov      line0q, src_line(2, 0)
    mov      line1q, src_line(2, 1)
    PREFETCH_LINES mmsize, line0q, line1q
    movu         m0, [line0q + xq]
    movu         m1, [line1q + xq]
    punpckhbw    m2, m0, m1                             ; (f2[row], f2[row + 1]) bytes
    punpcklbw    m0, m1
    punpckhbw    m1, m0, ZERO                           ; zero-extended, quarters 0..3
    punpcklbw    m0, ZERO
    punpckhbw    m3, m2, ZERO
    punpcklbw    m2, ZERO
    mova  [c_outq + 0 * mmsize], m0
    mova  [c_outq + 1 * mmsize], m1
    mova  [c_outq + 2 * mmsize], m2
    mova  [c_outq + 3 * mmsize], m3
    add      c_outq, PREP_ROW
%endmacro

; m%1 = sum of the u16 vectors at %2 and %3, both at x + %4
%macro LOAD_SUM 4 ; dst, line a, line b, offset
%if avx_enabled
    movu        m%1, [%2 + xq + %4]
    paddw       m%1, [%3 + xq + %4]
%else
    movu        m%1, [%2 + xq + %4]
    movu         m5, [%3 + xq + %4]
    paddw       m%1, m5
%endif
%endmacro

%macro PREP_AB_U16 1 ; row offset
    mov      line0q, src_line(0, %1)
    mov      line1q, src_line(4, %1)
    mov      line2q, src_line(1, %1)
    mov      line3q, src_line(3, %1)
    PREFETCH_LINES 2 * mmsize, line0q, line1q, line2q, line3q
    LOAD_SUM      0, line0q, line1q, 0                  ; A = f0 + f4
    LOAD_SUM      1, line0q, line1q, mmsize
    LOAD_SUM      2, line2q, line3q, 0                  ; B = f1 + f3
    LOAD_SUM      3, line2q, line3q, mmsize
    punpcklwd    m4, m0, m2                             ; (A, B), quarters 0..3
    punpckhwd    m0, m2
    punpcklwd    m2, m1, m3
    punpckhwd    m1, m3
    mova [ab_outq + 0 * mmsize], m4
    mova [ab_outq + 1 * mmsize], m0
    mova [ab_outq + 2 * mmsize], m2
    mova [ab_outq + 3 * mmsize], m1
    add     ab_outq, PREP_ROW
%endmacro

%macro PREP_C_U16 0
    mov      line0q, src_line(2, 0)
    mov      line1q, src_line(2, 1)
    PREFETCH_LINES 2 * mmsize, line0q, line1q
    movu         m0, [line0q + xq]
    movu         m1, [line0q + xq + mmsize]
    movu         m2, [line1q + xq]
    movu         m3, [line1q + xq + mmsize]
    punpcklwd    m4, m0, m2                             ; (f2[row], f2[row + 1]), quarters 0..3
    punpckhwd    m0, m2
    punpcklwd    m2, m1, m3
    punpckhwd    m1, m3
    mova  [c_outq + 0 * mmsize], m4
    mova  [c_outq + 1 * mmsize], m0
    mova  [c_outq + 2 * mmsize], m2
    mova  [c_outq + 3 * mmsize], m1
    add      c_outq, PREP_ROW
%endmacro

; u16n: 2 quarters
%macro PREP_AB_U16N 1 ; row offset
    mov      line0q, src_line(0, %1)
    mov      line1q, src_line(4, %1)
    mov      line2q, src_line(1, %1)
    mov      line3q, src_line(3, %1)
    PREFETCH_LINES mmsize, line0q, line1q, line2q, line3q
    LOAD_SUM      0, line0q, line1q, 0                  ; A = f0 + f4
    LOAD_SUM      2, line2q, line3q, 0                  ; B = f1 + f3
    punpcklwd    m4, m0, m2
    punpckhwd    m0, m2
    mova [ab_outq + 0 * mmsize], m4
    mova [ab_outq + 1 * mmsize], m0
    add     ab_outq, PREP_ROW
%endmacro

%macro PREP_C_U16N 0
    mov      line0q, src_line(2, 0)
    mov      line1q, src_line(2, 1)
    PREFETCH_LINES mmsize, line0q, line1q
    movu         m0, [line0q + xq]
    movu         m2, [line1q + xq]
    punpcklwd    m4, m0, m2
    punpckhwd    m0, m2
    mova  [c_outq + 0 * mmsize], m4
    mova  [c_outq + 1 * mmsize], m0
    add      c_outq, PREP_ROW
%endmacro

; ---------------------------------------------------------------------------
; Pass 2: output line k of the block
;   out[k] = clip((4096 + sum_v coef_ab(v) . (A, B)[2k + v]
;                       + sum_p coef_c(p)  . C pair[k + p]) >> 13)
; with v = 0..18, p = 0..9, and . the pmaddwd dot product of word pairs.
; ---------------------------------------------------------------------------

; set the %1 accumulators m0 .. m(%1 - 1) to the rounding term
%macro INIT_ACC 1 ; count
    mova         m0, [pd_4096]
%assign %%reg 1
%rep %1 - 1
    mova    m%[%%reg], m0
%assign %%reg %%reg + 1
%endrep
%endmacro

; >> 13, clamp and store the line held in the NQ accumulators (quarters)
; m%2 .. m%2 + NQ - 1
%macro STORE_LINE 2 ; type, first accumulator
%if NQ == 2
%assign %%acc0 %2
%assign %%acc1 %2 + 1
    psrad     m%[%%acc0], 13
    psrad     m%[%%acc1], 13
    packssdw  m%[%%acc0], m%[%%acc1]
    pxor      m%[%%acc1], m%[%%acc1]
    pmaxsw    m%[%%acc0], m%[%%acc1]
    pminsw    m%[%%acc0], [tmpq + SCRATCH_PEAK]
    movu [dst_lineq], m%[%%acc0]
%else
%assign %%acc0 %2
%assign %%acc1 %2 + 1
%assign %%acc2 %2 + 2
%assign %%acc3 %2 + 3
    psrad     m%[%%acc0], 13
    psrad     m%[%%acc1], 13
    psrad     m%[%%acc2], 13
    psrad     m%[%%acc3], 13
    packssdw  m%[%%acc0], m%[%%acc1]
    packssdw  m%[%%acc2], m%[%%acc3]
%ifidn %1, u8
    packuswb  m%[%%acc0], m%[%%acc2]
    movu [dst_lineq], m%[%%acc0]
%else
    pxor      m%[%%acc1], m%[%%acc1]
    pmaxsw    m%[%%acc0], m%[%%acc1]
    pmaxsw    m%[%%acc2], m%[%%acc1]
    pminsw    m%[%%acc0], [tmpq + SCRATCH_PEAK]
    pminsw    m%[%%acc2], [tmpq + SCRATCH_PEAK]
    movu [dst_lineq], m%[%%acc0]
    movu [dst_lineq + mmsize], m%[%%acc2]
%endif
%endif
%endmacro

; pmaddwd version, one line at a time:
;   m0 .. m(NQ - 1) accumulators (quarters), m4 coefficient pair, m5 product

; acc += coef . row, for prepared row %3 of the rows at %2
%macro PMADDWD_ROW 3 ; coefficient pair, rows base, row
    COEF_PD      m4, %1
%assign %%quarter 0
%rep NQ
    pmaddwd      m5, m4, [%2 + ((%3) * NQ + %%quarter) * mmsize]
    paddd   m%[%%quarter], m5
%assign %%quarter %%quarter + 1
%endrep
%endmacro

; acc += coef . ((A, B)[v] + (A, B)[18 - v]), for the symmetric set
%macro PMADDWD_MIRROR 1 ; v
    COEF_PD      m4, coef_ab(%1)
%assign %%quarter 0
%rep NQ
    mova         m5, [ab_inq + ((%1) * NQ + %%quarter) * mmsize]
    paddw        m5, [ab_inq + ((18 - (%1)) * NQ + %%quarter) * mmsize]
    pmaddwd      m5, m4
    paddd   m%[%%quarter], m5
%assign %%quarter %%quarter + 1
%endrep
%endmacro

; acc += coef . (A, B)[9], the centre row of the symmetric set
%macro PMADDWD_CENTRE 0
    COEF_PD      m4, coef_ab(9)
%assign %%quarter 0
%rep NQ
    pmaddwd      m5, m4, [ab_inq + (9 * NQ + %%quarter) * mmsize]
    paddd   m%[%%quarter], m5
%assign %%quarter %%quarter + 1
%endrep
%endmacro

%macro SUM_LINE_PMADDWD 2 ; type, vertically symmetric
    INIT_ACC      NQ
%if %2
%assign %%v 0
%rep 9
    PMADDWD_MIRROR %%v
%assign %%v %%v + 1
%endrep
    PMADDWD_CENTRE
%else
%assign %%v 0
%rep 19
    PMADDWD_ROW  coef_ab(%%v), ab_inq, %%v
%assign %%v %%v + 1
%endrep
%endif
%assign %%p 0
%rep 10
    PMADDWD_ROW  coef_c(%%p), c_inq, %%p
%assign %%p %%p + 1
%endrep
    STORE_LINE   %1, 0
%endmacro

; vpdpwssd version: a group of LINES output lines at once, so that
; 4 * LINES independent accumulator chains hide the vpdpwssd latency, and
; every prepared vector is loaded once for all the lines that use it.
; Line i of the group uses (A, B) rows 2i .. 2i + 18 and C pairs i .. i + 9.
; Register map, for LINES lines (4 at zmm, 3 at ymm, which fills its 16
; registers), with the zmm numbers for NQ = 4 on the right:
;   m(NQ * i + q)            accumulator of quarter q of line i   (m0-m15)
;   m(NQ * LINES)            the prepared vector being multiplied (m16)
;   m(NQ * LINES + 1 + i)    the coefficient pair of line i       (m17-m20)
;   m(NQ * LINES) and m(NQ * LINES + 2) also hold the pre-adds of the
;   symmetric set, alternately                                    (m16, m18)

; nasm picks the EVEX encoding unless told
%macro VNNI_DPWSSD 3
%if mmsize == 64
    vpdpwssd     %1, %2, %3
%else
    {vex} vpdpwssd %1, %2, %3
%endif
%endmacro

; broadcast coefficient pair %3 of the table into the register of line %2
%macro VNNI_LOAD_COEF 3 ; lines, line, table index
%assign %%coef_reg NQ * (%1) + 1 + (%2)
    vpbroadcastd m%[%%coef_reg], [tmpq + 4 * (%3)]
%endmacro

; accumulator of quarter %3 of line %2 += coefficient pair of line %4 . %5
%macro VNNI_DOT 5 ; lines, line, quarter, coefficient of line, source
%assign %%acc_reg  NQ * (%2) + (%3)
%assign %%coef_reg NQ * (%1) + 1 + (%4)
    VNNI_DPWSSD m%[%%acc_reg], m%[%%coef_reg], %5
%endmacro

; Prepared row %2 (of the rows at %6) into every line of the group that uses
; it: line i uses it as tap t = %2 - %3 * i if 0 <= t < %4, with coefficient
; pair %5 + t of the table.
%macro VNNI_ROW 6 ; lines, row, rows per line, taps, first coefficient, rows base
%assign %%line 0
%rep %1
%assign %%tap (%2) - (%3) * %%line
%if %%tap >= 0 && %%tap < (%4)
    VNNI_LOAD_COEF %1, %%line, (%5) + %%tap
%endif
%assign %%line %%line + 1
%endrep
%assign %%data_reg NQ * (%1)
%assign %%quarter 0
%rep NQ
    mova  m%[%%data_reg], [%6 + ((%2) * NQ + %%quarter) * mmsize]
%assign %%line 0
%rep %1
%assign %%tap (%2) - (%3) * %%line
%if %%tap >= 0 && %%tap < (%4)
    VNNI_DOT     %1, %%line, %%quarter, %%line, m%[%%data_reg]
%endif
%assign %%line %%line + 1
%endrep
%assign %%quarter %%quarter + 1
%endrep
%endmacro

; all (A, B) rows the group uses
%macro VNNI_SUM_AB 1 ; lines
%assign %%row 0
%rep 2 * ((%1) - 1) + 19
    VNNI_ROW     %1, %%row, 2, 19, 0, ab_inq
%assign %%row %%row + 1
%endrep
%endmacro

; symmetric set, tap v into every line i, one coefficient for all lines:
;   acc += coef_ab(v) . ((A, B)[2i + v] + (A, B)[2i + 18 - v])     v < 9
;   acc += coef_ab(9) . (A, B)[2i + 9]
%macro VNNI_MIRROR 2 ; lines, v
    VNNI_LOAD_COEF %1, 0, %2
%assign %%count 0
%assign %%quarter 0
%rep NQ
%assign %%line 0
%rep %1
%if (%2) < 9
%assign %%sum_reg NQ * (%1) + 2 * (%%count & 1)
    mova   m%[%%sum_reg], [ab_inq + ((2 * %%line + (%2)) * NQ + %%quarter) * mmsize]
    paddw  m%[%%sum_reg], [ab_inq + ((2 * %%line + 18 - (%2)) * NQ + %%quarter) * mmsize]
    VNNI_DOT     %1, %%line, %%quarter, 0, m%[%%sum_reg]
%else
    VNNI_DOT     %1, %%line, %%quarter, 0, [ab_inq + ((2 * %%line + 9) * NQ + %%quarter) * mmsize]
%endif
%assign %%count %%count + 1
%assign %%line %%line + 1
%endrep
%assign %%quarter %%quarter + 1
%endrep
%endmacro

%macro VNNI_SUM_AB_SYM 1 ; lines
%assign %%v 0
%rep 10
    VNNI_MIRROR  %1, %%v
%assign %%v %%v + 1
%endrep
%endmacro

; all C pairs the group uses
%macro VNNI_SUM_C 1 ; lines
%assign %%pair 0
%rep (%1) + 9
    VNNI_ROW     %1, %%pair, 1, 10, 19, c_inq
%assign %%pair %%pair + 1
%endrep
%endmacro

; store the lines of the group; it may run past nb_rows, and those lines
; are computed but not stored
%macro VNNI_STORE_LINES 2 ; type, lines
%assign %%line 0
%rep %2
%if %%line
    lea  next_lineq, [lineq + %%line]
    cmp  next_lineq, nb_rowsq
    jge %%done
%endif
    STORE_LINE   %1, NQ * %%line
    add   dst_lineq, dst_strideq
%assign %%line %%line + 1
%endrep
%%done:
%endmacro

%macro SUM_LINES_VNNI 3 ; type, lines, vertically symmetric
    INIT_ACC     NQ * (%2)
%if %3
    VNNI_SUM_AB_SYM %2
%else
    VNNI_SUM_AB  %2
%endif
    VNNI_SUM_C   %2
    VNNI_STORE_LINES %1, %2
%endmacro

; ---------------------------------------------------------------------------
; one x block: pass 1 over the window rows, then pass 2 over the lines
; %2: lines per vpdpwssd group, 0 for the pmaddwd version
; %3: vertically symmetric (A, B) coefficients
%macro BLOCK_XLOOP 3 ; type, lines, symmetric
.xloop_%2_%3:
    BLOCK_ARGS_PREP
%ifidn %1, u8
%if cpuflag(ssse3)
    mova       ONES, [pb_1]
%endif
    pxor       ZERO, ZERO
%endif
    lea     ab_outq, [tmpq + SCRATCH_AB]
    lea      c_outq, [tmpq + SCRATCH_C]
    xor        rowd, rowd
.prep_%2_%3:
%ifidn %1, u8
    PREP_AB_U8    0
    PREP_AB_U8    1
    PREP_C_U8
%elifidn %1, u16n
    PREP_AB_U16N  0
    PREP_AB_U16N  1
    PREP_C_U16N
%else
    PREP_AB_U16   0
    PREP_AB_U16   1
    PREP_C_U16
%endif
    add        rowd, 2
    cmp        rowd, nb_prepd
    jl .prep_%2_%3

    BLOCK_ARGS_SUM
    lea      ab_inq, [tmpq + SCRATCH_AB]
    lea       c_inq, [tmpq + SCRATCH_C]
    lea   dst_lineq, [dstq + xq]
    xor       lined, lined
.sum_%2_%3:
%if %2
    SUM_LINES_VNNI %1, %2, %3
    add      ab_inq, 2 * %2 * PREP_ROW
    add       c_inq, %2 * PREP_ROW
    add       lined, %2
%else
    SUM_LINE_PMADDWD %1, %3
    add   dst_lineq, dst_strideq
    add      ab_inq, 2 * PREP_ROW
    add       c_inq, PREP_ROW
    inc       lined
%endif
    cmp       lined, nb_rowsd
    jl .sum_%2_%3

    ; next x block; the last one is moved back to end at width
    cmp          xq, last_xq
    je .end_%2_%3
%ifidn %1, u8
    add          xq, mmsize
%else
    add          xq, NQ / 2 * mmsize
%endif
    cmp          xq, last_xq
    jle .xloop_%2_%3
    mov          xq, last_xq
    jmp .xloop_%2_%3
.end_%2_%3:
    RET
%endmacro

; nb_prep = rows to prepare: 2 * (nb_rows rounded up to the group) + 18
%macro COUNT_PREP_ROWS 1 ; lines per group
    BLOCK_ARGS_PREP
%if %1 > 1
    xor     nb_prepd, nb_prepd
%%round:
    add     nb_prepd, %1
    cmp     nb_prepd, nb_rowsd
    jl %%round
    lea     nb_prepd, [nb_prepq * 2 + 18]
%else
    lea     nb_prepd, [nb_rowsq * 2 + 18]
%endif
%endmacro

; %2: lines per vpdpwssd group, 0 for the pmaddwd version
%macro VT_BLOCK 2 ; type, lines
%ifidn %1, u16n
%define NQ 2
%else
%define NQ 4
%endif
%if %2
cglobal vt_filter_block_%1, 8, 15, NQ * %2 + 1 + %2, dst, dst_stride, src, tmp, width, nb_rows, peak, coefs, pair, pair_hi
%else
cglobal vt_filter_block_%1, 8, 15, 8, dst, dst_stride, src, tmp, width, nb_rows, peak, coefs, pair, pair_hi
%endif
    BLOCK_COEF_TABLE
    movsxdifnidn widthq, widthd
    movsxdifnidn nb_rowsq, nb_rowsd
    BLOCK_ARGS_SETUP
%ifidn %1, u8
    sub     last_xq, mmsize                     ; last_x = width - mmsize
%else
    movd        xm0, peakd
%if cpuflag(avx2)
    vpbroadcastw m0, xm0
%else
    punpcklwd    m0, m0
    pshufd       m0, m0, 0
%endif
    mova [tmpq + SCRATCH_PEAK], m0
    add     last_xq, last_xq                    ; last_x = 2 * width - bytes per x block
    sub     last_xq, NQ / 2 * mmsize
%endif
    xor          xd, xd
    ; vertically symmetric (A, B) coefficients, and room for the pre-add
%ifnidn %1, u8
    cmp       peakd, (1 << 13) - 1
    jg .generic
%endif
%assign %%v 0
%rep 9
    mov  coef_paird, coef_ab(%%v)
    cmp  coef_paird, coef_ab(18 - %%v)
    jne .generic
%assign %%v %%v + 1
%endrep
    COUNT_PREP_ROWS %2
    BLOCK_XLOOP  %1, %2, 1
.generic:
    COUNT_PREP_ROWS %2
    BLOCK_XLOOP  %1, %2, 0
%endmacro

; ---------------------------------------------------------------------------
; Float block kernels: the same two passes with float rows.
;
; void vt_filter_block_<f32|f16>(void *dst, ptrdiff_t dst_stride,
;                                const void *const *src, void *tmp, int width,
;                                int nb_rows, int peak, const float *coefs)
;
; Pass 1 stores, per input row, A = f0 + f4, B = f1 + f3 and C = f2 as FNQ
; float vectors each (f16 is converted here, once per input row). Pass 2
; sums 3 x 19 products per output line with FMA, coefficients straight from
; coefs: c[0][v] for A, c[1][v] for B, c[2][v] for C. A group of LINES
; output lines is summed at once so that FNQ * LINES accumulator chains
; hide the FMA latency. With the vertically symmetric set, rows v and
; 18 - v are added before the multiply.
;
; The sums are not in the tap order of the line kernels, so results differ
; from them by rounding.
;
; Scratch at tmp: the A rows, then the B rows, then the C rows, each set
; FLT_SET bytes: row j, vector q of set s is at
; tmp + s * FLT_SET + (j * FNQ + q) * mmsize.
; Registers: m(FNQ * i + q) accumulator of vector q of line i,
; m(FNQ * LINES) the row being multiplied, m(FNQ * LINES + 1 + i) the
; coefficient of line i; the first and third of those also hold the
; pre-adds of the symmetric set, alternately.
; ---------------------------------------------------------------------------

%define FLT_SET (BLOCK_ROWS * FNQ * mmsize)

; m%2 = vector %4 of the x block of line %3, as floats
%macro FLT_LOAD 4 ; type, dst, line, vector
%ifidn %1, f32
    movu        m%2, [%3 + xq + (%4) * mmsize]
%else
    vcvtph2ps   m%2, [%3 + xq + (%4) * mmsize / 2]
%endif
%endmacro

; pass 1, one input row
%macro FLT_PREP 1 ; type
    mov      line0q, src_line(0, 0)
    mov      line1q, src_line(4, 0)
    mov      line2q, src_line(1, 0)
    mov      line3q, src_line(3, 0)
    PREFETCH_LINES FLT_XBYTES, line0q, line1q, line2q, line3q
%assign %%q 0
%rep FNQ
%ifidn %1, f32
    movu         m0, [line0q + xq + %%q * mmsize]
    addps        m0, [line1q + xq + %%q * mmsize]       ; A = f0 + f4
    movu         m1, [line2q + xq + %%q * mmsize]
    addps        m1, [line3q + xq + %%q * mmsize]       ; B = f1 + f3
%else
    FLT_LOAD     %1, 0, line0q, %%q
    FLT_LOAD     %1, 2, line1q, %%q
    FLT_LOAD     %1, 1, line2q, %%q
    FLT_LOAD     %1, 3, line3q, %%q
    addps        m0, m2
    addps        m1, m3
%endif
    mova [outq + %%q * mmsize], m0
    mova [outq + FLT_SET + %%q * mmsize], m1
%assign %%q %%q + 1
%endrep
    mov      line0q, src_line(2, 0)
    PREFETCH_LINES FLT_XBYTES, line0q
%assign %%q 0
%rep FNQ
    FLT_LOAD     %1, 0, line0q, %%q                     ; C = f2
    mova [outq + 2 * FLT_SET + %%q * mmsize], m0
%assign %%q %%q + 1
%endrep
    add        outq, FNQ * mmsize
%endmacro

; prepared row %2 of set %3 into every line of the group that uses it:
; line i uses it as tap %2 - 2 * i
%macro FLT_ROW 3 ; lines, row, set
%assign %%line 0
%rep %1
%assign %%tap (%2) - 2 * %%line
%if %%tap >= 0 && %%tap < 19
%assign %%coef_reg FNQ * (%1) + 1 + %%line
    vbroadcastss m%[%%coef_reg], [coefsq + 4 * (19 * (%3) + %%tap)]
%endif
%assign %%line %%line + 1
%endrep
%assign %%data_reg FNQ * (%1)
%assign %%q 0
%rep FNQ
    mova  m%[%%data_reg], [inq + (%3) * FLT_SET + ((%2) * FNQ + %%q) * mmsize]
%assign %%line 0
%rep %1
%assign %%tap (%2) - 2 * %%line
%if %%tap >= 0 && %%tap < 19
%assign %%acc_reg  FNQ * %%line + %%q
%assign %%coef_reg FNQ * (%1) + 1 + %%line
    vfmadd231ps m%[%%acc_reg], m%[%%coef_reg], m%[%%data_reg]
%endif
%assign %%line %%line + 1
%endrep
%assign %%q %%q + 1
%endrep
%endmacro

; symmetric set, tap %2 of set %3 into every line, one coefficient for all
%macro FLT_MIRROR 3 ; lines, v, set
%assign %%coef_reg FNQ * (%1) + 1
    vbroadcastss m%[%%coef_reg], [coefsq + 4 * (19 * (%3) + (%2))]
%assign %%count 0
%assign %%q 0
%rep FNQ
%assign %%line 0
%rep %1
%assign %%acc_reg FNQ * %%line + %%q
%if (%2) < 9
%assign %%sum_reg FNQ * (%1) + 2 * (%%count & 1)
    mova   m%[%%sum_reg], [inq + (%3) * FLT_SET + ((2 * %%line + (%2)) * FNQ + %%q) * mmsize]
    addps  m%[%%sum_reg], [inq + (%3) * FLT_SET + ((2 * %%line + 18 - (%2)) * FNQ + %%q) * mmsize]
    vfmadd231ps m%[%%acc_reg], m%[%%coef_reg], m%[%%sum_reg]
%else
    vfmadd231ps m%[%%acc_reg], m%[%%coef_reg], [inq + (%3) * FLT_SET + ((2 * %%line + 9) * FNQ + %%q) * mmsize]
%endif
%assign %%count %%count + 1
%assign %%line %%line + 1
%endrep
%assign %%q %%q + 1
%endrep
%endmacro

; pass 2, a group of lines; the group may run past nb_rows, and those lines
; are computed but not stored
%macro FLT_SUM_LINES 3 ; type, lines, vertically symmetric
%assign %%reg 0
%rep FNQ * (%2)
    xorps   m%[%%reg], m%[%%reg]
%assign %%reg %%reg + 1
%endrep
%if %3
%assign %%v 0
%rep 10
    FLT_MIRROR   %2, %%v, 0
    FLT_MIRROR   %2, %%v, 1
    FLT_MIRROR   %2, %%v, 2
%assign %%v %%v + 1
%endrep
%else
%assign %%row 0
%rep 2 * ((%2) - 1) + 19
    FLT_ROW      %2, %%row, 0
    FLT_ROW      %2, %%row, 1
    FLT_ROW      %2, %%row, 2
%assign %%row %%row + 1
%endrep
%endif
%assign %%line 0
%rep %2
%if %%line
    lea  next_lineq, [lineq + %%line]
    cmp  next_lineq, nb_rowsq
    jge %%done
%endif
%assign %%q 0
%rep FNQ
%assign %%acc_reg FNQ * %%line + %%q
%ifidn %1, f32
    movu [dst_lineq + %%q * mmsize], m%[%%acc_reg]
%else
    vcvtps2ph [dst_lineq + %%q * mmsize / 2], m%[%%acc_reg], 4
%endif
%assign %%q %%q + 1
%endrep
    add   dst_lineq, dst_strideq
%assign %%line %%line + 1
%endrep
%%done:
%endmacro

%macro FLT_XLOOP 3 ; type, lines, symmetric
.xloop_%3:
    DEFINE_ARGS dst, dst_stride, src, tmp, last_x, nb_rows, nb_prep, coefs, x, row, line0, line1, line2, line3, out
    mov        outq, tmpq
    xor        rowd, rowd
.prep_%3:
    FLT_PREP     %1
    inc        rowd
    cmp        rowd, nb_prepd
    jl .prep_%3

    DEFINE_ARGS dst, dst_stride, src, tmp, last_x, nb_rows, nb_prep, coefs, x, line, in, dst_line, next_line
    mov         inq, tmpq
    lea   dst_lineq, [dstq + xq]
    xor       lined, lined
.sum_%3:
    FLT_SUM_LINES %1, %2, %3
    add         inq, 2 * %2 * FNQ * mmsize
    add       lined, %2
    cmp       lined, nb_rowsd
    jl .sum_%3

    ; next x block; the last one is moved back to end at width
    cmp          xq, last_xq
    je .end_%3
    add          xq, FLT_XBYTES
    cmp          xq, last_xq
    jle .xloop_%3
    mov          xq, last_xq
    jmp .xloop_%3
.end_%3:
    RET
%endmacro

%macro VT_BLOCK_FLT 3 ; type, vectors per x block, lines per group
%define FNQ %2
%ifidn %1, f32
%define FLT_XBYTES (%2 * mmsize)
%else
%define FLT_XBYTES (%2 * mmsize / 2)
%endif
cglobal vt_filter_block_%1, 8, 15, %2 * %3 + 1 + %3, dst, dst_stride, src, tmp, width, nb_rows, peak, coefs, x
    movsxdifnidn widthq, widthd
    movsxdifnidn nb_rowsq, nb_rowsd
    DEFINE_ARGS dst, dst_stride, src, tmp, last_x, nb_rows, nb_prep, coefs, x
%ifidn %1, f32
    shl     last_xq, 2
%else
    add     last_xq, last_xq
%endif
    sub     last_xq, FLT_XBYTES                 ; last_x = bytes of width - bytes per x block
    ; rows to prepare: 2 * (nb_rows rounded up to the group) + 18
    xor     nb_prepd, nb_prepd
.round:
    add     nb_prepd, %3
    cmp     nb_prepd, nb_rowsd
    jl .round
    lea     nb_prepd, [nb_prepq * 2 + 18]
    ; vertically symmetric coefficients
%assign %%i 0
%rep 3
%assign %%v 0
%rep 9
    mov          xd, [coefsq + 4 * (19 * %%i + %%v)]
    cmp          xd, [coefsq + 4 * (19 * %%i + 18 - %%v)]
    jne .generic
%assign %%v %%v + 1
%endrep
%assign %%i %%i + 1
%endrep
    xor          xd, xd
    FLT_XLOOP    %1, %3, 1
.generic:
    xor          xd, xd
    FLT_XLOOP    %1, %3, 0
%endmacro

; an x block is 64 bytes of each input row, which measured best for both
; types on every CPU tried; f16 at ymm has registers for 3 lines only
INIT_YMM avx2
VT_BLOCK_FLT f32, 2, 4
VT_BLOCK_FLT f16, 4, 3
INIT_ZMM avx512
VT_BLOCK_FLT f32, 1, 4
VT_BLOCK_FLT f16, 2, 4

INIT_XMM sse2
VT_BLOCK u8, 0
VT_BLOCK u16, 0
INIT_YMM avx2
VT_BLOCK u8, 0
VT_BLOCK u16, 0
INIT_YMM avxvnni
VT_BLOCK u8, 3
VT_BLOCK u16, 3
INIT_ZMM avx512
VT_BLOCK u8, 0
VT_BLOCK u16, 0
VT_BLOCK u16n, 0
INIT_ZMM avx512icl
VT_BLOCK u8, 4
VT_BLOCK u16, 4
VT_BLOCK u16n, 4
