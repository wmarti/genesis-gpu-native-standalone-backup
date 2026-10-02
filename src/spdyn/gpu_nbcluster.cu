/*
 * gpu_nbcluster.cu : the cluster-pair real-space kernel of the native core,
 * nonbond_precision = MIXED.
 *
 * This file is part of GENESIS.  GENESIS is released under the terms of the
 * GNU Lesser General Public License as published by the Free Software
 * Foundation; either version 3 of the License, or (at your option) any later
 * version.  See the files COPYING and COPYING.LESSER.
 *
 * Parts of this kernel's design (the cluster-pair mask and exclusion layout,
 * the grouping of j clusters, the force reduction and the in-kernel pruning)
 * are derived from GROMACS (https://www.gromacs.org), Copyright the GROMACS
 * development team, licensed under the GNU Lesser General Public License
 * version 2.1 or later and used here under version 3.  They are described in
 *
 *   S. Pall and B. Hess, "A flexible algorithm for calculating pair
 *   interactions on SIMD architectures", Comput. Phys. Commun. 184,
 *   2641-2650 (2013);
 *   S. Pall, A. Zhmurov, P. Bauer, M. Abraham, M. Lundborg, A. Gray,
 *   B. Hess and E. Lindahl, "Heterogeneous parallelization and acceleration
 *   of molecular dynamics simulations in GROMACS", J. Chem. Phys. 153,
 *   134110 (2020).
 *
 * The layout:
 *
 *  - Clusters of 8 atoms.  Each GENESIS cell's resident atoms are split by
 *    recursive bisection along the longest extent at multiples of 8, so
 *    every cluster is spatially compact and all but a cell's last are full
 *    (fillers are slot -1, parked far away).  An i "super-cluster" is up to
 *    8 clusters of one cell, so a rebuild cell pair gives the periodic move
 *    of every cluster pair, and ownership stays the rebuild's.
 *  - The outer list: per super-cluster, j clusters in groups of 4, each
 *    with a 32-bit mask (bit jm*8 + ic) of the i clusters whose bounding
 *    box lies within the list radius of its box, and an FP32 offset
 *    origin_cj - origin_ci - move.  Coordinates are FP32 relative to their
 *    cell's FP64 origin, so they carry a few angstroms, not the box.
 *    Excluded and 1-4 pairs come from the rebuild's exclusion mask, as
 *    64 32-bit words per group that has one.  Groups of the pairs whose j
 *    cell is owned come first in each super-cluster, then the ghost ones,
 *    so the interior and boundary parts are separate work ranges.
 *  - The inner list: per group and warp half, the 32-bit mask of the
 *    cluster pairs with an atom pair within r_in = r_eval + buffer (prune_buffer),
 *    where r_eval is the cut-off (analytic term) or the table's support,
 *    pruned again on the device whenever twice the largest displacement
 *    since the last prune reaches the buffer.  That is exact: a pair
 *    dropped at the prune was at r >= r_in and is still beyond r_eval.
 *    The prune is the force pass of that step: it walks the outer list
 *    instead of the inner one, and each warp writes its half's word of the
 *    atom-pair test r < r_in from the distances it computes anyway.  A
 *    cluster pair the inner list would not hold has no atom pair within
 *    r_eval (r_in > r_eval), so its terms are zero and skipped, and the
 *    forces are those of the inner list's pass.
 *  - The kernel: 64 threads per work item (lane = i atom x j atom; the two
 *    warps take j atoms 0-3 and 4-7), i data in shared memory, the i
 *    forces of all 8 i clusters in registers, the 4 j clusters of each
 *    group staged by cp.async with double buffering, a transposed
 *    shuffle reduction of the j forces; groups without excluded pairs skip
 *    the exclusion test.
 *  - The pair term ([ENERGY] ewald_evaluation, analytic unless TABLE is
 *    given): when the table is Ewald real space plus LJ with CHARMM's
 *    potential switch or force switch (vdw_force_switch; nbc_fit_ewald
 *    identifies it and checks every row inside the cut-off to 1e-8), it is
 *    evaluated analytically to the cut-off: erfc through FP32 Chebyshev
 *    corrections of FP32 accuracy (12 terms, or 18 when (beta rc)^2 is too
 *    large for 12; beyond that range the table stays), the switch
 *    branch-free.
 *    Against exact Ewald this
 *    is ~1e-6 rms in the forces where the linear table is ~3e-4.
 *    Otherwise the table is read as FP32 interval records to the table's support.  With vdw = CUTOFF (GC_TABLE_PME_ELEC)
 *    the table is the electrostatic column alone and the LJ pair is the
 *    plain r^-12/r^-6 inside the cutoff, as the FP64 walk's plain_lj form
 *    computes it; that table is never fitted (AUTO takes the table).
 *
 * Accumulation is the single-word fixed point of gpu_fixed_sum.cuh into
 * force_real_fx and acc_real, so results are identical run to run.
 */

#include "gpu_nbcluster.h"
#include "gpu_nbcluster_fit.h"
#include "gpu_fixed_sum.cuh"

#include <cub/cub.cuh>
#include <cuda_pipeline.h>

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gcn {

namespace {

const int kMaxCell = 512;   /* atoms per cell k_nbc_form can split      */
const int kCenter  = 13;
const int kGroups  = 16;
const int kForceSm = 16;
/* analytic LJ form: potential switch, plain r^-12/r^-6, force switch */
enum { kVdwPsw = 0, kVdwPlain = 1, kVdwFsw = 2 };
/* r_in - r_eval: kPruneBufferPerFs A per fs of timestep (so the prune interval
   in steps does not depend on the timestep), at most kPruneBufferSkin of the
   outer list's skin; kPruneBuffer without a timestep */
const float kPruneBuffer = 1.0f;
const float kPruneBufferPerFs = 0.5f;
const float kPruneBufferSkin = 0.75f;
const double kAkmaPs = 4.88882129e-02;   /* ps per AKMA time unit */

/* npk: the inner list's groups of the item, packed at its last prune
 * (k_nbc_mixed); they start at g0 in the packed arrays */
struct nbc_work { int sci, g0, g1, npk; };

/* A cell whose coordinates are all this rank's before the halo arrives: an owned
   cell.  Its pairs with such cells are the interior part; every other pair waits
   for the halo (gcn_pair_in_part). */
__device__ __forceinline__ bool nbc_interior(const struct gcn_layout &L, int c)
{
    return gcn_box_owned(L, c);
}

/* the state words (unsigned int), per part p = 0 interior/all, 1 boundary */
enum { W_DISP = 0, W_NEED = 2, W_FORCE = 4, W_NWORDS = 6 };

enum { T_NCL, T_NSCI, T_NGRP, T_NEXG, T_NWI, T_NW, T_ERR, T_OVF, T_NPENT, T_N };

}  /* anonymous namespace */
}  /* namespace gcn */

struct gcn_nbc {
    int ncls, nrow;
    float2 *ljf;                 /* [(ncls+1)^2] {c12, c6}              */
    float4 *tab;                 /* [4*nrow] grad, then energy records  */
    double *move;
    float  *movef;

    gc_i64 cap_cell, cap_cl, cap_sci, cap_pitch, cap_grp, cap_exg, cap_work;
    gc_i64 want_grp, want_exg;
    gc_i64 cap_pent, want_pent;
    int *cnt_cl, *cnt_sc, *cl_off, *sc_off;
    int2 *cpb;                   /* the cell's pairs [begin, end)       */
    double4 *origin;
    int *cl_slot, *cl_cell, *ty, *slot_pos;
    float4 *bb, *xq, *xref0, *xref1;  /* xref: each part's prune reference */
    float4 *cv;
    cudaEvent_t interior_pruned; /* boundary pack waits for the interior's launch */
    int4 *sci;
    int *ngi, *ngb, *gcnt, *goff, *wci, *woi, *wcb, *wob;
    int *pcnt, *pbase;           /* its cell's pair count, their scan   */
    int *pent;                   /* [cap_pent] each such pair's first entry */
    int *jent, *g_sci, *need, *eidx;
    int *xg;
    unsigned char *jm8;
    unsigned int *imask, *imask_in, *imask_out, *words;
    /* inner list packed at each prune (nbc_pack); its words are imask_in */
    float4 *pk_jo;
    int *pk_ex;
    float4 *joff;
    gcn::nbc_work *work;

    unsigned int *state;
    int *tot;
    int *pin;                    /* pinned [T_N]                        */
    void *scan_tmp;
    size_t scan_bytes;
    unsigned int *overflow;      /* this unit's gcn_fx::overflow        */
    int4 *xl;                    /* [cap_xl] the rebuild's exclusions   */
    int *xn;
    gc_i64 cap_xl;
    int listed;
    int formed;

    int ncl, nsci, ngrp, nexg, nwi, nw;
    float buffer;
    int nsm;
    int flags_grid;
    int ana;
    double qs;
    gcn::nbc_par par;
    int built;
    int refine[2];   /* per part: its next pass is the build's first (RF) */
    int plain;                   /* GC_TABLE_PME_ELEC: plain LJ pair    */
};

namespace gcn {
namespace {

__global__ void k_nbc_count(const struct gcn_cell *__restrict__ cell, int ncell,
                            int *__restrict__ ncl, int *__restrict__ nsc)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < ncell) {
        const int n = (cell[c].atom_count + 7) / 8;
        ncl[c] = n;
        nsc[c] = (n + 7) / 8;
    } else if (c == ncell) { ncl[c] = 0; nsc[c] = 0; }
}

/* Each cell's range of the rebuild's pair list (grouped by i cell, ascending; checked). */
__global__ void k_nbc_pair_range(const struct gcn_pair *__restrict__ pair, gc_i64 npair,
                                 int ncell, int2 *__restrict__ cpb, int *__restrict__ err)
{
    const gc_i64 p = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npair) return;
    const int ci = pair[p].ci;
    if (ci < 0 || ci >= ncell || pair[p].cj < 0 || pair[p].cj >= ncell) { atomicAdd(err, 1); return; }
    if (p == 0 || pair[p - 1].ci != ci) {
        if (p > 0 && pair[p - 1].ci > ci) atomicAdd(err, 1);
        cpb[ci].x = (int)p;
    }
    if (p == npair - 1 || pair[p + 1].ci != ci) cpb[ci].y = (int)p + 1;
}

/* One warp per cell: clusters of 8 by recursive bisection of the cell's atoms along
   the longest extent, at multiples of 8; atoms in a part are ordered by counting
   rank (ties by position), so the result is deterministic.  Also writes the cell's
   FP64 origin (box midpoint) and super-cluster records. */
__global__ void k_nbc_form(const struct gcn_cell *__restrict__ cell,
                           const gc_i32 *__restrict__ rslot,
                           const gc_f64 *__restrict__ coord, gc_i64 pitch, int ncell,
                           const int *__restrict__ cl_off, const int *__restrict__ sc_off,
                           int *__restrict__ cl_slot, int *__restrict__ slot_pos,
                           int *__restrict__ cl_cell, double4 *__restrict__ origin,
                           int4 *__restrict__ sci, int *__restrict__ err)
{
    __shared__ float s_p[2][3][kMaxCell];
    __shared__ short s_a[2][kMaxCell], s_b[2][kMaxCell];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int c = blockIdx.x * 2 + warp;
    if (c >= ncell) return;
    const int n = cell[c].atom_count, ab = cell[c].atom_begin;
    if (n > kMaxCell) { if (lane == 0) atomicAdd(err, 1); return; }
    double lo[3] = { 1e300, 1e300, 1e300 }, hi[3] = { -1e300, -1e300, -1e300 };
    for (int a = lane; a < n; a += 32) {
        const int s = rslot[ab + a];
        s_a[warp][a] = (short)a;
        for (int d = 0; d < 3; ++d) {
            const double x = coord[d * pitch + s];
            s_p[warp][d][a] = (float)x;
            lo[d] = fmin(lo[d], x);
            hi[d] = fmax(hi[d], x);
        }
    }
    for (int d = 0; d < 3; ++d)
        for (int o = 16; o > 0; o >>= 1) {
            lo[d] = fmin(lo[d], __shfl_xor_sync(0xffffffff, lo[d], o));
            hi[d] = fmax(hi[d], __shfl_xor_sync(0xffffffff, hi[d], o));
        }
    if (lane == 0)
        origin[c] = n > 0 ? make_double4(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]),
                                         0.5 * (lo[2] + hi[2]), 0.0)
                          : make_double4(0.0, 0.0, 0.0, 0.0);
    __syncwarp();
    short *A = s_a[warp], *B = s_b[warp];
    int stb[16], ste[16], sp = 0;   /* parts (b, e) still to split */
    if (n > 1) { stb[0] = 0; ste[0] = n; sp = 1; }
    while (sp > 0) {
        --sp;
        const int b = stb[sp], e = ste[sp], m = e - b;
        float l3[3] = { 3e38f, 3e38f, 3e38f }, h3[3] = { -3e38f, -3e38f, -3e38f };
        for (int k = b + lane; k < e; k += 32)
            for (int d = 0; d < 3; ++d) {
                const float x = s_p[warp][d][A[k]];
                l3[d] = fminf(l3[d], x);
                h3[d] = fmaxf(h3[d], x);
            }
        for (int d = 0; d < 3; ++d)
            for (int o = 16; o > 0; o >>= 1) {
                l3[d] = fminf(l3[d], __shfl_xor_sync(0xffffffff, l3[d], o));
                h3[d] = fmaxf(h3[d], __shfl_xor_sync(0xffffffff, h3[d], o));
            }
        int ax = 0;
        if (h3[1] - l3[1] > h3[ax] - l3[ax]) ax = 1;
        if (h3[2] - l3[2] > h3[ax] - l3[ax]) ax = 2;
        const float *key = s_p[warp][ax];
        for (int k = b + lane; k < e; k += 32) {
            const float kv = key[A[k]];
            int r = 0;
            for (int q = b; q < e; ++q) {
                const float kq = key[A[q]];
                r += (kq < kv) || (kq == kv && q < k);
            }
            B[b + r] = A[k];
        }
        __syncwarp();
        for (int k = b + lane; k < e; k += 32) A[k] = B[k];
        __syncwarp();
        /* a cluster (m <= 8) is sorted along its own longest axis so its halves 0-3 and 4-7
           are apart and the prune's half-cluster masks drop more */
        if (m <= 8) continue;
        int h = 8 * ((m + 8) / 16);
        h = max(8, min(h, 8 * ((m - 1) / 8)));
        if (e - (b + h) > 1) { stb[sp] = b + h; ste[sp] = e; ++sp; }
        if (h > 1) { stb[sp] = b; ste[sp] = b + h; ++sp; }
    }
    const int base = cl_off[c], ncl = cl_off[c + 1] - base;
    for (int k = lane; k < n; k += 32) {
        const int s = rslot[ab + A[k]];
        cl_slot[8 * base + k] = s;
        slot_pos[s] = 8 * base + k;
    }
    for (int k = n + lane; k < 8 * ncl; k += 32) cl_slot[8 * base + k] = -1;
    for (int k = lane; k < ncl; k += 32) cl_cell[base + k] = c;
    const int sb = sc_off[c], ns = sc_off[c + 1] - sb;
    for (int k = lane; k < ns; k += 32)
        sci[sb + k] = make_int4(c, base + 8 * k, min(8, ncl - 8 * k), 0);
}

/* Cluster boxes (absolute, FP32 rounded outward); fillers excluded. */
__global__ void k_nbc_bb(const int *__restrict__ cl_slot, const gc_f64 *__restrict__ coord,
                         gc_i64 pitch, const int *__restrict__ ncl_p, float4 *__restrict__ bb)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= *ncl_p) return;
    float lo[3] = { 3e38f, 3e38f, 3e38f }, hi[3] = { -3e38f, -3e38f, -3e38f };
    for (int k = 0; k < 8; ++k) {
        const int s = cl_slot[8 * c + k];
        if (s < 0) continue;
        for (int d = 0; d < 3; ++d) {
            const double x = coord[d * pitch + s];
            lo[d] = fminf(lo[d], __double2float_rd(x));
            hi[d] = fmaxf(hi[d], __double2float_ru(x));
        }
    }

    const float ext = fmaxf(fmaxf(fmaxf(-lo[0], hi[0]), fmaxf(-lo[1], hi[1])), fmaxf(-lo[2], hi[2]));
    bb[2 * c]     = make_float4(lo[0], lo[1], lo[2], ext);
    bb[2 * c + 1] = make_float4(hi[0], hi[1], hi[2], 0.f);
}

__device__ __forceinline__ int nbc_shift_code(const double *m)
{
    const int kx = m[0] > 0 ? 2 : (m[0] < 0 ? 0 : 1);
    const int ky = m[1] > 0 ? 2 : (m[1] < 0 ? 0 : 1);
    const int kz = m[2] > 0 ? 2 : (m[2] < 0 ? 0 : 1);
    return kx + 3 * ky + 9 * kz;
}

/* One warp per super-cluster: every j cluster of every cell pair of its cell, with an
   8-bit mask of the i clusters whose box is within rout of the j cluster's box.
   Part 0 (interior pairs; all pairs when `split` is 0) comes first, then part 1;
   entries keep pair order, j ascending.  FILL=false counts groups, work items and
   cell pairs (pcnt); FILL=true writes the entries (each part padded to a multiple
   of 4), group owners, work items and each cell pair's first entry (pent). */
template <bool FILL>
__global__ void k_nbc_search(const int4 *__restrict__ sci, const int *__restrict__ nsci_p,
                             const int2 *__restrict__ cpb, const struct gcn_pair *__restrict__ pairs,
                             const int *__restrict__ cl_off, const float4 *__restrict__ bb,
                             const double *__restrict__ move, struct gcn_layout L,
                             int split,
                             float rout2, int *__restrict__ ngi, int *__restrict__ ngb,
                             int *__restrict__ gcnt, int *__restrict__ wci, int *__restrict__ wcb,
                             int *__restrict__ pcnt,
                             const int *__restrict__ goff, const int *__restrict__ woi,
                             const int *__restrict__ wob, int cap_grp,
                             const int *__restrict__ pbase, gc_i64 cap_pent,
                             int *__restrict__ jent, unsigned char *__restrict__ jm8,
                             int *__restrict__ g_sci, nbc_work *__restrict__ work,
                             int *__restrict__ pent, int *__restrict__ err)
{
    __shared__ float4 s_bb[4][16];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int w = blockIdx.x * 4 + warp;
    const int nsci = *nsci_p;
    if (w >= nsci) return;
    if (FILL && (goff[nsci] > cap_grp || pbase[nsci] > cap_pent)) return;   /* the host grows and redoes */
    const int4 sc = sci[w];
    const int ci = sc.x, icl0 = sc.y, nci = sc.z;
    if (lane < 16)
        s_bb[warp][lane] = (lane >> 1) < nci ? bb[2 * icl0 + lane]
                                             : make_float4(3e38f, 3e38f, 3e38f, 0.f);
    __syncwarp();
    const int2 pr_range = cpb[ci];
    if (!FILL && lane == 0) pcnt[w] = pr_range.y - pr_range.x;
    const unsigned int lt = (1u << lane) - 1u;
    int e0 = FILL ? 4 * goff[w] : 0;
    for (int part = 0; part < 2; ++part) {
        int cnt = 0;
        for (int p0 = pr_range.x; p0 < pr_range.y; p0 += 32) {
            const int p = p0 + lane;
            int jb = 0, jn = 0, sh = kCenter, self = 0;
            float mx = 0.f, my = 0.f, mz = 0.f;
            if (p < pr_range.y) {
                const struct gcn_pair &pr = pairs[p];
                const int cj = pr.cj;
                if (pr.ix_n * pr.iy_n > 0 &&
                    !(split ? (nbc_interior(L, ci) && nbc_interior(L, cj) ? 1 : 0) != 1 - part
                            : part != 0)) {
                    sh = nbc_shift_code(pr.move);
                    if (FILL)
                        for (int d = 0; d < 3; ++d)
                            if (fabs(pr.move[d] - move[3 * sh + d]) > 1e-9 * fabs(move[3 * sh + d]))
                                atomicAdd(err, 1);
                    mx = (float)pr.move[0]; my = (float)pr.move[1]; mz = (float)pr.move[2];
                    jb = cl_off[cj]; jn = cl_off[cj + 1] - jb;
                    self = pr.self_pair;
                }
            }
            int end = jn;
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffff, end, o);
                if (lane >= o) end += v;
            }
            const int tot = __shfl_sync(0xffffffff, end, 31);
            for (int t0 = 0; t0 < tot; t0 += 32) {
                const int t = t0 + lane;
                int src = 0;
                for (int step = 16; step > 0; step >>= 1) {
                    const int v = __shfl_sync(0xffffffff, end, src + step - 1);
                    if (v <= t) src += step;
                }
                const int k = t - (__shfl_sync(0xffffffff, end, src) -
                                   __shfl_sync(0xffffffff, jn, src));
                const int jcl = __shfl_sync(0xffffffff, jb, src) + k;
                const int ssh = __shfl_sync(0xffffffff, sh, src);
                const int sself = __shfl_sync(0xffffffff, self, src);
                const float smx = __shfl_sync(0xffffffff, mx, src);
                const float smy = __shfl_sync(0xffffffff, my, src);
                const float smz = __shfl_sync(0xffffffff, mz, src);
                unsigned int m = 0;
                if (t < tot) {
                    const float4 jl = bb[2 * jcl], jh = bb[2 * jcl + 1];
#pragma unroll
                    for (int ic = 0; ic < 8; ++ic) {
                        if (ic >= nci || (sself && jcl < icl0 + ic)) continue;
                        const float4 il = s_bb[warp][2 * ic], ih = s_bb[warp][2 * ic + 1];
                        const float bx = fmaxf(0.f, fmaxf((jl.x - smx) - ih.x, il.x - (jh.x - smx)));
                        const float by = fmaxf(0.f, fmaxf((jl.y - smy) - ih.y, il.y - (jh.y - smy)));
                        const float bz = fmaxf(0.f, fmaxf((jl.z - smz) - ih.z, il.z - (jh.z - smz)));
                        if (bx * bx + by * by + bz * bz < rout2) m |= 1u << ic;
                    }
                }
                const unsigned int bal = __ballot_sync(0xffffffff, m != 0);
                const int pos = e0 + cnt + __popc(bal & lt);
                if (FILL && m) {
                    jent[pos] = (jcl << 5) | ssh;
                    jm8[pos] = (unsigned char)m;
                }
                if (FILL && t < tot && k == 0)
                    pent[pbase[w] + (p0 + src - pr_range.x)] = pos;
                cnt += __popc(bal);
            }
        }
        const int ng = (cnt + 3) >> 2;
        if (!FILL) {
            if (lane == 0) {
                if (part == 0) { ngi[w] = ng; wci[w] = (ng + kGroups - 1) / kGroups; }
                else {
                    ngb[w] = ng; wcb[w] = (ng + kGroups - 1) / kGroups;
                    gcnt[w] = ngi[w] + ng;
                }
            }
            __syncwarp();
            continue;
        }
        const int pad = 4 * ng - cnt;
        if (lane < pad) { jent[e0 + cnt + lane] = -1; jm8[e0 + cnt + lane] = 0; }
        const int g0 = e0 >> 2;
        for (int g = lane; g < ng; g += 32) g_sci[g0 + g] = w;
        const int wbase = part == 0 ? woi[w] : woi[nsci] + wob[w];
        for (int k = lane; k * kGroups < ng; k += 32) {
            nbc_work wk;
            wk.sci = w;
            wk.g0 = g0 + k * kGroups;
            wk.g1 = g0 + min(ng, (k + 1) * kGroups);
            wk.npk = 0;
            work[wbase + k] = wk;
        }
        e0 += 4 * ng;
    }
}

/* The selection's FP32 view of every cluster atom (gcn_kern_atom_view: x
 * NaN beyond c1), in cluster order; fillers NaN. */
__global__ void k_nbc_views(const int *__restrict__ cl_slot, const gc_f64 *__restrict__ coord,
                            gc_i64 pitch, const int *__restrict__ ncl_p, double c1,
                            float4 *__restrict__ cv)
{
    const gc_i64 k = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= 8LL * *ncl_p) return;
    const int sl = cl_slot[k];
    float4 v = make_float4(__int_as_float(0x7fc00000), 0.f, 0.f, 0.f);
    if (sl >= 0) {
        const double x = coord[sl], y = coord[pitch + sl], z = coord[2 * pitch + sl];
        const int ok = fabs(x) <= c1 && fabs(y) <= c1 && fabs(z) <= c1;
        v = make_float4(ok ? (float)x : __int_as_float(0x7fc00000), (float)y, (float)z, 0.f);
    }
    cv[k] = v;
}

/* The MIXED list's pair flags of a cell pair: non-empty (ix_n = iy_n = 1) when some
   atom pair decides each side within the list radius by the selection's rules, or
   a self pair of a cell with atoms.  A warp per pair takes its cluster pairs 32 at a
   time by their FP32 boxes: farthest distance within lo_in makes the pair non-empty,
   nearest beyond rpre makes it empty, the others take the atom-pair rules.  lo_in and
   rpre cover the FP32 rounding of boxes and views, so the flags equal those of all atom
   pairs (the checked build compares with the FP64 pass).  No runs or mask (offsets 0,
   mask_off -1). */
__global__ void k_nbc_flags(struct gcn_pair *__restrict__ pairs, gc_i64 npair,
                            const gc_i64 *__restrict__ np_dev,
                            const struct gcn_cell *__restrict__ cell,
                            const int *__restrict__ cl_off, const float4 *__restrict__ bb,
                            const int *__restrict__ cl_slot, const float4 *__restrict__ cv,
                            const gc_f64 *__restrict__ coord, gc_i64 pitch, double list2,
                            float lo, float hi, float lo_in, float rpre2, float c1f)
{
    const int lane = threadIdx.x & 31;
    if (np_dev) npair = min(npair, *np_dev);
    const gc_i64 nw = ((gc_i64)gridDim.x * blockDim.x) >> 5;
    for (gc_i64 p = ((gc_i64)blockIdx.x * blockDim.x + threadIdx.x) >> 5; p < npair; p += nw) {
        const struct gcn_pair pr = pairs[p];
        int f = 0;
        if (pr.self_pair) {
            f = cell[pr.ci].atom_count > 0;
        } else {
            const int ib = cl_off[pr.ci], ni = cl_off[pr.ci + 1] - ib;
            const int jb = cl_off[pr.cj], nj = cl_off[pr.cj + 1] - jb;
            const float mx = (float)pr.move[0], my = (float)pr.move[1], mz = (float)pr.move[2];
            /* t = i nj + j for t < ni nj <= 4096: the truncation of (t + 1/2) * rcp(nj) is exact in FP32 */
            const float rnj = __frcp_rn((float)nj);
            int f0 = 0, f1 = 0;
            for (int t0 = 0; t0 < ni * nj && !(f0 && f1); t0 += 32) {
                const int t = t0 + lane;
                int icl = 0, jcl = 0, near = 0, sure = 0;
                float bd2 = 0.f;
                if (t < ni * nj) {
                    const int q = (int)(((float)t + 0.5f) * rnj);
                    icl = ib + q;
                    jcl = jb + (t - q * nj);
                    const float4 il = bb[2 * icl], ih = bb[2 * icl + 1];
                    const float4 jl = bb[2 * jcl], jh = bb[2 * jcl + 1];
                    const float ax = (jl.x - mx) - ih.x, bx = il.x - (jh.x - mx);
                    const float ay = (jl.y - my) - ih.y, by = il.y - (jh.y - my);
                    const float az = (jl.z - mz) - ih.z, bz = il.z - (jh.z - mz);
                    const float nx = fmaxf(0.f, fmaxf(ax, bx)), ny = fmaxf(0.f, fmaxf(ay, by)),
                                nz = fmaxf(0.f, fmaxf(az, bz));
                    const float fx = fmaxf(ih.x - (jl.x - mx), (jh.x - mx) - il.x);
                    const float fy = fmaxf(ih.y - (jl.y - my), (jh.y - my) - il.y);
                    const float fz = fmaxf(ih.z - (jl.z - mz), (jh.z - mz) - il.z);
                    const float ext = fmaxf(il.w, jl.w);
                    const int inside = ext <= c1f;
                    bd2 = inside ? nx * nx + ny * ny + nz * nz : 0.f;
                    near = !inside || bd2 < rpre2;
                    sure = inside && fx * fx + fy * fy + fz * fz < lo_in;
                }
                if (__any_sync(0xffffffff, sure)) { f0 = 1; f1 = 1; break; }
                for (unsigned int q = __ballot_sync(0xffffffff, near); q && !(f0 && f1);) {
                    const unsigned int key = __reduce_min_sync(0xffffffff,
                        (q >> lane) & 1u ? (__float_as_uint(bd2) & ~31u) | (unsigned int)lane
                                         : 0xffffffffu);
                    const int src = (int)(key & 31u);
                    q &= ~(1u << src);
                    const int ic = __shfl_sync(0xffffffff, icl, src);
                    const int jc = __shfl_sync(0xffffffff, jcl, src);
                    const int ka = 8 * ic + (lane & 7);
                    const int sa = cl_slot[ka];
                    float4 v = cv[ka];
                    v.x = v.x + mx; v.y = v.y + my; v.z = v.z + mz;
                    int h0 = 0, h1 = 0;
                    for (int r = 0; r < 2; ++r) {
                        const int kb = 8 * jc + (lane >> 3) + 4 * r;
                        const int sb = cl_slot[kb];
                        if (sa < 0 || sb < 0) continue;
                        const float4 k = cv[kb];
                        if (v.x == v.x && k.x == k.x) {
                            const float dx = __fsub_rn(v.x, k.x);
                            const float dy = __fsub_rn(v.y, k.y);
                            const float dz = __fsub_rn(v.z, k.z);
                            const float d2 = __fmaf_rn(dz, dz,
                                             __fmaf_rn(dy, dy, __fmul_rn(dx, dx)));
                            if (d2 < lo) { h0 = 1; h1 = 1; continue; }
                            if (d2 > hi) continue;
                        }
                        const double xa = coord[sa], ya = coord[pitch + sa],
                                     za = coord[2 * pitch + sa];
                        const double xb = coord[sb], yb = coord[pitch + sb],
                                     zb = coord[2 * pitch + sb];
                        {
                            const double x = xa + pr.move[0], y = ya + pr.move[1],
                                         z = za + pr.move[2];
                            const double dx = x - xb, dy = y - yb, dz = z - zb;
                            h0 |= dx*dx + dy*dy + dz*dz < list2;
                        }
                        {
                            const double x = xb + (-pr.move[0]), y = yb + (-pr.move[1]),
                                         z = zb + (-pr.move[2]);
                            const double dx = x - xa, dy = y - ya, dz = z - za;
                            h1 |= dx*dx + dy*dy + dz*dz < list2;
                        }
                    }
                    f0 |= __any_sync(0xffffffff, h0);
                    f1 |= __any_sync(0xffffffff, h1);
                }
            }
            f = f0 && f1;
        }
        if (lane == 0) {
            pairs[p].ix_off = 0; pairs[p].iy_off = 0;
            pairs[p].ix_n = f;   pairs[p].iy_n = f;
            pairs[p].mask_off = -1;
        }
    }
}

/* Per group: the 32-bit outer mask, the FP32 j offsets (origin_cj - origin_ci - move,
   entry code in .w), and whether the group holds a diagonal (self) cluster pair. */
__global__ void k_nbc_group(const int *__restrict__ ngrp_p, int cap, const int *__restrict__ g_sci,
                            const int4 *__restrict__ sci, const int *__restrict__ jent,
                            const unsigned char *__restrict__ jm8, const int *__restrict__ cl_cell,
                            const double4 *__restrict__ origin, const double *__restrict__ move,
                            unsigned int *__restrict__ imask, float4 *__restrict__ joff,
                            int *__restrict__ need)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= *ngrp_p || *ngrp_p > cap) return;
    const int4 sc = sci[g_sci[g]];
    const double4 oi = origin[sc.x];
    unsigned int m = 0;
    int diag = 0;
    for (int jm = 0; jm < 4; ++jm) {
        const int e = jent[4 * g + jm];
        m |= (unsigned int)jm8[4 * g + jm] << (8 * jm);
        float4 o = make_float4(0.f, 0.f, 0.f, __int_as_float(e));
        if (e >= 0) {
            const int jcl = e >> 5, sh = e & 31;
            const double4 oj = origin[cl_cell[jcl]];
            o.x = (float)(oj.x - oi.x - move[3 * sh]);
            o.y = (float)(oj.y - oi.y - move[3 * sh + 1]);
            o.z = (float)(oj.z - oi.z - move[3 * sh + 2]);
            if (sh == kCenter && jcl >= sc.y && jcl < sc.y + sc.z) diag = 1;
        }
        joff[4 * g + jm] = o;
    }
    imask[g] = m;
    need[g] = diag;
}

/* The j offsets alone, after the box was scaled (NPT). */
__global__ void k_nbc_joff(int ngrp, const int *__restrict__ g_sci,
                           const int4 *__restrict__ sci, const int *__restrict__ cl_cell,
                           const double4 *__restrict__ origin, const double *__restrict__ move,
                           float4 *__restrict__ joff)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrp) return;
    const double4 oi = origin[sci[g_sci[g]].x];
    for (int jm = 0; jm < 4; ++jm) {
        float4 o = joff[4 * g + jm];
        const int e = __float_as_int(o.w);
        if (e < 0) continue;
        const int sh = e & 31;
        const double4 oj = origin[cl_cell[e >> 5]];
        o.x = (float)(oj.x - oi.x - move[3 * sh]);
        o.y = (float)(oj.y - oi.y - move[3 * sh + 1]);
        o.z = (float)(oj.z - oi.z - move[3 * sh + 2]);
        joff[4 * g + jm] = o;
    }
}

__global__ void k_nbc_rebox(double4 *__restrict__ origin, int ncell, double *__restrict__ move,
                            float *__restrict__ movef, unsigned int *__restrict__ state,
                            double sx, double sy, double sz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < 2) state[W_FORCE + c] = 1;   /* both parts prune again */
    if (c < ncell) {
        double4 o = origin[c];
        o.x *= sx; o.y *= sy; o.z *= sz;
        origin[c] = o;
    }
    if (c < 27) {
        move[3 * c] *= sx; move[3 * c + 1] *= sy; move[3 * c + 2] *= sz;
        for (int d = 0; d < 3; ++d) movef[3 * c + d] = (float)move[3 * c + d];
    }
}

/* The cluster places of an excluded pair {a, b} of rebuild pair pr, i first. */
__device__ __forceinline__ void nbc_excl_atoms(const struct gcn_pair &pr, int a, int b,
                                               const int *__restrict__ slot_pos, int *pa,
                                               int *pb)
{
    *pa = slot_pos[a];
    *pb = slot_pos[b];
    /* a self pair lists a cluster pair once, lower cluster as i; a
       diagonal one keeps its upper triangle */
    if (pr.self_pair && (*pa >> 3 > *pb >> 3 || (*pa >> 3 == *pb >> 3 && *pa > *pb))) {
        const int t = *pa; *pa = *pb; *pb = t;
    }
}

/* Exclusions: the rebuild's mask clear lists each excluded or 1-4 pair whose bit it
   cleared as {a, b, rebuild pair, -1} (a the i side's slot, a < b in a self pair).
   A thread per entry finds its cluster pair's entry among its cell pair's entries
   and keeps the entry index in .w for k_nbc_xclear; an entry in no group is a
   refusal. */
__global__ void k_nbc_xfind(int4 *__restrict__ xl, const int *__restrict__ xn, gc_i64 cap_xl,
                            const struct gcn_pair *__restrict__ pairs,
                            const int *__restrict__ slot_pos, const int *__restrict__ cl_cell,
                            const int *__restrict__ cl_off, const int *__restrict__ sc_off,
                            const int2 *__restrict__ cpb, const int *__restrict__ pbase,
                            const int *__restrict__ npent_p, const int *__restrict__ pent,
                            gc_i64 cap_pent,
                            const int *__restrict__ goff, const int *__restrict__ ngrp_p, int cap,
                            const int *__restrict__ jent,
                            const unsigned int *__restrict__ imask, int *__restrict__ need,
                            int *__restrict__ err)
{
    const gc_i64 k = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    const int n = *xn;
    if (*ngrp_p > cap || *npent_p > cap_pent) return;
    if ((gc_i64)n > cap_xl) {
        if (k == 0) atomicAdd(err, 1);
        return;
    }
    if (k >= n) return;
    const int4 x = xl[k];
    const struct gcn_pair &pr = pairs[x.z];
    int pa, pb;
    nbc_excl_atoms(pr, x.x, x.y, slot_pos, &pa, &pb);
    const int ca = pa >> 3, cb = pb >> 3, ci = cl_cell[ca];
    const int li = ca - cl_off[ci];
    const int s = sc_off[ci] + (li >> 3), ic = li & 7;
    const int key = (cb << 5) | nbc_shift_code(pr.move);
    const int e0 = 4 * goff[s], e1 = 4 * goff[s + 1];
    const int eb = ci == pr.ci ? max(e0, min(e1, pent[pbase[s] + (x.z - cpb[ci].x)])) : e1;
    const int ee = min(e1, eb + (cl_off[pr.cj + 1] - cl_off[pr.cj]));
    int found = -1;
    for (int e = eb; e < ee; ++e)
        if (jent[e] == key && (imask[e >> 2] & (1u << (8 * (e & 3) + ic)))) { found = e; break; }
    if (found < 0) atomicAdd(err, 1);
    else need[found >> 2] = 1;
    xl[k].w = found;
}

/* Clear each listed pair's bit in its group's words. */
__global__ void k_nbc_xclear(const int4 *__restrict__ xl, const int *__restrict__ xn,
                             gc_i64 cap_xl, const struct gcn_pair *__restrict__ pairs,
                             const int *__restrict__ slot_pos, const int *__restrict__ cl_off,
                             const int *__restrict__ cl_cell, const int *__restrict__ ngrp_p,
                             int cap, const int *__restrict__ nexg_p, int cap_exg,
                             const int *__restrict__ eidx, unsigned int *__restrict__ words)
{
    const gc_i64 k = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    const int n = *xn;
    if (k >= n || (gc_i64)n > cap_xl || *ngrp_p > cap || *nexg_p > cap_exg) return;
    const int4 x = xl[k];
    if (x.w < 0) return;
    int pa, pb;
    nbc_excl_atoms(pairs[x.z], x.x, x.y, slot_pos, &pa, &pb);
    const int ca = pa >> 3, li = ca - cl_off[cl_cell[ca]];
    const int bj = pb & 7;
    const unsigned int bit = 1u << (8 * (x.w & 3) + (li & 7));
    atomicAnd(&words[64 * (size_t)eidx[x.w >> 2] + 32 * (bj >> 2) + ((bj & 3) << 3) + (pa & 7)],
              ~bit);
}

/* The groups that have words, in word order (xg[eidx[g]] = g); eidx -1
 * for the others. */
__global__ void k_nbc_eidx(const int *__restrict__ ngrp_p, int cap, const int *__restrict__ nexg_p,
                           int cap_exg, const int *__restrict__ need, int *__restrict__ eidx,
                           int *__restrict__ xg)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= *ngrp_p || *ngrp_p > cap) return;
    if (!need[g]) eidx[g] = -1;
    else if (*nexg_p <= cap_exg) xg[eidx[g]] = g;
}

/* Initial words of the groups that have them: every pair on but the lower triangle
   and diagonal of a self cluster pair.  Sixteen threads per group, four words each. */
__global__ void k_nbc_winit(const int *__restrict__ ngrp_p, int cap, const int *__restrict__ nexg_p,
                            int cap_exg, const int *__restrict__ xg,
                            const int *__restrict__ g_sci, const int4 *__restrict__ sci,
                            const int *__restrict__ jent, unsigned int *__restrict__ words)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int nexg = *nexg_p;
    const int q = (int)(t >> 4), l0 = (int)(t & 15);
    if (q >= nexg || *ngrp_p > cap || nexg > cap_exg) return;
    const int g = xg[q];
    const int4 sc = sci[g_sci[g]];
    int e[4];
    for (int jm = 0; jm < 4; ++jm) e[jm] = jent[4 * g + jm];
    for (int w = 0; w < 4; ++w) {
        const int l = l0 + 16 * w;
        const int a = l & 7, b = ((l >> 5) << 2) | ((l >> 3) & 3);
        unsigned int wd = 0xffffffffu;
        for (int jm = 0; jm < 4; ++jm) {
            if (e[jm] < 0 || (e[jm] & 31) != kCenter) continue;
            const int jcl = e[jm] >> 5;
            for (int ic = 0; ic < sc.z; ++ic)
                if (jcl == sc.y + ic && b <= a) wd &= ~(1u << (8 * jm + ic));
        }
        words[64 * (size_t)q + l] = wd;
    }
}

/* classes, 0 based; fillers the null class ncls */
__global__ void k_nbc_types(const int *__restrict__ cl_slot, const gc_i32 *__restrict__ cls,
                            const int *__restrict__ ncl_p, int ncls, int *__restrict__ ty)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= 8LL * *ncl_p) return;
    const int s = cl_slot[p];
    ty[p] = s >= 0 ? cls[s] - 1 : ncls;
}

__global__ void k_nbc_totals(const int *__restrict__ ncl_p, const int *__restrict__ nsci_p,
                             const int *__restrict__ ngrp_p, const int *__restrict__ nwi_p,
                             const int *__restrict__ nwb_p, const int *__restrict__ nexg_p,
                             const int *__restrict__ npent_p, int cap_grp, int cap_exg,
                             gc_i64 cap_pent, const int *__restrict__ err,
                             int *__restrict__ tot)
{
    const int ngrp = *ngrp_p;
    const bool fits = ngrp <= cap_grp;
    tot[T_NCL] = *ncl_p;
    tot[T_NSCI] = *nsci_p;
    tot[T_NGRP] = ngrp;
    tot[T_NEXG] = fits ? *nexg_p : 0;
    tot[T_NWI] = *nwi_p;
    tot[T_NW] = *nwi_p + *nwb_p;
    tot[T_ERR] = *err;
    tot[T_OVF] = !fits || *nexg_p > cap_exg || *npent_p > cap_pent;
    tot[T_NPENT] = *npent_p;
}

/* The FP32 tables of the pair term: the LJ pairs with a null class (times 12 and 6
   for the analytic term) and the interval records of native_attach_tables
   (d->nb_mixed) as two float4 per record, record L at [2L], [2L+1], record 0 zero. */
__global__ void k_nbc_tables(const float2 *__restrict__ mixed, int ncls, int nrow, float s12, float s6,
                             float2 *__restrict__ ljf, float4 *__restrict__ tab)
{
    const int n1 = ncls + 1;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n1 * n1) {
        const int a = t / n1, b = t % n1;
        const float2 lj = (a < ncls && b < ncls) ? mixed[a * ncls + b] : make_float2(0.f, 0.f);
        ljf[t] = make_float2(s12 * lj.x, s6 * lj.y);
    }
    if (t < 2 * nrow) {
        const int e = t / nrow, L = t % nrow;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f), dl = v;
        if (L > 0) {
            const float2 *r = mixed + (size_t)ncls * ncls + 3 * ((size_t)e * nrow + L - 1);
            v = make_float4(r[0].x, r[0].y, r[1].x, 0.f);
            dl = make_float4(r[1].y, r[2].x, r[2].y, 0.f);
        }
        tab[2 * (e * nrow + L)] = v;
        tab[2 * (e * nrow + L) + 1] = dl;
    }
}

/* Fillers sit far away and apart (a 64 A lattice), so no pair with one is
 * ever inside the support. */
__device__ __forceinline__ float3 nbc_far(long long p)
{
    return make_float3(2.0e4f + 64.f * (float)(p & 1023), 2.0e4f + 64.f * (float)((p >> 10) & 1023),
                       2.0e4f + 64.f * (float)(p >> 20));
}

__device__ __forceinline__ float nbc_d2(float4 a, float4 b)
{
    const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

/* FP32 coordinates relative to the cell origin, w = charge * qs, for the clusters of
   owned (OWNED) or ghost cells.  The previous value becomes a part's prune reference
   when that part pruned; the largest displacement from the reference goes to the
   part's word (an order-independent maximum).  A thread loads all kPackItems slots
   before storing any. */
constexpr int kPackItems = 2;
template <bool OWNED>
__global__ void __launch_bounds__(256)
k_nbc_pack(const int *__restrict__ cl_slot, const int *__restrict__ cl_cell,
           const double4 *__restrict__ origin, const gc_f64 *__restrict__ coord,
           const gc_f64 *__restrict__ charge, double qs, gc_i64 pitch, int ncl, struct gcn_layout L,
           int split, int part, float4 *__restrict__ xq,
           float4 *__restrict__ xref0,
           float4 *__restrict__ xref1, unsigned int *__restrict__ state)
{
    __shared__ float s_m[2][8];
    const long long n = 8LL * ncl;
    const unsigned int need0 = state[W_NEED], need1 = state[W_NEED + 1];
    float m0 = 0.f, m1 = 0.f;
    for (long long base = (long long)blockIdx.x * (256 * kPackItems) + threadIdx.x; base < n;
         base += (long long)gridDim.x * (256 * kPackItems)) {
        long long p[kPackItems];
        int act[kPackItems];
        float4 v[kPackItems], old[kPackItems], r0[kPackItems], r1[kPackItems];
#pragma unroll
        for (int k = 0; k < kPackItems; ++k) {
            p[k] = base + 256 * k;
            act[k] = 0;
            if (p[k] >= n) continue;
            const int c = cl_cell[p[k] >> 3];
            const bool own = !split || nbc_interior(L, c);
            if (own == OWNED) {
                act[k] = 1;
                const int s = cl_slot[p[k]];
                if (s >= 0) {
                    const double4 o = origin[c];
                    v[k] = make_float4((float)(coord[s] - o.x), (float)(coord[pitch + s] - o.y),
                                       (float)(coord[2 * pitch + s] - o.z), (float)(charge[s] * qs));
                } else {
                    const float3 f = nbc_far(p[k]);
                    v[k] = make_float4(f.x, f.y, f.z, 0.f);
                }
                old[k] = xq[p[k]];
                if (OWNED) {
                    if (!need0) r0[k] = xref0[p[k]];
                    if (split && !need1) r1[k] = xref1[p[k]];
                } else if (!(part ? need1 : need0)) {
                    r0[k] = (part ? xref1 : xref0)[p[k]];
                }
            } else if (!OWNED && part == 1) {
                /* the owned clusters' distance from the boundary reference:
                   their coordinates are this step's (packed before the fork) */
                act[k] = 2;
                v[k] = xq[p[k]];
                r1[k] = xref1[p[k]];
            }
        }
#pragma unroll
        for (int k = 0; k < kPackItems; ++k) {
            if (act[k] == 2) {
                m1 = fmaxf(m1, nbc_d2(v[k], r1[k]));
                continue;
            }
            if (!act[k]) continue;
            xq[p[k]] = v[k];
            if (OWNED) {
                if (need0) { r0[k] = old[k]; xref0[p[k]] = old[k]; }
                m0 = fmaxf(m0, nbc_d2(v[k], r0[k]));
                if (split) {
                    if (need1) { r1[k] = old[k]; xref1[p[k]] = old[k]; }
                    m1 = fmaxf(m1, nbc_d2(v[k], r1[k]));
                }
            } else {
                if (part ? need1 : need0) { r0[k] = old[k]; (part ? xref1 : xref0)[p[k]] = old[k]; }
                const float d = nbc_d2(v[k], r0[k]);
                if (part) m1 = fmaxf(m1, d); else m0 = fmaxf(m0, d);
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, o));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, o));
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { s_m[0][w] = m0; s_m[1][w] = m1; }
    __syncthreads();
    if (threadIdx.x < 2) {
        float m = 0.f;
        for (int k = 0; k < 8; ++k) m = fmaxf(m, s_m[threadIdx.x][k]);
        if (m > 0.f) atomicMax(&state[W_DISP + threadIdx.x], __float_as_uint(m));
    }
}

/* A part prunes when built or reboxed, or when twice the largest displacement since
   its last prune reaches the buffer less an FP32 margin.  The force pass prunes
   (k_nbc_mixed). */
__global__ void k_nbc_decide(unsigned int *__restrict__ state, int part, float buffer)
{
    const float d = sqrtf(__uint_as_float(state[W_DISP + part]));
    const unsigned int force = state[W_FORCE + part];
    const unsigned int need = force || 2.f * d >= buffer - 0.01f;
    state[W_NEED + part] = need;
    state[W_FORCE + part] = 0;
    state[W_DISP + part] = 0;
}

/* A pointer or index the compiler must keep rather than rematerialise. */
template <typename T>
__device__ __forceinline__ T *nbc_opaque(T *p)
{
    asm volatile("" : "+l"(p));
    return p;
}
__device__ __forceinline__ int nbc_opaque(int i)
{
    asm volatile("" : "+r"(i));
    return i;
}

/* flush-to-zero approximations: an r2 inside the support is never
 * denormal, and an excluded r2 = 0 gives inf, clamped and masked */
__device__ __forceinline__ float nbc_rcp(float x)
{
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

/* The FP32 pair term, unmasked: the gradient coefficient (F_i = -gr * d) and, with E,
   the pair's elec and vdW energies.  PLAIN (GC_TABLE_PME_ELEC): the table's LJ columns
   are zero and the LJ pair is lj.x r^-12 - lj.y r^-6. */
template <bool E, bool PLAIN>
__device__ __forceinline__ float nbc_pair(float r2, float qq, float2 lj, const float4 *__restrict__ tab,
                                          const nbc_par &P, float &eel, float &evd)
{
    const float ri2 = nbc_rcp(r2);
    const float s = fminf(P.cdf * ri2, P.lmax);
    const int L = (int)s;
    const float Rf = s - (float)L;
    const float4 b = tab[2 * L], d = tab[2 * L + 1];
    float gr;
    if (PLAIN) {
        const float r6 = ri2 * ri2 * ri2, r12 = r6 * r6;
        gr = (6.f * lj.y * r6 - 12.f * lj.x * r12) * ri2 + qq * fmaf(Rf, d.z, b.z);
        if (E) evd = lj.x * r12 - lj.y * r6;
    } else {
        gr = fmaf(Rf, d.x, b.x) * lj.x - fmaf(Rf, d.y, b.y) * lj.y + qq * fmaf(Rf, d.z, b.z);
    }
    if (E) {
        const float4 be = tab[2 * (P.erow + L)], de = tab[2 * (P.erow + L) + 1];
        if (!PLAIN) evd = fmaf(Rf, de.x, be.x) * lj.x - fmaf(Rf, de.y, be.y) * lj.y;
        eel = qq * fmaf(Rf, de.z, be.z);
    }
    return gr;
}

/* The analytic pair term (charges carry sqrt(coulomb)): Ewald real space qq erfc(beta r)/r
   with polynomial corrections in t = r2*tA - 1 (Chebyshev fits, nbc_fit_ewald) and
   c12/r^12 - c6/r^6 with CHARMM's switch between ron and roff.  PLAIN has no switch:
   S = 1, dS = 0. */
template <int N>
__device__ __forceinline__ float nbc_poly(const float *c, float t)
{
    float p = c[N > 0 ? N - 1 : 0];   /* N = 0: the table kernel's dead branch */
#pragma unroll
    for (int k = N - 2; k >= 0; --k) p = fmaf(p, t, c[k]);
    return p;
}
__device__ __forceinline__ float nbc_rsqrt(float x)
{
    float y;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
template <bool E, int NP, int VDW>
__device__ __forceinline__ float nbc_pair_ana(float r2, float qq, float2 lj, const nbc_par &P,
                                              float &eel, float &evd)
{
    const float ri = nbc_rsqrt(r2);
    const float ri2 = ri * ri;
    float glj;
    if (VDW == kVdwFsw) {
        /* CHARMM's force switch (fs of nbc_par): with u = r^-6 and w = r^-3 the gradient
           coefficients are -12 c12 u h12(u) r^-2 and -6 c6 w h6(w) r^-2, h(x) = x below ron and
           k (x - roff^-n) above; the lines meet at ron and k > 1, so h is their minimum (no branch). */
        const float ri3 = ri2 * ri, ri6 = ri3 * ri3;
        const float u12 = lj.x * ri6, u6 = lj.y * ri3;
        const float h12 = fminf(ri6, fmaf(P.fs[0], ri6, -P.fs[1]));
        const float h6 = fminf(ri3, fmaf(P.fs[4], ri3, -P.fs[5]));
        glj = (u6 * h6 - u12 * h12) * ri2;
        if (E) {
            const bool out = r2 > P.ron2;
            const float a12 = out ? P.fs[0] : 1.f, b12 = out ? P.fs[1] : 0.f;
            const float a6 = out ? P.fs[4] : 1.f, b6 = out ? P.fs[5] : 0.f;
            /* c12 (a12 u^2 - 2 b12 u + c) - c6 (a6 w^2 - 2 b6 w + c), c = k roff^-2n
               above ron and -A below */
            const float c12 = out ? P.fs[2] : -P.fs[3], c6 = out ? P.fs[6] : -P.fs[7];
            const float e12 = fmaf(ri6, fmaf(a12, ri6, -2.f * b12), c12);
            const float e6 = fmaf(ri3, fmaf(a6, ri3, -2.f * b6), c6);
            evd = fmaf(lj.x * (1.f / 12.f), e12, -lj.y * (1.f / 6.f) * e6);
        }
    } else {
        const float ri6 = ri2 * ri2 * ri2;
        const float f6 = lj.y * ri6;
        const float f12 = lj.x * ri6 * ri6;
        if (VDW == kVdwPlain) {
            glj = -__fsub_rn(f12, f6) * ri2;
            if (E) evd = fmaf(f12, 1.f / 12.f, f6 * (-1.f / 6.f));
        } else {
            const float v = fmaf(f12, 1.f / 12.f, f6 * (-1.f / 6.f));
            /* the switch; S = 1 and dS = 0 exactly at r2 <= ron2 (selects, no
               branch: the polynomial at ron2 rounds to 1 + O(eps)) */
            const bool in = r2 > P.ron2;
            const float t1 = P.roff2 - r2;
            const float S = in ? t1 * t1 * fmaf(P.sw_a, r2, P.sw_b) : 1.f;
            const float dS = in ? t1 * fmaf(P.sw_d, r2, P.sw_e) : 0.f;
            glj = fmaf(dS, v, -S * (f12 - f6) * ri2);
            if (E) evd = S * v;
        }
    }
    const float tz = fmaf(r2, P.tA, -1.f);
    if (E) eel = qq * fmaf(-P.beta, nbc_poly<NP>(P.pe, tz), ri);
    return fmaf(qq, nbc_poly<NP>(P.pf, tz) - ri2 * ri, glj);
}

/* gcn_fx::add1<kForce1> of an FP32 term: the words of (double)v's add */
__device__ __forceinline__ void nbc_fadd(gcn_fx::word *f, gc_i64 i, float v)
{
    gcn_fx::add1<gcn_fx::kForce1>(f + i, v);
}

/* gcn_fx::add1<kForce1> of a j force term at f[i] (i 32-bit: native_nbc_force checks
   the pitch); a zero term is added too, as testing for it costs more than it saves. */
__device__ __forceinline__ void nbc_jadd(gcn_fx::word *f, unsigned int i, float v)
{
    const float t = v * (float)(1ull << gcn_fx::kForce1);   /* exact scale */
    if (!(fabsf(t) < 18014398509481984.0f)) {               /* 2^54        */
        atomicOr(&gcn_fx::overflow, 1u);
        return;
    }
    atomicAdd(f + i, (gcn_fx::word)__float2ll_rn(t));
}

/* Transposed 8-lane reduction of (x,y,z) over lane bits 0-2: lanes with
 * bits (b1 b0) = 00 hold the x total, 01 y, 1x z. */
__device__ __forceinline__ float nbc_red3(float x, float y, float z, int ti)
{
    const bool o1 = ti & 1, o2 = ti & 2;
    float keep = o1 ? y : x;
    keep += __shfl_xor_sync(0xffffffff, o1 ? x : y, 1);
    z += __shfl_xor_sync(0xffffffff, z, 1);
    float v = o2 ? z : keep;
    v += __shfl_xor_sync(0xffffffff, o2 ? keep : z, 2);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    return v;
}

/* A work item's i forces: the 4 j lanes of a warp by a transposed butterfly over lane
   bits 3 and 4, after which lane (ti, tjl) holds i clusters 2q and 2q + 1,
   q = 2 (tjl & 1) + (tjl >> 1); the second warp's sums go through shared memory onto
   the first's (fixed order), one fixed-point add per atom and component. */
__device__ __forceinline__ void nbc_ifold(float (&fi)[8][3], float (*s_fi)[64], int4 sc,
                                          const int *__restrict__ cl_slot,
                                          gcn_fx::word *__restrict__ force, gc_i64 pitch,
                                          int ti, int tjl, int w)
{
    const bool o3 = tjl & 1, o4 = tjl & 2;
    float h[12], g[6];
#pragma unroll
    for (int k = 0; k < 12; ++k) {
        const float a = fi[k / 3][k % 3], b = fi[(k + 12) / 3][(k + 12) % 3];
        h[k] = (o3 ? b : a) + __shfl_xor_sync(0xffffffff, o3 ? a : b, 8);
    }
#pragma unroll
    for (int k = 0; k < 6; ++k)
        g[k] = (o4 ? h[k + 6] : h[k]) + __shfl_xor_sync(0xffffffff, o4 ? h[k] : h[k + 6], 16);
    const int ic0 = 4 * (tjl & 1) + 2 * (tjl >> 1);
    if (w == 1)
#pragma unroll
        for (int k = 0; k < 6; ++k) s_fi[k % 3][8 * (ic0 + k / 3) + ti] = g[k];
    __syncthreads();
    if (w == 0)
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {
            const int ic = ic0 + h2;
            const int s = ic < sc.z ? cl_slot[8 * (sc.y + ic) + ti] : -1;
            if (s < 0) continue;
#pragma unroll
            for (int c = 0; c < 3; ++c)
                nbc_fadd(force, (gc_i64)c * pitch + s, g[3 * h2 + c] + s_fi[c][8 * ic + ti]);
        }
}

/* Energies and virial of a block as fixed-point sums, one add per slot.  Virial
   component c is non-zero only on lanes with ti = c, so only the two butterfly
   levels over the j rows add more than zeros; lanes 0-2 then hold the components. */
template <bool E>
__device__ __forceinline__ void nbc_block_acc(gcn_fx::word *acc, const double *v5, float virc,
                                              double *s_red)
{
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    if (E)
        for (int k = 0; k < 2; ++k) {
            double v = v5[k];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if (lane == 0) s_red[5 * w + k] = v;
        }
    double v = virc;
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    if (lane < 3) s_red[5 * w + 2 + lane] = v;
    __syncthreads();
    if (t < 5 && (E || t >= 2)) {
        const double v = s_red[t] + s_red[5 + t];
        if (v != 0.0) gcn_fx::add<gcn_fx::kEnergy>(acc + 2 * t, v);
    }
}

/* A prune's inner list, packed for the passes up to the next prune: j clusters with a
   pair in either warp half's new word fill groups of 4 in order.  A group with
   excluded pairs keeps its slots (its words are per slot), goes first, and is dropped
   only when both words are empty.  One thread per slot (4 kGroups = the block). */
static_assert(4 * kGroups == 64, "one thread per slot");
__device__ __forceinline__ void nbc_pack(nbc_work *wk, int g0, int ng, const float4 *s_jo,
                                         const int *s_ex, const unsigned int *s_nm,
                                         unsigned int *s_pm, unsigned int *pk_im,
                                         float4 *pk_jo, int *pk_ex)
{
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    const int g = t >> 2, jm = t & 3;
    __shared__ unsigned int s_b[3];
    const bool live = g < ng;
    const unsigned int b0 = live ? (s_nm[2 * g] >> (8 * jm)) & 0xffu : 0u;
    const unsigned int b1 = live ? (s_nm[2 * g + 1] >> (8 * jm)) & 0xffu : 0u;
    const bool exg = live && s_ex[g] >= 0;
    const bool keep_x = t < ng && s_ex[t] >= 0 && (s_nm[2 * t] | s_nm[2 * t + 1]) != 0u;
    const unsigned int mx = __ballot_sync(0xffffffffu, keep_x);
    const bool keep_f = live && !exg && (b0 | b1) != 0u;
    const unsigned int mf = __ballot_sync(0xffffffffu, keep_f);
    if (lane == 0) s_b[1 + w] = mf;
    if (t == 0) s_b[0] = mx;
    if (t < 2 * kGroups) s_pm[t] = 0u;
    __syncthreads();
    const unsigned int ax = s_b[0];
    const int nx = __popc(ax), nf = __popc(s_b[1]) + __popc(s_b[2]);
    const int npk = nx + (nf + 3) / 4;
    if (live && exg) {
        const unsigned int xg = __popc(ax & ((1u << g) - 1u));
        if ((ax >> g) & 1u) {
            pk_jo[4 * (g0 + xg) + jm] = s_jo[t];
            if (jm < 2) s_pm[2 * xg + jm] = s_nm[2 * g + jm];
            if (jm == 0) pk_ex[g0 + xg] = s_ex[g];
        }
    }
    const int q = 4 * nx + (w ? __popc(s_b[1]) : 0) + __popc(s_b[1 + w] & ((1u << lane) - 1u));
    if (keep_f) {
        pk_jo[4 * g0 + q] = s_jo[t];
        atomicOr(&s_pm[2 * (q >> 2)], b0 << (8 * (q & 3)));
        atomicOr(&s_pm[2 * (q >> 2) + 1], b1 << (8 * (q & 3)));
    }
    const int qe = 4 * nx + nf;
    if (t < 4 * npk - qe) pk_jo[4 * g0 + qe + t] = make_float4(0.f, 0.f, 0.f, __int_as_float(-1));
    if (t >= nx && t < npk) pk_ex[g0 + t] = -1;
    __syncthreads();
    if (t < 2 * npk) pk_im[2 * g0 + t] = s_pm[t];
    if (t == 0) wk->npk = npk;
}

/* The work item's j codes, offsets, masks and exclusion indices go to shared memory
   once; the 4 j clusters of each group are copied with cp.async, double buffered.
   A j force is committed only when some lane holds a non-zero one.  E: energies (the
   periodic-offset virial is always summed).  RF: the instance of a part's first pass
   after a build; it prunes the bounding-box list and refines it to the outer list. */
template <bool E, int NP, int VDW, bool RF>
__global__ void __launch_bounds__(64, kForceSm)
k_nbc_mixed(nbc_work *work, const int4 *__restrict__ sci,
            const float4 *__restrict__ xq, const int *__restrict__ ty, const int *__restrict__ cl_slot,
            const float4 *__restrict__ joff, const unsigned int *__restrict__ imask,
            unsigned int *imask_in, unsigned int *imask_out, float4 *pk_jo, int *pk_ex,
            const unsigned int *__restrict__ need, float rin2, float rl2,
            const int *__restrict__ eidx, const unsigned int *__restrict__ words,
            const float2 *__restrict__ ljf, const float4 *__restrict__ tab,
            const float *__restrict__ movef, gcn_fx::word *__restrict__ force,
            gcn_fx::word *__restrict__ acc, gc_i64 pitch, nbc_par P)
{
    __shared__ float4 s_x[64];
    __shared__ int    s_t[64];
    __shared__ float4 s_jo[4 * kGroups];
    __shared__ unsigned int s_im[2 * kGroups];
    __shared__ int    s_ex[kGroups];
    __shared__ __align__(16) float4 s_jx[2][32];
    __shared__ __align__(16) int    s_jt[2][32];
    __shared__ __align__(16) int    s_js[2][32];
    __shared__ union { float fi[3][64]; unsigned int nm[2 * kGroups]; } s_u;
    unsigned int *const s_nm = s_u.nm;
    __shared__ double s_red[10];
    __shared__ float  s_par[10];
    __shared__ float  s_mv[81];
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    const int ti = lane & 7, tjl = lane >> 3, tj = tjl + 4 * w;
    const nbc_work wk = work[blockIdx.x];
    if (NP && t == 0) {
        s_par[0] = P.tA; s_par[1] = P.roff2; s_par[2] = P.sw_a; s_par[3] = P.sw_b;
        s_par[4] = P.sw_d; s_par[5] = P.sw_e;
        s_par[6] = P.pf[NP > 2 ? NP - 2 : 0]; s_par[7] = P.pf[NP > 3 ? NP - 3 : 0];
        s_par[8] = P.fs[0]; s_par[9] = P.fs[4];
    }
for (int k = t; k < 81; k += 64) s_mv[k] = movef[k];
    const int4 sc = sci[wk.sci];
    /* a pruning pass walks the outer list (RF: the bounding-box list) and
       packs the inner list; the other passes walk the packed list */
    const bool prn = RF || *need;
    const int ng = prn ? wk.g1 - wk.g0 : wk.npk;
    const bool iv = (t >> 3) < sc.z;
    s_x[t] = iv ? xq[8 * sc.y + t] : make_float4(-3e4f, -3e4f, -3e4f, 0.f);
    s_t[t] = (iv ? ty[8 * sc.y + t] : P.ncls1 - 1) * P.ncls1;
    if (t < 4 * ng) s_jo[t] = prn ? joff[4 * wk.g0 + t] : pk_jo[4 * wk.g0 + t];
    if (t < 2 * ng)
        s_im[t] = RF ? imask[wk.g0 + (t >> 1)]
                : prn ? imask_out[2 * wk.g0 + t] : imask_in[2 * wk.g0 + t];
    if (t < ng) s_ex[t] = prn ? eidx[wk.g0 + t] : pk_ex[wk.g0 + t];
    __syncthreads();

    /* stage group gl into buffer b: threads 0-31 one atom's xq each, 32-39 four classes
       each, 40-47 four slots each.  kNB (sm_80): each warp stages its own j atoms (lanes
       l < 16 the xq of atom 4 w + (l & 3) of j cluster l >> 2 at [16 w + l], lanes 16-31
       class and slot) and the warps keep apart until the group loop ends */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
    constexpr bool kNB = true;
#else
    constexpr bool kNB = false;
#endif
    constexpr int js = kNB ? 4 : 8;
    auto stage = [&](int gl, int b) {
        if (kNB) {
            const int q = lane & 15, jm = q >> 2;
            const int code = __float_as_int(s_jo[4 * gl + jm].w);
            if (code >= 0) {
                const int a = 8 * (code >> 5) + 4 * w + (q & 3);
                if (lane < 16) __pipeline_memcpy_async(&s_jx[b][16 * w + q], &xq[a], 16);
                else {
                    __pipeline_memcpy_async(&s_jt[b][16 * w + q], &ty[a], 4);
                    __pipeline_memcpy_async(&s_js[b][16 * w + q], &cl_slot[a], 4);
                }
            }
        } else if (t < 48) {
            const int jm = t < 32 ? (t >> 3) : ((t & 7) >> 1);
            const int code = __float_as_int(s_jo[4 * gl + jm].w);
            if (code >= 0) {
                const int jcl = code >> 5;
                if (t < 32) __pipeline_memcpy_async(&s_jx[b][t], &xq[8 * jcl + (t & 7)], 16);
                else if (t < 40) __pipeline_memcpy_async(&s_jt[b][4 * (t - 32)], &ty[8 * jcl + 4 * (t & 1)], 16);
                else __pipeline_memcpy_async(&s_js[b][4 * (jm * 2 + (t & 1))], &cl_slot[8 * jcl + 4 * (t & 1)], 16);
            }
        }
        __pipeline_commit();
    };

    float fi[8][3];
#pragma unroll
    for (int k = 0; k < 8; ++k) fi[k][0] = fi[k][1] = fi[k][2] = 0.f;
    double v5[2] = { 0.0, 0.0 };
    float virc = 0.f;   /* lanes ti < 3: component ti of the periodic-offset virial */
    /* this lane's i and j entries; opaque indices keep one register each and the shared
       arrays' loads shared, not generic; the LJ pairs go through the read-only path */
    const int oi = nbc_opaque(ti);
    const int oj = nbc_opaque(kNB ? 16 * w + tjl : tj);
    const float2 *ljo = nbc_opaque(ljf);
    /* analytic parameters that take no constant operand slot, read back from shared memory
       so they stay in registers (ptxas reloads kernel parameters at each use) */
    nbc_par Q = P;
    if (NP) {
        Q.tA = s_par[0]; Q.roff2 = s_par[1]; Q.sw_a = s_par[2]; Q.sw_b = s_par[3];
        Q.sw_d = s_par[4]; Q.sw_e = s_par[5];
        Q.pf[NP > 2 ? NP - 2 : 0] = s_par[6]; Q.pf[NP > 3 ? NP - 3 : 0] = s_par[7];
        if (VDW == kVdwFsw) { Q.fs[0] = s_par[8]; Q.fs[4] = s_par[9]; }   /* fma(fs0, x, -fs1) */
    }
    const unsigned int fcomp = (unsigned int)(ti < 3 ? ti : 0) * (unsigned int)pitch;
    if (ng > 0) stage(0, 0);
    for (int gl = 0; gl < ng; ++gl) {
        const int b = gl & 1;
        if (gl + 1 < ng) stage(gl + 1, b ^ 1); else __pipeline_commit();
        const int ex = s_ex[gl];
        const unsigned int wx = ex < 0 ? 0xffffffffu : words[64 * (size_t)ex + t];
        __pipeline_wait_prior(1);
        if (kNB) {
            if (lane < 16) {
                const float4 o = s_jo[4 * gl + (lane >> 2)];
                float4 *v = &s_jx[b][16 * w + lane];
                v->x += o.x; v->y += o.y; v->z += o.z;
            }
            __syncwarp();
        } else {
            if (t < 32) {
                const float4 o = s_jo[4 * gl + (t >> 3)];
                float4 *v = &s_jx[b][t];
                v->x += o.x; v->y += o.y; v->z += o.z;
            }
            __syncthreads();
        }
        const unsigned int im = __reduce_or_sync(0xffffffffu, s_im[2 * gl + w]);
        unsigned int nm = 0u;
        unsigned int no = 0u;
        auto jclusters = [&](auto kMasked, auto kPrune) {
            constexpr bool masked = decltype(kMasked)::value;
            constexpr bool PR = decltype(kPrune)::value;
#pragma unroll 1
            for (int jm = 0; jm < 4; ++jm) {
                const unsigned int mb = (im >> (8 * jm)) & 0xffu;
                if (!mb) continue;
                const float4 xj = s_jx[b][js * jm + oj];
                const float2 *ljj = nbc_opaque(ljo + s_jt[b][js * jm + oj]);
                const float jx = xj.x, jy = xj.y, jz = xj.z;
                float fx = 0.f, fy = 0.f, fz = 0.f, ee = 0.f, ev = 0.f;
                unsigned int any = 0u;
                const unsigned int wj = wx >> (8 * jm);
#pragma unroll
                for (int ic = 0; ic < 8; ++ic) {
                    if (!(mb & (1u << ic))) continue;
                    const float4 xi = s_x[oi + 8 * ic];
                    const float dx = xi.x - jx, dy = xi.y - jy, dz = xi.z - jz;
                    const float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                    const bool on = (r2 < P.rc2) && (!masked || ((wj >> ic) & 1u));
                    if (PR && __any_sync(0xffffffffu, r2 < rin2)) nm |= 1u << (8 * jm + ic);
                    /* RF: the pair stays in the outer list when an atom pair
                       is within the pair-list distance (as GENESIS lists pairs) */
                    if (RF && __any_sync(0xffffffffu, r2 < rl2)) no |= 1u << (8 * jm + ic);
                    if (!__any_sync(0xffffffffu, on)) continue;
                    any = 1u;
                    const float2 lj = __ldg(ljj + (unsigned int)s_t[oi + 8 * ic]);
                    float eel = 0.f, evd = 0.f;
                    float gr = NP ? nbc_pair_ana<E, NP, VDW>(r2, xi.w * xj.w, lj, Q, eel, evd)
                                  : nbc_pair<E, VDW == kVdwPlain>(r2, xi.w * xj.w, lj, tab, P, eel, evd);
                    gr = on ? gr : 0.f;
                    if (E) { ee += on ? eel : 0.f; ev += on ? evd : 0.f; }
                    const float wxf = gr * dx, wyf = gr * dy, wzf = gr * dz;
                    fi[ic][0] -= wxf; fi[ic][1] -= wyf; fi[ic][2] -= wzf;
                    fx += wxf; fy += wyf; fz += wzf;
                }
                if (E) { v5[0] += ee; v5[1] += ev; }
                /* a j cluster with no evaluated tile has zero force */
                if (!any) continue;
                const float r = nbc_red3(fx, fy, fz, ti);
                if (ti < 3) {
                    const int sh = __float_as_int(s_jo[4 * gl + jm].w) & 31;
                    virc -= r * s_mv[3 * sh + ti];
                    const int s = s_js[b][js * jm + oj];
                    if (s >= 0) nbc_jadd(force, fcomp + (unsigned int)s, r);
                }
            }
        };
        if (prn) {
            if (ex < 0) jclusters(std::false_type(), std::true_type());
            else jclusters(std::true_type(), std::true_type());
            if (lane == 0) {
                s_nm[2 * gl + w] = nm;
                if (RF) imask_out[2 * (wk.g0 + gl) + w] = no;
            }
        } else {
            if (ex < 0) jclusters(std::false_type(), std::false_type());
            else jclusters(std::true_type(), std::false_type());
        }
        if (kNB) __syncwarp(); else __syncthreads();
    }
    if (kNB && prn) __syncthreads();
    if (prn) nbc_pack(work + blockIdx.x, wk.g0, ng, s_jo, s_ex, s_nm, s_im, imask_in, pk_jo, pk_ex);
    nbc_ifold(fi, s_u.fi, sc, cl_slot, force, pitch, ti, tjl, w);
    nbc_block_acc<E>(acc, v5, virc, s_red);
}

/* The force kernel of this run's pair term: the table (NP = 0) or the analytic form
   with its correction length and LJ form; RF for a part's first pass after a build. */
using nbc_kfn = decltype(&k_nbc_mixed<false, 0, kVdwPsw, false>);
template <bool E, int NP, bool RF>
nbc_kfn nbc_kernel_np(int vdw)
{
    return vdw == kVdwPlain ? k_nbc_mixed<E, NP, kVdwPlain, RF>
         : vdw == kVdwFsw ? (NP ? k_nbc_mixed<E, NP, kVdwFsw, RF> : k_nbc_mixed<E, NP, kVdwPsw, RF>)
                          : k_nbc_mixed<E, NP, kVdwPsw, RF>;
}
template <bool E, bool RF>
nbc_kfn nbc_kernel_e(const struct gcn_nbc *c)
{
    const int vdw = c->plain ? kVdwPlain : c->par.fsw ? kVdwFsw : kVdwPsw;
    return !c->ana ? nbc_kernel_np<E, 0, RF>(vdw)
         : c->par.np == kNpLong ? nbc_kernel_np<E, kNpLong, RF>(vdw)
                                : nbc_kernel_np<E, kNpShort, RF>(vdw);
}
nbc_kfn nbc_kernel(const struct gcn_nbc *c, bool want_energy, bool refine)
{
    return refine ? (want_energy ? nbc_kernel_e<true, true>(c) : nbc_kernel_e<false, true>(c))
                  : (want_energy ? nbc_kernel_e<true, false>(c) : nbc_kernel_e<false, false>(c));
}

unsigned int grid_of(gc_i64 n, int block)
{
    return (unsigned int)std::max<gc_i64>(1, (n + block - 1) / block);
}

/* Arrays that share one capacity: regrown together, with a quarter of headroom, only
   when `need` exceeds it.  `unit` is the bytes one unit takes in that array. */
struct nbc_arr { void **p; size_t unit; };

gc_status regrow(gc_i64 *cap, gc_i64 need, std::initializer_list<nbc_arr> arrs)
{
    if (*cap >= need) return GC_OK;
    const gc_i64 n = need + need / 4 + 16;
    *cap = 0;
    for (const nbc_arr &a : arrs) {
        dev_release(a.p);
        gc_status st = dev_calloc(a.p, n * (gc_i64)a.unit + 64);
        if (st != GC_OK) return st;
    }
    *cap = n;
    return GC_OK;
}

gc_status scan(struct gcn_nbc *c, const int *in, int *out, gc_i64 n, cudaStream_t s)
{
    size_t need = 0;
    if (cub::DeviceScan::ExclusiveSum(0, need, in, out, (int)n, s) != cudaSuccess)
        return GC_E_DEVICE;
    if (need > c->scan_bytes) {
        dev_release(&c->scan_tmp);
        c->scan_bytes = 0;
        gc_status st = dev_calloc(&c->scan_tmp, (gc_i64)need);
        if (st != GC_OK) return st;
        c->scan_bytes = need;
    }
    size_t b = c->scan_bytes;
    return cub::DeviceScan::ExclusiveSum(c->scan_tmp, b, in, out, (int)n, s) == cudaSuccess
               ? GC_OK : GC_E_DEVICE;
}

/* The per-context constants: tables, pinned totals, the unit's overflow
 * flag, the SM count and the inner list's buffer. */
gc_status nbc_create(struct gcn_device *d)
{
    struct gcn_nbc *c = new gcn_nbc();
    std::memset(c, 0, sizeof *c);
    d->nbc = c;
    c->ncls = d->tab.num_atom_cls;
    c->nrow = (int)d->tab.cutoff_int;
    const gc_i64 n1 = c->ncls + 1;
    gc_status st;
    if ((st = dev_calloc((void **)&c->ljf, n1 * n1 * (gc_i64)sizeof(float2))) != GC_OK ||
        (st = dev_calloc((void **)&c->tab, 4LL * c->nrow * (gc_i64)sizeof(float4))) != GC_OK ||
        (st = dev_calloc((void **)&c->move, 81 * (gc_i64)sizeof(double))) != GC_OK ||
        (st = dev_calloc((void **)&c->movef, 81 * (gc_i64)sizeof(float))) != GC_OK ||
        (st = dev_calloc((void **)&c->state, W_NWORDS * (gc_i64)sizeof(unsigned int))) != GC_OK ||
        (st = dev_calloc((void **)&c->tot, T_N * (gc_i64)sizeof(int))) != GC_OK ||
        (st = dev_calloc((void **)&c->xn, (gc_i64)sizeof(int))) != GC_OK)
        return st;
    if (cudaMallocHost((void **)&c->pin, T_N * sizeof(int)) != cudaSuccess) return GC_E_NOMEM;
    if (cudaEventCreateWithFlags(&c->interior_pruned, cudaEventDisableTiming) != cudaSuccess)
        return GC_E_DEVICE;
    int dev = 0;
    if (cudaGetSymbolAddress((void **)&c->overflow, gcn_fx::overflow) != cudaSuccess ||
        cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&c->nsm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
        return GC_E_DEVICE;
    int nb = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_nbc_flags, 128, 0) != cudaSuccess)
        return GC_E_DEVICE;
    c->flags_grid = std::max(1, nb) * c->nsm;
    /* the pair term: analytic when the table is its form, unless the input asks for the
       table (ewald_evaluation = TABLE); an explicit ANALYTIC on another table stops the run */
    std::memset(&c->par, 0, sizeof c->par);
    c->par.rc2 = (float)d->prune_r2;
    c->par.cdf = (float)(d->cutoff2 * d->tab.density);
    c->par.lmax = (float)(c->nrow - 1);
    c->par.erow = c->nrow;
    c->par.ncls1 = c->ncls + 1;
    c->plain = (d->tab.table_form == GC_TABLE_PME_ELEC);
    c->qs = 1.0;
    double coul = 0;
    if (d->tab.ewald_evaluation != GC_EWALD_TABLE) {
        const gc_i64 n3 = (c->plain ? 1LL : 3LL) * c->nrow;
        std::vector<double> tg((size_t)n3), te((size_t)n3);
        if (cudaMemcpy(tg.data(), d->table_grad, n3 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(te.data(), d->table_ene, n3 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
            return GC_E_DEVICE;
        const bool fit = c->plain
            ? nbc_fit_ewald_elec(tg.data(), te.data(), c->nrow, d->tab.density, d->cutoff2, &c->par, &coul)
            : nbc_fit_ewald(tg.data(), te.data(), c->nrow, d->tab.density, d->cutoff2, &c->par, &coul);
        if (fit) {
            c->ana = 1;
            c->qs = std::sqrt(coul);
            c->par.rc2 = (float)d->cutoff2;
        } else if (d->tab.ewald_evaluation == GC_EWALD_ANALYTIC) {
            std::fprintf(stderr, "GPU_Core_Error> ewald_evaluation = ANALYTIC needs PME with "
                         "CHARMM's potential or force switch, or plain LJ; this run's table is not that form\n");
            return GC_E_UNSUPPORTED;
        }
    }
    c->buffer = 0.f;
    k_nbc_tables<<<grid_of(std::max<gc_i64>(n1 * n1, 2LL * c->nrow), 256), 256, 0, d->stream>>>(
        d->nb_mixed, c->ncls, c->nrow, c->ana ? 12.f : 1.f, c->ana ? 6.f : 1.f, c->ljf, c->tab);
    return dev_launched("nbcluster_tables");
}

}  /* anonymous namespace */

bool native_nbc_active(const struct gcn_device *d)
{
    return d && d->tab.nonbond_precision == GC_NONBOND_MIXED &&
           d->tab.table_form != GC_TABLE_CUTOFF_CUBIC && d->nb_mixed;
}

void native_nbc_release(struct gcn_device *d)
{
    struct gcn_nbc *c = d ? d->nbc : 0;
    if (!c) return;
    void **p[] = {
        (void **)&c->ljf, (void **)&c->tab, (void **)&c->move, (void **)&c->movef,
        (void **)&c->cnt_cl, (void **)&c->cnt_sc, (void **)&c->cl_off, (void **)&c->sc_off,
        (void **)&c->cpb, (void **)&c->origin, (void **)&c->cl_slot, (void **)&c->cl_cell,
        (void **)&c->ty, (void **)&c->slot_pos, (void **)&c->bb, (void **)&c->xq,
        (void **)&c->xref0, (void **)&c->xref1, (void **)&c->cv,
        (void **)&c->sci, (void **)&c->ngi,
        (void **)&c->ngb, (void **)&c->gcnt, (void **)&c->goff, (void **)&c->wci,
        (void **)&c->woi, (void **)&c->wcb, (void **)&c->wob, (void **)&c->jent,
        (void **)&c->g_sci, (void **)&c->need, (void **)&c->eidx, (void **)&c->jm8,
        (void **)&c->imask, (void **)&c->imask_in, (void **)&c->imask_out, (void **)&c->pk_jo,
        (void **)&c->pk_ex, (void **)&c->words, (void **)&c->joff,
        (void **)&c->work, (void **)&c->state, (void **)&c->tot, &c->scan_tmp,
        (void **)&c->xl, (void **)&c->xn, (void **)&c->xg, (void **)&c->pcnt,
        (void **)&c->pbase, (void **)&c->pent
    };
    for (unsigned i = 0; i < sizeof(p) / sizeof(p[0]); ++i) dev_release(p[i]);
    if (c->pin) cudaFreeHost(c->pin);
    if (c->interior_pruned) cudaEventDestroy(c->interior_pruned);
    delete c;
    d->nbc = 0;
}

gc_status native_nbc_excl_list(struct gcn_device *d, gc_i64 cap, cudaStream_t s,
                               int4 **list, int **count, gc_i64 *room)
{
    *list = 0; *count = 0; *room = 0;
    if (!native_nbc_active(d)) return GC_OK;
    gc_status st = GC_OK;
    if (!d->nbc && (st = nbc_create(d)) != GC_OK) return st;
    struct gcn_nbc *c = d->nbc;
    if ((st = regrow(&c->cap_xl, std::max<gc_i64>(cap, 1),
                     { { (void **)&c->xl, sizeof(int4) } })) != GC_OK)
        return st;
    if (cudaMemsetAsync(c->xn, 0, sizeof(int), s) != cudaSuccess) return GC_E_DEVICE;
    *list = c->xl; *count = c->xn; *room = c->cap_xl;
    c->listed = 1;
    return GC_OK;
}

/* The clusters of the rebuild's cells (slots, boxes, classes, super-clusters) and the
   MIXED list's pair flags (k_nbc_flags), in place of the rebuild's selection count. */
gc_status native_nbc_flags(gc_context *ctx, gc_i64 pcap, const gc_i64 *np_dev,
                           float lo, float hi, double cmax)
{
    struct gcn_device *d = ctx->native;
    if (!native_nbc_active(d) || d->ncell <= 0) return GC_E_STATE;
    cudaStream_t s = d->stream;
    gc_status st = GC_OK;
    if (!d->nbc && (st = nbc_create(d)) != GC_OK) return st;
    struct gcn_nbc *c = d->nbc;

    {
        double mv[81];
        float mf[81];
        for (int k = 0; k < 27; ++k) {
            const int kk[3] = { k % 3, (k / 3) % 3, k / 9 };
            for (int e = 0; e < 3; ++e) {
                mv[3 * k + e] = (double)(kk[e] - 1) * d->box[e];
                mf[3 * k + e] = (float)mv[3 * k + e];
            }
        }
        if (cudaMemcpyAsync(c->move, mv, sizeof mv, cudaMemcpyHostToDevice, s) != cudaSuccess ||
            cudaMemcpyAsync(c->movef, mf, sizeof mf, cudaMemcpyHostToDevice, s) != cudaSuccess)
            return GC_E_DEVICE;
    }

    const int ncell = (int)d->ncell;
    const gc_i64 bcl = d->num_resident / 8 + ncell + 1;
    const gc_i64 bsci = bcl / 8 + ncell + 1;
    if ((st = regrow(&c->cap_cell, ncell + 1, {
             { (void **)&c->cnt_cl, sizeof(int) }, { (void **)&c->cnt_sc, sizeof(int) },
             { (void **)&c->cl_off, sizeof(int) }, { (void **)&c->sc_off, sizeof(int) },
             { (void **)&c->cpb, sizeof(int2) }, { (void **)&c->origin, sizeof(double4) } })) != GC_OK ||
        (st = regrow(&c->cap_cl, bcl, {
             { (void **)&c->cl_cell, sizeof(int) }, { (void **)&c->cl_slot, 8 * sizeof(int) },
             { (void **)&c->ty, 8 * sizeof(int) }, { (void **)&c->bb, 2 * sizeof(float4) },
             { (void **)&c->xq, 8 * sizeof(float4) }, { (void **)&c->xref0, 8 * sizeof(float4) },
             { (void **)&c->xref1, 8 * sizeof(float4) }, { (void **)&c->cv, 8 * sizeof(float4) } })) != GC_OK ||
        (st = regrow(&c->cap_pitch, d->pitch, { { (void **)&c->slot_pos, sizeof(int) } })) != GC_OK ||
        (st = regrow(&c->cap_sci, bsci, {
             { (void **)&c->sci, sizeof(int4) }, { (void **)&c->ngi, sizeof(int) },
             { (void **)&c->ngb, sizeof(int) }, { (void **)&c->gcnt, sizeof(int) },
             { (void **)&c->goff, sizeof(int) }, { (void **)&c->wci, sizeof(int) },
             { (void **)&c->woi, sizeof(int) }, { (void **)&c->wcb, sizeof(int) },
             { (void **)&c->wob, sizeof(int) }, { (void **)&c->pcnt, sizeof(int) },
             { (void **)&c->pbase, sizeof(int) } })) != GC_OK)
        return st;

    if (cudaMemsetAsync(c->tot, 0, T_N * sizeof(int), s) != cudaSuccess) return GC_E_DEVICE;
    int *err = c->tot + T_ERR;
    k_nbc_count<<<grid_of(ncell + 1, 256), 256, 0, s>>>(d->cell, ncell, c->cnt_cl, c->cnt_sc);
    if ((st = scan(c, c->cnt_cl, c->cl_off, ncell + 1, s)) != GC_OK ||
        (st = scan(c, c->cnt_sc, c->sc_off, ncell + 1, s)) != GC_OK)
        return st;
    k_nbc_form<<<grid_of(ncell, 2), 64, 0, s>>>(d->cell, d->resident_slot, d->force_coord, d->pitch,
                                                ncell, c->cl_off, c->sc_off, c->cl_slot, c->slot_pos,
                                                c->cl_cell, c->origin, c->sci, err);
    const int *ncl_p = c->cl_off + ncell;
    k_nbc_bb<<<grid_of(c->cap_cl, 128), 128, 0, s>>>(c->cl_slot, d->force_coord, d->pitch, ncl_p, c->bb);
    k_nbc_types<<<grid_of(8 * c->cap_cl, 256), 256, 0, s>>>(c->cl_slot, d->cls, ncl_p, c->ncls, c->ty);

    /* the flags (k_nbc_flags): e = 8 u cmax exceeds the FP32 rounding of a box and of a view
       distance, so a farthest box distance within sqrt(lo) / (1 + 8 u) - 2 e puts every view
       distance of the cluster pair within lo, and a nearest one beyond sqrt(hi) + e puts every
       atom pair beyond the rules' reach */
    const double u = std::ldexp(1.0, -24), e = 8.0 * u * cmax;
    const double rpre = std::sqrt((double)hi) + e;
    float rpre2 = (float)(rpre * rpre);
    if ((double)rpre2 < rpre * rpre) rpre2 = std::nextafter(rpre2, 3e38f);
    const double rin = std::sqrt((double)lo) / (1.0 + 8.0 * u) - 2.0 * e;
    float lo_in = rin > 0.0 ? (float)(rin * rin) : 0.f;
    if ((double)lo_in > rin * rin) lo_in = std::nextafter(lo_in, 0.f);
    k_nbc_views<<<grid_of(8 * c->cap_cl, 256), 256, 0, s>>>(c->cl_slot, d->force_coord, d->pitch,
                                                          ncl_p, 0.5 * cmax, c->cv);
    /* one resident wave: blocks waiting for a slot would take their pairs after the others */
    const gc_i64 fw = std::min<gc_i64>(pcap, 4LL * c->flags_grid);
    if (fw > 0)
        k_nbc_flags<<<(unsigned)((fw + 3) / 4), 128, 0, s>>>(
            d->pair, pcap, np_dev, d->cell, c->cl_off, c->bb, c->cl_slot, c->cv, d->force_coord,
            d->pitch, d->pairlistdist2, lo, hi, lo_in, rpre2, (float)(0.5 * cmax));
    c->formed = 1;
    return dev_launched("nbcluster_flags");
}

gc_status native_nbc_build(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;
    if (!native_nbc_active(d) || d->ncell <= 0) return GC_OK;
    cudaStream_t s = d->stream;
    gc_status st = GC_OK;
    if (!d->nbc && (st = nbc_create(d)) != GC_OK) return st;
    struct gcn_nbc *c = d->nbc;
    if (!c->listed || !c->formed) return GC_E_STATE;
    c->listed = 0;
    c->formed = 0;

    const int ncell = (int)d->ncell;
    const gc_i64 bcl = d->num_resident / 8 + ncell + 1;
    const gc_i64 bsci = bcl / 8 + ncell + 1;
    const gc_i64 S = c->cap_sci;

    /* the counts past nsci stay zero, so the scan totals at nsci and at S alike */
    if (cudaMemsetAsync(c->cpb, 0, (size_t)ncell * sizeof(int2), s) != cudaSuccess ||
        cudaMemsetAsync(c->gcnt, 0, (size_t)(S + 1) * sizeof(int), s) != cudaSuccess ||
        cudaMemsetAsync(c->wci, 0, (size_t)(S + 1) * sizeof(int), s) != cudaSuccess ||
        cudaMemsetAsync(c->wcb, 0, (size_t)(S + 1) * sizeof(int), s) != cudaSuccess ||
        cudaMemsetAsync(c->pcnt, 0, (size_t)(S + 1) * sizeof(int), s) != cudaSuccess)
        return GC_E_DEVICE;
    int *err = c->tot + T_ERR;
    if (d->num_pairs > 0)
        k_nbc_pair_range<<<grid_of(d->num_pairs, 256), 256, 0, s>>>(d->pair, d->num_pairs, ncell,
                                                                     c->cpb, err);
    const int *ncl_p = c->cl_off + ncell, *nsci_p = c->sc_off + ncell;
    const int split = ctx->nproc > 1;
    const float rout = (float)std::sqrt(d->pairlistdist2) + 1e-3f;
    k_nbc_search<false><<<grid_of(S, 4), 128, 0, s>>>(
        c->sci, nsci_p, c->cpb, d->pair, c->cl_off, c->bb, c->move, d->layout, split, rout * rout,
        c->ngi, c->ngb, c->gcnt, c->wci, c->wcb, c->pcnt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, err);
    if ((st = scan(c, c->gcnt, c->goff, S + 1, s)) != GC_OK ||
        (st = scan(c, c->pcnt, c->pbase, S + 1, s)) != GC_OK ||
        (st = scan(c, c->wci, c->woi, S + 1, s)) != GC_OK ||
        (st = scan(c, c->wcb, c->wob, S + 1, s)) != GC_OK)
        return st;
    const int *ngrp_p = c->goff + S, *npent_p = c->pbase + S;

    /* entries, groups, exclusions and work items; redone once when the group or word
       capacity was short (the kernels then write nothing) */
    if (c->cap_grp == 0) c->want_grp = 80 * bsci;
    if (c->cap_pent == 0) c->want_pent = d->num_pairs + 1;
    for (int pass = 0; pass < 3; ++pass) {
        if ((st = regrow(&c->cap_grp, c->want_grp, {
                 { (void **)&c->g_sci, sizeof(int) }, { (void **)&c->jent, 4 * sizeof(int) },
                 { (void **)&c->jm8, 4 }, { (void **)&c->imask, sizeof(unsigned int) },
                 { (void **)&c->imask_in, 2 * sizeof(unsigned int) },
                 { (void **)&c->imask_out, 2 * sizeof(unsigned int) },
                 { (void **)&c->pk_jo, 4 * sizeof(float4) }, { (void **)&c->pk_ex, sizeof(int) },
                 { (void **)&c->joff, 4 * sizeof(float4) }, { (void **)&c->need, sizeof(int) },
                 { (void **)&c->eidx, sizeof(int) } })) != GC_OK ||
            (st = regrow(&c->cap_exg, std::max(c->want_exg, c->cap_grp / 16),
                         { { (void **)&c->words, 64 * sizeof(unsigned int) },
                           { (void **)&c->xg, sizeof(int) } })) != GC_OK ||
            (st = regrow(&c->cap_work, 2 * S + c->cap_grp / kGroups + 1,
                         { { (void **)&c->work, sizeof(nbc_work) } })) != GC_OK ||
            (st = regrow(&c->cap_pent, c->want_pent, { { (void **)&c->pent, sizeof(int) } })) != GC_OK)
            return st;
        const gc_i64 G = c->cap_grp;
        if (cudaMemsetAsync(c->need, 0, (size_t)(G + 1) * sizeof(int), s) != cudaSuccess)
            return GC_E_DEVICE;
        k_nbc_search<true><<<grid_of(S, 4), 128, 0, s>>>(
            c->sci, nsci_p, c->cpb, d->pair, c->cl_off, c->bb, c->move, d->layout, split, rout * rout,
            c->ngi, c->ngb, c->gcnt, c->wci, c->wcb, 0, c->goff, c->woi, c->wob, (int)G,
            c->pbase, c->cap_pent, c->jent, c->jm8, c->g_sci, c->work, c->pent, err);
        k_nbc_group<<<grid_of(G, 128), 128, 0, s>>>(ngrp_p, (int)G, c->g_sci, c->sci, c->jent, c->jm8,
                                                     c->cl_cell, c->origin, c->move, c->imask,
                                                     c->joff, c->need);
        k_nbc_xfind<<<grid_of(c->cap_xl, 128), 128, 0, s>>>(
            c->xl, c->xn, c->cap_xl, d->pair, c->slot_pos, c->cl_cell, c->cl_off, c->sc_off,
            c->cpb, c->pbase, npent_p, c->pent, c->cap_pent, c->goff, ngrp_p, (int)G, c->jent,
            c->imask, c->need, err);
        if ((st = scan(c, c->need, c->eidx, G + 1, s)) != GC_OK) return st;
        const int *nexg_p = c->eidx + G;
        k_nbc_eidx<<<grid_of(G, 256), 256, 0, s>>>(ngrp_p, (int)G, nexg_p, (int)c->cap_exg,
                                                    c->need, c->eidx, c->xg);
        k_nbc_winit<<<grid_of(16 * c->cap_exg, 128), 128, 0, s>>>(
            ngrp_p, (int)G, nexg_p, (int)c->cap_exg, c->xg, c->g_sci, c->sci, c->jent, c->words);
        k_nbc_xclear<<<grid_of(c->cap_xl, 256), 256, 0, s>>>(
            c->xl, c->xn, c->cap_xl, d->pair, c->slot_pos, c->cl_off, c->cl_cell, ngrp_p, (int)G,
            nexg_p, (int)c->cap_exg, c->eidx, c->words);
        k_nbc_totals<<<1, 1, 0, s>>>(ncl_p, nsci_p, ngrp_p, c->woi + S, c->wob + S, nexg_p,
                                     npent_p, (int)G, (int)c->cap_exg, c->cap_pent, err, c->tot);
        if (cudaMemcpyAsync(c->pin, c->tot, T_N * sizeof(int), cudaMemcpyDeviceToHost, s) != cudaSuccess)
            return GC_E_DEVICE;
        if ((st = dev_phase(s, "nbcluster_build")) != GC_OK) return st;
        if (c->pin[T_ERR] != 0) {
            std::fprintf(stderr, "GPU_Core_Error> phase=nbcluster_build rank=%d refused=%d "
                         "(a cell over %d atoms, pairs out of cell order, a move off the "
                         "box, or an excluded pair outside the cluster list)\n",
                         (int)ctx->rank, c->pin[T_ERR], kMaxCell);
            return GC_E_UNSUPPORTED;
        }
        c->want_grp = c->pin[T_NGRP];
        c->want_exg = c->pin[T_NEXG];
        c->want_pent = c->pin[T_NPENT];
        if (!c->pin[T_OVF]) break;
        if (pass == 2) return GC_E_CAPACITY;
    }
    c->ncl = c->pin[T_NCL];
    c->nsci = c->pin[T_NSCI];
    c->ngrp = c->pin[T_NGRP];
    c->nexg = c->pin[T_NEXG];
    c->nwi = c->pin[T_NWI];
    c->nw = c->pin[T_NW];
    const unsigned int w[W_NWORDS] = { 0, 0, 0, 0, 1, 1 };
    c->refine[0] = c->refine[1] = 1;
    if (cudaMemcpyAsync(c->state, w, sizeof w, cudaMemcpyHostToDevice, s) != cudaSuccess)
        return GC_E_DEVICE;
    c->built = 1;
    return dev_launched("nbcluster_build");
}

gc_status native_nbc_pack_owned(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;
    struct gcn_nbc *c = d->nbc;
    if (!c || !c->built || c->ncl <= 0) return GC_OK;
    k_nbc_pack<true><<<grid_of(8LL * c->ncl, 256 * kPackItems), 256, 0, d->stream>>>(
        c->cl_slot, c->cl_cell, c->origin, d->force_coord, d->charge, c->qs, d->pitch, c->ncl,
        d->layout, ctx->nproc > 1, 0, c->xq, c->xref0, c->xref1, c->state);
    return dev_launched("nbcluster_pack");
}

/* The inner list's buffer (kPruneBufferPerFs): from the step plan's
 * timestep, within the outer list's skin */
static float prune_buffer(const struct gcn_device *d, const struct gcn_nbc *c)
{
    const double dt_fs = d->plan.dt * kAkmaPs * 1e3;
    float b = dt_fs > 0.0 ? (float)(kPruneBufferPerFs * dt_fs) : kPruneBuffer;
    const double skin = std::sqrt(d->pairlistdist2) - std::sqrt((double)c->par.rc2);
    if (skin > 0.0) b = std::min(b, (float)(kPruneBufferSkin * skin));
    return b > 0.f ? b : kPruneBuffer;
}

gc_status native_nbc_force(gc_context *ctx, gc_i32 want_energy, cudaStream_t s, int part)
{
    struct gcn_device *d = ctx->native;
    struct gcn_nbc *c = d->nbc;
    if (!c || !c->built) return GC_E_STATE;
    if (c->ncl <= 0) return GC_OK;
    const int p = part == GCN_REAL_BOUNDARY ? 1 : 0;
    const int split = ctx->nproc > 1;
    if (part == GCN_REAL_BOUNDARY) {
        if (cudaStreamWaitEvent(s, c->interior_pruned, 0) != cudaSuccess)
            return GC_E_DEVICE;
        k_nbc_pack<false><<<grid_of(8LL * c->ncl, 256 * kPackItems), 256, 0, s>>>(
            c->cl_slot, c->cl_cell, c->origin, d->force_coord, d->charge, c->qs, d->pitch, c->ncl,
            d->layout, split, p, c->xq, c->xref0, c->xref1, c->state);
    }
    if (c->buffer <= 0.f) c->buffer = prune_buffer(d, c);
    k_nbc_decide<<<1, 1, 0, s>>>(c->state, p, c->buffer);
    /* the refine pass follows its build in the same (eager) step; a
       captured plain step replays one instance, so it may not be RF */
    const bool refine = c->refine[p] != 0;
    if (refine && d->graph_mode == GCN_GRAPH_CAPTURE) return GC_E_STATE;
    c->refine[p] = 0;
    const int w0 = part == GCN_REAL_BOUNDARY ? c->nwi : 0;
    const int w1 = part == GCN_REAL_INTERIOR ? c->nwi : c->nw;
    if (3 * d->pitch >= (gc_i64)1 << 32) return GC_E_UNSUPPORTED;   /* nbc_jadd's index */
    if (w1 > w0) {
        const float rin = std::sqrt(c->par.rc2) + c->buffer;
        const float rl = (float)std::sqrt(d->pairlistdist2) + 1e-3f;
        if (part == GCN_REAL_INTERIOR &&
            cudaEventRecord(c->interior_pruned, s) != cudaSuccess)
            return GC_E_DEVICE;
        nbc_kernel(c, want_energy, refine)
            <<<(unsigned)(w1 - w0), 64, 0, s>>>(
            c->work + w0, c->sci, c->xq, c->ty, c->cl_slot, c->joff, c->imask, c->imask_in,
            c->imask_out, c->pk_jo, c->pk_ex, c->state + W_NEED + p, rin * rin, rl * rl, c->eidx,
            c->words, c->ljf, c->tab, c->movef, d->force_real_fx, d->acc_real, d->pitch, c->par);
    } else if (part == GCN_REAL_INTERIOR &&
               cudaEventRecord(c->interior_pruned, s) != cudaSuccess)
        return GC_E_DEVICE;
    return dev_launched("nbcluster_force");
}

const unsigned int *native_nbc_overflow(const struct gcn_device *d)
{
    return d && d->nbc ? d->nbc->overflow : 0;
}

gc_status native_nbc_rebox(gc_context *ctx, const double scale[3])
{
    struct gcn_device *d = ctx->native;
    struct gcn_nbc *c = d->nbc;
    if (!c || !c->built) return GC_OK;
    cudaStream_t s = d->stream;
    const int ncell = (int)d->ncell;
    k_nbc_rebox<<<grid_of(std::max(ncell, 27), 128), 128, 0, s>>>(c->origin, ncell, c->move, c->movef,
                                                                 c->state, scale[0], scale[1], scale[2]);
    if (c->ngrp > 0)
        k_nbc_joff<<<grid_of(c->ngrp, 128), 128, 0, s>>>(c->ngrp, c->g_sci, c->sci, c->cl_cell,
                                                         c->origin, c->move, c->joff);
    return dev_launched("nbcluster_rebox");
}

}  /* namespace gcn */
