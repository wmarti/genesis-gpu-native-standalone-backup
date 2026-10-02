/* gpu_force.cu : the native real-space, bonded, 1-4 and excluded-pair kernels of the
 * device-native core.  A cell pair is a self-contained gcn_pair whose atoms are runs of
 * absolute slot indices, and the exclusion mask is one bit per pair-local (i,j).
 * Expression, operand and accumulation order within a warp, table indices, the prune
 * boundary and signs follow the upstream table kernels.  Nothing here reads the
 * environment. */

#include "gpu_core_native.h"
#include "gpu_nbcluster.h"
#include "gpu_fixed_sum.cuh"

#include <cstring>
#include <cmath>
#include <vector>

namespace {

template <typename T>
__device__ __forceinline__ T warp_sum(T v)
{
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

/* The order-independent adds (gpu_fixed_sum.cuh): force component i of a
 * field (FP64 or FP32), and energy or virial slot `slot` of an accumulator. */
template <typename Real>
__device__ __forceinline__ void force_add(gcn_fx::word *f, gc_i64 i, Real v)
{
    gcn_fx::add1<gcn_fx::kForce1>(f + i, v);
}

__device__ __forceinline__ void slot_add(gcn_fx::word *acc, int slot, double v)
{
    gcn_fx::add<gcn_fx::kEnergy>(acc + 2 * slot, v);
}

/* A block's energy and virial terms: each thread's term is encoded to fixed point
 * (gpu_fixed_sum.cuh), the words are summed as integers over the warp and then the
 * block in shared memory, then one add per word and block.  Integer sums are exact, so
 * the slots do not depend on term order.  Each warp and block folds its lo word's carry
 * into hi, so an add to w[1] stays below 2^32.  Every thread of the block takes part. */
struct ene_t { int slot; double v; };

template <int N>
__device__ __forceinline__ void ene_block(gcn_fx::word *acc, const ene_t (&e)[N])
{
    __shared__ gcn_fx::word s_w[2 * N];
    __shared__ unsigned int s_bad;
    const int t = threadIdx.x;
    if (t < 2 * N) s_w[t] = 0;
    if (t == 0) s_bad = 0;
    __syncthreads();
    for (int k = 0; k < N; ++k) {
        gcn_fx::word hi, lo;
        const bool bad = __any_sync(0xffffffff,
                                    gcn_fx::encode<gcn_fx::kEnergy>(e[k].v, &hi, &lo));
        hi = warp_sum(hi);
        lo = warp_sum(lo);
        if ((t & 31) != 0) continue;
        if (bad) {
            atomicOr(&s_bad, 1u << k);
            continue;
        }
        hi += lo >> 32;
        lo &= 0xffffffffull;
        if (hi != 0) atomicAdd(&s_w[2 * k], hi);
        if (lo != 0) atomicAdd(&s_w[2 * k + 1], lo);
    }
    __syncthreads();
    if (t < N) {
        gcn_fx::word *w = acc + 2 * e[t].slot;
        const gcn_fx::word lo = s_w[2 * t + 1];
        const gcn_fx::word hi = s_w[2 * t] + (lo >> 32);
        if ((s_bad >> t) & 1u) atomicOr(&w[1], 1ull << 63);
        if (hi != 0) atomicAdd(&w[0], hi);
        if ((lo & 0xffffffffull) != 0) atomicAdd(&w[1], lo & 0xffffffffull);
    }
}

/* GENESIS's pbc code 0..26 as a base-3 number, each digit mapped to {-1,0,+1}. */
__device__ __forceinline__ void pbc_decode(int pbc, int *k1, int *k2, int *k3)
{
    int t3 = pbc / 9;
    int r  = pbc - t3 * 9;
    int t2 = r / 3;
    int t1 = r - t2 * 3;
    *k1 = t1 - 1;
    *k2 = t2 - 1;
    *k3 = t3 - 1;
}

/* Scatter one force triple into the component-major accumulator (an FP32
 * triple by the FP32 add: the same words). */
template <typename Real>
__device__ __forceinline__ void scatter3(gcn_fx::word *f, gc_i64 pitch, int a,
                                         Real fx, Real fy, Real fz)
{
    force_add(f, a,             fx);
    force_add(f, pitch + a,     fy);
    force_add(f, 2 * pitch + a, fz);
}

}  /* anonymous namespace */

/* Is cell pair `pr` in launch part `part` (GCN_REAL_*)?  The i cell is always owned
 * (gcn_kern_pair_flag), so the j cell decides: an owned j cell reads no ghost
 * coordinate. */
__device__ __forceinline__ bool gcn_pair_in_part(const struct gcn_layout &L,
                                                 const struct gcn_pair &pr,
                                                 int part)
{
    const bool interior = gcn_box_owned(L, pr.cj);
    return part == gcn::GCN_REAL_ALL ||
           interior == (part == gcn::GCN_REAL_INTERIOR);
}

/* One warp per cell pair: the 32 lanes are 8 i-atom groups by 4 cached j slabs; lane l
 * owns i group l/4 and j slab l&3.  The shuffle distances (4, 8, 16 on the j side; 1, 2
 * on the i side) follow that mapping, and changing them changes the rounding. */
/* The FP32 prefilter of the real-space kernel: before the FP64 work the warp tests its
 * slots in FP32.  d2f > prune_hi proves the FP64 rij2 is at least prune_r2
 * (gcn_sel_bounds brackets the FP32 error), where grad_coef is +0.0, and in a self pair
 * a slot with islot >= jslot has its mask bit clear; a step where every lane is one of
 * those adds nothing, so skipping it is bit-identical.  Compiled for sm_86 and sm_89
 * only, where FP64 issue is scarce. */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890)
#define GCN_RS_PREFILTER 1
#else
#define GCN_RS_PREFILTER 0
#endif

/* The FP32 view of a point for that test: x NaN (every test then undecided) when a
 * component is outside the range the bounds hold for. */
__device__ __forceinline__ float gcn_view(double x, double y, double z, double c1)
{
    return (fabs(x) <= c1 && fabs(y) <= c1 && fabs(z) <= c1)
         ? (float)x : __int_as_float(0x7fc00000);
}

/* cd / r2 for the pair term: rcp.approx refined by two Newton steps, the quotient from
 * it, and one residual correction.  The operands are positive and normal, so no range
 * dispatch is needed. */
__device__ __forceinline__ double gcn_rs_quot(double cd, double r2)
{
    double y;
    asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(y) : "d"(r2));
    double e = fma(-r2, y, 1.0);
    y = fma(y, e, y);
    e = fma(-r2, y, 1.0);
    y = fma(y, e, y);
    const double q = cd * y;
    return fma(y, fma(-r2, q, cd), q);
}

/* The energy and virial partials of an energy step, staged per warp as fixed-point
 * words (gcn_fx::encode) and committed once per block, the words the per-pair adds
 * give.  Slot k's words sit at [2k], [2k+1]; bit k of [GCN_RS_STAGE - 1] marks an
 * unrepresentable term. */
#define GCN_RS_STAGE    (2 * GCN_RE_NSLOT + 1)

__device__ __forceinline__ void stage_add(gcn_fx::word *stage, int slot, double v)
{
    gcn_fx::word hi, lo;
    if (gcn_fx::encode<gcn_fx::kEnergy>(v, &hi, &lo))
        stage[GCN_RS_STAGE - 1] |= 1ull << slot;
    stage[2 * slot]     += hi;
    stage[2 * slot + 1] += lo;
}

/* The block's commit of its four warps' staged words after a block barrier;
 * `stride` is the words per warp. */
__device__ __forceinline__ void stage_commit(const gcn_fx::word *s, int stride,
                                             gcn_fx::word *acc)
{
    const int lane = threadIdx.x;
    if (threadIdx.y != 0 || lane >= 2 * GCN_RE_NSLOT) return;
    gcn_fx::word sum = 0, bad = 0;
    for (int k = 0; k < GCN_RS_WARPS; ++k) {
        sum += s[k * stride + lane];
        bad |= s[k * stride + GCN_RS_STAGE - 1];
    }
    if (sum != 0) atomicAdd(acc + lane, sum);
    if ((lane & 1) && ((bad >> (lane >> 1)) & 1ull))
        atomicOr(acc + lane, 1ull << 63);
}

/* The pieces both pair walks share (gcn_kern_real_space, gcn_kern_real_space_cutoff).
 * A j tile: its force sums cleared, its coordinates, charges, classes and slots cached,
 * and with the prefilter their FP32 views. */
template <bool prefilter>
__device__ __forceinline__ void rs_load_j(const gc_f64 *coord,
                                          const gc_f64 *charge,
                                          const gc_i32 *cls,
                                          const gc_i32 *sel,
                                          int off, gc_i64 pitch, int n_iy, double c1,
                                          int lane, double *fj, double *jc, int *jcls,
                                          int *jslot, float *jf)
{
    for (int k = lane; k < 3 * n_iy; k += 32) fj[k] = 0.0;
    for (int k = lane; k < n_iy; k += 32) {
        int s = sel[off + k];
        jc[k]                    = coord[s];
        jc[GCN_RS_JBLOCK + k]    = coord[pitch + s];
        jc[2 * GCN_RS_JBLOCK + k]= coord[2 * pitch + s];
        jc[3 * GCN_RS_JBLOCK + k]= charge[s];
        jcls[k]                  = cls[s];
        jslot[k]                 = s;
        if (prefilter) {
            jf[k] = gcn_view(jc[k], jc[GCN_RS_JBLOCK + k],
                             jc[2 * GCN_RS_JBLOCK + k], c1);
            jf[GCN_RS_JBLOCK + k]     = (float)jc[GCN_RS_JBLOCK + k];
            jf[2 * GCN_RS_JBLOCK + k] = (float)jc[2 * GCN_RS_JBLOCK + k];
        }
    }
}

/* Sums over the i groups (lane bits 2..4) and over an i row's j slabs (bits 0..1). */
__device__ __forceinline__ double rs_sum_i(double v)
{
    v += __shfl_xor_sync(0xffffffff, v,  4, 32);
    v += __shfl_xor_sync(0xffffffff, v,  8, 32);
    v += __shfl_xor_sync(0xffffffff, v, 16, 32);
    return v;
}

__device__ __forceinline__ double rs_sum_j(double v)
{
    v += __shfl_xor_sync(0xffffffff, v, 1, 32);
    v += __shfl_xor_sync(0xffffffff, v, 2, 32);
    return v;
}

/* The j tile's force sums into the field. */
__device__ __forceinline__ void rs_flush_j(gcn_fx::word *force, gc_i64 pitch,
                                           const double *fj, const int *jslot, int n_iy,
                                           int lane)
{
    for (int k = lane; k < 3 * n_iy; k += 32) {
        double val = fj[k];
        if (val != 0.0) {
            int c = k % 3;
            int j = k / 3;
            force_add(force, (gc_i64)c * pitch + jslot[j], val);
        }
    }
}

/* One cell pair, `w` in execution order, on one warp, with the warp's own staging
 * arrays (gcn_kern_real_space).  plain_lj: GC_TABLE_PME_ELEC (vdw = CUTOFF, which AMBER
 * forces): the table holds the electrostatic column only and the LJ pair is the plain
 * r^-12/r^-6 inside the cutoff (rij2 < cutoff2 is prune_r2 here). */
template <bool want_energy, bool plain_lj>
__device__ __forceinline__ void
gcn_rs_pair(const gc_f64 *__restrict__ coord,
            const gc_f64 *__restrict__ charge,
            const gc_i32 *__restrict__ cls,
            gcn_fx::word *__restrict__ force,
            gcn_fx::word *__restrict__ acc,
            const struct gcn_pair *__restrict__ pairs,
            const gc_i32 *__restrict__ order,
            const gc_i32 *__restrict__ sel,
            const gc_u64 *__restrict__ mask,
            const gc_f64 *__restrict__ lj12,
            const gc_f64 *__restrict__ lj6,
            const gc_f64 *__restrict__ table_grad,
            const gc_f64 *__restrict__ table_ene,
            gc_i64 pitch, int ncls,
            double density, double cutoff2, double prune_r2,
            float prune_hi, double c1,
            const struct gcn_layout &L, int part, gc_i64 w,
            double *fj, double *jc, int *jcls, int *jslot, float *jf,
            gcn_fx::word *stage)
{
    const int lane    = threadIdx.x;
    const int id_xx   = lane / GCN_RS_IY;
    const int id_xy   = lane & (GCN_RS_IY - 1);

    const struct gcn_pair pr = pairs[order[w]];
    const int ix_n = pr.ix_n;
    const int iy_n = pr.iy_n;
    if (ix_n * iy_n <= 0 || !gcn_pair_in_part(L, pr, part)) return;

    const double move_x = pr.move[0];
    const double move_y = pr.move[1];
    const double move_z = pr.move[2];
    const bool   masked = (pr.mask_off >= 0);

    double vsum = 0.0;          /* this lane's virial component, id_xy < 3 */
    double eelec = 0.0, evdw = 0.0;
    const bool want_virial = (pr.virial != 0);

    for (int iiy_s = 0; iiy_s < iy_n; iiy_s += GCN_RS_JBLOCK) {

        int n_iy = iy_n - iiy_s;
        if (n_iy > GCN_RS_JBLOCK) n_iy = GCN_RS_JBLOCK;

        rs_load_j<GCN_RS_PREFILTER>(coord, charge, cls, sel, pr.iy_off + iiy_s, pitch,
                                    n_iy, c1, lane, fj, jc, jcls, jslot, jf);
        __syncwarp();

        const int iix_nit = ((ix_n + GCN_RS_IX - 1) / GCN_RS_IX) * GCN_RS_IX;
        const int iiy_nit = ((n_iy + GCN_RS_IY - 1) / GCN_RS_IY) * GCN_RS_IY;

        for (int iix = id_xx; iix < iix_nit; iix += GCN_RS_IX) {

            int    islot = 0, iatmcls = 0;
            double rtmp1 = 0.0, rtmp2 = 0.0, rtmp3 = 0.0, iqtmp = 0.0;

            if (iix < ix_n) {
                islot   = sel[pr.ix_off + iix];
                iatmcls = cls[islot];
                rtmp1   = coord[islot]             + move_x;
                rtmp2   = coord[pitch + islot]     + move_y;
                rtmp3   = coord[2 * pitch + islot] + move_z;
                iqtmp   = charge[islot];
            }

            const float if1 = gcn_view(rtmp1, rtmp2, rtmp3, c1);
            const float if2 = (float)rtmp2, if3 = (float)rtmp3;
            double fl0 = 0.0, fl1 = 0.0, fl2 = 0.0;
            const gc_i64 bit0 = masked
                ? pr.mask_off + (gc_i64)iix * iy_n + iiy_s : (gc_i64)0;

            for (int iiy = id_xy; iiy < iiy_nit; iiy += GCN_RS_IY) {

                if (GCN_RS_PREFILTER) {
                    bool near = false;
                    if (iix < ix_n && iiy < n_iy &&
                        (!pr.self_pair || islot < jslot[iiy])) {
                        const float ex = if1 - jf[iiy];
                        const float ey = if2 - jf[GCN_RS_JBLOCK + iiy];
                        const float ez = if3 - jf[2 * GCN_RS_JBLOCK + iiy];
                        near = !(ex * ex + ey * ey + ez * ez > prune_hi);
                    }
                    if (!__any_sync(0xffffffff, near)) continue;
                }

                double grad_coef = 0.0;
                double dij1 = 0.0, dij2 = 0.0, dij3 = 0.0, rij2 = 0.0;

                if (iix < ix_n && iiy < n_iy) {

                    dij1 = rtmp1 - jc[iiy];
                    dij2 = rtmp2 - jc[GCN_RS_JBLOCK + iiy];
                    dij3 = rtmp3 - jc[2 * GCN_RS_JBLOCK + iiy];
                    rij2 = dij1 * dij1 + dij2 * dij2 + dij3 * dij3;

                    /* the mask read is deferred past the distance test: a masked pair keeps grad_coef at
                     * +0.0 either way, and most slots are outside prune_r2 */
                    bool active = (rij2 < prune_r2);
                    if (active && masked) {
                        gc_i64 b = bit0 + iiy;
                        active = ((mask[b >> 6] >> (b & 63)) & 1ull) != 0ull;
                    }

                    if (active && plain_lj) {
                        double rij2_inv = 1.0 / rij2;
                        rij2 = cutoff2 * density * rij2_inv;

                        double jqtmp  = jc[3 * GCN_RS_JBLOCK + iiy];
                        int    jatmcls = jcls[iiy];
                        double c12 = lj12[(jatmcls - 1) + ncls * (iatmcls - 1)];
                        double c6  = lj6 [(jatmcls - 1) + ncls * (iatmcls - 1)];

                        int    L  = (int)rij2;
                        double R  = rij2 - (double)L;

                        double term_lj6  = rij2_inv * rij2_inv * rij2_inv;
                        double term_lj12 = term_lj6 * term_lj6;
                        if (want_energy) {
                            double te0 = table_ene[L - 1];
                            double te1 = table_ene[L];
                            evdw  += term_lj12 * c12 - term_lj6 * c6;
                            eelec += iqtmp * jqtmp * (te0 + R * (te1 - te0));
                        }
                        double tg0 = table_grad[L - 1];
                        double tg1 = table_grad[L];
                        double term_elec = tg0 + R * (tg1 - tg0);
                        term_lj12 = -12.0 * term_lj12 * rij2_inv;
                        term_lj6  = -6.0 * term_lj6 * rij2_inv;
                        grad_coef = term_lj12 * c12 - term_lj6 * c6
                                  + iqtmp * jqtmp * term_elec;
                    } else if (active) {
                        rij2 = gcn_rs_quot(cutoff2 * density, rij2);

                        double jqtmp  = jc[3 * GCN_RS_JBLOCK + iiy];
                        int    jatmcls = jcls[iiy];
                        double c12 = lj12[(jatmcls - 1) + ncls * (iatmcls - 1)];
                        double c6  = lj6 [(jatmcls - 1) + ncls * (iatmcls - 1)];

                        int    L  = (int)rij2;
                        double R  = rij2 - (double)L;
                        int    L1 = 3 * L - 3;

                        double tg0 = table_grad[L1];
                        double tg1 = table_grad[L1 + 1];
                        double tg2 = table_grad[L1 + 2];
                        double tg3 = table_grad[L1 + 3];
                        double tg4 = table_grad[L1 + 4];
                        double tg5 = table_grad[L1 + 5];
                        double term_lj12 = tg0 + R * (tg3 - tg0);
                        double term_lj6  = tg1 + R * (tg4 - tg1);
                        double term_elec = tg2 + R * (tg5 - tg2);

                        grad_coef = term_lj12 * c12 - term_lj6 * c6
                                  + iqtmp * jqtmp * term_elec;

                        if (want_energy) {
                            double te0 = table_ene[L1];
                            double te1 = table_ene[L1 + 1];
                            double te2 = table_ene[L1 + 2];
                            double te3 = table_ene[L1 + 3];
                            double te4 = table_ene[L1 + 4];
                            double te5 = table_ene[L1 + 5];
                            double e12 = te0 + R * (te3 - te0);
                            double e6  = te1 + R * (te4 - te1);
                            double eel = te2 + R * (te5 - te2);
                            evdw  += e12 * c12 - e6 * c6;
                            eelec += iqtmp * jqtmp * eel;
                        }
                    }
                }

                if (__any_sync(0xffffffff, grad_coef != 0.0)) {
                    double work1 = grad_coef * dij1;
                    double work2 = grad_coef * dij2;
                    double work3 = grad_coef * dij3;
                    fl0 -= work1;
                    fl1 -= work2;
                    fl2 -= work3;
                    work1 = rs_sum_i(work1);
                    work2 = rs_sum_i(work2);
                    work3 = rs_sum_i(work3);
                    if (id_xx == 0 && iix < ix_n && iiy < n_iy) {
                        fj[3 * iiy]     += work1;
                        fj[3 * iiy + 1] += work2;
                        fj[3 * iiy + 2] += work3;
                    }
                }
            }

            fl0 = rs_sum_j(fl0);
            fl1 = rs_sum_j(fl1);
            fl2 = rs_sum_j(fl2);

            /* all four lanes of an i row hold its totals: lane c adds component c, so the three
             * adds run side by side; the same lane keeps that component's periodic-virial sum, so
             * the walk carries one virial register */
            if (id_xy < 3 && iix < ix_n) {
                const double fl = id_xy == 0 ? fl0 : (id_xy == 1 ? fl1 : fl2);
                if (fl != 0.0) force_add(force, id_xy * pitch + islot, fl);
                if (want_virial) vsum += fl;
            }
        }

        /* the shared j accumulator was written by the id_xx == 0 lanes and is read by a
         * different distribution of lanes; shuffles are not memory barriers, and a block
         * barrier would not do: whole cell-pair warps return early */
        __syncwarp();

        rs_flush_j(force, pitch, fj, jslot, n_iy, lane);
        __syncwarp();
    }

    /* the periodic-offset virial correction; with the sum over owned slots of coord*force
     * (gcn_kern_coord_force_virial) it is the full pair virial */
    if (want_virial) {
        vsum = rs_sum_i(vsum);
        const double sumval[3] = { vsum, __shfl_sync(0xffffffff, vsum, 1, 32),
                                   __shfl_sync(0xffffffff, vsum, 2, 32) };
        if (lane == 0 && want_energy) {
            stage_add(stage, GCN_RE_VIRX, sumval[0] * move_x);
            stage_add(stage, GCN_RE_VIRY, sumval[1] * move_y);
            stage_add(stage, GCN_RE_VIRZ, sumval[2] * move_z);
        } else if (lane == 0) {
            slot_add(acc, GCN_RE_VIRX, sumval[0] * move_x);
            slot_add(acc, GCN_RE_VIRY, sumval[1] * move_y);
            slot_add(acc, GCN_RE_VIRZ, sumval[2] * move_z);
        }
    }

    if (want_energy) {
        eelec = warp_sum(eelec);
        evdw  = warp_sum(evdw);
        if (lane == 0) {
            if (eelec != 0.0) stage_add(stage, GCN_RE_ELEC, eelec);
            if (evdw  != 0.0) stage_add(stage, GCN_RE_VDW,  evdw);
        }
    }
}

/* A warp runs GCN_RS_PAIRS(want_energy) cell pairs of the execution order one after
 * another: four on an energy step (each block's energy words then cover sixteen pairs),
 * one on a force-only step, where there is nothing to stage. */
#define GCN_RS_PAIRS(e) ((e) ? 4 : 1)

/* Resident blocks per SM the walk is compiled for: 7 (72 registers) on sm_90 and later,
 * 6 elsewhere. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define GCN_RS_MINB 7
#else
#define GCN_RS_MINB 6
#endif

template <bool want_energy, bool plain_lj>
__global__ void __launch_bounds__(GCN_RS_BLOCK, GCN_RS_MINB)
gcn_kern_real_space(const gc_f64 *__restrict__ coord,
                    const gc_f64 *__restrict__ charge,
                    const gc_i32 *__restrict__ cls,
                    gcn_fx::word *__restrict__ force,
                    gcn_fx::word *__restrict__ acc,
                    const struct gcn_pair *__restrict__ pairs,
                    const gc_i32 *__restrict__ order,
                    const gc_i32 *__restrict__ sel,
                    const gc_u64 *__restrict__ mask,
                    const gc_f64 *__restrict__ lj12,
                    const gc_f64 *__restrict__ lj6,
                    const gc_f64 *__restrict__ table_grad,
                    const gc_f64 *__restrict__ table_ene,
                    gc_i64 npair, gc_i64 pitch, int ncls,
                    double density, double cutoff2, double prune_r2,
                    float prune_hi, double c1,
                    struct gcn_layout L, int part)
{
    __shared__ double s_fj[GCN_RS_WARPS][3 * GCN_RS_JBLOCK];
    __shared__ double s_jc[GCN_RS_WARPS][4 * GCN_RS_JBLOCK];
    __shared__ int    s_jcls[GCN_RS_WARPS][GCN_RS_JBLOCK];
    __shared__ int    s_jslot[GCN_RS_WARPS][GCN_RS_JBLOCK];
    __shared__ float  s_jf[GCN_RS_WARPS][GCN_RS_PREFILTER ? 3 * GCN_RS_JBLOCK : 1];
    __shared__ gcn_fx::word s_stage[GCN_RS_WARPS][want_energy ? GCN_RS_STAGE : 1];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    if (want_energy) {
        if (lane < GCN_RS_STAGE) s_stage[warp][lane] = 0;
        __syncwarp();
    }

    for (int t = 0; t < GCN_RS_PAIRS(want_energy); ++t) {
        const gc_i64 w = ((gc_i64)blockIdx.x * GCN_RS_PAIRS(want_energy) + t)
                       * GCN_RS_WARPS + warp;
        if (w >= npair) break;
        gcn_rs_pair<want_energy, plain_lj>(coord, charge, cls, force, acc, pairs, order,
                                 sel, mask, lj12, lj6, table_grad, table_ene,
                                 pitch, ncls, density, cutoff2, prune_r2,
                                 prune_hi, c1, L, part, w,
                                 s_fj[warp], s_jc[warp], s_jcls[warp],
                                 s_jslot[warp], s_jf[warp], s_stage[warp]);
    }

    if (want_energy) {
        __syncthreads();
        stage_commit(&s_stage[0][0], sizeof(s_stage[0]) / sizeof(gcn_fx::word), acc);
    }
}

/* The real-space kernel for electrostatic = CUTOFF: GENESIS's CPU cutoff pair
 * arithmetic (sp_energy_table_cubic.fpp, compute_energy_nonbond_table) on the pair walk
 * of gcn_kern_real_space, reading the cubic Hermite table built by
 * setup_table_general_cutoff_cubic, which carries the switching, shifting or force
 * switching. */
template <bool want_energy>
__global__ void __launch_bounds__(GCN_RS_BLOCK, 6)
gcn_kern_real_space_cutoff(const gc_f64 *__restrict__ coord,
                    const gc_f64 *__restrict__ charge,
                    const gc_i32 *__restrict__ cls,
                    gcn_fx::word *__restrict__ force,
                    gcn_fx::word *__restrict__ acc,
                    const struct gcn_pair *__restrict__ pairs,
                    const gc_i32 *__restrict__ order,
                    const gc_i32 *__restrict__ sel,
                    const gc_u64 *__restrict__ mask,
                    const gc_f64 *__restrict__ lj12,
                    const gc_f64 *__restrict__ lj6,
                    const gc_f64 *__restrict__ table_grad,
                    const gc_f64 *__restrict__ table_ene,
                    gc_i64 npair, gc_i64 pitch, int ncls,
                    double density, double cubic_lim, double cutoff2,
                    struct gcn_layout L, int part)
{
    __shared__ double s_fj[GCN_RS_WARPS][3 * GCN_RS_JBLOCK];
    __shared__ double s_jc[GCN_RS_WARPS][4 * GCN_RS_JBLOCK];
    __shared__ int    s_jcls[GCN_RS_WARPS][GCN_RS_JBLOCK];
    __shared__ int    s_jslot[GCN_RS_WARPS][GCN_RS_JBLOCK];

    const int lane    = threadIdx.x;
    const int warp    = threadIdx.y;
    const int id_xx   = lane / GCN_RS_IY;
    const int id_xy   = lane & (GCN_RS_IY - 1);

    const gc_i64 w = (gc_i64)blockIdx.x * GCN_RS_WARPS + warp;
    if (w >= npair) return;

    const struct gcn_pair pr = pairs[order[w]];
    const int ix_n = pr.ix_n;
    const int iy_n = pr.iy_n;
    if (ix_n * iy_n <= 0 || !gcn_pair_in_part(L, pr, part)) return;

    double *fj    = s_fj[warp];
    double *jc    = s_jc[warp];
    int    *jcls  = s_jcls[warp];
    int    *jslot = s_jslot[warp];

    const double move_x = pr.move[0];
    const double move_y = pr.move[1];
    const double move_z = pr.move[2];
    const bool   masked = (pr.mask_off >= 0);

    double sumval[3] = { 0.0, 0.0, 0.0 };
    double eelec = 0.0, evdw = 0.0;
    const bool want_virial = (pr.virial != 0);

    for (int iiy_s = 0; iiy_s < iy_n; iiy_s += GCN_RS_JBLOCK) {

        int n_iy = iy_n - iiy_s;
        if (n_iy > GCN_RS_JBLOCK) n_iy = GCN_RS_JBLOCK;

        rs_load_j<false>(coord, charge, cls, sel, pr.iy_off + iiy_s, pitch, n_iy, 0.0,
                         lane, fj, jc, jcls, jslot, 0);
        __syncwarp();

        const int iix_nit = ((ix_n + GCN_RS_IX - 1) / GCN_RS_IX) * GCN_RS_IX;
        const int iiy_nit = ((n_iy + GCN_RS_IY - 1) / GCN_RS_IY) * GCN_RS_IY;

        for (int iix = id_xx; iix < iix_nit; iix += GCN_RS_IX) {

            int    islot = 0, iatmcls = 0;
            double rtmp1 = 0.0, rtmp2 = 0.0, rtmp3 = 0.0, iqtmp = 0.0;

            if (iix < ix_n) {
                islot   = sel[pr.ix_off + iix];
                iatmcls = cls[islot];
                rtmp1   = coord[islot]             + move_x;
                rtmp2   = coord[pitch + islot]     + move_y;
                rtmp3   = coord[2 * pitch + islot] + move_z;
                iqtmp   = charge[islot];
            }

            double fl0 = 0.0, fl1 = 0.0, fl2 = 0.0;
            const gc_i64 bit0 = masked
                ? pr.mask_off + (gc_i64)iix * iy_n + iiy_s : (gc_i64)0;

            for (int iiy = id_xy; iiy < iiy_nit; iiy += GCN_RS_IY) {

                double grad_coef = 0.0;
                double dij1 = 0.0, dij2 = 0.0, dij3 = 0.0, rij2 = 0.0;

                if (iix < ix_n && iiy < n_iy) {

                    dij1 = rtmp1 - jc[iiy];
                    dij2 = rtmp2 - jc[GCN_RS_JBLOCK + iiy];
                    dij3 = rtmp3 - jc[2 * GCN_RS_JBLOCK + iiy];
                    rij2 = dij1 * dij1 + dij2 * dij2 + dij3 * dij3;

                    /* the CPU kernel's table coordinate: a pair is evaluated while L = int(density*r2)
                     * still reaches a non-zero node (L or L+1); beyond that every Hermite sum is zero */
                    double rij2d = density * rij2;
                    bool active = (rij2d < cubic_lim);
                    /* A force-only step is stock's compute_force_nonbond_table, whose within-cell loop
                     * clamps rij2 = min(cutoff2, rij2) before the lookup (its between-cell loop and the
                     * energy routine do not).  A clamped pair reads the table at the cutoff, where
                     * native_attach_tables has checked every gradient column is zero, so it is skipped;
                     * unclamped, a pair just past the cutoff gets the Hermite slope term of the cutoff
                     * node, as in the energy routine. */
                    if (!want_energy && pr.self_pair && rij2 > cutoff2)
                        active = false;
                    if (active && masked) {
                        gc_i64 b = bit0 + iiy;
                        active = ((mask[b >> 6] >> (b & 63)) & 1ull) != 0ull;
                    }

                    if (active) {
                        /* sp_energy_table_cubic.fpp (compute_energy_nonbond_table, within a cell and between cells), in
                         * the CPU's operand order */
                        double jqtmp  = jc[3 * GCN_RS_JBLOCK + iiy];
                        int    jatmcls = jcls[iiy];
                        double c6  = lj6 [(jatmcls - 1) + ncls * (iatmcls - 1)];
                        double c12 = lj12[(jatmcls - 1) + ncls * (iatmcls - 1)];

                        int    L   = (int)rij2d;
                        double R   = rij2d - (double)L;
                        double h00 = (1.0 + 2.0 * R) * (1.0 - R) * (1.0 - R);
                        double h10 = R * (1.0 - R) * (1.0 - R);
                        double h01 = R * R * (3.0 - 2.0 * R);
                        double h11 = R * R * (R - 1.0);
                        int    L1  = 6 * L - 6;     /* Fortran 6*L-5 */

                        if (want_energy) {
                            double e12 = table_ene[L1]     * h00 + table_ene[L1 + 1]  * h10;
                            double e6  = table_ene[L1 + 2] * h00 + table_ene[L1 + 3]  * h10;
                            double eel = table_ene[L1 + 4] * h00 + table_ene[L1 + 5]  * h10;
                            e12 = e12 + table_ene[L1 + 6]  * h01 + table_ene[L1 + 7]  * h11;
                            e6  = e6  + table_ene[L1 + 8]  * h01 + table_ene[L1 + 9]  * h11;
                            eel = eel + table_ene[L1 + 10] * h01 + table_ene[L1 + 11] * h11;
                            evdw  += e12 * c12 - e6 * c6;
                            eelec += iqtmp * jqtmp * eel;
                        }

                        double g12 = table_grad[L1]     * h00 + table_grad[L1 + 1]  * h10;
                        double g6  = table_grad[L1 + 2] * h00 + table_grad[L1 + 3]  * h10;
                        double gel = table_grad[L1 + 4] * h00 + table_grad[L1 + 5]  * h10;
                        g12 = g12 + table_grad[L1 + 6]  * h01 + table_grad[L1 + 7]  * h11;
                        g6  = g6  + table_grad[L1 + 8]  * h01 + table_grad[L1 + 9]  * h11;
                        gel = gel + table_grad[L1 + 10] * h01 + table_grad[L1 + 11] * h11;
                        grad_coef = g12 * c12 - g6 * c6 + iqtmp * jqtmp * gel;
                    }
                }

                if (__any_sync(0xffffffff, grad_coef != 0.0)) {
                    double work1 = grad_coef * dij1;
                    double work2 = grad_coef * dij2;
                    double work3 = grad_coef * dij3;
                    fl0 -= work1;
                    fl1 -= work2;
                    fl2 -= work3;
                    work1 = rs_sum_i(work1);
                    work2 = rs_sum_i(work2);
                    work3 = rs_sum_i(work3);
                    if (id_xx == 0 && iix < ix_n && iiy < n_iy) {
                        fj[3 * iiy]     += work1;
                        fj[3 * iiy + 1] += work2;
                        fj[3 * iiy + 2] += work3;
                    }
                }
            }

            if (want_virial) {
                sumval[0] += fl0;
                sumval[1] += fl1;
                sumval[2] += fl2;
            }

            fl0 = rs_sum_j(fl0);
            fl1 = rs_sum_j(fl1);
            fl2 = rs_sum_j(fl2);

            if (id_xy < 3 && iix < ix_n) {
                const double fl = id_xy == 0 ? fl0 : (id_xy == 1 ? fl1 : fl2);
                if (fl != 0.0) force_add(force, id_xy * pitch + islot, fl);
            }
        }

        __syncwarp();

        rs_flush_j(force, pitch, fj, jslot, n_iy, lane);
        __syncwarp();
    }

    if (want_virial) {
        for (int ii = 0; ii < 3; ++ii) {
            sumval[ii] += __shfl_xor_sync(0xffffffff, sumval[ii],  1, 32);
            sumval[ii] += __shfl_xor_sync(0xffffffff, sumval[ii],  2, 32);
            sumval[ii] += __shfl_xor_sync(0xffffffff, sumval[ii],  4, 32);
            sumval[ii] += __shfl_xor_sync(0xffffffff, sumval[ii],  8, 32);
            sumval[ii] += __shfl_xor_sync(0xffffffff, sumval[ii], 16, 32);
        }
        if (lane == 0) {
            slot_add(acc, GCN_RE_VIRX, sumval[0] * move_x);
            slot_add(acc, GCN_RE_VIRY, sumval[1] * move_y);
            slot_add(acc, GCN_RE_VIRZ, sumval[2] * move_z);
        }
    }

    if (want_energy) {
        eelec = warp_sum(eelec);
        evdw  = warp_sum(evdw);
        if (lane == 0) {
            if (eelec != 0.0) slot_add(acc, GCN_RE_ELEC, eelec);
            if (evdw  != 0.0) slot_add(acc, GCN_RE_VDW,  evdw);
        }
    }
}

/* The second half of the real-space virial: the sum over owned slots of coord * force
 * over the real-space accumulator only (the bonded and reciprocal terms report their
 * own full pair virial). */
__global__ void gcn_kern_coord_force_virial(const gc_f64 *__restrict__ coord,
                                            const gc_f64 *__restrict__ force,
                                            gc_i64 n, gc_i64 pitch,
                                            gc_f64 *__restrict__ part,
                                            int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double k[3] = { 0.0, 0.0, 0.0 };

    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        k[0] += coord[s]                 * force[s];
        k[1] += coord[pitch + s]         * force[pitch + s];
        k[2] += coord[2 * pitch + s]     * force[2 * pitch + s];
    }
    block_reduce_sum(k, 3, sh, part, nblk);
}

namespace {

/* calculate_dihedral_2 of sp_energy_dihedrals.fpp.  The cross-product operand order and
 * the single sqrt over the product of reciprocals are part of the rounding contract
 * shared by the dihedral, improper and CMAP terms. */
template <typename Real>
__device__ __forceinline__ void dihedral_2(
    const double cx[4], const double cy[4], const double cz[4],
    const int pbc[3], double sx, double sy, double sz,
    Real *cos_dih, Real *sin_dih, Real grad[9], Real v[3])
{
    int k1, k2, k3;
    Real dij[3], djk[3], dlk[3], aijk[3], ajkl[3];

    pbc_decode(pbc[0], &k1, &k2, &k3);
    dij[0] = (Real)(cx[0] - cx[1] + sx * (double)k1);
    dij[1] = (Real)(cy[0] - cy[1] + sy * (double)k2);
    dij[2] = (Real)(cz[0] - cz[1] + sz * (double)k3);

    pbc_decode(pbc[1], &k1, &k2, &k3);
    djk[0] = (Real)(cx[1] - cx[2] + sx * (double)k1);
    djk[1] = (Real)(cy[1] - cy[2] + sy * (double)k2);
    djk[2] = (Real)(cz[1] - cz[2] + sz * (double)k3);

    pbc_decode(pbc[2], &k1, &k2, &k3);
    dlk[0] = (Real)(cx[3] - cx[2] + sx * (double)k1);
    dlk[1] = (Real)(cy[3] - cy[2] + sy * (double)k2);
    dlk[2] = (Real)(cz[3] - cz[2] + sz * (double)k3);

    aijk[0] = dij[1] * djk[2] - dij[2] * djk[1];
    aijk[1] = dij[2] * djk[0] - dij[0] * djk[2];
    aijk[2] = dij[0] * djk[1] - dij[1] * djk[0];

    ajkl[0] = dlk[1] * djk[2] - dlk[2] * djk[1];
    ajkl[1] = dlk[2] * djk[0] - dlk[0] * djk[2];
    ajkl[2] = dlk[0] * djk[1] - dlk[1] * djk[0];

    Real raijk2 = aijk[0]*aijk[0] + aijk[1]*aijk[1] + aijk[2]*aijk[2];
    Real rajkl2 = ajkl[0]*ajkl[0] + ajkl[1]*ajkl[1] + ajkl[2]*ajkl[2];

    Real inv_raijk2 = Real(1.0) / raijk2;
    Real inv_rajkl2 = Real(1.0) / rajkl2;
    Real inv_raijkl = sqrt(inv_raijk2 * inv_rajkl2);

    *cos_dih = (aijk[0]*ajkl[0] + aijk[1]*ajkl[1] + aijk[2]*ajkl[2])
             * inv_raijkl;

    Real rjk = sqrt(djk[0]*djk[0] + djk[1]*djk[1] + djk[2]*djk[2]);
    Real t1  = aijk[0]*dlk[0] + aijk[1]*dlk[1] + aijk[2]*dlk[2];
    *sin_dih = t1 * rjk * inv_raijkl;

    Real inv_rjk  = Real(1.0) / rjk;
    Real dot_ijk  = dij[0]*djk[0] + dij[1]*djk[1] + dij[2]*djk[2];
    Real dot_jkl  = djk[0]*dlk[0] + djk[1]*dlk[1] + djk[2]*dlk[2];

    Real tmp1 = rjk * inv_raijk2;
    Real tmp2 = rjk * inv_rajkl2;
    Real tmp3 = dot_ijk * inv_raijk2 * inv_rjk;
    Real tmp4 = dot_jkl * inv_rajkl2 * inv_rjk;

    grad[0] =  tmp1 * aijk[0];
    grad[1] =  tmp1 * aijk[1];
    grad[2] =  tmp1 * aijk[2];
    grad[3] = -tmp3 * aijk[0] + tmp4 * ajkl[0];
    grad[4] = -tmp3 * aijk[1] + tmp4 * ajkl[1];
    grad[5] = -tmp3 * aijk[2] + tmp4 * ajkl[2];
    grad[6] =                 - tmp2 * ajkl[0];
    grad[7] =                 - tmp2 * ajkl[1];
    grad[8] =                 - tmp2 * ajkl[2];

    v[0] = grad[0]*dij[0] + grad[3]*djk[0] + grad[6]*dlk[0];
    v[1] = grad[1]*dij[1] + grad[4]*djk[1] + grad[7]*dlk[1];
    v[2] = grad[2]*dij[2] + grad[5]*djk[2] + grad[8]*dlk[2];
}

/* (cos,sin) to the CMAP grid angle in degrees. */
__device__ __forceinline__ double cmap_angle(double cos_dih, double sin_dih,
                                             double rad)
{
    double dihed;
    if (fabs(cos_dih) > 1.0e-1) {
        dihed = asin(sin_dih) / rad;
        if (cos_dih < 0.0) {
            if (dihed > 0.0) dihed =  180.0 - dihed;
            else             dihed = -180.0 - dihed;
        }
    } else {
        dihed = copysign(1.0, sin_dih) * acos(cos_dih) / rad;
    }
    if      (dihed < -180.0) dihed += 360.0;
    else if (dihed >  180.0) dihed -= 360.0;
    return dihed;
}

}  /* anonymous namespace */

/* One kernel per purpose; the purposes share the accumulator and force array but launch
 * separately.  They accumulate in fixed point, so they run side by side on the step's
 * lanes (native_force_bonded) in any order. */

template <typename Real>
__global__ void gcn_kern_bond(const gc_f64 *__restrict__ c,
                              gcn_fx::word *__restrict__ f, gcn_fx::word *__restrict__ acc,
                              const struct gcn_term_pair *__restrict__ t,
                              const gc_f64 *__restrict__ par,
                              gc_i64 n, gc_i64 pitch,
                              double sx, double sy, double sz,
                              int sums)
{
    gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    Real e = Real(0.0), vx = Real(0.0), vy = Real(0.0), vz = Real(0.0);

    if (i < n) {
        int a1 = t[i].a0, a2 = t[i].a1;
        int k1, k2, k3;
        pbc_decode(t[i].pbc, &k1, &k2, &k3);

        const gc_f64 *p = par + 2 * (gc_i64)t[i].param;
        Real fc = (Real)p[0], r0 = (Real)p[1];
        Real d1 = (Real)(c[a1]                 - c[a2]                 + sx*(double)k1);
        Real d2 = (Real)(c[pitch + a1]         - c[pitch + a2]         + sy*(double)k2);
        Real d3 = (Real)(c[2 * pitch + a1]     - c[2 * pitch + a2]     + sz*(double)k3);

        Real r12   = sqrt(d1*d1 + d2*d2 + d3*d3);
        Real r_dif = r12 - r0;
        e = fc * r_dif * r_dif;

        Real cc = (Real(2.0) * fc * r_dif) / r12;
        Real w1 = cc*d1, w2 = cc*d2, w3 = cc*d3;

        vx = d1*w1; vy = d2*w2; vz = d3*w3;

        scatter3(f, pitch, a1, -w1, -w2, -w3);
        scatter3(f, pitch, a2,  w1,  w2,  w3);
    }
    if (sums) {
        const ene_t es[4] = { {GCN_BE_BOND, e}, {GCN_BE_VIRX, vx}, {GCN_BE_VIRY, vy},
                              {GCN_BE_VIRZ, vz} };
        ene_block(acc, es);
    }
}

/* The angle and its Urey-Bradley 1-3 term are one purpose over one atom triple: they
 * share the virial accumulation. */
template <typename Real>
__global__ void gcn_kern_angle(const gc_f64 *__restrict__ c,
                               gcn_fx::word *__restrict__ f, gcn_fx::word *__restrict__ acc,
                               const struct gcn_term_exec *__restrict__ t,
                               const gc_f64 *__restrict__ par,
                               gc_i64 n, gc_i64 pitch,
                               double sx, double sy, double sz, double eps,
                               int sums)
{
    gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    Real ea = Real(0.0), eu = Real(0.0), vx = Real(0.0), vy = Real(0.0), vz = Real(0.0);

    if (i < n) {
        int a1 = t[i].a[0], a2 = t[i].a[1], a3 = t[i].a[2];
        int k1, k2, k3;
        Real d12[3], d32[3], d13[3], w[9];
        const gc_f64 *p = par + 4 * (gc_i64)t[i].param;

        pbc_decode(t[i].pbc[0], &k1, &k2, &k3);
        d12[0] = (Real)(c[a1]             - c[a2]             + sx*(double)k1);
        d12[1] = (Real)(c[pitch + a1]     - c[pitch + a2]     + sy*(double)k2);
        d12[2] = (Real)(c[2 * pitch + a1] - c[2 * pitch + a2] + sz*(double)k3);

        pbc_decode(t[i].pbc[1], &k1, &k2, &k3);
        d32[0] = (Real)(c[a3]             - c[a2]             + sx*(double)k1);
        d32[1] = (Real)(c[pitch + a3]     - c[pitch + a2]     + sy*(double)k2);
        d32[2] = (Real)(c[2 * pitch + a3] - c[2 * pitch + a2] + sz*(double)k3);

        Real r12_2  = d12[0]*d12[0] + d12[1]*d12[1] + d12[2]*d12[2];
        Real r32_2  = d32[0]*d32[0] + d32[1]*d32[1] + d32[2]*d32[2];
        Real r12r32 = sqrt(r12_2 * r32_2);

        Real inv_r12r32 = Real(1.0) / r12r32;
        Real inv_r12_2  = Real(1.0) / r12_2;
        Real inv_r32_2  = Real(1.0) / r32_2;

        Real cos_t = (d12[0]*d32[0] + d12[1]*d32[1] + d12[2]*d32[2])
                     * inv_r12r32;
        cos_t = fmin( Real(1.0), cos_t);
        cos_t = fmax(-Real(1.0), cos_t);
        Real t123  = acos(cos_t);
        Real fc    = (Real)p[0], theta0 = (Real)p[1];
        Real t_dif = t123 - theta0;
        ea = fc * t_dif * t_dif;

        Real sin_t = sin(t123);
        sin_t = fmax((Real)eps, sin_t);
        Real cc_frc  = -(Real(2.0) * fc * t_dif) / sin_t;
        Real cc_frc2 = cos_t * inv_r12_2;
        Real cc_frc3 = cos_t * inv_r32_2;
        w[0] = cc_frc * (d32[0]*inv_r12r32 - d12[0]*cc_frc2);
        w[1] = cc_frc * (d32[1]*inv_r12r32 - d12[1]*cc_frc2);
        w[2] = cc_frc * (d32[2]*inv_r12r32 - d12[2]*cc_frc2);
        w[3] = cc_frc * (d12[0]*inv_r12r32 - d32[0]*cc_frc3);
        w[4] = cc_frc * (d12[1]*inv_r12r32 - d32[1]*cc_frc3);
        w[5] = cc_frc * (d12[2]*inv_r12r32 - d32[2]*cc_frc3);

        vx = d12[0]*w[0] + d32[0]*w[3];
        vy = d12[1]*w[1] + d32[1]*w[4];
        vz = d12[2]*w[2] + d32[2]*w[5];

        Real fc_ub = (Real)p[2];
        if (fabs(fc_ub) > eps) {
            pbc_decode(t[i].pbc[2], &k1, &k2, &k3);
            d13[0] = (Real)(c[a1]             - c[a3]             + sx*(double)k1);
            d13[1] = (Real)(c[pitch + a1]     - c[pitch + a3]     + sy*(double)k2);
            d13[2] = (Real)(c[2 * pitch + a1] - c[2 * pitch + a3] + sz*(double)k3);
            Real r13 = sqrt(d13[0]*d13[0] + d13[1]*d13[1] + d13[2]*d13[2]);
            Real ub_dif = r13 - (Real)p[3];
            eu = fc_ub * ub_dif * ub_dif;
            Real cc_ub = (Real(2.0) * fc_ub * ub_dif) / r13;
            w[6] = cc_ub * d13[0];
            w[7] = cc_ub * d13[1];
            w[8] = cc_ub * d13[2];
            vx += d13[0]*w[6];
            vy += d13[1]*w[7];
            vz += d13[2]*w[8];
        } else {
            w[6] = Real(0.0); w[7] = Real(0.0); w[8] = Real(0.0);
        }

        scatter3(f, pitch, a1, -w[0]-w[6], -w[1]-w[7], -w[2]-w[8]);
        scatter3(f, pitch, a2,  w[0]+w[3],  w[1]+w[4],  w[2]+w[5]);
        scatter3(f, pitch, a3, -w[3]+w[6], -w[4]+w[7], -w[5]+w[8]);
    }
    if (sums) {
        const ene_t es[5] = { {GCN_BE_ANGLE, ea}, {GCN_BE_UREY, eu}, {GCN_BE_VIRX, vx},
                              {GCN_BE_VIRY, vy}, {GCN_BE_VIRZ, vz} };
        ene_block(acc, es);
    }
}

/* The Fourier term k(1 + cos(n phi - phi0)): every dihedral, and the impropers of
 * GC_IMPROPER_FOURIER (AMBER; stock compute_energy_improp_cos).  eslot is the energy it
 * sums into. */
template <typename Real>
__global__ void gcn_kern_dihedral(const gc_f64 *__restrict__ c,
                                  gcn_fx::word *__restrict__ f,
                                  gcn_fx::word *__restrict__ acc,
                                  const struct gcn_term_exec *__restrict__ t,
                                  const gc_f64 *__restrict__ par,
                                  const gc_i32 *__restrict__ pint,
                                  gc_i64 n, gc_i64 pitch,
                                  double sx, double sy, double sz,
                                  int pmod, int eslot,
                                  int sums)
{
    gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    Real e = Real(0.0), vx = Real(0.0), vy = Real(0.0), vz = Real(0.0);

    if (i < n) {
        int a[4], pbc[3];
        double cx[4], cy[4], cz[4];
        Real grad[9], v[3];
        for (int k = 0; k < 4; ++k) {
            a[k]  = t[i].a[k];
            cx[k] = c[a[k]];
            cy[k] = c[pitch + a[k]];
            cz[k] = c[2 * pitch + a[k]];
        }
        pbc[0] = t[i].pbc[0]; pbc[1] = t[i].pbc[1]; pbc[2] = t[i].pbc[2];
        /* the periodicity modulo notation_14types, as stock reads it */
        int nrot = pint[t[i].param];
        if (pmod > 0) nrot %= pmod;

        Real cos_dih, sin_dih;
        dihedral_2(cx, cy, cz, pbc, sx, sy, sz, &cos_dih, &sin_dih, grad, v);

        Real cosnt = Real(1.0), sinnt = Real(0.0);
        for (int krot = 0; krot < nrot; ++krot) {
            Real tt = cosnt*cos_dih - sinnt*sin_dih;
            sinnt = sinnt*cos_dih + cosnt*sin_dih;
            cosnt = tt;
        }

        const gc_f64 *p = par + 3 * (gc_i64)t[i].param;
        Real fc = (Real)p[0], cospha = (Real)p[1], sinpha = (Real)p[2];
        e = fc * (Real(1.0) + cospha*cosnt + sinnt*sinpha);

        Real gc_ = fc * (Real)nrot * (cospha*sinnt - cosnt*sinpha);
        Real w[9];
        for (int k = 0; k < 9; ++k) w[k] = gc_ * grad[k];

        vx = gc_*v[0]; vy = gc_*v[1]; vz = gc_*v[2];

        scatter3(f, pitch, a[0], -w[0], -w[1], -w[2]);
        scatter3(f, pitch, a[1], w[0]-w[3], w[1]-w[4], w[2]-w[5]);
        scatter3(f, pitch, a[2], w[3]+w[6], w[4]+w[7], w[5]+w[8]);
        scatter3(f, pitch, a[3], -w[6], -w[7], -w[8]);
    }
    if (sums) {
        const ene_t es[4] = { {eslot, e}, {GCN_BE_VIRX, vx}, {GCN_BE_VIRY, vy},
                              {GCN_BE_VIRZ, vz} };
        ene_block(acc, es);
    }
}

template <typename Real>
__global__ void gcn_kern_improper(const gc_f64 *__restrict__ c,
                                  gcn_fx::word *__restrict__ f,
                                  gcn_fx::word *__restrict__ acc,
                                  const struct gcn_term_exec *__restrict__ t,
                                  const gc_f64 *__restrict__ par,
                                  gc_i64 n, gc_i64 pitch,
                                  double sx, double sy, double sz,
                                  int sums)
{
    gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    Real e = Real(0.0), vx = Real(0.0), vy = Real(0.0), vz = Real(0.0);

    if (i < n) {
        int a[4], pbc[3];
        double cx[4], cy[4], cz[4];
        Real grad[9], v[3];
        for (int k = 0; k < 4; ++k) {
            a[k]  = t[i].a[k];
            cx[k] = c[a[k]];
            cy[k] = c[pitch + a[k]];
            cz[k] = c[2 * pitch + a[k]];
        }
        pbc[0] = t[i].pbc[0]; pbc[1] = t[i].pbc[1]; pbc[2] = t[i].pbc[2];

        Real cos_dih, sin_dih;
        dihedral_2(cx, cy, cz, pbc, sx, sy, sz, &cos_dih, &sin_dih, grad, v);

        const gc_f64 *p = par + 3 * (gc_i64)t[i].param;
        Real fc = (Real)p[0], cospha = (Real)p[1], sinpha = (Real)p[2];
        Real cosdif = cos_dih*cospha + sin_dih*sinpha;
        Real sindif = cos_dih*sinpha - sin_dih*cospha;
        Real diffphi;
        if (cosdif > Real(1.0e-1)) diffphi = asin(sindif);
        else                 diffphi = copysign(Real(1.0), sindif) * acos(cosdif);

        e = fc * diffphi * diffphi;
        Real gc_ = Real(2.0) * fc * diffphi;

        Real w[9];
        for (int k = 0; k < 9; ++k) w[k] = gc_ * grad[k];

        vx = gc_*v[0]; vy = gc_*v[1]; vz = gc_*v[2];

        scatter3(f, pitch, a[0], -w[0], -w[1], -w[2]);
        scatter3(f, pitch, a[1], w[0]-w[3], w[1]-w[4], w[2]-w[5]);
        scatter3(f, pitch, a[2], w[3]+w[6], w[4]+w[7], w[5]+w[8]);
        scatter3(f, pitch, a[3], -w[6], -w[7], -w[8]);
    }
    if (sums) {
        const ene_t es[4] = { {GCN_BE_IMPR, e}, {GCN_BE_VIRX, vx}, {GCN_BE_VIRY, vy},
                              {GCN_BE_VIRZ, vz} };
        ene_block(acc, es);
    }
}

/* CMAP is its own launch: its register pressure is about twice the next worst term's,
 * and a fused grid would pay that allocation on every block. */
template <typename Real>
__global__ void gcn_kern_cmap(const gc_f64 *__restrict__ c,
                              gcn_fx::word *__restrict__ f, gcn_fx::word *__restrict__ acc,
                              const struct gcn_term_exec *__restrict__ t,
                              const gc_f64 *__restrict__ coef,
                              const gc_i32 *__restrict__ res,
                              const gc_i32 *__restrict__ type,
                              int ngrid_dim, gc_i64 n, gc_i64 pitch,
                              double sx, double sy, double sz, double rad,
                              int sums)
{
    gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    Real e = Real(0.0), vx = Real(0.0), vy = Real(0.0), vz = Real(0.0);

    if (i < n) {
        int a[8], pbc[3];
        double cx[8], cy[8], cz[8];
        for (int k = 0; k < 8; ++k) {
            a[k]  = t[i].a[k];
            cx[k] = c[a[k]];
            cy[k] = c[pitch + a[k]];
            cz[k] = c[2 * pitch + a[k]];
        }
        int itype  = type[t[i].param] - 1;   /* GENESIS's type, one based */
        int ngrid0 = res[itype];
        double delta     = 360.0 / (double)ngrid0;
        double inv_delta = 1.0 / delta;

        Real gradphi[9], gradpsi[9], vphi[3], vpsi[3];
        Real cos_dih, sin_dih;
        double dgrid[2];
        int    igrid[2];

        /* phi over endpoints 1..4 with image codes 0..2, psi over 5..8
           with codes 3..5. */
        pbc[0] = t[i].pbc[0]; pbc[1] = t[i].pbc[1]; pbc[2] = t[i].pbc[2];
        dihedral_2(cx, cy, cz, pbc, sx, sy, sz,
                   &cos_dih, &sin_dih, gradphi, vphi);
        for (int k = 0; k < 9; ++k) gradphi[k] /= (Real)rad;
        double dihed = cmap_angle((double)cos_dih, (double)sin_dih, rad);
        int igr = (int)((dihed + 180.0) * inv_delta);
        dgrid[0] = (dihed - (delta * (double)igr - 180.0)) * inv_delta;
        igrid[0] = igr;

        pbc[0] = t[i].pbc[3]; pbc[1] = t[i].pbc[4]; pbc[2] = t[i].pbc[5];
        dihedral_2(cx + 4, cy + 4, cz + 4, pbc, sx, sy, sz,
                   &cos_dih, &sin_dih, gradpsi, vpsi);
        for (int k = 0; k < 9; ++k) gradpsi[k] /= (Real)rad;
        dihed = cmap_angle((double)cos_dih, (double)sin_dih, rad);
        igr = (int)((dihed + 180.0) * inv_delta);
        dgrid[1] = (dihed - (delta * (double)igr - 180.0)) * inv_delta;
        igrid[1] = igr;

        Real gp[2][4], dgp[2][4];
        gp[0][0] = Real(1.0); gp[1][0] = Real(1.0);
        for (int ip = 1; ip < 4; ++ip) {
            gp[0][ip] = gp[0][ip-1] * (Real)dgrid[0];
            gp[1][ip] = gp[1][ip-1] * (Real)dgrid[1];
        }
        dgp[0][0] = Real(0.0);       dgp[1][0] = Real(0.0);
        dgp[0][1] = (Real)inv_delta; dgp[1][1] = (Real)inv_delta;
        for (int ip = 2; ip < 4; ++ip) {
            dgp[0][ip] = gp[0][ip-1] * (Real)ip * (Real)inv_delta;
            dgp[1][ip] = gp[1][ip-1] * (Real)ip * (Real)inv_delta;
        }

        const gc_f64 *cf = coef + (size_t)16 * (igrid[1]
                         + ngrid_dim * (igrid[0] + ngrid_dim * (size_t)itype));
        Real g0 = Real(0.0), g1 = Real(0.0), ecm = Real(0.0);
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                Real ctmp = (Real)cf[k + 4*j];
                ecm += gp[1][k] * gp[0][j] * ctmp;
                g0  += -dgp[0][j] * gp[1][k] * ctmp;
                g1  += -dgp[1][k] * gp[0][j] * ctmp;
            }
        }
        e = ecm;

        Real w1[9], w2[9];
        for (int k = 0; k < 9; ++k) { w1[k] = g0*gradphi[k]; w2[k] = g1*gradpsi[k]; }

        scatter3(f, pitch, a[0], -w1[0], -w1[1], -w1[2]);
        scatter3(f, pitch, a[1], w1[0]-w1[3], w1[1]-w1[4], w1[2]-w1[5]);
        scatter3(f, pitch, a[2], w1[3]+w1[6], w1[4]+w1[7], w1[5]+w1[8]);
        scatter3(f, pitch, a[3], -w1[6], -w1[7], -w1[8]);
        scatter3(f, pitch, a[4], -w2[0], -w2[1], -w2[2]);
        scatter3(f, pitch, a[5], w2[0]-w2[3], w2[1]-w2[4], w2[2]-w2[5]);
        scatter3(f, pitch, a[6], w2[3]+w2[6], w2[4]+w2[7], w2[5]+w2[8]);
        scatter3(f, pitch, a[7], -w2[6], -w2[7], -w2[8]);

        vx = (g0*vphi[0] + g1*vpsi[0]) / (Real)rad;
        vy = (g0*vphi[1] + g1*vpsi[1]) / (Real)rad;
        vz = (g0*vphi[2] + g1*vpsi[2]) / (Real)rad;
    }
    if (sums) {
        const ene_t es[4] = { {GCN_BE_CMAP, e}, {GCN_BE_VIRX, vx}, {GCN_BE_VIRY, vy},
                              {GCN_BE_VIRZ, vz} };
        ene_block(acc, es);
    }
}

/* The 1-4 term, in the two gc_nb14_form forms.  TABLE (CHARMM): table-driven through
 * the linear tables of the real-space kernel with the 1-4 LJ pair table, so its table
 * indices must match the real-space ones (sp_energy_table_linear.fpp).  SCALED (AMBER,
 * GROMACS): plain r^-12/r^-6 with the 1-4 LJ pair table times the pair's lj_scale, the
 * table's electrostatic column times qq_scale (estride is that column stride), and the
 * pair's share of the reciprocal sum removed with weight 1 - qq_scale on the correction
 * table (a weight of zero skips it).  par holds each record's (qq_scale, lj_scale). */
template <bool scaled, typename Real>
__global__ void gcn_kern_nb14(const gc_f64 *__restrict__ c,
                              const gc_f64 *__restrict__ charge,
                              const gc_i32 *__restrict__ cls,
                              gcn_fx::word *__restrict__ f, gcn_fx::word *__restrict__ acc,
                              const struct gcn_term_pair *__restrict__ t,
                              const gc_f64 *__restrict__ lj12,
                              const gc_f64 *__restrict__ lj6,
                              const gc_f64 *__restrict__ tab_ene,
                              const gc_f64 *__restrict__ tab_grad,
                              const gc_f64 *__restrict__ par,
                              const gc_f64 *__restrict__ tab_ecor,
                              const gc_f64 *__restrict__ tab_decor,
                              int estride,
                              int ncls, gc_i64 n, gc_i64 pitch,
                              double sx, double sy, double sz,
                              double density, double cutoff2,
                              int sums)
{
    gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    Real ee = Real(0.0), ev = Real(0.0), ec = Real(0.0), vx = Real(0.0), vy = Real(0.0), vz = Real(0.0);

    if (i < n) {
        int ai = t[i].a0, aj = t[i].a1;
        int k1, k2, k3;
        pbc_decode(t[i].pbc, &k1, &k2, &k3);

        Real d1 = (Real)(c[ai]             - c[aj]             + sx*(double)k1);
        Real d2 = (Real)(c[pitch + ai]     - c[pitch + aj]     + sy*(double)k2);
        Real d3 = (Real)(c[2 * pitch + ai] - c[2 * pitch + aj] + sz*(double)k3);
        Real rij2 = d1*d1 + d2*d2 + d3*d3;

        int ic = cls[ai] - 1;
        int jc = cls[aj] - 1;
        Real c6  = (Real)lj6 [ic + ncls*jc];
        Real c12 = (Real)lj12[ic + ncls*jc];
        Real gc_, coef = Real(0.0);

        if (scaled) {
            const gc_f64 *p = par + 2 * (gc_i64)t[i].param;
            Real qq_scale = (Real)p[0], lj_scale = (Real)p[1];
            Real inv_r12   = Real(1.0) / rij2;
            Real inv_r121  = sqrt(inv_r12);
            Real inv_r123  = inv_r12 * inv_r121;
            Real inv_r126  = inv_r123 * inv_r123;
            Real inv_r1212 = inv_r126 * inv_r126;

            rij2 = (Real)(cutoff2 * density) / rij2;
            int    L  = (int)rij2;
            Real R  = rij2 - (Real)L;
            int    e0 = estride * L - 1;
            Real cc = (Real)charge[ai] * (Real)charge[aj] * qq_scale;

            Real tel = (Real)tab_ene[e0] + R*((Real)tab_ene[e0+estride]-(Real)tab_ene[e0]);
            ev = (inv_r1212*c12 - inv_r126*c6) * lj_scale;
            ee = cc*tel;

            Real t12 = -Real(12.0) * inv_r1212 * inv_r12;
            Real t6  = -Real(6.0) * inv_r126 * inv_r12;
            tel = (Real)tab_grad[e0] + R*((Real)tab_grad[e0+estride]-(Real)tab_grad[e0]);
            gc_ = (t12*c12 - t6*c6) * lj_scale + cc*tel;

            Real qcor = -qq_scale + Real(1.0);
            if (qcor != Real(0.0)) {
                Real ccor = (Real)charge[ai] * (Real)charge[aj] * qcor;
                Real term = (Real)tab_ecor[L-1] + R*((Real)tab_ecor[L]-(Real)tab_ecor[L-1]);
                ec = term*ccor;
                term = (Real)tab_decor[L-1] + R*((Real)tab_decor[L]-(Real)tab_decor[L-1]);
                coef = ccor*term;
            }
        } else {
            Real qq = (Real)charge[ai] * (Real)charge[aj];

            rij2 = (Real)(cutoff2 * density) / rij2;
            int    L  = (int)rij2;
            Real R  = rij2 - (Real)L;
            int    L1 = 3*L - 3;

            Real t12 = (Real)tab_ene[L1]     + R*((Real)tab_ene[L1+3]-(Real)tab_ene[L1]);
            Real t6  = (Real)tab_ene[L1+1]   + R*((Real)tab_ene[L1+4]-(Real)tab_ene[L1+1]);
            Real tel = (Real)tab_ene[L1+2]   + R*((Real)tab_ene[L1+5]-(Real)tab_ene[L1+2]);
            ev = t12*c12 - t6*c6;
            ee = qq*tel;

            t12 = (Real)tab_grad[L1]   + R*((Real)tab_grad[L1+3]-(Real)tab_grad[L1]);
            t6  = (Real)tab_grad[L1+1] + R*((Real)tab_grad[L1+4]-(Real)tab_grad[L1+1]);
            tel = (Real)tab_grad[L1+2] + R*((Real)tab_grad[L1+5]-(Real)tab_grad[L1+2]);
            gc_ = t12*c12 - t6*c6 + qq*tel;
        }

        Real w1 = gc_*d1, w2 = gc_*d2, w3 = gc_*d3;
        if (scaled) { w1 += coef*d1; w2 += coef*d2; w3 += coef*d3; }
        vx = d1*w1; vy = d2*w2; vz = d3*w3;

        scatter3(f, pitch, ai, -w1, -w2, -w3);
        scatter3(f, pitch, aj,  w1,  w2,  w3);
    }
    if (sums) {
        const ene_t es[6] = { {GCN_BE_ELEC14, ee}, {GCN_BE_VDW14, ev}, {GCN_BE_ELECCOR, scaled ? ec : 0.0},
                              {GCN_BE_VIRX, vx}, {GCN_BE_VIRY, vy}, {GCN_BE_VIRZ, vz} };
        ene_block(acc, es);
    }
}

/* Flexible water: when water is not constrained, GENESIS evaluates each water_list
 * molecule's two O-H bonds, optionally its H-H bond, and its H-O-H angle from scalar
 * parameters (sp_energy_bonds.fpp, sp_energy_angles.fpp), ported term by term.  A
 * native water group is its three members in water_list order (O, H1, H2); a molecule
 * never straddles an image, as in the CPU routines. */
__global__ void gcn_kern_water_flex(const gc_f64 *__restrict__ c,
                                    gcn_fx::word *__restrict__ f,
                                    gcn_fx::word *__restrict__ acc,
                                    const gc_i64 *__restrict__ goff,
                                    const gc_i32 *__restrict__ gmem,
                                    const gc_u8 *__restrict__ gkind,
                                    gc_i64 ngroup, gc_i64 pitch,
                                    int bond_calc, int bond_hh,
                                    int angle_calc,
                                    double oh_bond, double oh_force,
                                    double hh_bond, double hh_force,
                                    double hoh_angle, double hoh_force,
                                    double eps)
{
    gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    double eb = 0.0, ea = 0.0, vx = 0.0, vy = 0.0, vz = 0.0;

    if (g < ngroup && gkind[g] == GC_GROUP_WATER) {
        const gc_i64 o = goff[g];
        const int list[3] = { gmem[o], gmem[o + 1], gmem[o + 2] };

        if (bond_calc) {
            const int pa[3] = { list[0], list[0], list[1] };
            const int pb[3] = { list[1], list[2], list[2] };
            const int nb = bond_hh ? 3 : 2;
            for (int k = 0; k < nb; ++k) {
                const double r0 = (k < 2) ? oh_bond  : hh_bond;
                const double kf = (k < 2) ? oh_force : hh_force;
                double d1 = c[pa[k]]             - c[pb[k]];
                double d2 = c[pitch + pa[k]]     - c[pitch + pb[k]];
                double d3 = c[2 * pitch + pa[k]] - c[2 * pitch + pb[k]];
                double r12   = sqrt(d1*d1 + d2*d2 + d3*d3);
                double r_dif = r12 - r0;
                eb += kf*r_dif*r_dif;
                double cc = (2.0*kf*r_dif) / r12;
                double w1 = cc*d1, w2 = cc*d2, w3 = cc*d3;
                vx += d1*w1; vy += d2*w2; vz += d3*w3;
                scatter3(f, pitch, pa[k], -w1, -w2, -w3);
                scatter3(f, pitch, pb[k],  w1,  w2,  w3);
            }
        }

        if (angle_calc) {
            const int i2 = list[0], i1 = list[1], i3 = list[2];
            double d12[3], d32[3];
            for (int k = 0; k < 3; ++k) {
                d12[k] = c[(gc_i64)k * pitch + i1] - c[(gc_i64)k * pitch + i2];
                d32[k] = c[(gc_i64)k * pitch + i3] - c[(gc_i64)k * pitch + i2];
            }
            double r12_2  = d12[0]*d12[0] + d12[1]*d12[1] + d12[2]*d12[2];
            double r32_2  = d32[0]*d32[0] + d32[1]*d32[1] + d32[2]*d32[2];
            double r12r32 = sqrt(r12_2*r32_2);
            double inv_r12r32 = 1.0 / r12r32;
            double inv_r12_2  = 1.0 / r12_2;
            double inv_r32_2  = 1.0 / r32_2;
            double cos_t = (d12[0]*d32[0] + d12[1]*d32[1] + d12[2]*d32[2])
                         * inv_r12r32;
            cos_t = fmin( 1.0, cos_t);
            cos_t = fmax(-1.0, cos_t);
            double t123  = acos(cos_t);
            double t_dif = t123 - hoh_angle;
            ea += hoh_force*t_dif*t_dif;
            double sin_t = sin(t123);
            sin_t = fmax(eps, sin_t);
            double cc  = -(2.0*hoh_force*t_dif) / sin_t;
            double cc2 = cos_t * inv_r12_2;
            double cc3 = cos_t * inv_r32_2;
            double w[6];
            for (int k = 0; k < 3; ++k) {
                w[k]     = cc * (d32[k] * inv_r12r32 - d12[k] * cc2);
                w[k + 3] = cc * (d12[k] * inv_r12r32 - d32[k] * cc3);
            }
            vx += d12[0]*w[0] + d32[0]*w[3];
            vy += d12[1]*w[1] + d32[1]*w[4];
            vz += d12[2]*w[2] + d32[2]*w[5];
            scatter3(f, pitch, i1, -w[0], -w[1], -w[2]);
            scatter3(f, pitch, i2, w[0] + w[3], w[1] + w[4], w[2] + w[5]);
            scatter3(f, pitch, i3, -w[3], -w[4], -w[5]);
        }
    }
    {
        const ene_t es[5] = { {GCN_BE_BOND, eb}, {GCN_BE_ANGLE, ea}, {GCN_BE_VIRX, vx},
                              {GCN_BE_VIRY, vy}, {GCN_BE_VIRZ, vz} };
        ene_block(acc, es);
    }
}

/* The CHARMM 1-4 pairs for electrostatic = CUTOFF: sp_energy_table_cubic.fpp
 * compute_energy_nonbond14_table_charmm, term by term against the cubic table of the
 * pair kernel and CHARMM's 1-4 LJ table.  The CPU routine forms coord(ix) - coord(iy)
 * with no image, the same number whenever the term's image code is zero. */
__global__ void gcn_kern_nb14_cutoff(const gc_f64 *__restrict__ c,
                                     const gc_f64 *__restrict__ charge,
                                     const gc_i32 *__restrict__ cls,
                                     gcn_fx::word *__restrict__ f,
                                     gcn_fx::word *__restrict__ acc,
                                     const struct gcn_term_pair *__restrict__ t,
                                     const gc_f64 *__restrict__ lj12,
                                     const gc_f64 *__restrict__ lj6,
                                     const gc_f64 *__restrict__ tab_ene,
                                     const gc_f64 *__restrict__ tab_grad,
                                     int ncls, gc_i64 n, gc_i64 pitch,
                                     double sx, double sy, double sz,
                                     double density)
{
    gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    double ee = 0.0, ev = 0.0, vx = 0.0, vy = 0.0, vz = 0.0;

    if (i < n) {
        int ai = t[i].a0, aj = t[i].a1;
        int k1, k2, k3;
        pbc_decode(t[i].pbc, &k1, &k2, &k3);

        double d1 = c[ai]             - c[aj]             + sx*(double)k1;
        double d2 = c[pitch + ai]     - c[pitch + aj]     + sy*(double)k2;
        double d3 = c[2 * pitch + ai] - c[2 * pitch + aj] + sz*(double)k3;
        double rij2 = d1*d1 + d2*d2 + d3*d3;

        rij2 = density * rij2;
        int    L   = (int)rij2;
        double R   = rij2 - (double)L;
        int    ic  = cls[ai] - 1;
        int    jc  = cls[aj] - 1;
        double c6  = lj6 [ic + ncls*jc];
        double c12 = lj12[ic + ncls*jc];
        double h00 = (1.0 + 2.0*R)*(1.0 - R)*(1.0 - R);
        double h10 = R*(1.0 - R)*(1.0 - R);
        double h01 = R*R*(3.0 - 2.0*R);
        double h11 = R*R*(R - 1.0);
        const gc_f64 *te = tab_ene  + (6*L - 6);   /* table(1:12) */
        const gc_f64 *tg = tab_grad + (6*L - 6);

        double t12 = te[0]*h00 + te[1]*h10 + te[6]*h01  + te[7]*h11;
        double t6  = te[2]*h00 + te[3]*h10 + te[8]*h01  + te[9]*h11;
        double tel = te[4]*h00 + te[5]*h10 + te[10]*h01 + te[11]*h11;
        ev = t12*c12 - t6*c6;
        ee = charge[ai]*charge[aj]*tel;

        t12 = tg[0]*h00 + tg[1]*h10 + tg[6]*h01  + tg[7]*h11;
        t6  = tg[2]*h00 + tg[3]*h10 + tg[8]*h01  + tg[9]*h11;
        tel = tg[4]*h00 + tg[5]*h10 + tg[10]*h01 + tg[11]*h11;
        double gc_ = t12*c12 - t6*c6 + charge[ai]*charge[aj]*tel;

        double w1 = gc_*d1, w2 = gc_*d2, w3 = gc_*d3;
        vx = d1*w1; vy = d2*w2; vz = d3*w3;

        scatter3(f, pitch, ai, -w1, -w2, -w3);
        scatter3(f, pitch, aj,  w1,  w2,  w3);
    }
    {
        const ene_t es[5] = { {GCN_BE_ELEC14, ee}, {GCN_BE_VDW14, ev}, {GCN_BE_VIRX, vx},
                              {GCN_BE_VIRY, vy}, {GCN_BE_VIRZ, vz} };
        ene_block(acc, es);
    }
}

/* The excluded-pair correction that removes the reciprocal sum's contribution from
 * bonded neighbours (sp_energy_table_linear_bondcorr.fpp; the stock GPU path runs it on
 * the CPU). */
template <typename Real>
__global__ void gcn_kern_excluded(const gc_f64 *__restrict__ c,
                                  const gc_f64 *__restrict__ charge,
                                  gcn_fx::word *__restrict__ f,
                                  gcn_fx::word *__restrict__ acc,
                                  const struct gcn_term_pair *__restrict__ t,
                                  const gc_f64 *__restrict__ tab_ecor,
                                  const gc_f64 *__restrict__ tab_decor,
                                  gc_i64 n, gc_i64 pitch,
                                  double sx, double sy, double sz,
                                  double density, double cutoff2,
                                  int sums)
{
    gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    Real ee = Real(0.0), vx = Real(0.0), vy = Real(0.0), vz = Real(0.0);

    if (i < n) {
        int ai = t[i].a0, aj = t[i].a1;
        int k1, k2, k3;
        pbc_decode(t[i].pbc, &k1, &k2, &k3);

        Real d1 = (Real)(c[ai]             - c[aj]             + sx*(double)k1);
        Real d2 = (Real)(c[pitch + ai]     - c[pitch + aj]     + sy*(double)k2);
        Real d3 = (Real)(c[2 * pitch + ai] - c[2 * pitch + aj] + sz*(double)k3);
        Real rij2 = d1*d1 + d2*d2 + d3*d3;
        Real qq   = (Real)charge[ai] * (Real)charge[aj];

        rij2 = (Real)(cutoff2 * density) / rij2;
        int    L = (int)rij2;
        Real R = rij2 - (Real)L;

        Real term = (Real)tab_ecor[L-1] + R*((Real)tab_ecor[L]-(Real)tab_ecor[L-1]);
        ee = qq * term;

        term = (Real)tab_decor[L-1] + R*((Real)tab_decor[L]-(Real)tab_decor[L-1]);
        Real coef = qq * term;

        Real w1 = coef*d1, w2 = coef*d2, w3 = coef*d3;
        vx = d1*w1; vy = d2*w2; vz = d3*w3;

        scatter3(f, pitch, ai, -w1, -w2, -w3);
        scatter3(f, pitch, aj,  w1,  w2,  w3);
    }
    if (sums) {
        const ene_t es[4] = { {GCN_BE_ELECCOR, ee}, {GCN_BE_VIRX, vx}, {GCN_BE_VIRY, vy},
                              {GCN_BE_VIRZ, vz} };
        ene_block(acc, es);
    }
}

/* Positional restraints, one thread per owned slot (sp_energy_restraints.fpp
 * compute_energy_restraints_pos): d = x - ref, E = k (w_x d_x^2 + w_y d_y^2 + w_z
 * d_z^2), work = 2k w d, force -= work and virial_ext(j,j) -= x_j work_j.  The table is
 * probed by GID.  d is taken at its nearest image and x rebuilt as ref + d, since
 * stock's x never wraps. */
__device__ __forceinline__ gc_i32 posres_find(const gc_gid *__restrict__ key,
                                              const gc_i32 *__restrict__ val,
                                              gc_i64 mask, gc_gid g)
{
    gc_i64 h = (gc_i64)(((unsigned long long)g * 11400714819323198485ull)
                        >> 20) & mask;
    for (;;) {
        const gc_gid k = key[h];
        if (k == g) return val[h];
        if (k == 0) return -1;
        h = (h + 1) & mask;
    }
}

__global__ void gcn_kern_posres(const gc_f64 *__restrict__ c,
                                gcn_fx::word *__restrict__ f,
                                gcn_fx::word *__restrict__ acc,
                                const gc_gid *__restrict__ gid,
                                const gc_gid *__restrict__ key,
                                const gc_i32 *__restrict__ val,
                                gc_i64 mask,
                                const gc_f64 *__restrict__ par,
                                gc_i64 n, gc_i64 pitch,
                                double sx, double sy, double sz)
{
    gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0.0, vx = 0.0, vy = 0.0, vz = 0.0;

    const gc_i32 r = (s < n) ? posres_find(key, val, mask, gid[s]) : -1;
    if (r >= 0) {
        const gc_f64 *p = par + 7 * (gc_i64)r;
        double d1 = c[s]             - p[4];
        double d2 = c[pitch + s]     - p[5];
        double d3 = c[2 * pitch + s] - p[6];
        d1 -= sx * rint(d1 / sx);
        d2 -= sy * rint(d2 / sy);
        d3 -= sz * rint(d3 / sz);

        e = p[0] * (p[1]*d1*d1 + p[2]*d2*d2 + p[3]*d3*d3);
        const double g = p[0] * 2.0;
        const double w1 = g*p[1]*d1, w2 = g*p[2]*d2, w3 = g*p[3]*d3;

        vx = -(p[4] + d1) * w1;
        vy = -(p[5] + d2) * w2;
        vz = -(p[6] + d3) * w3;

        scatter3(f, pitch, (int)s, -w1, -w2, -w3);
    }
    {
        const ene_t es[4] = { {GCN_BE_POSRES, e}, {GCN_BE_PVIRX, vx}, {GCN_BE_PVIRY, vy},
                              {GCN_BE_PVIRZ, vz} };
        ene_block(acc, es);
    }
}

/* The join: three accumulators become one force array in a fixed order (real space,
 * bonded, reciprocal), which makes the sum reproducible run to run on one device. */
template <class S>
__global__ void gcn_kern_join_force(S *__restrict__ f,
                                    const gc_f64 *__restrict__ fr,
                                    const gc_f64 *__restrict__ fb,
                                    const gc_f64 *__restrict__ fp,
                                    gc_i64 n3)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n3;
         s += (gc_i64)gridDim.x * blockDim.x)
        f[s] = fr[s] + fb[s] + fp[s];
}

/* One rank's join straight from the real-space and bonded fixed-point fields
 * (gcn_kern_force_fold's conversion and gcn_kern_join_force's sum, in one pass), when
 * nothing reads force_real or force_bond after the join. */
template <class S>
__global__ void gcn_kern_join_force_fx(S *__restrict__ f,
                                       const gcn_fx::word *__restrict__ frx,
                                       const gcn_fx::word *__restrict__ fbx,
                                       const gc_f64 *__restrict__ fp,
                                       gc_i64 n3)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n3;
         s += (gc_i64)gridDim.x * blockDim.x)
        f[s] = gcn_fx::value1<gcn_fx::kForce1>(frx[s])
             + gcn_fx::value1<gcn_fx::kForce1>(fbx[s]) + fp[s];
}

__global__ void gcn_kern_zero(gc_f64 *__restrict__ p, gc_i64 n)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x)
        p[s] = 0.0;
}

__global__ void gcn_kern_force_fold(const gcn_fx::word *__restrict__ fx,
                                    gc_f64 *__restrict__ f, gc_i64 pitch,
                                    gc_i64 lo, gc_i64 hi)
{
    for (gc_i64 s = lo + (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < hi;
         s += (gc_i64)gridDim.x * blockDim.x)
        for (int c = 0; c < 3; ++c)
            f[c * pitch + s] = gcn_fx::value1<gcn_fx::kForce1>(fx[c * pitch + s]);
}

namespace gcn {

static void force_fold(const gcn_fx::word *fx, gc_f64 *f, gc_i64 pitch,
                       gc_i64 lo, gc_i64 hi, cudaStream_t s)
{
    if (hi <= lo) return;
    gc_i64 nb = (hi - lo + GCN_BLOCK - 1) / GCN_BLOCK;
    if (nb > GCN_MAX_BLOCKS) nb = GCN_MAX_BLOCKS;
    gcn_kern_force_fold<<<(unsigned)nb, GCN_BLOCK, 0, s>>>(fx, f, pitch, lo, hi);
}

static gc_status upload_table(gc_f64 **dst, const gc_f64 *src, gc_i64 n)
{
    if (src == 0 || n <= 0) { *dst = 0; return GC_OK; }
    gc_status st = dev_calloc((void **)dst, n * (gc_i64)sizeof(gc_f64));
    if (st != GC_OK) return st;
    if (cudaMemcpy(*dst, src, (size_t)n * sizeof(gc_f64),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return GC_E_DEVICE;
    return GC_OK;
}

gc_status native_posres_upload(gc_context *ctx, gc_i64 n, const gc_gid *gid,
                               const gc_f64 *par7)
{
    struct gcn_device *d = ctx->native;
    if (!d) return GC_E_STATE;
    dev_release((void **)&d->posres_key);
    dev_release((void **)&d->posres_val);
    dev_release((void **)&d->posres_par);
    d->posres_count = 0;
    d->posres_mask  = 0;
    if (n == 0) return GC_OK;

    gc_i64 cap = 2;
    while (cap < 2 * n) cap *= 2;
    const gc_i64 mask = cap - 1;
    std::vector<gc_gid> key((size_t)cap, 0);
    std::vector<gc_i32> val((size_t)cap, -1);
    for (gc_i64 i = 0; i < n; ++i) {
        gc_i64 h = (gc_i64)(((unsigned long long)gid[i] *
                             11400714819323198485ull) >> 20) & mask;
        while (key[(size_t)h] != 0) {
            if (key[(size_t)h] == gid[i]) return GC_E_MISMATCH;  /* twice */
            h = (h + 1) & mask;
        }
        key[(size_t)h] = gid[i];
        val[(size_t)h] = (gc_i32)i;
    }
    gc_status st = dev_calloc((void **)&d->posres_key, cap * (gc_i64)sizeof(gc_gid));
    if (st == GC_OK)
        st = dev_calloc((void **)&d->posres_val, cap * (gc_i64)sizeof(gc_i32));
    if (st == GC_OK) st = upload_table(&d->posres_par, par7, 7 * n);
    if (st != GC_OK) return st;
    if (cudaMemcpy(d->posres_key, &key[0], (size_t)cap * sizeof(gc_gid),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d->posres_val, &val[0], (size_t)cap * sizeof(gc_i32),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return GC_E_DEVICE;
    d->posres_count = n;
    d->posres_mask  = mask;
    return GC_OK;
}

gc_status native_attach_tables(gc_context *ctx, const gc_table_desc *t,
                               const gc_constraint_desc *c)
{
    struct gcn_device *d = ctx->native;
    gc_i64 ncls2 = (gc_i64)t->num_atom_cls * (gc_i64)t->num_atom_cls;
    gc_status st;

    d->tab = *t;
    d->con = *c;

    st = upload_table(&d->lj12,      t->nonb_lj12,  ncls2);           if (st) return st;
    st = upload_table(&d->lj6,       t->nonb_lj6,   ncls2);           if (st) return st;
    st = upload_table(&d->nb14_lj12, t->nb14_lj12,  ncls2);           if (st) return st;
    st = upload_table(&d->nb14_lj6,  t->nb14_lj6,   ncls2);           if (st) return st;
    if (t->table_form != GC_TABLE_PME_LINEAR &&
        t->table_form != GC_TABLE_PME_ELEC &&
        t->table_form != GC_TABLE_CUTOFF_CUBIC) return GC_E_UNSUPPORTED;
    if ((t->improper_form != GC_IMPROPER_HARMONIC &&
         t->improper_form != GC_IMPROPER_FOURIER) ||
        (t->nb14_form != GC_NB14_TABLE && t->nb14_form != GC_NB14_SCALED) ||
        t->periodicity_mod < 0) return GC_E_UNSUPPORTED;
    const bool cubic = (t->table_form == GC_TABLE_CUTOFF_CUBIC);
    /* GENESIS allocates 6*cutoff_int for either form (sp_enefunc_str.fpp); the linear PME
     * form fills 3 per row, the cubic cutoff form 6 per node. */
    const gc_i64 per_row = cubic ? 6 : 3;
    st = upload_table(&d->table_ene,  t->table_ene,  per_row*(gc_i64)t->cutoff_int);
    if (st) return st;
    st = upload_table(&d->table_grad, t->table_grad, per_row*(gc_i64)t->cutoff_int);
    if (st) return st;
    if (!cubic) {
        st = upload_table(&d->table_ecor,  t->table_ecor,  (gc_i64)t->cutoff_int);
        if (st) return st;
        st = upload_table(&d->table_decor, t->table_decor, (gc_i64)t->cutoff_int);
        if (st) return st;
    }
    if (t->nonbond_precision != GC_NONBOND_DOUBLE &&
        t->nonbond_precision != GC_NONBOND_MIXED) return GC_E_UNSUPPORTED;
    if (t->nonbond_precision == GC_NONBOND_MIXED) {
        if (cubic || t->nonb_lj12 == 0 || t->table_grad == 0 ||
            t->table_ene == 0) return GC_E_UNSUPPORTED;
        /* Interval records of the cluster kernel (gpu_nbcluster.cu): row L-1 of each table as
         * {base, next row - base}, the difference taken in FP64 and rounded once.  The last row
         * has no successor and no pair inside the prune radius reads it.  GC_TABLE_PME_ELEC has
         * the electrostatic column alone; its LJ columns stay zero. */
        const gc_i64 nrow = t->cutoff_int;
        const bool elec_only = (t->table_form == GC_TABLE_PME_ELEC);
        std::vector<float2> h((size_t)(ncls2 + 6 * nrow), make_float2(0.f, 0.f));
        for (gc_i64 k = 0; k < ncls2; ++k)
            h[k] = make_float2((float)t->nonb_lj12[k], (float)t->nonb_lj6[k]);
        for (int e = 0; e < 2; ++e) {
            const gc_f64 *src = e ? t->table_ene : t->table_grad;
            float2 *r = &h[(size_t)(ncls2 + 3 * e * nrow)];
            for (gc_i64 L = 0; L + 1 < nrow; ++L) {
                if (elec_only) {
                    r[3 * L + 1] = make_float2((float)src[L], 0.f);
                    r[3 * L + 2] = make_float2(0.f, (float)(src[L + 1] - src[L]));
                    continue;
                }
                const gc_f64 *a = src + 3 * L, *b = a + 3;
                r[3 * L]     = make_float2((float)a[0], (float)a[1]);
                r[3 * L + 1] = make_float2((float)a[2], (float)(b[0] - a[0]));
                r[3 * L + 2] = make_float2((float)(b[1] - a[1]), (float)(b[2] - a[2]));
            }
        }
        st = dev_calloc((void **)&d->nb_mixed, (gc_i64)(h.size() * sizeof(float2)));
        if (st) return st;
        if (cudaMemcpy(d->nb_mixed, h.data(), h.size() * sizeof(float2),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return GC_E_DEVICE;
    }

    if (t->cmap_ntype > 0 && t->cmap_coef != 0) {
        gc_i64 n = (gc_i64)16 * t->cmap_ngrid * t->cmap_ngrid * t->cmap_ntype;
        st = upload_table(&d->cmap_coef, t->cmap_coef, n);
        if (st) return st;
        st = dev_calloc((void **)&d->cmap_resolution,
                        (gc_i64)t->cmap_ntype * (gc_i64)sizeof(gc_i32));
        if (st) return st;
        if (cudaMemcpy(d->cmap_resolution, t->cmap_resolution,
                       (size_t)t->cmap_ntype * sizeof(gc_i32),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return GC_E_DEVICE;
    }

    /* The safe table-support prune: below int(density) the table holds the zeros
     * setup_table_general_pme_linear left, so both rows of a pair beyond this radius are
     * zero and skipping it is bit-identical.  Its denominator (density - 2) is one less
     * than the (density - 1) of the true support radius the list guard uses, so it is
     * deliberately generous. */
    d->cutoff2 = t->cutoffdist * t->cutoffdist;
    d->prune_r2 = (t->density > 3.0)
                ? d->cutoff2 * t->density / (t->density - 2.0)
                : 1.0e30;
    /* The plain LJ pair ends at the cutoff itself (rij2 < cutoff2), whatever the
     * electrostatic column holds beyond it. */
    if (t->table_form == GC_TABLE_PME_ELEC) d->prune_r2 = d->cutoff2;
    d->cubic_lim = 0.0;
    if (cubic) {
        /* The cubic pair term reads nodes L and L+1 of L = int(density*r2).  With nz the last
         * node carrying any non-zero value, a pair with L > nz is a Hermite sum of zeros:
         * evaluate exactly those with density*r2 < nz+1 (nz+1 is also the last node a lookup
         * may touch). */
        gc_i64 nz = 0;
        if (t->table_ene == 0 || t->table_grad == 0) return GC_E_ARG;
        for (gc_i64 node = 1; node <= (gc_i64)t->cutoff_int; ++node)
            for (int k = 0; k < 6; ++k)
                if (t->table_ene [6*(node-1) + k] != 0.0 ||
                    t->table_grad[6*(node-1) + k] != 0.0) nz = node;
        if (nz <= 0 || nz >= (gc_i64)t->cutoff_int) return GC_E_UNSUPPORTED;
        d->cubic_lim = (double)(nz + 1);
        /* compute_force_nonbond_table's within-cell clamp (see gcn_kern_real_space_cutoff)
         * evaluates a pair past the cutoff at density*cutoff2; the native walk skips such
         * pairs, which is exact only when the clamped term is zero in every gradient column.
         * Otherwise decline. */
        {
            const double x  = t->density * d->cutoff2;
            const gc_i64 L  = (gc_i64)(int)x;
            const double R  = x - (double)L;
            const double h00 = (1.0 + 2.0 * R) * (1.0 - R) * (1.0 - R);
            const double h10 = R * (1.0 - R) * (1.0 - R);
            const double h01 = R * R * (3.0 - 2.0 * R);
            const double h11 = R * R * (R - 1.0);
            const gc_i64 L1 = 6 * L - 6;
            if (L < 1 || L + 1 > (gc_i64)t->cutoff_int) return GC_E_UNSUPPORTED;
            for (int k = 0; k < 6; k += 2) {
                const double g = t->table_grad[L1 + k]     * h00
                               + t->table_grad[L1 + k + 1] * h10
                               + t->table_grad[L1 + k + 6] * h01
                               + t->table_grad[L1 + k + 7] * h11;
                if (g != 0.0) return GC_E_UNSUPPORTED;
            }
        }
        d->prune_r2  = d->cubic_lim / t->density;   /* the guard's radius^2 */
    }
    d->tables_attached = 1;
    return GC_OK;
}

__global__ void gcn_kern_fx_clear(gcn_fx::word *__restrict__ f, gc_i64 pitch,
                                  gc_i64 n)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < 3 * n;
         s += (gc_i64)gridDim.x * blockDim.x)
        f[(s / n) * pitch + s % n] = 0;
}

/* Clears a fixed-point force field (k: 0 real, 1 bonded) on d->stream.  Its writers add
 * to resident slots only, so between whole clears only those need clearing.  A kernel
 * rather than a memset, so a captured step's graph takes a new count as a parameter
 * update. */
static cudaError_t fx_clear(struct gcn_device *d, gcn_fx::word *f, int k)
{
    const gc_i64 n = d->num_resident;
    const bool whole = d->fx_clear_pitch[k] != d->pitch || n < d->fx_clear_n[k];
    const gc_i64 m = whole ? d->pitch : n;
    d->fx_clear_pitch[k] = d->pitch;
    d->fx_clear_n[k] = n;
    if (m <= 0) return cudaSuccess;
    const gc_i64 nb = (3 * m + GCN_BLOCK - 1) / GCN_BLOCK;
    gcn_kern_fx_clear<<<(unsigned)(nb > GCN_MAX_BLOCKS ? GCN_MAX_BLOCKS : nb),
                        GCN_BLOCK, 0, d->stream>>>(f, d->pitch, m);
    return cudaGetLastError();
}

/* Zero the real-space fixed-point field and accumulator on d->stream, before any part
 * of the real-space sum is launched. */
gc_status native_force_real_clear(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;

    gc_i64 n3 = 3 * d->pitch;
    if (fx_clear(d, d->force_real_fx, 0) != cudaSuccess ||
        cudaMemsetAsync(d->acc_real, 0,
                        2 * GCN_RE_NSLOT * sizeof(gcn_fx::word),
                        d->stream) != cudaSuccess)
        return GC_E_DEVICE;
    if (!d->reciprocal_ready)
        gcn_kern_zero<<<(unsigned)((n3 + GCN_BLOCK - 1) / GCN_BLOCK > GCN_MAX_BLOCKS
                                   ? GCN_MAX_BLOCKS
                                   : (n3 + GCN_BLOCK - 1) / GCN_BLOCK),
                        GCN_BLOCK, 0, d->stream>>>(d->force_recip, n3);
    if (native_nbc_active(d)) {
        gc_status st = native_nbc_pack_owned(ctx);
        if (st != GC_OK) return st;
    }
    return dev_launched("force_real_clear");
}

/* The cluster kernels' fixed-point overflow, carried into this unit's flag,
 * which the folds read. */
__global__ void gcn_kern_overflow_join(const unsigned int *__restrict__ flag)
{
    if (*flag) atomicOr(&gcn_fx::overflow, 1u);
}

/* One part (GCN_REAL_*) of the real-space sum on stream `s`. */
gc_status native_force_real(gc_context *ctx, gc_i32 want_energy,
                            cudaStream_t s, int part)
{
    struct gcn_device *d = ctx->native;

    if (d->num_pairs > 0 && native_nbc_active(d)) {
        gc_status st = native_nbc_force(ctx, want_energy, s, part);
        if (st != GC_OK) return st;
        gcn_kern_overflow_join<<<1, 1, 0, s>>>(native_nbc_overflow(d));
    } else if (d->num_pairs > 0 && d->tab.table_form == GC_TABLE_CUTOFF_CUBIC) {
        gc_i64 nblk = (d->num_pairs + GCN_RS_WARPS - 1) / GCN_RS_WARPS;
        dim3 blk(32, GCN_RS_WARPS);
        (want_energy ? gcn_kern_real_space_cutoff<true>
                     : gcn_kern_real_space_cutoff<false>)
            <<<(unsigned)nblk, blk, 0, s>>>(
            d->force_coord, d->charge, d->cls, d->force_real_fx, d->acc_real,
            d->pair, d->pair_order, d->sel, d->mask, d->lj12, d->lj6,
            d->table_grad, d->table_ene,
            d->num_pairs, d->pitch, d->tab.num_atom_cls,
            d->tab.density, d->cubic_lim, d->cutoff2, d->layout, part);
    } else if (d->num_pairs > 0) {
        float prune_lo, prune_hi;
        double cmax;
        gcn_sel_bounds(d->prune_r2, d->box, &prune_lo, &prune_hi, &cmax);
        const gc_i64 per_blk = (gc_i64)GCN_RS_WARPS * GCN_RS_PAIRS(want_energy);
        gc_i64 nblk = (d->num_pairs + per_blk - 1) / per_blk;
        dim3 blk(32, GCN_RS_WARPS);
        const bool plain = (d->tab.table_form == GC_TABLE_PME_ELEC);
        (want_energy ? (plain ? gcn_kern_real_space<true, true>
                              : gcn_kern_real_space<true, false>)
                     : (plain ? gcn_kern_real_space<false, true>
                              : gcn_kern_real_space<false, false>))
            <<<(unsigned)nblk, blk, 0, s>>>(
            d->force_coord, d->charge, d->cls, d->force_real_fx, d->acc_real,
            d->pair, d->pair_order, d->sel, d->mask, d->lj12, d->lj6,
            d->table_grad, d->table_ene,
            d->num_pairs, d->pitch, d->tab.num_atom_cls,
            d->tab.density, d->cutoff2, d->prune_r2, prune_hi, 0.5 * cmax,
            d->layout, part);
    }
    /* the boundary part is the last writer of the ghost slots, which the halo return reads
     * next on this stream: fold them now; the owned slots wait for the interior part, in
     * the join */
    if (part == GCN_REAL_BOUNDARY && !native_dist_mixed_words(d))
        force_fold(d->force_real_fx, d->force_real, d->pitch, d->num_owned,
                   d->pitch, s);
    return dev_launched("force_real");
}

gc_status native_force_bonded(gc_context *ctx, gc_i32 want_energy)
{
    struct gcn_device *d = ctx->native;
    const double sx = d->box[0], sy = d->box[1], sz = d->box[2];
    const double eps = 1.0e-10;
    const double rad = 3.14159265358979323846 / 180.0;

    /* nonbond_precision = MIXED evaluates the bonded, 1-4 and excluded-pair terms in FP32
     * on FP64 displacements (formed in FP64, rounded once); force and energy/virial adds
     * stay fixed point, so the sums stay reproducible.  CMAP's grid angle and cell stay
     * FP64. */
    const bool fp32 = d->tab.nonbond_precision == GC_NONBOND_MIXED;

    const int sums = want_energy != 0;

    if (fx_clear(d, d->force_bond_fx, 1) != cudaSuccess ||
        cudaMemsetAsync(d->acc_bond, 0,
                        2 * GCN_BE_NSLOT * sizeof(gcn_fx::word),
                        d->stream) != cudaSuccess)
        return GC_E_DEVICE;

    /* purposes on four streams: dihedrals and restraints on d->stream; excluded and 1-4
     * pairs with the bonds on two lanes; angles, flexible water, impropers and CMAP on the
     * third */
    if (native_lanes_fork(d, GCN_LANES) != GC_OK) return GC_E_DEVICE;
    const cudaStream_t s_dihe = d->stream, s_excl = d->lane[0], s_nb14 = d->lane[1],
                       s_angle = d->lane[2];

#define GCN_GRID(n) (unsigned)(((n) + GCN_BLOCK - 1) / GCN_BLOCK)
#define GCN_LAUNCH(kern, n, s, ...)                                         \
    do {                                                                    \
        if (fp32) kern<float><<<GCN_GRID(n), GCN_BLOCK, 0, (s)>>>(__VA_ARGS__); \
        else      kern<double><<<GCN_GRID(n), GCN_BLOCK, 0, (s)>>>(__VA_ARGS__); \
    } while (0)

    gc_i64 n;

    if (d->term_pair_n[GC_TERM_BOND] != d->term_count[GC_TERM_BOND] ||
        d->term_pair_n[GC_TERM_NB14] != d->term_count[GC_TERM_NB14] ||
        d->term_pair_n[GC_TERM_EXCL] != d->term_count[GC_TERM_EXCL])
        return GC_E_STATE;

    n = d->term_count[GC_TERM_BOND];
    if (n > 0)
        GCN_LAUNCH(gcn_kern_bond, n, s_nb14,
            d->coord, d->force_bond_fx, d->acc_bond,
            d->term_pair[GC_TERM_BOND], d->term_param[GC_TERM_BOND],
            n, d->pitch, sx, sy, sz, sums);

    n = d->term_count[GC_TERM_ANGLE];
    if (n > 0)
        GCN_LAUNCH(gcn_kern_angle, n, s_angle,
            d->coord, d->force_bond_fx, d->acc_bond,
            d->term[GC_TERM_ANGLE], d->term_param[GC_TERM_ANGLE],
            n, d->pitch, sx, sy, sz, eps, sums);
    if ((d->tab.water_bond_calc || d->tab.water_angle_calc) &&
        d->num_groups > 0)
        gcn_kern_water_flex<<<GCN_GRID(d->num_groups), GCN_BLOCK, 0,
                              s_angle>>>(
            d->coord, d->force_bond_fx, d->acc_bond,
            d->group_offset, d->group_member, d->group_kind,
            d->num_groups, d->pitch,
            d->tab.water_bond_calc, d->tab.water_bond_hh,
            d->tab.water_angle_calc,
            d->tab.water_oh_bond, d->tab.water_oh_force,
            d->tab.water_hh_bond, d->tab.water_hh_force,
            d->tab.water_hoh_angle, d->tab.water_hoh_force, eps);

    n = d->term_count[GC_TERM_DIHEDRAL];
    if (n > 0)
        GCN_LAUNCH(gcn_kern_dihedral, n, s_dihe,
            d->coord, d->force_bond_fx, d->acc_bond,
            d->term[GC_TERM_DIHEDRAL], d->term_param[GC_TERM_DIHEDRAL],
            d->term_param_int[GC_TERM_DIHEDRAL], n, d->pitch, sx, sy, sz,
            d->tab.periodicity_mod, GCN_BE_DIHE, sums);

    n = d->term_count[GC_TERM_IMPROPER];
    if (n > 0 && d->tab.improper_form == GC_IMPROPER_FOURIER)
        GCN_LAUNCH(gcn_kern_dihedral, n, s_angle,
            d->coord, d->force_bond_fx, d->acc_bond,
            d->term[GC_TERM_IMPROPER], d->term_param[GC_TERM_IMPROPER],
            d->term_param_int[GC_TERM_IMPROPER], n, d->pitch, sx, sy, sz,
            d->tab.periodicity_mod, GCN_BE_IMPR, sums);
    else if (n > 0)
        GCN_LAUNCH(gcn_kern_improper, n, s_angle,
            d->coord, d->force_bond_fx, d->acc_bond,
            d->term[GC_TERM_IMPROPER], d->term_param[GC_TERM_IMPROPER],
            n, d->pitch, sx, sy, sz, sums);

    n = d->term_count[GC_TERM_CMAP];
    if (n > 0 && d->cmap_coef != 0)
        GCN_LAUNCH(gcn_kern_cmap, n, s_angle,
            d->coord, d->force_bond_fx, d->acc_bond,
            d->term[GC_TERM_CMAP], d->cmap_coef, d->cmap_resolution,
            d->term_param_int[GC_TERM_CMAP], d->tab.cmap_ngrid,
            n, d->pitch, sx, sy, sz, rad, sums);

    n = d->term_count[GC_TERM_NB14];
    if (n > 0 && d->tab.table_form == GC_TABLE_CUTOFF_CUBIC)
        gcn_kern_nb14_cutoff<<<GCN_GRID(n), GCN_BLOCK, 0, s_nb14>>>(
            d->force_coord, d->charge, d->cls, d->force_bond_fx, d->acc_bond,
            d->term_pair[GC_TERM_NB14], d->nb14_lj12, d->nb14_lj6,
            d->table_ene, d->table_grad, d->tab.num_atom_cls,
            n, d->pitch, sx, sy, sz, d->tab.density);
    else if (n > 0)
        (d->tab.nb14_form == GC_NB14_SCALED
             ? (fp32 ? gcn_kern_nb14<true, float> : gcn_kern_nb14<true, double>)
             : (fp32 ? gcn_kern_nb14<false, float> : gcn_kern_nb14<false, double>))
            <<<GCN_GRID(n), GCN_BLOCK, 0, s_nb14>>>(
            d->force_coord, d->charge, d->cls, d->force_bond_fx, d->acc_bond,
            d->term_pair[GC_TERM_NB14], d->nb14_lj12, d->nb14_lj6,
            d->table_ene, d->table_grad, d->term_param[GC_TERM_NB14],
            d->table_ecor, d->table_decor,
            d->tab.table_form == GC_TABLE_PME_ELEC ? 1 : 3,
            d->tab.num_atom_cls,
            n, d->pitch, sx, sy, sz, d->tab.density, d->cutoff2, sums);

    /* a cutoff run has no reciprocal sum, and GENESIS's cutoff path has no excluded-pair
     * correction */
    n = d->term_count[GC_TERM_EXCL];
    if (d->tab.table_form == GC_TABLE_CUTOFF_CUBIC) n = 0;
    /* a force-only step of the mixed mode: the pairs outside rigid waters
       (excl_keep_launch, gpu_step.cu) */
    const bool keep = !sums && d->excl_keep_valid && n > 0;
    if (keep) n = d->excl_keep_n;
    if (n > 0)
        GCN_LAUNCH(gcn_kern_excluded, n, s_excl,
            d->force_coord, d->charge, d->force_bond_fx, d->acc_bond,
            keep ? d->excl_keep : d->term_pair[GC_TERM_EXCL],
            d->table_ecor, d->table_decor,
            n, d->pitch, sx, sy, sz, d->tab.density, d->cutoff2, sums);

    /* positional restraints: the table the bridge set must be the one the device holds */
    if (d->posres_count != ctx->posres_count) return GC_E_STATE;
    if (d->posres_count > 0 && d->num_owned > 0)
        gcn_kern_posres<<<GCN_GRID(d->num_owned), GCN_BLOCK, 0, d->stream>>>(
            d->coord, d->force_bond_fx, d->acc_bond, d->gid,
            d->posres_key, d->posres_val, d->posres_mask, d->posres_par,
            d->num_owned, d->pitch, sx, sy, sz);

#undef GCN_GRID
#undef GCN_LAUNCH

    if (native_lanes_join(d, GCN_LANES) != GC_OK)
        return GC_E_DEVICE;
    /* the halo return adds the ghosts' returned totals to force_bond next on this stream,
     * as doubles, one landing per slot per launch; one rank has no return, and its join
     * folds force_bond itself */
    if (ctx->nproc > 1 && !native_dist_mixed_words(d))
        force_fold(d->force_bond_fx, d->force_bond, d->pitch, 0, d->pitch,
                   d->stream);
    return dev_launched("force_bonded");
}

/* The owned slots of force_bond from force_bond_fx, on d->stream: after
 * an exact halo return (native_dist_reverse) has added the ghosts' words. */
void native_force_bond_fold_owned(struct gcn_device *d)
{
    force_fold(d->force_bond_fx, d->force_bond, d->pitch, 0, d->num_owned,
               d->stream);
}

gc_status native_force_join_view(gc_context *ctx, struct gcn_join *j)
{
    struct gcn_device *d = ctx->native;
    static unsigned int *flag = 0;
    if (!flag && cudaGetSymbolAddress((void **)&flag, gcn_fx::overflow) !=
                     cudaSuccess)
        return GC_E_DEVICE;
    j->f = d->force.p;
    j->frx = d->force_real_fx;
    j->fbx = ctx->nproc == 1 || native_dist_mixed_words(d) ? d->force_bond_fx
                                                            : 0;
    j->fb = d->force_bond;
    j->fp = d->force_recip;
    j->overflow = flag;
    return GC_OK;
}

static_assert(GC_EXACT_NSLOT == GCN_BE_NSLOT &&
              GC_EXACT_VIRX == GCN_BE_VIRX &&
              GC_EXACT_POSRES == GCN_BE_POSRES &&
              GC_EXACT_PVIRX == GCN_BE_PVIRX &&
              GC_EXACT_ELECCOR == GCN_BE_ELECCOR,
              "gc_exact_slot follows the bonded slots");

/* One fixed-point sum as three integer-valued doubles (gc_exact_slot):
 * the canonical word pair's signed high and low 32 bits of w0, and w1. */
static void exact_pack(gcn_fx::word w0, gcn_fx::word w1, gc_f64 *o)
{
    if (w1 >> 63) {
        o[0] = o[1] = o[2] = (double)NAN;
        return;
    }
    w0 += w1 >> 32;
    w1 &= 0xffffffffull;
    o[0] = (double)((long long)w0 >> 32);
    o[1] = (double)(w0 & 0xffffffffull);
    o[2] = (double)w1;
}

}  /* namespace gcn */

/* Each double of a summed triple is an integer below 2^53 (fewer than 2^21
 * ranks), so the conversions are exact; the words are the sums' modulo
 * 2^64, which value() reads as the one-rank sum would be read. */
extern "C" void gpu_core_exact_decode(const gc_f64 *words, gc_i32 nslot,
                                      gc_f64 *value)
{
    for (gc_i32 k = 0; k < nslot; ++k) {
        const double a = words[3 * k], b = words[3 * k + 1],
                     c = words[3 * k + 2];
        if (!(std::isfinite(a) && std::isfinite(b) && std::isfinite(c))) {
            value[k] = (double)NAN;
            continue;
        }
        const gcn_fx::word w0 = ((gcn_fx::word)(long long)a << 32)
                              + (gcn_fx::word)(long long)b;
        const gcn_fx::word w1 = (gcn_fx::word)(long long)c;
        value[k] = gcn_fx::value<gcn_fx::kEnergy>(w0, w1);
    }
}

namespace gcn {

gc_status native_force_join(gc_context *ctx, gc_i32 want_virial,
                            gc_i32 publish, gc_i32 keep,
                            gc_step_result *out)
{
    struct gcn_device *d = ctx->native;
    gc_i64 n3 = 3 * d->pitch;
    gc_i64 nb = (n3 + GCN_BLOCK - 1) / GCN_BLOCK;
    if (nb > GCN_MAX_BLOCKS) nb = GCN_MAX_BLOCKS;

    /* The owned slots' real-space sum is complete once d->stream has waited for the
     * interior part; one rank has no boundary part, so its fold is every slot (and its
     * bonded fold, native_force_bonded).  When nothing reads the two double fields after
     * the join (`keep` clear), one rank sums the fixed-point fields straight into force. */
    /* the mixed mode's multi-rank return lands in the words too (native_dist_mixed_words):
     * its doubles are folded here, with the ghosts' real-space parts (the virial reads
     * them) */
    const int words = ctx->nproc == 1 || native_dist_mixed_words(d);
    if (words && !keep) {
        vf_each(d->vf32, [&](auto z) {
            gcn_kern_join_force_fx<<<(unsigned)nb, GCN_BLOCK, 0, d->stream>>>(
                d->force.as<decltype(z)>(), d->force_real_fx,
                d->force_bond_fx, d->force_recip, n3);
        });
    } else {
        force_fold(d->force_real_fx, d->force_real, d->pitch, 0,
                   words ? d->pitch : d->num_owned, d->stream);
        if (words)
            force_fold(d->force_bond_fx, d->force_bond, d->pitch, 0,
                       d->pitch, d->stream);
        vf_each(d->vf32, [&](auto z) {
            gcn_kern_join_force<<<(unsigned)nb, GCN_BLOCK, 0, d->stream>>>(
                d->force.as<decltype(z)>(), d->force_real, d->force_bond,
                d->force_recip, n3);
        });
    }
    if (!publish) return dev_launched("force_join");

    /* every resident slot's own real-space force: the halo return adds a ghost's total into
     * its owner's force_bond, not force_real, so the sum of view*force over this rank's
     * slots is its share of the pair virial */
    double cfv[3] = { 0.0, 0.0, 0.0 };
    if (want_virial) {
        gc_i64 rb = (d->num_resident + GCN_RED_BLOCK - 1) / GCN_RED_BLOCK;
        if (rb > GCN_MAX_BLOCKS) rb = GCN_MAX_BLOCKS;
        if (rb < 1) rb = 1;
        gcn_kern_coord_force_virial<<<(unsigned)rb, GCN_RED_BLOCK, 0,
                                      d->stream>>>(
            d->force_coord, d->force_real, d->num_resident, d->pitch,
            d->reduce_partial, (int)rb);
        gc_status st = native_reduce_n(ctx, d->num_resident, cfv, 3);
        if (st != GC_OK) return st;
    }

    /* pull the three small accumulators: seventeen fixed-point sums and four doubles */
    gcn_fx::word rew[2 * GCN_RE_NSLOT], bew[2 * GCN_BE_NSLOT];
    double re[GCN_RE_NSLOT], be[GCN_BE_NSLOT], pe[GCN_PE_NSLOT];
    if (cudaMemcpyAsync(rew, d->acc_real, sizeof(rew), cudaMemcpyDeviceToHost,
                        d->stream) != cudaSuccess) return GC_E_DEVICE;
    if (cudaMemcpyAsync(bew, d->acc_bond, sizeof(bew), cudaMemcpyDeviceToHost,
                        d->stream) != cudaSuccess) return GC_E_DEVICE;
    if (cudaMemcpyAsync(pe, d->acc_recip, sizeof(pe), cudaMemcpyDeviceToHost,
                        d->stream) != cudaSuccess) return GC_E_DEVICE;
    {
        gc_status js = dev_phase(d->stream, "force_join");
        if (js != GC_OK) return js;
    }
    for (int k = 0; k < GCN_RE_NSLOT; ++k)
        re[k] = gcn_fx::value<gcn_fx::kEnergy>(rew[2 * k], rew[2 * k + 1]);
    for (int k = 0; k < GCN_BE_NSLOT; ++k)
        be[k] = gcn_fx::value<gcn_fx::kEnergy>(bew[2 * k], bew[2 * k + 1]);

    std::memset(out->energy, 0, sizeof(out->energy));
    out->energy[GC_ENE_BOND]       = be[GCN_BE_BOND];
    out->energy[GC_ENE_ANGLE]      = be[GCN_BE_ANGLE];
    out->energy[GC_ENE_UREY]       = be[GCN_BE_UREY];
    out->energy[GC_ENE_DIHEDRAL]   = be[GCN_BE_DIHE];
    out->energy[GC_ENE_IMPROPER]   = be[GCN_BE_IMPR];
    out->energy[GC_ENE_CMAP]       = be[GCN_BE_CMAP];
    out->energy[GC_ENE_ELEC14]     = be[GCN_BE_ELEC14];
    out->energy[GC_ENE_VDW14]      = be[GCN_BE_VDW14];
    out->energy[GC_ENE_ELEC_CORR]  = be[GCN_BE_ELECCOR];
    out->energy[GC_ENE_POSRES]     = be[GCN_BE_POSRES];
    out->energy[GC_ENE_ELEC_REAL]  = re[GCN_RE_ELEC];
    out->energy[GC_ENE_VDW_REAL]   = re[GCN_RE_VDW];
    out->energy[GC_ENE_ELEC_RECIP] = pe[GCN_PE_ENE];
    if (d->reciprocal_ready && d->self_epoch != ctx->epoch) {
        d->self_energy = native_pme_self_energy(ctx);
        d->self_epoch  = ctx->epoch;
    }
    out->energy[GC_ENE_ELEC_SELF]  = d->reciprocal_ready ? d->self_energy : 0.0;

    /* The three conventions, joined here and nowhere else:
     *   real space  the periodic-offset correction plus sum coord*force;
     *   bonded      minus the sum of d*w, which is what every GENESIS
     *               bonded routine's `virial = virial - viri` means;
     *   reciprocal  the diagonal already in GENESIS's sign. */
    for (int k = 0; k < 3; ++k)
        out->virial[k] = re[GCN_RE_VIRX + k] + cfv[k]
                       - be[GCN_BE_VIRX + k] + pe[GCN_PE_VIRX + k];
    for (int k = 0; k < 3; ++k)
        out->virial_ext[k] = be[GCN_BE_PVIRX + k];
    for (int k = 0; k < 3; ++k)
        out->virial_nb[k] = re[GCN_RE_VIRX + k] + cfv[k] + pe[GCN_PE_VIRX + k];
    for (int k = 0; k < GCN_BE_NSLOT; ++k)
        exact_pack(bew[2 * k], bew[2 * k + 1], out->bonded_exact + 3 * k);

    return GC_OK;
}

}  /* namespace gcn */
