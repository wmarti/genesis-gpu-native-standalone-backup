/* gpu_step.cu : the native velocity-Verlet step, its constraints, its
 * observables and the ABI entry points that drive them. Kick, drift, group
 * rescale, SETTLE, RATTLE and SHAKE follow stock spdyn (sp_md_vverlet.fpp,
 * sp_constraints.fpp) expression for expression: the rescale-then-kick-then-
 * drift order, the two kinetic conventions, SETTLE's rotation order, iteration
 * tolerances and virial signs are unchanged. */

#include "gpu_core_native.h"
#include "gpu_nbcluster.h"
#include "gpu_fixed_sum.cuh"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#ifdef HAVE_MPI_GENESIS
#include <mpi.h>
#endif

/* The fixed-order final reduction: thread c adds the per-block partials of
 * quantity c (part[c*nblk + b]) in increasing block order, so the result does
 * not depend on which block finished first. */
#define GCN_FIN_TILE 2048
__device__ __forceinline__ void reduce_finalize_block(
    const gc_f64 *__restrict__ part, int nblk, int ncomp,
    gc_f64 *__restrict__ out, double *tile)
{
    const int per = GCN_FIN_TILE / ncomp;
    double s = 0.0;
    for (int b0 = 0; b0 < nblk; b0 += per) {
        const int nb = min(per, nblk - b0);
        for (int k = threadIdx.x; k < ncomp * nb; k += blockDim.x) {
            const int c = k / nb, j = k - c * nb;
            tile[c * per + j] = __ldcg(&part[(gc_i64)c * nblk + b0 + j]);
        }
        __syncthreads();
        if ((int)threadIdx.x < ncomp)
            for (int j = 0; j < nb; ++j) s += tile[threadIdx.x * per + j];
        __syncthreads();
    }
    if ((int)threadIdx.x < ncomp) out[threadIdx.x] = s;
}

__global__ void gcn_kern_reduce_finalize(const gc_f64 *__restrict__ part,
                                         int nblk, int ncomp,
                                         gc_f64 *__restrict__ out)
{
    __shared__ double tile[GCN_FIN_TILE];
    reduce_finalize_block(part, nblk, ncomp, out, tile);
}

/* a / b rounded to nearest, from r near 1/b: a*r corrected once by its
 * remainder, accepted only when the corrected q's exact remainder is under
 * half an ulp of q times |b| (an integer test on the bit patterns); q with a
 * zero mantissa, or q or b outside [2^-400, 2^401), takes the division. The
 * result equals the division bit for bit; one reciprocal serves several
 * quotients by the same b. */
__device__ __forceinline__ double div_rn_by(double a, double b, double r)
{
    const double q0 = a * r;
    const double q = __fma_rn(__fma_rn(-q0, b, a), r, q0);
    const long long e = __double_as_longlong(__fma_rn(-q, b, a)) &
                        0x7fffffffffffffffLL;
    const long long iq = __double_as_longlong(q) & 0x7fffffffffffffffLL;
    const long long ib = __double_as_longlong(b) & 0x7fffffffffffffffLL;
    const long long eq = iq >> 52, eb = ib >> 52;
    if ((iq & 0xfffffffffffffLL) != 0 && eq >= 623 && eq <= 1423 &&
        eb >= 623 && eb <= 1423 &&
        e < ib + ((eq - 1023 - 53) << 52))
        return q;
    return __ddiv_rn(a, b);
}

template <class S>
__global__ void gcn_kern_save_ref(gc_f64 *__restrict__ xref,
                                  gc_f64 *__restrict__ vref,
                                  const gc_f64 *__restrict__ x,
                                  const S *__restrict__ v,
                                  gc_i64 n, gc_i64 pitch)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x)
        for (int c = 0; c < 3; ++c) {
            xref[(gc_i64)c * pitch + s] = x[(gc_i64)c * pitch + s];
            vref[(gc_i64)c * pitch + s] = v[(gc_i64)c * pitch + s];
        }
}

/* nve_vv1's energy-output preparation: velocity_half <- half_dt/m * force.
 * It really does overwrite velocity_half. */
template <class S>
__global__ void gcn_kern_kin_prep(gc_f64 *__restrict__ vhalf,
                                  const S *__restrict__ f,
                                  const gc_f64 *__restrict__ minv,
                                  double half_dt, gc_i64 n, gc_i64 pitch)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double factor = half_dt * minv[s];
        for (int c = 0; c < 3; ++c)
            vhalf[(gc_i64)c * pitch + s] = factor * f[(gc_i64)c * pitch + s];
    }
}

/* Generic NVT VV1 uses the preceding VV2 half velocity minus the current
 * velocity for its thermostat kinetic term.  VV2 refreshes vhalf later. */
template <class S>
__global__ void gcn_kern_nvt_half(gc_f64 *__restrict__ vhalf,
                                  const S *__restrict__ v,
                                  gc_i64 n, gc_i64 pitch)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x)
        for (int c = 0; c < 3; ++c) {
            const gc_i64 at = (gc_i64)c * pitch + s;
            vhalf[at] -= v[at];
        }
}

/* calc_kinetic: the per-axis sum of mass * v * v, formed as (mass*v)*v in
 * the CPU's source order. */
template <class S>
__global__ void gcn_kern_kinetic_flat(const S *__restrict__ v,
                                      const gc_f64 *__restrict__ m,
                                      gc_i64 n, gc_i64 pitch,
                                      gc_f64 *__restrict__ part, int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double k[3] = { 0.0, 0.0, 0.0 };
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double mm = m[s];
        for (int c = 0; c < 3; ++c) {
            double vv = v[(gc_i64)c * pitch + s];
            k[c] += (mm * vv) * vv;
        }
    }
    block_reduce_sum(k, 3, sh, part, nblk);
}

/* The two local sums compute_dynvars forms after VV1.  Preserve the stock
 * per-atom expression order; only the cross-atom reduction tree differs. */
template <class S>
__global__ void gcn_kern_dynvars_sums(const S *__restrict__ force,
                                     const gc_f64 *__restrict__ vref,
                                     const gc_f64 *__restrict__ mass,
                                     gc_i64 n, gc_i64 pitch,
                                     gc_f64 *__restrict__ part, int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double sums[2] = { 0.0, 0.0 };
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        const double fx = force[s], fy = force[pitch + s];
        const double fz = force[2 * pitch + s];
        const double vx = vref[s], vy = vref[pitch + s];
        const double vz = vref[2 * pitch + s];
        sums[0] += (fx * fx + fy * fy) + fz * fz;
        sums[1] += mass[s] * ((vx * vx + vy * vy) + vz * vz);
    }
    block_reduce_sum(sums, 2, sh, part, nblk);
}

/* compute_kin_group: a free solute atom contributes m*v*v directly and is
 * deliberately NOT folded into the group's mass-weighted centre form,
 * because the CPU reference keeps the two apart. */
template <class S>
__global__ void gcn_kern_kinetic_group(const S *__restrict__ v,
                                       const gc_f64 *__restrict__ m,
                                       const gc_i64 *__restrict__ goff,
                                       const gc_i32 *__restrict__ gmem,
                                       const gc_u8 *__restrict__ gkind,
                                       gc_i64 ngroup, gc_i64 pitch,
                                       gc_f64 *__restrict__ part, int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double k[3] = { 0.0, 0.0, 0.0 };

    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        gc_i64 b = goff[g], e = goff[g + 1];
        if (gkind[g] == GC_GROUP_SINGLE) {
            int s = gmem[b];
            double mm = m[s];
            for (int c = 0; c < 3; ++c) {
                double vv = v[(gc_i64)c * pitch + s];
                k[c] += (mm * vv) * vv;
            }
        } else {
            double tm = 0.0, cf[3] = { 0.0, 0.0, 0.0 };
            for (gc_i64 i = b; i < e; ++i) {
                int s = gmem[i];
                double mm = m[s];
                tm += mm;
                for (int c = 0; c < 3; ++c)
                    cf[c] += mm * v[(gc_i64)c * pitch + s];
            }
            for (int c = 0; c < 3; ++c) {
                cf[c] /= tm;
                k[c] += (tm * cf[c]) * cf[c];
            }
        }
    }
    block_reduce_sum(k, 3, sh, part, nblk);
}

/* Rescale, then kick, then drift: the rescale uses the pre-kick velocity and
 * the drift the post-kick one. The rescale is its own launch (grid-wide
 * ordering); skipping it for a unit factor is bit-identical. The group arm
 * rescales only a rigid group's centre-of-mass motion (update_vel_group). */
template <class S>
__global__ void gcn_kern_rescale(S *__restrict__ v,
                                 const gc_f64 *__restrict__ m,
                                 const gc_i64 *__restrict__ goff,
                                 const gc_i32 *__restrict__ gmem,
                                 const gc_u8 *__restrict__ gkind,
                                 gc_i64 ngroup, gc_i64 n, gc_i64 pitch,
                                 double scale,
                                 const double *__restrict__ scale_dev,
                                 int group_tp)
{
    if (scale_dev) {
        scale = *scale_dev;
        if (scale == 1.0) return;
    }
    if (group_tp) {
        double sm1 = scale - 1.0;
        for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
             g < ngroup; g += (gc_i64)gridDim.x * blockDim.x) {
            gc_i64 b = goff[g], e = goff[g + 1];
            if (gkind[g] == GC_GROUP_SINGLE) {
                int s = gmem[b];
                for (int c = 0; c < 3; ++c)
                    v[(gc_i64)c * pitch + s] *= scale;
            } else {
                double tm = 0.0, cm[3] = { 0.0, 0.0, 0.0 };
#pragma unroll 3
                for (gc_i64 i = b; i < e; ++i) {
                    int s = gmem[i];
                    double mm = m[s];
                    tm += mm;
                    for (int c = 0; c < 3; ++c)
                        cm[c] += mm * v[(gc_i64)c * pitch + s];
                }
                const double rt = 1.0 / tm;
                for (int c = 0; c < 3; ++c) cm[c] = div_rn_by(cm[c], tm, rt);
#pragma unroll 3
                for (gc_i64 i = b; i < e; ++i) {
                    int s = gmem[i];
                    for (int c = 0; c < 3; ++c)
                        v[(gc_i64)c * pitch + s] += sm1 * cm[c];
                }
            }
        }
    } else {
        for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
             s < n; s += (gc_i64)gridDim.x * blockDim.x)
            for (int c = 0; c < 3; ++c)
                v[(gc_i64)c * pitch + s] *= scale;
    }
}

/* This rank's kinetic sums as fixed-point words, so that the sum over the
 * ranks is exact integer arithmetic and the same bits in any order. */
__device__ __forceinline__ void thermo_encode(struct gcn_thermo *t, int c)
{
    t->words[2 * c] = t->words[2 * c + 1] = 0;
    gcn_fx::add<gcn_fx::kEnergy>(&t->words[2 * c], t->kin_local[c]);
}

__global__ void gcn_kern_thermo_encode(struct gcn_thermo *t)
{
    if (threadIdx.x < 6) thermo_encode(t, threadIdx.x);
}

/* vel_scale_nhc (sp_md_vverlet.fpp (vel_scale_nhc)) on the chain the device
 * keeps; the step sizes and masses are the host's own values. */
__device__ static double thermo_nhc(struct gcn_thermo *t,
                                    const struct gcn_thermo_args &a,
                                    double ekin)
{
    const int n = a.nh_length;
    double *vel = t->nh_vel, *frc = t->nh_force, *cf = t->nh_coef;
    const double *ms = a.nh_mass;
    const double kbt = a.kbt;
    double scale = 1.0, ekf = 2.0 * ekin;
    for (int i = 0; i < a.nh_step; ++i)
        for (int j = 0; j < 3; ++j) {
            const double dt_1 = a.nh_dt[4 * j], dt_2 = a.nh_dt[4 * j + 1];
            const double dt_4 = a.nh_dt[4 * j + 2], dt_8 = a.nh_dt[4 * j + 3];
            frc[n - 1] = ms[n - 2] * vel[n - 2] * vel[n - 2] - kbt;
            frc[n - 1] = frc[n - 1] / ms[n - 1];
            vel[n - 1] = vel[n - 1] + frc[n - 1] * dt_4;
            for (int k = n - 2; k >= 1; --k) {
                frc[k] = ms[k - 1] * vel[k - 1] * vel[k - 1] - kbt;
                frc[k] = frc[k] / ms[k];
                cf[k]  = exp(-vel[k + 1] * dt_8);
                vel[k] = vel[k] * cf[k];
                vel[k] = vel[k] + frc[k] * dt_4;
                vel[k] = vel[k] * cf[k];
            }
            frc[0] = ekf - a.degree * kbt;
            frc[0] = frc[0] / ms[0];
            cf[0]  = exp(-vel[1] * dt_8);
            vel[0] = vel[0] * cf[0];
            vel[0] = vel[0] + frc[0] * dt_4;
            vel[0] = vel[0] * cf[0];
            const double scale_kin = exp(-vel[0] * dt_1);
            scale = scale * exp(-vel[0] * dt_2);
            ekf = ekf * scale_kin;
            frc[0] = (ekf - a.degree * kbt) / ms[0];
            vel[0] = vel[0] * cf[0];
            vel[0] = vel[0] + frc[0] * dt_4;
            vel[0] = vel[0] * cf[0];
            for (int k = 1; k <= n - 2; ++k) {
                frc[k] = ms[k - 1] * vel[k - 1] * vel[k - 1] - kbt;
                frc[k] = frc[k] / ms[k];
                vel[k] = vel[k] * cf[k];
                vel[k] = vel[k] + frc[k] * dt_4;
                vel[k] = vel[k] * cf[k];
            }
            frc[n - 1] = ms[n - 2] * vel[n - 2] * vel[n - 2] - kbt;
            frc[n - 1] = frc[n - 1] / ms[n - 1];
            vel[n - 1] = vel[n - 1] + frc[n - 1] * dt_4;
        }
    return scale;
}

/* The tick: global kinetic tensors, GENESIS's thermostat kinetic energy and
 * the scale (vel_scale_bussi / _berendsen / _nhc) formed from it. Every rank
 * runs this on the same words and draws, so all hold the same scale without a
 * broadcast. */
__device__ static void thermo_tick(struct gcn_thermo *t,
                                   const struct gcn_thermo_args &a,
                                   int summed)
{
    double k[6];
    for (int c = 0; c < 6; ++c)
        k[c] = summed ? gcn_fx::value<gcn_fx::kEnergy>(t->words[2 * c],
                                                       t->words[2 * c + 1])
                      : t->kin_local[c];
    for (int c = 0; c < 6; ++c) t->kin[c] = k[c];
    const double ekin_half = 0.5 * (k[0] + k[1] + k[2]);
    const double ekin_full = 0.5 * (k[3] + k[4] + k[5]);
    const double ekin = ekin_full + 2.0 * ekin_half / 3.0;
    double scale = 1.0;
    if (a.kind == GC_THERMOSTAT_BUSSI) {
        if (t->next >= t->count) {          /* no draw: fail loudly */
            t->scale = (double)NAN;
            return;
        }
        const double rr = t->draw[2 * t->next];
        const double sg = t->draw[2 * t->next + 1];
        t->next++;
        const double tempf = 2.0 * ekin / (a.degree * a.kboltz);
        const double tempt = tempf * a.factor
            + a.temp0 / a.degree * (1.0 - a.factor) * (sg + rr * rr)
            + 2.0 * sqrt(tempf * a.temp0 / a.degree * (1.0 - a.factor)
                         * a.factor) * rr;
        scale = sqrt(tempt / tempf);
    } else if (a.kind == GC_THERMOSTAT_BERENDSEN) {
        const double tempf = 2.0 * ekin / (a.degree * a.kboltz);
        scale = sqrt(1.0 + a.dt_tau * (a.temp0 / tempf - 1.0));
    } else if (a.kind == GC_THERMOSTAT_NHC) {
        scale = thermo_nhc(t, a, ekin);
    }
    t->scale = scale;
}

__global__ void gcn_kern_thermostat(struct gcn_thermo *t,
                                    struct gcn_thermo_args a, int summed)
{
    thermo_tick(t, a, summed);
}

/* A thermostat tick's device work in one pass (GROUP: compute_kin_group, else
 * calc_kinetic): step_begin's velocity save, nvt_half's velocity_half <-
 * velocity_half - velocity, and per-block partials of both kinetic tensors.
 * The last block to finish forms the fixed-order sums into kin_local and runs
 * the tick, or encodes this rank's words for the sum over ranks (run by the
 * same block when a device route exists). Which block is last does not change
 * any sum. */
__device__ __forceinline__ void tick_finish(struct gcn_thermo *t,
                                            const struct gcn_thermo_args &a,
                                            int encode,
                                            const struct gcx_wsum_view &ws)
{
    if (encode) {
        if (threadIdx.x < 6) thermo_encode(t, threadIdx.x);
        if (!ws.boards && !ws.post) return;
        __syncthreads();
        gcx_wsum_sum((gc_u64 *)t->words, ws);
        __syncthreads();
        if (threadIdx.x == 0) thermo_tick(t, a, 1);
    } else if (threadIdx.x == 0) {
        thermo_tick(t, a, 0);
    }
}

template <int GROUP, class S>
__global__ void __launch_bounds__(GCN_RED_BLOCK)
gcn_kern_tick_kinetic(const S *__restrict__ v,
                      gc_f64 *__restrict__ vref,
                      gc_f64 *__restrict__ vhalf,
                      const gc_f64 *__restrict__ m,
                      const gc_i64 *__restrict__ goff,
                      const gc_i32 *__restrict__ gmem,
                      const gc_u8 *__restrict__ gkind,
                      gc_i64 nitem, gc_i64 pitch,
                      gc_f64 *__restrict__ part, int nblk,
                      struct gcn_thermo *t, struct gcn_thermo_args a,
                      int encode, const struct gcx_wsum_view ws)
{
    __shared__ double sh[GCN_FIN_TILE];
    __shared__ int last;
    double k[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < nitem;
         g += (gc_i64)gridDim.x * blockDim.x) {
        if (!GROUP || gkind[g] == GC_GROUP_SINGLE) {
            const int s = GROUP ? gmem[goff[g]] : (int)g;
            const double mm = m[s];
            for (int c = 0; c < 3; ++c) {
                const gc_i64 at = (gc_i64)c * pitch + s;
                const double vv = v[at];
                const double vh = vhalf[at] - vv;
                vref[at] = vv;
                vhalf[at] = vh;
                k[c]     += (mm * vh) * vh;
                k[3 + c] += (mm * vv) * vv;
            }
        } else {
            const gc_i64 b = goff[g], e = goff[g + 1];
            double tm = 0.0, ch[3] = { 0.0, 0.0, 0.0 };
            double cr[3] = { 0.0, 0.0, 0.0 };
#pragma unroll 3
            for (gc_i64 i = b; i < e; ++i) {
                const int s = gmem[i];
                const double mm = m[s];
                tm += mm;
                for (int c = 0; c < 3; ++c) {
                    const gc_i64 at = (gc_i64)c * pitch + s;
                    const double vv = v[at];
                    const double vh = vhalf[at] - vv;
                    vref[at] = vv;
                    vhalf[at] = vh;
                    ch[c] += mm * vh;
                    cr[c] += mm * vv;
                }
            }
            const double rt = 1.0 / tm;
            for (int c = 0; c < 3; ++c) {
                ch[c] = div_rn_by(ch[c], tm, rt);
                cr[c] = div_rn_by(cr[c], tm, rt);
                k[c]     += (tm * ch[c]) * ch[c];
                k[3 + c] += (tm * cr[c]) * cr[c];
            }
        }
    }
    block_reduce_sum(k, 6, sh, part, nblk);

    if (threadIdx.x == 0) {
        __threadfence();
        last = atomicAdd(&t->tick_blocks, 1u) == (unsigned)(nblk - 1);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    reduce_finalize_block(part, nblk, 6, t->kin_local, sh);
    __syncthreads();
    if (threadIdx.x == 0) t->tick_blocks = 0;
    tick_finish(t, a, encode, ws);
}

/* A tick whose pass the last VV2's solvers took (gcn_tick_prep, mixed mode):
 * sums their per-block partials in fixed order (thread t adds blocks t, t +
 * blockDim, ...; block_reduce_sum adds the threads), then runs the tick; one
 * block. */
__global__ void __launch_bounds__(GCN_RED_BLOCK)
gcn_kern_tick_finalize(const gc_f64 *__restrict__ part, int nblk,
                       struct gcn_thermo *t, struct gcn_thermo_args a,
                       int encode, const struct gcx_wsum_view ws)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double k[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (int b = threadIdx.x; b < nblk; b += blockDim.x)
        for (int c = 0; c < 6; ++c) k[c] += __ldcg(&part[(gc_i64)c * nblk + b]);
    block_reduce_sum(k, 6, sh, t->kin_local, 1);
    __syncthreads();
    tick_finish(t, a, encode, ws);
}

/* nve_vv1's kick then drift of atom s: v kicked in place, xr the
 * pre-drift coordinate (coord_ref), x the drifted one */
template <class S>
__device__ __forceinline__ void kick_drift_atom(double v[3], double xr[3],
                                                double x[3],
                                                const S *__restrict__ f,
                                                const gc_f64 *__restrict__ minv,
                                                gc_i64 s, gc_i64 pitch,
                                                double dt, double half_dt)
{
    const double factor = half_dt * minv[s];
    for (int c = 0; c < 3; ++c) {
        const double vv = v[c] + factor * f[(gc_i64)c * pitch + s];
        v[c] = vv;
        x[c] = xr[c] + dt * vv;
    }
}

/* The force of atom slot s: read (JOIN 0), or formed as native_force_join
 * forms it (JOIN 1): real, bonded, reciprocal. All loads precede any store;
 * force3_store writes the formed force once a group's atoms are read. Only
 * owned slots are formed; nothing reads a ghost slot's joined force. */
template <int JOIN, class S>
__device__ __forceinline__ void force3_load(const S *__restrict__ f,
                                            const struct gcn_join &j,
                                            unsigned int ovf, gc_i64 s,
                                            gc_i64 pitch, double fs[3])
{
    for (int c = 0; c < 3; ++c) {
        const gc_i64 at = (gc_i64)c * pitch + s;
        if (!JOIN) { fs[c] = f[at]; continue; }
        const double fr =
            gcn_fx::value1_of<gcn_fx::kForce1>(__ldg(j.frx + at), ovf);
        const double fb = j.fbx
            ? gcn_fx::value1_of<gcn_fx::kForce1>(__ldg(j.fbx + at), ovf)
            : __ldg(j.fb + at);
        fs[c] = fr + fb + __ldg(j.fp + at);
    }
}

template <int JOIN, class S>
__device__ __forceinline__ void force3_store(const struct gcn_join &j,
                                             gc_i64 s, gc_i64 pitch,
                                             const double fs[3])
{
    if (JOIN)
        for (int c = 0; c < 3; ++c)
            ((S *)j.f)[(gc_i64)c * pitch + s] = (S)fs[c];
}

/* The next tick's pass taken into a VV2 solver pass (mixed mode,
 * GC_CONSTRAIN_TICK_NEXT): part, when non-null, receives the solvers' kinetic
 * partials from block `off` of `nblk`. With keep clear neither velocity_ref
 * nor velocity_half is stored; the tick takes its kinetic terms from the
 * partials. */
struct gcn_tick_prep {
    gc_f64       *vref;
    gc_f64       *part;
    const gc_f64 *mass;
    int           off, nblk, group_tp, keep;
    struct gcn_thermo     *t;
    struct gcn_thermo_args a;
    int                    encode;
    struct gcx_wsum_view   ws;
};

/* thermo_tick, or, for a Bussi tick whose draw is not uploaded yet, a note
 * that gcn_kern_tick_pending runs it after the upload */
__device__ __forceinline__ void tick_or_defer(struct gcn_thermo *t,
                                              const struct gcn_thermo_args &a,
                                              int summed)
{
    if (a.kind == GC_THERMOSTAT_BUSSI && t->next >= t->count) {
        t->pending = 1u + (unsigned)summed;
        return;
    }
    thermo_tick(t, a, summed);
}

__global__ void gcn_kern_tick_pending(struct gcn_thermo *t,
                                      struct gcn_thermo_args a)
{
    if (!t->pending) return;
    const int summed = (int)t->pending - 1;
    t->pending = 0;
    thermo_tick(t, a, summed);
}

/* gcn_kern_tick_finalize taken into the VV2 solver passes: the last of tp.nblk
 * blocks to finish sums the partials and runs the tick; sums are formed by
 * this block's threads and added in block_reduce_sum's tree; sh holds
 * GCN_RED_BLOCK doubles. */
__device__ __forceinline__ void tick_fold(const struct gcn_tick_prep &tp,
                                          double *sh)
{
    __shared__ int last;
    if (!tp.t) return;
    if (threadIdx.x == 0) {
        __threadfence();
        last = atomicAdd(&tp.t->tick_blocks, 1u) == (unsigned)(tp.nblk - 1);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    struct gcn_thermo *t = tp.t;
    for (int c = 0; c < 6; ++c) {
        for (int v = threadIdx.x; v < GCN_RED_BLOCK; v += blockDim.x) {
            double k = 0.0;
            for (int b = v; b < tp.nblk; b += GCN_RED_BLOCK)
                k += __ldcg(&tp.part[(gc_i64)c * tp.nblk + b]);
            sh[v] = k;
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            double w[GCN_RED_BLOCK / 32];
#pragma unroll
            for (int j = 0; j < GCN_RED_BLOCK / 32; ++j)
                w[j] = sh[threadIdx.x + 32 * j];
#pragma unroll
            for (int h = GCN_RED_BLOCK / 64; h > 0; h >>= 1)
#pragma unroll
                for (int j = 0; j < h; ++j) w[j] += w[j + h];
            double r = w[0];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                r += __shfl_down_sync(0xffffffffu, r, o);
            if (threadIdx.x == 0) t->kin_local[c] = r;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) t->tick_blocks = 0;
    if (tp.encode) {
        if (threadIdx.x < 6) thermo_encode(t, threadIdx.x);
        __syncthreads();
        gcx_wsum_sum((gc_u64 *)t->words, tp.ws);
        __syncthreads();
        if (threadIdx.x == 0) tick_or_defer(t, tp.a, 1);
    } else if (threadIdx.x == 0) {
        tick_or_defer(t, tp.a, 0);
    }
}

/* gcn_kern_tick_kinetic's work on one group of n members in registers: vel(k,
 * c) the post-constraint velocity, pre(k, c) the pre-kick one VV2 saves as
 * velocity_half; the tick's save, half velocity and kinetic terms. */
template <int N, class V, class P>
__device__ __forceinline__ void tick_members(const struct gcn_tick_prep &tp,
                                             const int *s, int n,
                                             gc_f64 *__restrict__ vhalf,
                                             gc_i64 pitch, V &&vel, P &&pre,
                                             double k[6],
                                             const double *mp = 0)
{
    if (!tp.group_tp || n == 1) {
        for (int i = 0; i < N && i < n; ++i) {
            const double mm = mp ? mp[i] : __ldg(tp.mass + s[i]);
            for (int c = 0; c < 3; ++c) {
                const gc_i64 at = (gc_i64)c * pitch + s[i];
                const double vv = vel(i, c);
                const double vh = pre(i, c) - vv;
                if (tp.keep) { tp.vref[at] = vv; vhalf[at] = vh; }
                k[c]     += (mm * vh) * vh;
                k[3 + c] += (mm * vv) * vv;
            }
        }
        return;
    }
    double tm = 0.0, ch[3] = { 0.0, 0.0, 0.0 };
    double cr[3] = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < N && i < n; ++i) {
        const double mm = mp ? mp[i] : __ldg(tp.mass + s[i]);
        tm += mm;
        for (int c = 0; c < 3; ++c) {
            const gc_i64 at = (gc_i64)c * pitch + s[i];
            const double vv = vel(i, c);
            const double vh = pre(i, c) - vv;
            if (tp.keep) { tp.vref[at] = vv; vhalf[at] = vh; }
            ch[c] += mm * vh;
            cr[c] += mm * vv;
        }
    }
    const double rt = 1.0 / tm;
    for (int c = 0; c < 3; ++c) {
        ch[c] = div_rn_by(ch[c], tm, rt);
        cr[c] = div_rn_by(cr[c], tm, rt);
        k[c]     += (tm * ch[c]) * ch[c];
        k[3 + c] += (tm * cr[c]) * cr[c];
    }
}

/* nve_vv2's half kick of atom s from its factor half_dt * minv[s] and
 * force fs, velocity_half saved first (with TICK into pre instead: the
 * tick pass stores it after the constraint) */
template <int TICK = 0>
__device__ __forceinline__ void vv2_kick(double v[3],
                                         gc_f64 *__restrict__ vhalf,
                                         gc_f64 *__restrict__ vfull,
                                         double factor, const double fs[3],
                                         gc_i64 s, gc_i64 pitch, int full,
                                         double *pre = 0)
{
    for (int c = 0; c < 3; ++c) {
        const gc_i64 at = (gc_i64)c * pitch + s;
        double vv = v[c];
        if (TICK) pre[c] = vv; else vhalf[at] = vv;
        vv += factor * fs[c];
        v[c] = vv;
        if (full) vfull[at] = vv;
    }
}

/* nve_vv2's half kick of atom s: its reads, then vv2_kick and the join's
 * store */
template <int JOIN, int TICK = 0, class S>
__device__ __forceinline__ void vv2_atom(double v[3],
                                         gc_f64 *__restrict__ vhalf,
                                         gc_f64 *__restrict__ vfull,
                                         const S *__restrict__ f,
                                         const struct gcn_join &j,
                                         unsigned int ovf,
                                         const gc_f64 *__restrict__ minv,
                                         gc_i64 s, gc_i64 pitch,
                                         double half_dt, int full,
                                         double *pre = 0)
{
    const double factor = half_dt * minv[s];
    double fs[3];
    force3_load<JOIN>(f, j, ovf, s, pitch, fs);
    vv2_kick<TICK>(v, vhalf, vfull, factor, fs, s, pitch, full, pre);
    force3_store<JOIN, S>(j, s, pitch, fs);
}

template <class S>
__global__ void gcn_kern_kick_drift(S *__restrict__ v,
                                    gc_f64 *__restrict__ x,
                                    gc_f64 *__restrict__ xref,
                                    const S *__restrict__ f,
                                    const gc_f64 *__restrict__ minv,
                                    gc_i64 n, gc_i64 pitch,
                                    double dt, double half_dt)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double vv[3], xr[3], xn[3];
        for (int c = 0; c < 3; ++c) {
            vv[c] = v[(gc_i64)c * pitch + s];
            xr[c] = x[(gc_i64)c * pitch + s];
        }
        kick_drift_atom(vv, xr, xn, f, minv, s, pitch, dt, half_dt);
        for (int c = 0; c < 3; ++c) {
            v[(gc_i64)c * pitch + s]    = vv[c];
            xref[(gc_i64)c * pitch + s] = xr[c];
            x[(gc_i64)c * pitch + s]    = xn[c];
        }
    }
}

/* The iterated constrained-thermostat sweep: velocity and position are rebuilt
 * from the references each time, so it can rerun with a new scale after the
 * constraints moved them (sp_md_vverlet.fpp (vel_rescaling_thermostat_vv1_cons)). */
template <class S>
__global__ void gcn_kern_vv1_from_ref(S *__restrict__ v,
                                      const gc_f64 *__restrict__ vref,
                                      gc_f64 *__restrict__ x,
                                      const gc_f64 *__restrict__ xref,
                                      const S *__restrict__ f,
                                      const gc_f64 *__restrict__ minv,
                                      gc_i64 n, gc_i64 pitch,
                                      double scale, double dt, double half_dt)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double factor = half_dt * minv[s];
        for (int c = 0; c < 3; ++c) {
            double vv = scale * vref[(gc_i64)c * pitch + s]
                      + factor * f[(gc_i64)c * pitch + s];
            v[(gc_i64)c * pitch + s] = vv;
            x[(gc_i64)c * pitch + s] = xref[(gc_i64)c * pitch + s] + dt * vv;
        }
    }
}

/* nve_vv2: save velocity_half, half kick, and under rigid_bond save
 * velocity_full before the RATTLE that follows. */
template <class S>
__global__ void gcn_kern_vv2(S *__restrict__ v,
                             gc_f64 *__restrict__ vhalf,
                             gc_f64 *__restrict__ vfull,
                             const S *__restrict__ f,
                             const gc_f64 *__restrict__ minv,
                             double half_dt, gc_i64 n, gc_i64 pitch, int rigid)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double vv[3];
        for (int c = 0; c < 3; ++c) vv[c] = v[(gc_i64)c * pitch + s];
        vv2_atom<0>(vv, vhalf, vfull, f, gcn_join(), 0u, minv, s, pitch,
                    half_dt, rigid);
        for (int c = 0; c < 3; ++c) v[(gc_i64)c * pitch + s] = vv[c];
    }
}

/* r-RESPA (sp_md_respa.fpp) kicks. The short force is real space plus bonded,
 * the long force the reciprocal sum, each in its own accumulator. The long
 * kick comes first and only at the outer boundary (half_dt_long > 0). VV1
 * drifts from the saved reference; VV2 keeps velocity_full for its RATTLE. */
template <class S>
__global__ void gcn_kern_respa_kick(S *__restrict__ v,
                                    gc_f64 *__restrict__ x,
                                    gc_f64 *__restrict__ xref,
                                    gc_f64 *__restrict__ vfull,
                                    const gc_f64 *__restrict__ fr,
                                    const gc_f64 *__restrict__ fb,
                                    const gc_f64 *__restrict__ fp,
                                    const gc_f64 *__restrict__ minv,
                                    gc_i64 n, gc_i64 pitch, double dt,
                                    double half_dt, double half_dt_long,
                                    int drift, int rigid)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double fl = half_dt_long * minv[s];
        double fs = half_dt * minv[s];
        for (int c = 0; c < 3; ++c) {
            gc_i64 k = (gc_i64)c * pitch + s;
            double vv = v[k];
            if (half_dt_long != 0.0) vv += fl * fp[k];
            vv += fs * (fr[k] + fb[k]);
            v[k] = vv;
            if (drift) {
                double xr = x[k];
                xref[k] = xr;
                x[k] = xr + dt * vv;
            } else if (rigid) {
                vfull[k] = vv;
            }
        }
    }
}

/* nve_vv2's constraint virial: the diagonal sum of
 * mass*(vel - vel_full)*coord/half_dt.  The caller halves it, as the CPU
 * does. */
template <class S>
__global__ void gcn_kern_vv2_virial(const S *__restrict__ v,
                                    const gc_f64 *__restrict__ vfull,
                                    const gc_f64 *__restrict__ x,
                                    const gc_f64 *__restrict__ m,
                                    double half_dt, gc_i64 n, gc_i64 pitch,
                                    gc_f64 *__restrict__ part, int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double k[3] = { 0.0, 0.0, 0.0 };
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double mm = m[s];
        for (int c = 0; c < 3; ++c) {
            double dv = v[(gc_i64)c * pitch + s] - vfull[(gc_i64)c * pitch + s];
            k[c] += mm * dv * x[(gc_i64)c * pitch + s] / half_dt;
        }
    }
    block_reduce_sum(k, 3, sh, part, nblk);
}

/* compute_virial_group: over rigid groups only, the sum of
 * (r - r_cm) * f from the reference coordinates.  The caller subtracts. */
template <class S>
__global__ void gcn_kern_group_virial(const gc_f64 *__restrict__ xref,
                                      const S *__restrict__ f,
                                      const gc_f64 *__restrict__ m,
                                      const gc_i64 *__restrict__ goff,
                                      const gc_i32 *__restrict__ gmem,
                                      const gc_u8 *__restrict__ gkind,
                                      gc_i64 ngroup, gc_i64 pitch,
                                      gc_f64 *__restrict__ part, int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double k[3] = { 0.0, 0.0, 0.0 };

    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        if (gkind[g] == GC_GROUP_SINGLE) continue;
        gc_i64 b = goff[g], e = goff[g + 1];
        double tm = 0.0, rcm[3] = { 0.0, 0.0, 0.0 };
        for (gc_i64 i = b; i < e; ++i) {
            int s = gmem[i];
            double mm = m[s];
            tm += mm;
            for (int c = 0; c < 3; ++c)
                rcm[c] += mm * xref[(gc_i64)c * pitch + s];
        }
        for (int c = 0; c < 3; ++c) rcm[c] /= tm;
        for (gc_i64 i = b; i < e; ++i) {
            int s = gmem[i];
            for (int c = 0; c < 3; ++c)
                k[c] += (xref[(gc_i64)c * pitch + s] - rcm[c])
                      * f[(gc_i64)c * pitch + s];
        }
    }
    block_reduce_sum(k, 3, sh, part, nblk);
}

template <class S>
__global__ void gcn_kern_com_partials(const gc_f64 *__restrict__ x,
                                      const S *__restrict__ v,
                                      const gc_f64 *__restrict__ m,
                                      gc_i64 n, gc_i64 pitch,
                                      gc_f64 *__restrict__ part, int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double k[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double mm = m[s];
        for (int c = 0; c < 3; ++c) {
            k[c]     += x[(gc_i64)c * pitch + s] * mm;
            k[3 + c] += v[(gc_i64)c * pitch + s] * mm;
        }
        k[6] += mm;
    }
    block_reduce_sum(k, 7, sh, part, nblk);
}

template <class S>
__global__ void gcn_kern_com_rot_partials(const gc_f64 *__restrict__ x,
                                          const S *__restrict__ v,
                                          const gc_f64 *__restrict__ m,
                                          const gc_f64 *__restrict__ cv,
                                          gc_i64 n, gc_i64 pitch,
                                          gc_f64 *__restrict__ part, int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double k[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const double cx = cv[0], cy = cv[1], cz = cv[2];
    const double ux = cv[3], uy = cv[4], uz = cv[5];

    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double mm = m[s];
        double c1 = x[s] - cx, c2 = x[pitch + s] - cy, c3 = x[2*pitch + s] - cz;
        double v1 = v[s] - ux, v2 = v[pitch + s] - uy, v3 = v[2*pitch + s] - uz;
        k[0] += (c2*v3 - c3*v2) * mm;
        k[1] += (c3*v1 - c1*v3) * mm;
        k[2] += (c1*v2 - c2*v1) * mm;
        k[3] += c1*c1*mm; k[4] += c1*c2*mm; k[5] += c1*c3*mm;
        k[6] += c2*c2*mm; k[7] += c2*c3*mm; k[8] += c3*c3*mm;
    }
    block_reduce_sum(k, 9, sh, part, nblk);
}

template <class S>
__global__ void gcn_kern_com_apply(S *__restrict__ v,
                                   const gc_f64 *__restrict__ x,
                                   const gc_f64 *__restrict__ par,
                                   gc_i64 n, gc_i64 pitch,
                                   int do_trans, int do_rot)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double v1 = v[s], v2 = v[pitch + s], v3 = v[2 * pitch + s];
        if (do_trans) { v1 -= par[0]; v2 -= par[1]; v3 -= par[2]; }
        if (do_rot) {
            double c1 = x[s]             - par[3];
            double c2 = x[pitch + s]     - par[4];
            double c3 = x[2 * pitch + s] - par[5];
            double o1 = par[6], o2 = par[7], o3 = par[8];
            v1 -= (o2*c3 - o3*c2);
            v2 -= (o3*c1 - o1*c3);
            v3 -= (o1*c2 - o2*c1);
        }
        v[s] = v1; v[pitch + s] = v2; v[2 * pitch + s] = v3;
    }
}

/* The water solvers' quotient by b with r = 1/b: correctly rounded in
 * FP64, the product in FP32. */
__device__ __forceinline__ double water_div(double a, double b, double r)
{
    return div_rn_by(a, b, r);
}
__device__ __forceinline__ float water_div(float a, float, float r)
{
    return a * r;
}

/* SETTLE, the Miyamoto-Kollman analytic solve (J. Comput. Chem. 13, 952,
 * 1992), on one water in registers: x0r the reference and x1 the unconstrained
 * coordinates ([axis][member]), x the constrained ones out, v corrected in
 * place. The virial uses the UNTRANSLATED reference coordinate. R is the
 * solve's arithmetic: in FP32 (mixed mode) the water is taken in FP64 relative
 * to its reference oxygen before rounding, and the constrained coordinates,
 * virial and velocity correction are formed back in FP64. */
template <typename R = double>
__device__ __forceinline__ void settle_solve(const double x0r[3][3],
                                             const double x1in[3][3],
                                             double x[3][3], double v[3][3],
                                             int vel_update,
                                             R massO, R massH,
                                             R mass_H2O, R ra,
                                             R rb, R rc,
                                             R rHH, R inv_ra,
                                             double inv_dt, double acc[3])
{
    const bool local = sizeof(R) < sizeof(double);
    R mass[3];
    R x0[3][3], x1[3][3], x3[3][3], delt[3][3];
    R com0[3], com1[3], oh1[3], oh2[3];
    R Xax[3], Yax[3], Zax[3], mtrx[3][3];
    R xp0[3][3], xp1[3][3], xp2[3][3], xp3[3][3];
    R dxp21[2], dxp23[2], dxp31[2];
    double o[3];
    int c, k, i1, j1, k1;

    mass[0] = massO; mass[1] = massH; mass[2] = massH;

    for (c = 0; c < 3; ++c) {
        com0[c] = 0.0; com1[c] = 0.0;
        o[c] = local ? x0r[c][0] : 0.0;
    }
    for (k = 0; k < 3; ++k) {
        for (c = 0; c < 3; ++c) {
            x0[c][k] = local ? R(x0r[c][k] - o[c]) : R(x0r[c][k]);
            x1[c][k] = local ? R(x1in[c][k] - o[c]) : R(x1in[c][k]);
        }
        for (c = 0; c < 3; ++c) {
            com0[c] += x0[c][k] * mass[k];
            com1[c] += x1[c][k] * mass[k];
        }
    }
    {
        const R r = R(1) / mass_H2O;
        for (c = 0; c < 3; ++c) {
            com0[c] = water_div(com0[c], mass_H2O, r);
            com1[c] = water_div(com1[c], mass_H2O, r);
        }
    }
    for (k = 0; k < 3; ++k)
        for (c = 0; c < 3; ++c) { x0[c][k] -= com0[c]; x1[c][k] -= com1[c]; }

    for (c = 0; c < 3; ++c) {
        oh1[c] = x0[c][1] - x0[c][0];
        oh2[c] = x0[c][2] - x0[c][0];
    }
    Zax[0] = oh1[1]*oh2[2] - oh1[2]*oh2[1];
    Zax[1] = oh1[2]*oh2[0] - oh1[0]*oh2[2];
    Zax[2] = oh1[0]*oh2[1] - oh1[1]*oh2[0];
    Xax[0] = x1[1][0]*Zax[2] - x1[2][0]*Zax[1];
    Xax[1] = x1[2][0]*Zax[0] - x1[0][0]*Zax[2];
    Xax[2] = x1[0][0]*Zax[1] - x1[1][0]*Zax[0];
    Yax[0] = Zax[1]*Xax[2] - Zax[2]*Xax[1];
    Yax[1] = Zax[2]*Xax[0] - Zax[0]*Xax[2];
    Yax[2] = Zax[0]*Xax[1] - Zax[1]*Xax[0];

    {
        R nx = sqrt(Xax[0]*Xax[0] + Xax[1]*Xax[1] + Xax[2]*Xax[2]);
        R ny = sqrt(Yax[0]*Yax[0] + Yax[1]*Yax[1] + Yax[2]*Yax[2]);
        R nz = sqrt(Zax[0]*Zax[0] + Zax[1]*Zax[1] + Zax[2]*Zax[2]);
        const R rx = R(1) / nx, ry = R(1) / ny, rz = R(1) / nz;
        for (c = 0; c < 3; ++c) {
            mtrx[c][0] = water_div(Xax[c], nx, rx);
            mtrx[c][1] = water_div(Yax[c], ny, ry);
            mtrx[c][2] = water_div(Zax[c], nz, rz);
        }
    }

    for (i1 = 0; i1 < 3; ++i1)
        for (k1 = 0; k1 < 3; ++k1) { xp0[k1][i1] = 0.0; xp1[k1][i1] = 0.0; }
    for (i1 = 0; i1 < 3; ++i1)
        for (k1 = 0; k1 < 3; ++k1)
            for (j1 = 0; j1 < 3; ++j1) {
                xp0[k1][i1] += mtrx[j1][i1] * x0[j1][k1];
                xp1[k1][i1] += mtrx[j1][i1] * x1[j1][k1];
            }

    R sin_phi, cos_phi, sin_psi, cos_psi, tmp;
    sin_phi = xp1[0][2] * inv_ra;
    tmp     = R(1) - sin_phi * sin_phi;
    cos_phi = (tmp > R(0)) ? sqrt(tmp) : R(0);
    sin_psi = (xp1[1][2] - xp1[2][2]) / (rHH * cos_phi);
    tmp     = R(1) - sin_psi * sin_psi;
    cos_psi = (tmp > R(0)) ? sqrt(tmp) : R(0);

    R rb_cosphi = rb * cos_phi;
    R rb_sinphi = rb * sin_phi;
    R rc_ss     = rc * sin_psi * sin_phi;
    R rc_sc     = rc * sin_psi * cos_phi;

    xp2[0][0] = 0.0;
    xp2[0][1] =   ra * cos_phi;
    xp2[0][2] =   ra * sin_phi;
    xp2[1][0] = - rc * cos_psi;
    xp2[1][1] = - rb_cosphi - rc_ss;
    xp2[1][2] = - rb_sinphi + rc_sc;
    xp2[2][0] = - xp2[1][0];
    xp2[2][1] = - rb_cosphi + rc_ss;
    xp2[2][2] = - rb_sinphi - rc_sc;

    for (c = 0; c < 2; ++c) {
        dxp21[c] = xp0[1][c] - xp0[0][c];
        dxp23[c] = xp0[1][c] - xp0[2][c];
        dxp31[c] = xp0[2][c] - xp0[0][c];
    }
    R alpha =   xp2[1][0]*dxp23[0] + dxp21[1]*xp2[1][1]
                   + dxp31[1]*xp2[2][1];
    R beta  = - xp2[1][0]*dxp23[1] + dxp21[0]*xp2[1][1]
                   + dxp31[0]*xp2[2][1];
    R gamma =   dxp21[0]*xp1[1][1] - xp1[1][0]*dxp21[1]
                   + dxp31[0]*xp1[2][1] - xp1[2][0]*dxp31[1];

    R al2bt2    = alpha*alpha + beta*beta;
    R sin_theta = (alpha*gamma - beta*sqrt(al2bt2 - gamma*gamma))
                     / al2bt2;
    tmp = R(1) - sin_theta*sin_theta;
    R cos_theta = (tmp > R(0)) ? sqrt(tmp) : R(0);

    xp3[0][0] = - xp2[0][1]*sin_theta;
    xp3[0][1] =   xp2[0][1]*cos_theta;
    xp3[0][2] =   xp2[0][2];
    xp3[1][0] =   xp2[1][0]*cos_theta - xp2[1][1]*sin_theta;
    xp3[1][1] =   xp2[1][0]*sin_theta + xp2[1][1]*cos_theta;
    xp3[1][2] =   xp2[1][2];
    xp3[2][0] =   xp2[2][0]*cos_theta - xp2[2][1]*sin_theta;
    xp3[2][1] =   xp2[2][0]*sin_theta + xp2[2][1]*cos_theta;
    xp3[2][2] =   xp2[2][2];

    for (k1 = 0; k1 < 3; ++k1)
        for (i1 = 0; i1 < 3; ++i1) x3[i1][k1] = 0.0;
    for (k1 = 0; k1 < 3; ++k1) {
        for (i1 = 0; i1 < 3; ++i1)
            for (j1 = 0; j1 < 3; ++j1)
                x3[i1][k1] += mtrx[i1][j1] * xp3[k1][j1];
        for (c = 0; c < 3; ++c)
            x[c][k1] = local ? double(x3[c][k1] + com1[c]) + o[c]
                             : x3[c][k1] + com1[c];
    }

    for (k1 = 0; k1 < 3; ++k1)
        for (c = 0; c < 3; ++c) delt[c][k1] = x3[c][k1] - x1[c][k1];

    for (k1 = 0; k1 < 3; ++k1)
        for (c = 0; c < 3; ++c)
            acc[c] += x0r[c][k1] * double(delt[c][k1]) * double(mass[k1]);

    if (vel_update)
        for (k1 = 0; k1 < 3; ++k1)
            for (c = 0; c < 3; ++c)
                v[c][k1] += double(delt[c][k1]) * inv_dt;
}

/* water RATTLE on one water held in registers: v corrected in place, the
 * solve in R from the FP64 differences (as settle_solve) */
template <typename R = double>
__device__ __forceinline__ void rattle_water_solve(const double x[3][3],
                                                   double v[3][3],
                                                   R massO,
                                                   R massH)
{
    R rab[3], rbc[3], rca[3], vab[3], vbc[3], vca[3];
    R uab[3], ubc[3], uca[3];
    R massOH  = massO + massH;
    R massHH  = massH + massH;
    R mass2OH = R(2) * massOH;
    int c, k;

    for (c = 0; c < 3; ++c) {
        double xa = x[c][0], xb = x[c][1], xc = x[c][2];
        double va = v[c][0], vb = v[c][1], vc = v[c][2];
        rab[c] = R(xb - xa); rbc[c] = R(xc - xb); rca[c] = R(xa - xc);
        vab[c] = R(vb - va); vbc[c] = R(vc - vb); vca[c] = R(va - vc);
    }

    R lab = sqrt(rab[0]*rab[0] + rab[1]*rab[1] + rab[2]*rab[2]);
    R lbc = sqrt(rbc[0]*rbc[0] + rbc[1]*rbc[1] + rbc[2]*rbc[2]);
    R lca = sqrt(rca[0]*rca[0] + rca[1]*rca[1] + rca[2]*rca[2]);
    {
        const R rl = R(1) / lab, rm = R(1) / lbc, rn = R(1) / lca;
        for (c = 0; c < 3; ++c) {
            uab[c] = water_div(rab[c], lab, rl);
            ubc[c] = water_div(rbc[c], lbc, rm);
            uca[c] = water_div(rca[c], lca, rn);
        }
    }

    R vab0 = 0.0, vbc0 = 0.0, vca0 = 0.0;
    for (k = 0; k < 3; ++k) {
        vab0 += vab[k]*uab[k];
        vbc0 += vbc[k]*ubc[k];
        vca0 += vca[k]*uca[k];
    }
    R cosA = 0.0, cosB = 0.0, cosC = 0.0;
    for (k = 0; k < 3; ++k) {
        cosA -= uab[k]*uca[k];
        cosB -= ubc[k]*uab[k];
        cosC -= uca[k]*ubc[k];
    }

    R MbCA = massH*cosC*cosA - massOH*cosB;
    R MaBC = massO*cosB*cosC - massHH*cosA;
    R McAB = massH*cosA*cosB - massOH*cosC;

    R T_AB = vab0*(mass2OH - massO*cosC*cosC) + vbc0*MbCA + vca0*MaBC;
    T_AB = T_AB * massO;
    R T_BC = vbc0*(massOH*massOH - (massH*cosA)*(massH*cosA));
    T_BC = T_BC + vca0*massO*McAB + vab0*massO*MbCA;
    R T_CA = vca0*(mass2OH - massO*cosB*cosB) + vab0*MaBC + vbc0*McAB;
    T_CA = T_CA * massO;
    R Det = R(2)*massOH*massOH + R(2)*massO*massH*cosA*cosB*cosC;
    Det = Det - R(2)*massH*massH*cosA*cosA
        - massO*massOH*(cosB*cosB + cosC*cosC);
    const R rO = R(1) / massO, rH = R(1) / massH;
    Det = water_div(Det, massH, rH);

    const R rD = R(1) / Det;
    T_AB = water_div(T_AB, Det, rD);
    T_BC = water_div(T_BC, Det, rD);
    T_CA = water_div(T_CA, Det, rD);

    for (c = 0; c < 3; ++c) {
        v[c][0] += double(water_div(T_AB*uab[c] - T_CA*uca[c], massO, rO));
        v[c][1] += double(water_div(T_BC*ubc[c] - T_AB*uab[c], massH, rH));
        v[c][2] += double(water_div(T_CA*uca[c] - T_BC*ubc[c], massH, rH));
    }
}

template <int M>
__device__ __forceinline__ bool hg_in(int ih, int J)
{
    return M < GC_MAX_HGROUP_H ? (ih < M && ih < J) : ih < J;
}

/* x*u + y*v + z*w rounded as the solvers' source expression compiles (x and z
 * products fused into the sum, y product rounded), and dist2 - r*r with r*r
 * fused: written out so fixed-arity groups form them the same way. */
__device__ __forceinline__ double dot3(double x, double y, double z,
                                       double u, double v, double w)
{
    return __fma_rn(z, w, __fma_rn(x, u, __dmul_rn(y, v)));
}

/* SHAKE on one hydrogen group held in registers: heavy atom c0/v0, its J
 * hydrogens ch/vh, the reference bond vectors b*, target distances r and
 * inverse masses.  Returns 1 when every bond converged; sf gets each
 * bond's accumulated multiplier. */
template <int M>
__device__ __forceinline__ int shake_solve(int J, double c0[3],
                                           double ch[][3], double v0[3],
                                           double vh[][3],
                                           const double *bx,
                                           const double *by,
                                           const double *bz,
                                           const double *r,
                                           const double *im2,
                                           double imass1, double *sf,
                                           int vel_update, double dt,
                                           int iteration, double tolerance)
{
    int ih, it, shake_end = 1;
    const double rdt = 1.0 / dt;
    for (it = 1; it <= iteration; ++it) {
        shake_end = 1;
        for (ih = 0; hg_in<M>(ih, J); ++ih) {
            double x12 = c0[0] - ch[ih][0];
            double y12 = c0[1] - ch[ih][1];
            double z12 = c0[2] - ch[ih][2];
            double dist2 = dot3(x12, y12, z12, x12, y12, z12);
            double diff  = __fma_rn(-r[ih], r[ih], dist2);
            if (fabs(diff) >= 2.0*tolerance*r[ih]) {
                shake_end = 0;
                double x12o = bx[ih], y12o = by[ih], z12o = bz[ih];
                double factor = dot3(x12, y12, z12, x12o, y12o, z12o)
                              * (imass1 + im2[ih]);
                double g12   = 0.5*diff/factor;
                double g12m1 = g12 * imass1;
                double g12m2 = g12 * im2[ih];
                double v12m1 = div_rn_by(g12m1, dt, rdt);
                double v12m2 = div_rn_by(g12m2, dt, rdt);
                sf[ih] += g12;
                c0[0] -= g12m1 * x12o;
                c0[1] -= g12m1 * y12o;
                c0[2] -= g12m1 * z12o;
                ch[ih][0] += g12m2 * x12o;
                ch[ih][1] += g12m2 * y12o;
                ch[ih][2] += g12m2 * z12o;
                if (vel_update) {
                    v0[0] -= v12m1 * x12o;
                    v0[1] -= v12m1 * y12o;
                    v0[2] -= v12m1 * z12o;
                    vh[ih][0] += v12m2 * x12o;
                    vh[ih][1] += v12m2 * y12o;
                    vh[ih][2] += v12m2 * z12o;
                }
            }
        }
        if (shake_end) break;
    }
    return shake_end;
}

/* RATTLE on one hydrogen group held in registers; returns 1 when every
 * bond converged. */
template <int M>
__device__ __forceinline__ int rattle_hgroup_solve(int J, const double c0[3],
                                                   double ch[][3],
                                                   double v0[3],
                                                   double vh[][3],
                                                   const double *r,
                                                   const double *im2,
                                                   double imass1,
                                                   int iteration,
                                                   double tolerance)
{
    int ih, it, rattle_end = 1;
    for (it = 1; it <= iteration; ++it) {
        rattle_end = 1;
        for (ih = 0; hg_in<M>(ih, J); ++ih) {
            double x12 = c0[0] - ch[ih][0];
            double y12 = c0[1] - ch[ih][1];
            double z12 = c0[2] - ch[ih][2];
            double vx  = v0[0] - vh[ih][0];
            double vy  = v0[1] - vh[ih][1];
            double vz  = v0[2] - vh[ih][2];
            double dot = dot3(x12, y12, z12, vx, vy, vz);
            if (fabs(dot) >= tolerance) {
                rattle_end = 0;
                double g12   = dot / ((imass1 + im2[ih]) * r[ih] * r[ih]);
                double g12m1 = g12 * imass1;
                double g12m2 = g12 * im2[ih];
                v0[0] -= g12m1 * x12;
                v0[1] -= g12m1 * y12;
                v0[2] -= g12m1 * z12;
                vh[ih][0] += g12m2 * x12;
                vh[ih][1] += g12m2 * y12;
                vh[ih][2] += g12m2 * z12;
            }
        }
        if (rattle_end) break;
    }
    return rattle_end;
}

/* The mixed mode's hydrogen-group SHAKE and RATTLE (shake_solve and
 * rattle_hgroup_solve in FP32). Each bond is the FP64 difference of its
 * endpoints rounded to FP32; FP32 corrections are added to the FP64
 * coordinates and velocities once at the end, and multipliers and virial are
 * formed in FP64. A bond converges at shake_tolerance or at GCN_MIXED_CON_REL
 * of its own scale, whichever is looser. */
#define GCN_MIXED_CON_REL 2.0e-6f

template <int M>
__device__ __forceinline__ int shake_solve_f32(int J, double c0[3],
                                               double ch[][3], double v0[3],
                                               double vh[][3],
                                               const double *bx,
                                               const double *by,
                                               const double *bz,
                                               const double *r,
                                               const double *im2,
                                               double imass1, double *sf,
                                               int vel_update, double dt,
                                               int iteration, double tolerance)
{
    float d0[3] = { 0.0f, 0.0f, 0.0f }, u0[3] = { 0.0f, 0.0f, 0.0f };
    float b[M][3], dh[M][3], uh[M][3], e[M][3], rr[M], tol[M], im[M], g[M];
    const float im1 = (float)imass1, rdt = (float)(1.0 / dt);
    int ih, c, it, shake_end = 1;
    for (ih = 0; hg_in<M>(ih, J); ++ih) {
        const float rf = (float)r[ih];
        b[ih][0] = (float)bx[ih]; b[ih][1] = (float)by[ih];
        b[ih][2] = (float)bz[ih];
        for (c = 0; c < 3; ++c) {
            e[ih][c]  = (float)(c0[c] - ch[ih][c]);
            dh[ih][c] = 0.0f; uh[ih][c] = 0.0f;
        }
        rr[ih]  = rf * rf;
        tol[ih] = fmaxf((float)(2.0 * tolerance * r[ih]),
                        2.0f * GCN_MIXED_CON_REL * rr[ih]);
        im[ih]  = im1 + (float)im2[ih];
        g[ih]   = 0.0f;
    }
    for (it = 1; it <= iteration; ++it) {
        shake_end = 1;
        for (ih = 0; hg_in<M>(ih, J); ++ih) {
            float x[3];
            for (c = 0; c < 3; ++c) x[c] = e[ih][c] + (d0[c] - dh[ih][c]);
            const float diff = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] - rr[ih];
            if (fabsf(diff) >= tol[ih]) {
                shake_end = 0;
                const float factor = (x[0]*b[ih][0] + x[1]*b[ih][1] +
                                      x[2]*b[ih][2]) * im[ih];
                const float g12 = 0.5f * diff / factor;
                const float g1 = g12 * im1, g2 = g12 * (float)im2[ih];
                g[ih] += g12;
                for (c = 0; c < 3; ++c) {
                    d0[c]     -= g1 * b[ih][c];
                    dh[ih][c] += g2 * b[ih][c];
                    u0[c]     -= g1 * rdt * b[ih][c];
                    uh[ih][c] += g2 * rdt * b[ih][c];
                }
            }
        }
        if (shake_end) break;
    }
    for (c = 0; c < 3; ++c) {
        c0[c] += (double)d0[c];
        if (vel_update) v0[c] += (double)u0[c];
    }
    for (ih = 0; hg_in<M>(ih, J); ++ih) {
        sf[ih] += (double)g[ih];
        for (c = 0; c < 3; ++c) {
            ch[ih][c] += (double)dh[ih][c];
            if (vel_update) vh[ih][c] += (double)uh[ih][c];
        }
    }
    return shake_end;
}

template <int M>
__device__ __forceinline__ int rattle_hgroup_solve_f32(int J,
                                                       const double c0[3],
                                                       double ch[][3],
                                                       double v0[3],
                                                       double vh[][3],
                                                       const double *r,
                                                       const double *im2,
                                                       double imass1,
                                                       int iteration,
                                                       double tolerance)
{
    float u0[3] = { 0.0f, 0.0f, 0.0f };
    float x[M][3], w[M][3], uh[M][3], den[M], rr[M], w2[M];
    const float im1 = (float)imass1, tol2 = (float)(tolerance * tolerance);
    const float rel2 = GCN_MIXED_CON_REL * GCN_MIXED_CON_REL;
    int ih, c, it, rattle_end = 1;
    for (ih = 0; hg_in<M>(ih, J); ++ih) {
        const float rf = (float)r[ih];
        for (c = 0; c < 3; ++c) {
            x[ih][c]  = (float)(c0[c] - ch[ih][c]);
            w[ih][c]  = (float)(v0[c] - vh[ih][c]);
            uh[ih][c] = 0.0f;
        }
        rr[ih]  = rf * rf;
        den[ih] = (im1 + (float)im2[ih]) * rr[ih];
        w2[ih]  = w[ih][0]*w[ih][0] + w[ih][1]*w[ih][1] + w[ih][2]*w[ih][2];
    }
    for (it = 1; it <= iteration; ++it) {
        rattle_end = 1;
        for (ih = 0; hg_in<M>(ih, J); ++ih) {
            float v[3], u[3];
            for (c = 0; c < 3; ++c) {
                u[c] = u0[c] - uh[ih][c];
                v[c] = w[ih][c] + u[c];
            }
            const float dot = x[ih][0]*v[0] + x[ih][1]*v[1] + x[ih][2]*v[2];
            const float v2  = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
            const float u2  = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
            const float lim = fmaxf(tol2, rel2 * rr[ih] *
                                          fmaxf(v2, fmaxf(w2[ih], u2)));
            if (dot * dot >= lim) {
                rattle_end = 0;
                const float g12 = dot / den[ih];
                const float g1 = g12 * im1, g2 = g12 * (float)im2[ih];
                for (c = 0; c < 3; ++c) {
                    u0[c]     -= g1 * x[ih][c];
                    uh[ih][c] += g2 * x[ih][c];
                }
            }
        }
        if (rattle_end) break;
    }
    for (c = 0; c < 3; ++c) v0[c] += (double)u0[c];
    for (ih = 0; hg_in<M>(ih, J); ++ih)
        for (c = 0; c < 3; ++c) vh[ih][c] += (double)uh[ih][c];
    return rattle_end;
}

/* A failing group is counted with the least global id among them. */
__device__ __forceinline__ void con_fail_note(gc_i64 *fail, gc_gid g)
{
    atomicAdd((unsigned long long *)&fail[0], 1ull);
    atomicMin((unsigned long long *)&fail[1], (unsigned long long)g);
}

/* One hydrogen group's table row: its heavy atom a[0], hydrogens a[1..J],
 * bond distances and inverse masses; returns J. */
template <int M = GC_MAX_HGROUP_H>
__device__ __forceinline__ int hgroup_row(gc_i64 g, gc_i64 ngrp,
                                          const gc_i32 *__restrict__ heavy,
                                          const gc_i32 *__restrict__ hslot,
                                          const gc_i32 *__restrict__ arity,
                                          const gc_f64 *__restrict__ dist,
                                          const gc_f64 *__restrict__ imh,
                                          int *a, double *r, double *im2)
{
    const int J = arity[g];
    a[0] = heavy[g];
    for (int ih = 0; hg_in<M>(ih, J); ++ih) {
        a[ih + 1] = hslot[(gc_i64)ih * ngrp + g];
        r[ih]     = dist[(gc_i64)ih * ngrp + g];
        im2[ih]   = imh[(gc_i64)ih * ngrp + g];
    }
    return J;
}

template <class S>
__global__ void gcn_kern_settle_vv1(gc_i64 nwater,
                                    const gc_i32 *__restrict__ wslot,
                                    const gc_f64 *__restrict__ xref,
                                    gc_f64 *__restrict__ x,
                                    S *__restrict__ v,
                                    gc_i64 pitch, int vel_update,
                                    double massO, double massH,
                                    double mass_H2O, double ra, double rb,
                                    double rc, double rHH, double inv_ra,
                                    double inv_dt,
                                    gc_f64 *__restrict__ part, int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double acc[3] = { 0.0, 0.0, 0.0 };

    for (gc_i64 w = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; w < nwater;
         w += (gc_i64)gridDim.x * blockDim.x) {
        int ia[3];
        double x0r[3][3], x1[3][3], xn[3][3], vw[3][3];
        for (int k = 0; k < 3; ++k) ia[k] = wslot[(gc_i64)k * nwater + w];
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c) {
                x0r[c][k] = xref[(gc_i64)c * pitch + ia[k]];
                x1[c][k]  = x[(gc_i64)c * pitch + ia[k]];
                vw[c][k]  = v[(gc_i64)c * pitch + ia[k]];
            }
        settle_solve(x0r, x1, xn, vw, vel_update, massO, massH, mass_H2O,
                     ra, rb, rc, rHH, inv_ra, inv_dt, acc);
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c) {
                x[(gc_i64)c * pitch + ia[k]] = xn[c][k];
                if (vel_update) v[(gc_i64)c * pitch + ia[k]] = vw[c][k];
            }
    }

    block_reduce_sum(acc, 3, sh, part, nblk);
}

template <class S>
__global__ void gcn_kern_rattle_water(gc_i64 nwater,
                                      const gc_i32 *__restrict__ wslot,
                                      const gc_f64 *__restrict__ x,
                                      S *__restrict__ v,
                                      gc_i64 pitch,
                                      double massO, double massH)
{
    for (gc_i64 w = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; w < nwater;
         w += (gc_i64)gridDim.x * blockDim.x) {
        int ia[3];
        double xw[3][3], vw[3][3];
        for (int k = 0; k < 3; ++k) ia[k] = wslot[(gc_i64)k * nwater + w];
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c) {
                xw[c][k] = x[(gc_i64)c * pitch + ia[k]];
                vw[c][k] = v[(gc_i64)c * pitch + ia[k]];
            }
        rattle_water_solve(xw, vw, massO, massH);
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c)
                v[(gc_i64)c * pitch + ia[k]] = vw[c][k];
    }
}

/* SHAKE over hydrogen groups. Arity is data (loop bounded by GC_MAX_HGROUP_H),
 * so one kernel serves every arity. A group that does not converge is
 * reported, never left silently wrong. */
template <class S>
__global__ void gcn_kern_shake_vv1(gc_i64 ngrp,
                                   const gc_i32 *__restrict__ heavy,
                                   const gc_i32 *__restrict__ hslot,
                                   const gc_i32 *__restrict__ arity,
                                   const gc_f64 *__restrict__ dist,
                                   const gc_f64 *__restrict__ imh,
                                   const gc_f64 *__restrict__ im1,
                                   const gc_f64 *__restrict__ xref,
                                   gc_f64 *__restrict__ x,
                                   S *__restrict__ v,
                                   gc_i64 pitch, int vel_update, double dt,
                                   int iteration, double tolerance,
                                   const gc_gid *__restrict__ gid,
                                   gc_i64 *__restrict__ fail,
                                   gc_f64 *__restrict__ part, int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double acc[3] = { 0.0, 0.0, 0.0 };

    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngrp;
         g += (gc_i64)gridDim.x * blockDim.x) {

        int    a[GC_MAX_HGROUP_H + 1];
        double c0[3], ch[GC_MAX_HGROUP_H][3], v0[3], vh[GC_MAX_HGROUP_H][3];
        double bx[GC_MAX_HGROUP_H], by[GC_MAX_HGROUP_H], bz[GC_MAX_HGROUP_H];
        double sf[GC_MAX_HGROUP_H], r[GC_MAX_HGROUP_H], im2[GC_MAX_HGROUP_H];
        const int J = hgroup_row(g, ngrp, heavy, hslot, arity, dist, imh,
                                 a, r, im2);
        int ih, c;

        for (ih = 0; ih < J; ++ih) sf[ih] = 0.0;
        {
            double ox = xref[a[0]];
            double oy = xref[pitch + a[0]];
            double oz = xref[2 * pitch + a[0]];
            for (ih = 0; ih < J; ++ih) {
                bx[ih] = ox - xref[a[ih+1]];
                by[ih] = oy - xref[pitch + a[ih+1]];
                bz[ih] = oz - xref[2 * pitch + a[ih+1]];
            }
        }
        for (c = 0; c < 3; ++c) {
            c0[c] = x[(gc_i64)c*pitch + a[0]];
            v0[c] = v[(gc_i64)c*pitch + a[0]];
        }
        for (ih = 0; ih < J; ++ih)
            for (c = 0; c < 3; ++c) {
                ch[ih][c] = x[(gc_i64)c*pitch + a[ih+1]];
                vh[ih][c] = v[(gc_i64)c*pitch + a[ih+1]];
            }

        const int shake_end = shake_solve<GC_MAX_HGROUP_H>(
                                          J, c0, ch, v0, vh, bx, by, bz, r,
                                          im2, im1[g], sf, vel_update, dt,
                                          iteration, tolerance);

        for (c = 0; c < 3; ++c) {
            x[(gc_i64)c*pitch + a[0]] = c0[c];
            v[(gc_i64)c*pitch + a[0]] = v0[c];
        }
        for (ih = 0; ih < J; ++ih)
            for (c = 0; c < 3; ++c) {
                x[(gc_i64)c*pitch + a[ih+1]] = ch[ih][c];
                v[(gc_i64)c*pitch + a[ih+1]] = vh[ih][c];
            }

        if (!shake_end) con_fail_note(fail, gid[a[0]]);

        for (ih = 0; ih < J; ++ih) {
            acc[0] -= bx[ih] * (sf[ih] * bx[ih]);
            acc[1] -= by[ih] * (sf[ih] * by[ih]);
            acc[2] -= bz[ih] * (sf[ih] * bz[ih]);
        }
    }

    block_reduce_sum(acc, 3, sh, part, nblk);
}

template <class S>
__global__ void gcn_kern_rattle_hgroup(gc_i64 ngrp,
                                       const gc_i32 *__restrict__ heavy,
                                       const gc_i32 *__restrict__ hslot,
                                       const gc_i32 *__restrict__ arity,
                                       const gc_f64 *__restrict__ dist,
                                       const gc_f64 *__restrict__ imh,
                                       const gc_f64 *__restrict__ im1,
                                       const gc_f64 *__restrict__ x,
                                       S *__restrict__ v,
                                       gc_i64 pitch, int iteration,
                                       double tolerance,
                                       const gc_gid *__restrict__ gid,
                                       gc_i64 *__restrict__ fail)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngrp;
         g += (gc_i64)gridDim.x * blockDim.x) {

        int    a[GC_MAX_HGROUP_H + 1];
        double c0[3], ch[GC_MAX_HGROUP_H][3], v0[3], vh[GC_MAX_HGROUP_H][3];
        double r[GC_MAX_HGROUP_H], im2[GC_MAX_HGROUP_H];
        const int J = hgroup_row(g, ngrp, heavy, hslot, arity, dist, imh,
                                 a, r, im2);
        int ih, c;
        for (c = 0; c < 3; ++c) {
            c0[c] = x[(gc_i64)c*pitch + a[0]];
            v0[c] = v[(gc_i64)c*pitch + a[0]];
        }
        for (ih = 0; ih < J; ++ih)
            for (c = 0; c < 3; ++c) {
                ch[ih][c] = x[(gc_i64)c*pitch + a[ih+1]];
                vh[ih][c] = v[(gc_i64)c*pitch + a[ih+1]];
            }

        const int rattle_end = rattle_hgroup_solve<GC_MAX_HGROUP_H>(
                                                   J, c0, ch, v0, vh, r, im2,
                                                   im1[g], iteration,
                                                   tolerance);

        for (c = 0; c < 3; ++c) v[(gc_i64)c*pitch + a[0]] = v0[c];
        for (ih = 0; ih < J; ++ih)
            for (c = 0; c < 3; ++c)
                v[(gc_i64)c*pitch + a[ih+1]] = vh[ih][c];

        if (!rattle_end) con_fail_note(fail, gid[a[0]]);
    }
}

/* One pass per rigid-group kind does what gcn_kern_kick_drift (or
 * gcn_kern_vv2) and the kind's solver did in turn: a thread takes a group's
 * atoms into registers, kicks, drifts, solves, and writes each coordinate and
 * velocity once. Arithmetic and launch shapes (hence virial partials) match
 * the separate kernels. In mixed mode a device tick's rescale is taken into
 * the VV1 pass (gcn_rescale); in double mode it stays its own launch
 * (gcn_kern_rescale), whose compiled rounding is the reference. */

/* Singles: kick and drift.  One thread per group; non-singles are the
 * solvers'. */
/* What a VV1 pass does with the coordinates it moved (VIEW 1): the step's
 * force coordinates (as gcn_kern_force_coord, capture 0; a non-finite move or
 * coordinate is counted in `bad`) and the list guard's two distances
 * (gcn_kern_displacement), whose maxima go to gmax. VIEW 0 does neither. */
struct gcn_view {
    const gc_f64 *move;
    gc_f64       *view;
    gc_i64       *bad;
    const gc_f64 *lref;
    gc_f64       *gmax;
    double        box[3];
};

/* gcn_kern_rescale taken into a VV1 solver pass (mixed mode): scale, when non-
 * null, is the device tick's factor, applied to each group before its kick. */
struct gcn_rescale {
    const double *scale;
    const gc_f64 *mass;
    int           group_tp;
};

/* gcn_kern_rescale's arms on one group of n members in registers, vel(k, c)
 * member k's component c, with its expressions and explicit roundings; a unit
 * factor is no rescale */
template <int N, class V>
__device__ __forceinline__ void rescale_members(const struct gcn_rescale &rs,
                                                const int *s, int n, V &&vel)
{
    if (!rs.scale) return;
    const double scale = *rs.scale;
    if (scale == 1.0) return;
    if (!rs.group_tp || n == 1) {
        for (int k = 0; k < N && k < n; ++k)
            for (int c = 0; c < 3; ++c) vel(k, c) = __dmul_rn(vel(k, c), scale);
        return;
    }
    double sm1 = scale - 1.0;
    double tm = 0.0, cm[3] = { 0.0, 0.0, 0.0 };
    for (int k = 0; k < N && k < n; ++k) {
        double mm = __ldg(rs.mass + s[k]);
        tm += mm;
        for (int c = 0; c < 3; ++c) cm[c] = __fma_rn(mm, vel(k, c), cm[c]);
    }
    const double rt = 1.0 / tm;
    for (int c = 0; c < 3; ++c) cm[c] = __dmul_rn(sm1, div_rn_by(cm[c], tm, rt));
    for (int k = 0; k < N && k < n; ++k)
        for (int c = 0; c < 3; ++c) vel(k, c) = __dadd_rn(vel(k, c), cm[c]);
}

__device__ __forceinline__ void view_guard(const struct gcn_view &vw,
                                           double x0, double x1, double x2,
                                           double r0, double r1, double r2,
                                           gc_i64 s, gc_i64 pitch,
                                           double &m, double &ms)
{
    const double x[3] = { x0, x1, x2 }, p[3] = { r0, r1, r2 };
    double ref[3], d2, s2;
    for (int c = 0; c < 3; ++c) ref[c] = vw.lref[(gc_i64)c * pitch + s];
    guard_d2(x, ref, p, vw.box, &d2, &s2);
    m  = fmax(m, d2);
    ms = fmax(ms, s2);
}

template <int VIEW>
__device__ __forceinline__ int view_move(const struct gcn_view &vw, gc_i64 g,
                                         double mv[3])
{
    if (!VIEW) return 0;
    for (int k = 0; k < 3; ++k) {
        mv[k] = vw.move[3 * g + k];
        if (!isfinite(mv[k])) {
            atomicAdd((unsigned long long *)vw.bad, 1ull);
            return 0;
        }
    }
    return 1;
}

template <int VIEW>
__device__ __forceinline__ int view_move_regs(const struct gcn_view &vw,
                                              const double mv[3])
{
    if (!VIEW) return 0;
    for (int k = 0; k < 3; ++k)
        if (!isfinite(mv[k])) {
            atomicAdd((unsigned long long *)vw.bad, 1ull);
            return 0;
        }
    return 1;
}

__device__ __forceinline__ void view_guard_ref(const struct gcn_view &vw,
                                               const double ref[3],
                                               double x0, double x1,
                                               double x2, double r0,
                                               double r1, double r2,
                                               double &m, double &ms)
{
    const double x[3] = { x0, x1, x2 }, p[3] = { r0, r1, r2 };
    double d2, s2;
    guard_d2(x, ref, p, vw.box, &d2, &s2);
    m  = fmax(m, d2);
    ms = fmax(ms, s2);
}

__device__ __forceinline__ void view_atom(const struct gcn_view &vw,
                                          const double mv[3], double x0,
                                          double x1, double x2, gc_i64 s,
                                          gc_i64 pitch)
{
    const double x[3] = { x0, x1, x2 };
    for (int k = 0; k < 3; ++k) {
        if (!isfinite(x[k]) || !isfinite(x[k] + mv[k])) {
            atomicAdd((unsigned long long *)vw.bad, 1ull);
            continue;
        }
        vw.view[(gc_i64)k * pitch + s] = x[k] + mv[k];
    }
}

template <int VIEW, class S>
__global__ void gcn_kern_vv1_single(S *__restrict__ v,
                                    gc_f64 *__restrict__ x,
                                    gc_f64 *__restrict__ xref,
                                    const S *__restrict__ f,
                                    const gc_f64 *__restrict__ minv,
                                    const gc_i64 *__restrict__ goff,
                                    const gc_i32 *__restrict__ gmem,
                                    const gc_u8 *__restrict__ gkind,
                                    gc_i64 ngroup, gc_i64 pitch,
                                    double dt, double half_dt,
                                    const struct gcn_view vw,
                                    const struct gcn_rescale rs)
{
    double m = 0.0, ms = 0.0;
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        if (gkind[g] != GC_GROUP_SINGLE) continue;
        const int si = gmem[goff[g]];
        const gc_i64 s = si;
        double vv[3], xr[3], xn[3];
        for (int c = 0; c < 3; ++c) {
            vv[c] = v[(gc_i64)c * pitch + s];
            xr[c] = x[(gc_i64)c * pitch + s];
        }
        rescale_members<1>(rs, &si, 1,
                           [&](int, int c) -> double & { return vv[c]; });
        kick_drift_atom(vv, xr, xn, f, minv, s, pitch, dt, half_dt);
        for (int c = 0; c < 3; ++c) {
            v[(gc_i64)c * pitch + s]    = vv[c];
            xref[(gc_i64)c * pitch + s] = xr[c];
            x[(gc_i64)c * pitch + s]    = xn[c];
        }
        double mv[3];
        if (view_move<VIEW>(vw, g, mv))
            view_atom(vw, mv, xn[0], xn[1], xn[2], s, pitch);
        if (VIEW)
            view_guard(vw, xn[0], xn[1], xn[2], xr[0], xr[1], xr[2], s,
                       pitch, m, ms);
    }
    if (VIEW) guard_max_warp(m, ms, vw.gmax);
}

/* Waters: kick, drift and SETTLE (gcn_kern_settle_vv1's launch shape, so
 * its partials), the solve in R. */
template <int VIEW, typename R>
__global__ void __launch_bounds__(GCN_RED_BLOCK)
gcn_kern_vv1_water(gc_i64 nwater, const gc_i32 *__restrict__ wslot,
                   R *__restrict__ v, gc_f64 *__restrict__ x,
                   gc_f64 *__restrict__ xref,
                   const R *__restrict__ f,
                   const gc_f64 *__restrict__ minv,
                   gc_i64 pitch, double dt, double half_dt,
                   double massO, double massH, double mass_H2O, double ra,
                   double rb, double rc, double rHH, double inv_ra,
                   double inv_dt, gc_f64 *__restrict__ part, int nblk,
                   const gc_i32 *__restrict__ wgrp,
                   const struct gcn_view vwv, const struct gcn_rescale rs)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double acc[3] = { 0.0, 0.0, 0.0 };
    double m = 0.0, ms = 0.0;

    for (gc_i64 w = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; w < nwater;
         w += (gc_i64)gridDim.x * blockDim.x) {
        int ia[3];
        double vw[3][3], xr[3][3], x1[3][3], xn[3][3];
        double fw[3][3], fac[3], mv[3], lr[VIEW ? 3 : 1][3];
        for (int k = 0; k < 3; ++k) ia[k] = wslot[(gc_i64)k * nwater + w];
        for (int k = 0; k < 3; ++k) {
            for (int c = 0; c < 3; ++c) {
                vw[c][k] = v[(gc_i64)c * pitch + ia[k]];
                xr[c][k] = x[(gc_i64)c * pitch + ia[k]];
                fw[c][k] = __ldg(f + (gc_i64)c * pitch + ia[k]);
            }
            fac[k] = half_dt * __ldg(minv + ia[k]);
        }
        if (VIEW) {
            const gc_i64 g = wgrp[w];
            for (int k = 0; k < 3; ++k) {
                mv[k] = __ldg(vwv.move + 3 * g + k);
                for (int c = 0; c < 3; ++c)
                    lr[k][c] = __ldg(vwv.lref + (gc_i64)c * pitch + ia[k]);
            }
        }
        rescale_members<3>(rs, ia, 3,
                           [&](int k, int c) -> double & { return vw[c][k]; });
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c) {
                const double vv = vw[c][k] + fac[k] * fw[c][k];
                vw[c][k] = vv;
                x1[c][k] = xr[c][k] + dt * vv;
            }
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c)
                xref[(gc_i64)c * pitch + ia[k]] = xr[c][k];
        settle_solve<R>(xr, x1, xn, vw, 1, massO, massH, mass_H2O, ra, rb,
                        rc, rHH, inv_ra, inv_dt, acc);
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c) {
                x[(gc_i64)c * pitch + ia[k]] = xn[c][k];
                v[(gc_i64)c * pitch + ia[k]] = vw[c][k];
            }
        if (view_move_regs<VIEW>(vwv, mv))
            for (int k = 0; k < 3; ++k)
                view_atom(vwv, mv, xn[0][k], xn[1][k], xn[2][k], ia[k], pitch);
        if (VIEW)
            for (int k = 0; k < 3; ++k)
                view_guard_ref(vwv, lr[k], xn[0][k], xn[1][k], xn[2][k],
                               xr[0][k], xr[1][k], xr[2][k], m, ms);
    }
    if (VIEW) guard_max_warp(m, ms, vwv.gmax);
    block_reduce_sum(acc, 3, sh, part, nblk);
}

/* One hydrogen group's kick, drift and SHAKE for at most M hydrogens. With M a
 * small constant the loops unroll and the per-group arrays are registers; a
 * shorter group skips the rest of each loop. */
template <int M, int VIEW, typename R>
__device__ __forceinline__ void vv1_hgroup_row(
    gc_i64 g, gc_i64 ngrp, const gc_i32 *__restrict__ heavy,
    const gc_i32 *__restrict__ hslot, const gc_i32 *__restrict__ arity,
    const gc_f64 *__restrict__ dist, const gc_f64 *__restrict__ imh,
    const gc_f64 *__restrict__ im1, R *__restrict__ v,
    gc_f64 *__restrict__ x, gc_f64 *__restrict__ xref,
    const R *__restrict__ f, const gc_f64 *__restrict__ minv,
    gc_i64 pitch, double dt, double half_dt, double con_dt, int iteration,
    double tolerance, const gc_gid *__restrict__ gid,
    gc_i64 *__restrict__ fail, const gc_i32 *__restrict__ hgrp,
    const struct gcn_view &vw, const struct gcn_rescale &rs, double acc[3],
    double &m, double &ms)
{
    int    a[M + 1];
    double xg[3][M + 1];
    double c0[3], ch[M][3], v0[3], vh[M][3];
    double bx[M], by[M], bz[M];
    double sf[M], r[M], im2[M];
    const int J = hgroup_row<M>(g, ngrp, heavy, hslot, arity, dist, imh,
                                 a, r, im2);
    int ih, c;

    for (ih = 0; hg_in<M + 1>(ih, J + 1); ++ih)
        for (c = 0; c < 3; ++c) {
            const double vv = v[(gc_i64)c * pitch + a[ih]];
            if (ih == 0) v0[c] = vv; else vh[ih - 1][c] = vv;
        }
    rescale_members<M + 1>(rs, a, J + 1, [&](int k, int c) -> double & {
        return k == 0 ? v0[c] : vh[k - 1][c];
    });
    for (ih = 0; hg_in<M + 1>(ih, J + 1); ++ih) {
        double va[3], xa[3], xb[3];
        for (c = 0; c < 3; ++c) {
            va[c] = ih == 0 ? v0[c] : vh[ih - 1][c];
            xa[c] = xg[c][ih] = x[(gc_i64)c * pitch + a[ih]];
        }
        kick_drift_atom(va, xa, xb, f, minv, a[ih], pitch, dt, half_dt);
        for (c = 0; c < 3; ++c) {
            xref[(gc_i64)c * pitch + a[ih]] = xa[c];
            if (ih == 0) { v0[c] = va[c]; c0[c] = xb[c]; }
            else { vh[ih - 1][c] = va[c]; ch[ih - 1][c] = xb[c]; }
        }
    }
    for (ih = 0; hg_in<M>(ih, J); ++ih) {
        sf[ih] = 0.0;
        bx[ih] = xg[0][0] - xg[0][ih + 1];
        by[ih] = xg[1][0] - xg[1][ih + 1];
        bz[ih] = xg[2][0] - xg[2][ih + 1];
    }

    const int shake_end = sizeof(R) < sizeof(double)
        ? shake_solve_f32<M>(J, c0, ch, v0, vh, bx, by, bz, r, im2, im1[g],
                             sf, 1, con_dt, iteration, tolerance)
        : shake_solve<M>(J, c0, ch, v0, vh, bx, by, bz, r, im2, im1[g], sf,
                         1, con_dt, iteration, tolerance);

    for (c = 0; c < 3; ++c) {
        x[(gc_i64)c*pitch + a[0]] = c0[c];
        v[(gc_i64)c*pitch + a[0]] = v0[c];
    }
    for (ih = 0; hg_in<M>(ih, J); ++ih)
        for (c = 0; c < 3; ++c) {
            x[(gc_i64)c*pitch + a[ih+1]] = ch[ih][c];
            v[(gc_i64)c*pitch + a[ih+1]] = vh[ih][c];
        }
    if (!shake_end) con_fail_note(fail, gid[a[0]]);
    double mv[3];
    if (view_move<VIEW>(vw, VIEW ? hgrp[g] : 0, mv)) {
        view_atom(vw, mv, c0[0], c0[1], c0[2], a[0], pitch);
        for (ih = 0; hg_in<M>(ih, J); ++ih)
            view_atom(vw, mv, ch[ih][0], ch[ih][1], ch[ih][2], a[ih + 1],
                      pitch);
    }
    if (VIEW) {
        view_guard(vw, c0[0], c0[1], c0[2], xg[0][0], xg[1][0], xg[2][0],
                   a[0], pitch, m, ms);
        for (ih = 0; hg_in<M>(ih, J); ++ih)
            view_guard(vw, ch[ih][0], ch[ih][1], ch[ih][2], xg[0][ih + 1],
                       xg[1][ih + 1], xg[2][ih + 1], a[ih + 1], pitch, m, ms);
    }
    for (ih = 0; hg_in<M>(ih, J); ++ih) {
        acc[0] -= bx[ih] * (sf[ih] * bx[ih]);
        acc[1] -= by[ih] * (sf[ih] * by[ih]);
        acc[2] -= bz[ih] * (sf[ih] * bz[ih]);
    }
}

/* Hydrogen groups: kick, drift and SHAKE (gcn_kern_shake_vv1's launch
 * shape, so its partials), the solve in R; groups of up to four hydrogens
 * take the register body. */
template <int VIEW, typename R, int WIDE>
__global__ void __launch_bounds__(GCN_RED_BLOCK)
gcn_kern_vv1_hgroup(gc_i64 ngrp, const gc_i32 *__restrict__ heavy,
                    const gc_i32 *__restrict__ hslot,
                    const gc_i32 *__restrict__ arity,
                    const gc_f64 *__restrict__ dist,
                    const gc_f64 *__restrict__ imh,
                    const gc_f64 *__restrict__ im1,
                    R *__restrict__ v, gc_f64 *__restrict__ x,
                    gc_f64 *__restrict__ xref,
                    const R *__restrict__ f,
                    const gc_f64 *__restrict__ minv,
                    gc_i64 pitch, double dt, double half_dt, double con_dt,
                    int iteration, double tolerance,
                    const gc_gid *__restrict__ gid,
                    gc_i64 *__restrict__ fail,
                    gc_f64 *__restrict__ part, int nblk,
                    const gc_i32 *__restrict__ hgrp,
                    const struct gcn_view vw, const struct gcn_rescale rs)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double acc[3] = { 0.0, 0.0, 0.0 };
    double m = 0.0, ms = 0.0;

    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngrp;
         g += (gc_i64)gridDim.x * blockDim.x) {
#define GCN_VV1_HGROUP(M)                                                    \
        vv1_hgroup_row<M, VIEW, R>(g, ngrp, heavy, hslot, arity, dist, imh,   \
                                 im1, v, x, xref, f, minv, pitch, dt,       \
                                 half_dt, con_dt, iteration, tolerance,     \
                                 gid, fail, hgrp, vw, rs, acc, m, ms)
        if (arity[g] <= 4) GCN_VV1_HGROUP(4);
        else if (WIDE)     GCN_VV1_HGROUP(GC_MAX_HGROUP_H);
        else               con_fail_note(fail, gid[heavy[g]]);
#undef GCN_VV1_HGROUP
    }
    if (VIEW) guard_max_warp(m, ms, vw.gmax);
    block_reduce_sum(acc, 3, sh, part, nblk);
}

/* Singles' VV2 half kick, with JOIN their force's join */
template <int JOIN, int TICK, class S>
__global__ void gcn_kern_vv2_single(S *__restrict__ v,
                                    gc_f64 *__restrict__ vhalf,
                                    gc_f64 *__restrict__ vfull,
                                    const S *__restrict__ f,
                                    const gc_f64 *__restrict__ minv,
                                    const gc_i64 *__restrict__ goff,
                                    const gc_i32 *__restrict__ gmem,
                                    const gc_u8 *__restrict__ gkind,
                                    gc_i64 ngroup, gc_i64 pitch,
                                    double half_dt, int full,
                                    const struct gcn_join j,
                                    const struct gcn_tick_prep tp)
{
    __shared__ double sh[TICK ? GCN_RED_BLOCK : 1];
    double kin[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const unsigned int ovf = JOIN ? *j.overflow : 0u;
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        if (gkind[g] != GC_GROUP_SINGLE) continue;
        const int si = gmem[goff[g]];
        const gc_i64 s = si;
        double vv[3], pre[3];
        for (int c = 0; c < 3; ++c) vv[c] = v[(gc_i64)c * pitch + s];
        vv2_atom<JOIN, TICK>(vv, vhalf, vfull, f, j, ovf, minv, s, pitch,
                             half_dt, full, pre);
        for (int c = 0; c < 3; ++c) v[(gc_i64)c * pitch + s] = vv[c];
        if (TICK)
            tick_members<1>(tp, &si, 1, vhalf, pitch,
                            [&](int, int c) { return vv[c]; },
                            [&](int, int c) { return pre[c]; }, kin);
    }
    if (TICK) {
        block_reduce_sum(kin, 6, sh, tp.part + tp.off, tp.nblk);
        tick_fold(tp, sh);
    }
}

/* Waters' VV2 half kick and water RATTLE, the solve in R */
template <int JOIN, typename R, int TICK>
__global__ void gcn_kern_vv2_water(gc_i64 nwater,
                                   const gc_i32 *__restrict__ wslot,
                                   R *__restrict__ v,
                                   gc_f64 *__restrict__ vhalf,
                                   gc_f64 *__restrict__ vfull,
                                   const gc_f64 *__restrict__ x,
                                   const R *__restrict__ f,
                                   const gc_f64 *__restrict__ minv,
                                   gc_i64 pitch, double half_dt, int full,
                                   double massO, double massH,
                                   const struct gcn_join j,
                                   const struct gcn_tick_prep tp)
{
    __shared__ double sh[TICK ? GCN_RED_BLOCK : 1];
    double kin[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const unsigned int ovf = JOIN ? *j.overflow : 0u;
    for (gc_i64 w = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; w < nwater;
         w += (gc_i64)gridDim.x * blockDim.x) {
        int ia[3];
        double xw[3][3], vw[3][3], pre[3][3], fs[3][3], fac[3], mw[3];
        for (int k = 0; k < 3; ++k) ia[k] = wslot[(gc_i64)k * nwater + w];
        for (int k = 0; k < 3; ++k) {
            for (int c = 0; c < 3; ++c) {
                vw[c][k] = v[(gc_i64)c * pitch + ia[k]];
                xw[c][k] = x[(gc_i64)c * pitch + ia[k]];
            }
            fac[k] = half_dt * minv[ia[k]];
            force3_load<JOIN>(f, j, ovf, ia[k], pitch, fs[k]);
            if (TICK) mw[k] = __ldg(tp.mass + ia[k]);
        }
        for (int k = 0; k < 3; ++k) {
            double va[3], pa[3];
            for (int c = 0; c < 3; ++c) va[c] = vw[c][k];
            vv2_kick<TICK>(va, vhalf, vfull, fac[k], fs[k], ia[k], pitch,
                           full, pa);
            force3_store<JOIN, R>(j, ia[k], pitch, fs[k]);
            for (int c = 0; c < 3; ++c) { vw[c][k] = va[c]; pre[c][k] = pa[c]; }
        }
        rattle_water_solve<R>(xw, vw, massO, massH);
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < 3; ++c)
                v[(gc_i64)c * pitch + ia[k]] = vw[c][k];
        if (TICK)
            tick_members<3>(tp, ia, 3, vhalf, pitch,
                            [&](int i, int c) { return vw[c][i]; },
                            [&](int i, int c) { return pre[c][i]; }, kin, mw);
    }
    if (TICK) {
        block_reduce_sum(kin, 6, sh, tp.part + tp.off, tp.nblk);
        tick_fold(tp, sh);
    }
}

/* One hydrogen group's VV2 half kick and RATTLE, its arity as in
 * vv1_hgroup_row. */
template <int M, int JOIN, typename R, int TICK>
__device__ __forceinline__ void vv2_hgroup_row(
    gc_i64 g, gc_i64 ngrp, const gc_i32 *__restrict__ heavy,
    const gc_i32 *__restrict__ hslot, const gc_i32 *__restrict__ arity,
    const gc_f64 *__restrict__ dist, const gc_f64 *__restrict__ imh,
    const gc_f64 *__restrict__ im1, R *__restrict__ v,
    gc_f64 *__restrict__ vhalf, gc_f64 *__restrict__ vfull,
    const gc_f64 *__restrict__ x, const R *__restrict__ f,
    const gc_f64 *__restrict__ minv, gc_i64 pitch, double half_dt, int full,
    int iteration, double tolerance, const gc_gid *__restrict__ gid,
    gc_i64 *__restrict__ fail, const struct gcn_join &j, unsigned int ovf,
    const struct gcn_tick_prep &tp, double kin[6])
{
    int    a[M + 1];
    double c0[3], ch[M][3], v0[3], vh[M][3];
    double p0[3], ph[TICK ? M : 1][3];
    double r[M], im2[M];
    const int J = hgroup_row<M>(g, ngrp, heavy, hslot, arity, dist, imh,
                                 a, r, im2);
    int ih, c;
    double fs[M + 1][3], fac[M + 1];
    for (ih = 0; hg_in<M + 1>(ih, J + 1); ++ih) {
        fac[ih] = half_dt * minv[a[ih]];
        for (c = 0; c < 3; ++c) {
            const double va = v[(gc_i64)c * pitch + a[ih]];
            const double xa = x[(gc_i64)c * pitch + a[ih]];
            if (ih == 0) { v0[c] = va; c0[c] = xa; }
            else { vh[ih - 1][c] = va; ch[ih - 1][c] = xa; }
        }
        force3_load<JOIN>(f, j, ovf, a[ih], pitch, fs[ih]);
    }
    for (ih = 0; hg_in<M + 1>(ih, J + 1); ++ih) {
        double va[3], pa[3];
        for (c = 0; c < 3; ++c) va[c] = ih == 0 ? v0[c] : vh[ih - 1][c];
        vv2_kick<TICK>(va, vhalf, vfull, fac[ih], fs[ih], a[ih],
                       pitch, full, pa);
        force3_store<JOIN, R>(j, a[ih], pitch, fs[ih]);
        for (c = 0; c < 3; ++c) {
            if (ih == 0) v0[c] = va[c]; else vh[ih - 1][c] = va[c];
            if (TICK) { if (ih == 0) p0[c] = pa[c]; else ph[ih - 1][c] = pa[c]; }
        }
    }

    const int rattle_end = sizeof(R) < sizeof(double)
        ? rattle_hgroup_solve_f32<M>(J, c0, ch, v0, vh, r, im2, im1[g],
                                     iteration, tolerance)
        : rattle_hgroup_solve<M>(J, c0, ch, v0, vh, r, im2, im1[g],
                                 iteration, tolerance);

    for (c = 0; c < 3; ++c) v[(gc_i64)c*pitch + a[0]] = v0[c];
    for (ih = 0; hg_in<M>(ih, J); ++ih)
        for (c = 0; c < 3; ++c)
            v[(gc_i64)c*pitch + a[ih+1]] = vh[ih][c];
    if (!rattle_end) con_fail_note(fail, gid[a[0]]);
    if (TICK)
        tick_members<M + 1>(tp, a, J + 1, vhalf, pitch,
            [&](int k, int c) { return k == 0 ? v0[c] : vh[k - 1][c]; },
            [&](int k, int c) { return k == 0 ? p0[c] : ph[k - 1][c]; }, kin);
}

/* Hydrogen groups' VV2 half kick and RATTLE, the solve in R */
template <int JOIN, typename R, int TICK, int WIDE>
__global__ void gcn_kern_vv2_hgroup(gc_i64 ngrp,
                                    const gc_i32 *__restrict__ heavy,
                                    const gc_i32 *__restrict__ hslot,
                                    const gc_i32 *__restrict__ arity,
                                    const gc_f64 *__restrict__ dist,
                                    const gc_f64 *__restrict__ imh,
                                    const gc_f64 *__restrict__ im1,
                                    R *__restrict__ v,
                                    gc_f64 *__restrict__ vhalf,
                                    gc_f64 *__restrict__ vfull,
                                    const gc_f64 *__restrict__ x,
                                    const R *__restrict__ f,
                                    const gc_f64 *__restrict__ minv,
                                    gc_i64 pitch, double half_dt, int full,
                                    int iteration, double tolerance,
                                    const gc_gid *__restrict__ gid,
                                    gc_i64 *__restrict__ fail,
                                    const struct gcn_join j,
                                    const struct gcn_tick_prep tp)
{
    __shared__ double sh[TICK ? GCN_RED_BLOCK : 1];
    double kin[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const unsigned int ovf = JOIN ? *j.overflow : 0u;
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngrp;
         g += (gc_i64)gridDim.x * blockDim.x) {
#define GCN_VV2_HGROUP(M)                                                    \
        vv2_hgroup_row<M, JOIN, R, TICK>(g, ngrp, heavy, hslot, arity, dist,  \
                                 imh, im1, v, vhalf, vfull, x, f, minv,     \
                                 pitch, half_dt, full, iteration, tolerance,\
                                 gid, fail, j, ovf, tp, kin)
        if (arity[g] <= 4) GCN_VV2_HGROUP(4);
        else if (WIDE)     GCN_VV2_HGROUP(GC_MAX_HGROUP_H);
        else               con_fail_note(fail, gid[heavy[g]]);
#undef GCN_VV2_HGROUP
    }
    if (TICK) {
        block_reduce_sum(kin, 6, sh, tp.part + tp.off, tp.nblk);
        tick_fold(tp, sh);
    }
}

/* The hydrogen-group solvers without their wide body when no group has
 * more than four hydrogens (the wide body's registers set the kernels'
 * occupancy) */
template <int VIEW, typename R>
static inline decltype(&gcn_kern_vv1_hgroup<VIEW, R, 1>)
vv1_hgroup_kernel(int wide)
{
    return wide ? gcn_kern_vv1_hgroup<VIEW, R, 1> : gcn_kern_vv1_hgroup<VIEW, R, 0>;
}

template <int JOIN, typename R, int TICK>
static inline decltype(&gcn_kern_vv2_hgroup<JOIN, R, TICK, 1>)
vv2_hgroup_kernel(int wide)
{
    return wide ? gcn_kern_vv2_hgroup<JOIN, R, TICK, 1>
                : gcn_kern_vv2_hgroup<JOIN, R, TICK, 0>;
}

/* Flatten the sorted groups into the solvers' inputs: a water is its three
 * members in water_list order; a hydrogen group is its heavy atom and the
 * hydrogens that follow, with bond distances from the rigid descriptor by
 * representative GID. Rebuilt whenever the groups are sorted. */
__global__ void gcn_kern_rigid_flag(const gc_u8 *__restrict__ gkind,
                                    gc_i32 *__restrict__ fw,
                                    gc_i32 *__restrict__ fh, gc_i64 ngroup)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        fw[g] = (gkind[g] == GC_GROUP_WATER)  ? 1 : 0;
        fh[g] = (gkind[g] == GC_GROUP_HGROUP) ? 1 : 0;
    }
}

__global__ void gcn_kern_rigid_fill(const gc_u8 *__restrict__ gkind,
                                    const gc_i64 *__restrict__ goff,
                                    const gc_i32 *__restrict__ gmem,
                                    const gc_gid *__restrict__ ggid,
                                    const gc_i32 *__restrict__ fw,
                                    const gc_i32 *__restrict__ fh,
                                    const gc_i64 *__restrict__ ow,
                                    const gc_i64 *__restrict__ oh,
                                    const gc_f64 *__restrict__ inv_mass,
                                    const gc_gid *__restrict__ rg_gid,
                                    const gc_f64 *__restrict__ rg_dist,
                                    gc_i64 rg_n,
                                    gc_i32 *__restrict__ wslot,
                                    gc_i32 *__restrict__ wgrp,
                                    gc_i32 *__restrict__ heavy,
                                    gc_i32 *__restrict__ hgrp,
                                    gc_i32 *__restrict__ hslot,
                                    gc_i32 *__restrict__ arity,
                                    gc_f64 *__restrict__ hdist,
                                    gc_f64 *__restrict__ imh,
                                    gc_f64 *__restrict__ im1,
                                    gc_i64 ngroup, gc_i64 nwater,
                                    gc_i64 nhgroup,
                                    gc_i64 *__restrict__ bad)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        gc_i64 b = goff[g], e = goff[g + 1];
        if (fw[g]) {
            gc_i64 w = ow[g];
            wgrp[w] = (gc_i32)g;
            for (int k = 0; k < 3 && b + k < e; ++k)
                wslot[(gc_i64)k * nwater + w] = gmem[b + k];
        } else if (fh[g]) {
            gc_i64 h = oh[g];
            int J = (int)(e - b) - 1;
            if (J < 1 || J > GC_MAX_HGROUP_H) {
                atomicAdd((unsigned long long *)bad, 1ull);
                continue;
            }
            heavy[h] = gmem[b];
            hgrp[h]  = (gc_i32)g;
            arity[h] = J;
            im1[h]   = inv_mass[gmem[b]];
            gc_i64 lo = 0, hi = rg_n;
            while (lo < hi) {
                gc_i64 m = (lo + hi) >> 1;
                if (rg_gid[m] < ggid[g]) lo = m + 1; else hi = m;
            }
            if (lo >= rg_n || rg_gid[lo] != ggid[g]) {
                atomicAdd((unsigned long long *)bad, 1ull);
                continue;
            }
            for (int k = 0; k < J; ++k) {
                hslot[(gc_i64)k * nhgroup + h] = gmem[b + 1 + k];
                hdist[(gc_i64)k * nhgroup + h] =
                    rg_dist[(gc_i64)GC_MAX_HGROUP_H * lo + k];
                imh[(gc_i64)k * nhgroup + h]   = inv_mass[gmem[b + 1 + k]];
            }
        }
    }
}

namespace gcn {

static gc_i64 grid_of(gc_i64 n, gc_i64 block)
{
    gc_i64 g = (n + block - 1) / block;
    if (g > GCN_MAX_BLOCKS) g = GCN_MAX_BLOCKS;
    if (g < 1) g = 1;
    return g;
}

/* The final sums of `ncomp` quantities over the partials of `nitem` items at
 * part, written by the finishing launch into pinned out_pin + off and read
 * once d->stream has run to them. */
static void finalize_launch(struct gcn_device *d, const gc_f64 *part,
                            gc_i64 nitem, gc_i32 ncomp, gc_f64 *out)
{
    gc_i64 nb = grid_of(nitem, GCN_RED_BLOCK);
    gcn_kern_reduce_finalize<<<1, GCN_RED_BLOCK, 0, d->stream>>>(
        part, (int)nb, ncomp, out);
}

static gc_status read_out(struct gcn_device *d, gc_f64 *dst, gc_i32 n)
{
    gc_status st = dev_sync(d->stream);
    if (st != GC_OK) return st;
    for (int c = 0; c < n; ++c) dst[c] = d->out_pin[c];
    return GC_OK;
}

gc_status native_reduce(gc_context *ctx, gc_f64 *dst, gc_i32 ncomp)
{
    struct gcn_device *d = ctx->native;
    finalize_launch(d, d->reduce_partial, d->num_owned, ncomp,
                    d->out_pin);
    return read_out(d, dst, ncomp);
}

static gc_status reduce_n(gc_context *ctx, gc_i64 nitem, gc_f64 *dst,
                          gc_i32 ncomp)
{
    struct gcn_device *d = ctx->native;
    finalize_launch(d, d->reduce_partial, nitem, ncomp, d->out_pin);
    return read_out(d, dst, ncomp);
}

gc_status native_reduce_n(gc_context *ctx, gc_i64 nitem, gc_f64 *dst,
                          gc_i32 ncomp)
{
    return reduce_n(ctx, nitem, dst, ncomp);
}

/* reduce_n, then the sum over every rank of the simulation communicator, for
 * quantities stock takes globally (kinetic tensors, centre-of-mass sums); per-
 * rank partials that compute_dynvars reduces itself use reduce_n. */
static gc_status reduce_n_global(gc_context *ctx, gc_i64 nitem, gc_f64 *dst,
                                 gc_i32 ncomp)
{
    gc_status st = reduce_n(ctx, nitem, dst, ncomp);
    if (st != GC_OK || ctx->nproc <= 1) return st;
#ifdef HAVE_MPI_GENESIS
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)ctx->comm);
    if (MPI_Allreduce(MPI_IN_PLACE, dst, ncomp, MPI_DOUBLE, MPI_SUM, comm)
            != MPI_SUCCESS) return GC_E_STATE;
    return GC_OK;
#else
    return GC_E_UNSUPPORTED;
#endif
}

/* The step's lanes: independent launches run beside d->stream on lane[0..n-1],
 * forked from and joined back to it by events; a lane changes when work runs,
 * never what it computes. */
gc_status native_lanes_fork(struct gcn_device *d, int n)
{
    if (cudaEventRecord(d->lane_fork, d->stream) != cudaSuccess)
        return GC_E_DEVICE;
    for (int k = 0; k < n; ++k)
        if (cudaStreamWaitEvent(d->lane[k], d->lane_fork, 0) != cudaSuccess)
            return GC_E_DEVICE;
    return GC_OK;
}

gc_status native_lanes_join(struct gcn_device *d, int n)
{
    for (int k = 0; k < n; ++k)
        if (cudaEventRecord(d->lane_done[k], d->lane[k]) != cudaSuccess ||
            cudaStreamWaitEvent(d->stream, d->lane_done[k], 0) != cudaSuccess)
            return GC_E_DEVICE;
    return GC_OK;
}

/* The step's own streams and small buffers, once per device state. */
static gc_status step_words(struct gcn_device *d, int nproc)
{
    if (!d->con_fail) {
        const gc_i64 seed[2] = { 0, (gc_i64)0x7fffffffffffffffLL };
        if (dev_calloc((void **)&d->con_fail, sizeof(seed)) != GC_OK ||
            cudaMemcpy(d->con_fail, seed, sizeof(seed),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return GC_E_NOMEM;
    }
    if (!d->out_pin &&
        cudaHostAlloc((void **)&d->out_pin, 16 * sizeof(gc_f64),
                      cudaHostAllocDefault) != cudaSuccess)
        return GC_E_NOMEM;
    if (!d->guard_idx &&
        dev_calloc((void **)&d->guard_idx, 2 * sizeof(gc_i64)) != GC_OK)
        return GC_E_NOMEM;
    if (!d->guard_max &&
        dev_calloc((void **)&d->guard_max, 2 * sizeof(gc_f64)) != GC_OK)
        return GC_E_NOMEM;
    if (!d->thermo &&
        (dev_calloc((void **)&d->thermo, sizeof(struct gcn_thermo)) != GC_OK ||
         cudaHostAlloc((void **)&d->thermo_pin, sizeof(struct gcn_thermo),
                       cudaHostAllocDefault) != cudaSuccess ||
         cudaEventCreateWithFlags(&d->thermo_up, cudaEventDisableTiming) !=
             cudaSuccess))
        return GC_E_NOMEM;
    if (!d->aux_stream) {
        int lo = 0, hi = 0;
        if (cudaDeviceGetStreamPriorityRange(&lo, &hi) != cudaSuccess ||
            cudaStreamCreateWithPriority(&d->aux_stream, cudaStreamNonBlocking,
                                         lo) != cudaSuccess ||
            cudaEventCreateWithFlags(&d->aux_fork, cudaEventDisableTiming) !=
                cudaSuccess ||
            cudaEventCreateWithFlags(&d->aux_done, cudaEventDisableTiming) !=
                cudaSuccess ||
            cudaEventCreateWithFlags(&d->lane_fork, cudaEventDisableTiming) !=
                cudaSuccess ||
            cudaStreamCreateWithPriority(&d->bnd_stream, cudaStreamNonBlocking,
                                         gcn_stream_priority(lo, hi,
                                                             nproc > 2 ? 2 : 1))
                != cudaSuccess ||
            cudaEventCreateWithFlags(&d->bnd_fork, cudaEventDisableTiming) !=
                cudaSuccess ||
            cudaEventCreateWithFlags(&d->bnd_done, cudaEventDisableTiming) !=
                cudaSuccess)
            return GC_E_DEVICE;
        for (int k = 0; k < GCN_LANES; ++k)
            if (cudaStreamCreateWithPriority(&d->lane[k], cudaStreamNonBlocking,
                                             gcn_stream_priority(lo, hi, 1)) !=
                    cudaSuccess ||
                cudaEventCreateWithFlags(&d->lane_done[k],
                                         cudaEventDisableTiming) !=
                    cudaSuccess)
                return GC_E_DEVICE;
    }
    return GC_OK;
}

void native_step_release(struct gcn_device *d)
{
    for (int k = 0; k < GCN_GRAPH_KINDS; ++k)
        for (int j = 0; j < GCX_SLOTS; ++j) {
            if (d->graph[k][j]) cudaGraphExecDestroy(d->graph[k][j]);
            d->graph[k][j] = 0;
        }
    d->graph_pending = 0;
    dev_release((void **)&d->con_fail);
    if (d->out_pin) { cudaFreeHost(d->out_pin); d->out_pin = 0; }
    dev_release((void **)&d->thermo);
    dev_release((void **)&d->guard_idx);
    dev_release((void **)&d->guard_max);
    d->guard_ready = 0;
    gcx_wsum_destroy(d->thermo_wsum);
    d->thermo_wsum = 0;
    d->thermo_wsum_tried = 0;
    if (d->thermo_pin) { cudaFreeHost(d->thermo_pin); d->thermo_pin = 0; }
    if (d->thermo_up) { cudaEventDestroy(d->thermo_up); d->thermo_up = 0; }
    for (int k = 0; k < GCN_LANES; ++k) {
        if (d->lane[k])      cudaStreamDestroy(d->lane[k]);
        if (d->lane_done[k]) cudaEventDestroy(d->lane_done[k]);
        d->lane[k] = 0; d->lane_done[k] = 0;
    }
    if (d->lane_fork)  cudaEventDestroy(d->lane_fork);
    if (d->aux_stream) cudaStreamDestroy(d->aux_stream);
    if (d->aux_fork)   cudaEventDestroy(d->aux_fork);
    if (d->aux_done)   cudaEventDestroy(d->aux_done);
    d->lane_fork = 0; d->aux_stream = 0; d->aux_fork = d->aux_done = 0;
    if (d->bnd_stream) cudaStreamDestroy(d->bnd_stream);
    if (d->bnd_fork)   cudaEventDestroy(d->bnd_fork);
    if (d->bnd_done)   cudaEventDestroy(d->bnd_done);
    d->bnd_stream = 0; d->bnd_fork = d->bnd_done = 0;
}

/* The guard's measurement joins the step before d->stream reads reduce_partial
 * or the host reads guard_pin, and before the next step moves the coordinates. */
static gc_status aux_join(struct gcn_device *d)
{
    return cudaStreamWaitEvent(d->stream, d->aux_done, 0) == cudaSuccess
           ? GC_OK : GC_E_DEVICE;
}

}  /* namespace gcn */

namespace gcn {

__global__ void gcn_kern_excl_wid(const gc_i32 *__restrict__ wslot,
                                  gc_i64 nwater, gc_i32 *__restrict__ wid)
{
    for (gc_i64 w = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; w < nwater;
         w += (gc_i64)gridDim.x * blockDim.x)
        for (int k = 0; k < 3; ++k) wid[wslot[(gc_i64)k * nwater + w]] = (gc_i32)w;
}

/* The excluded pairs whose endpoints are not the same rigid water, in any
 * order (the pair kernel's adds are fixed point) */
__global__ void gcn_kern_excl_keep(const struct gcn_term_pair *__restrict__ t,
                                   gc_i64 n, const gc_i32 *__restrict__ wid,
                                   struct gcn_term_pair *__restrict__ keep,
                                   unsigned long long *__restrict__ count)
{
    const int lane = threadIdx.x & 31;
    for (gc_i64 b = (gc_i64)blockIdx.x * blockDim.x; b < n;
         b += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 i = b + threadIdx.x;
        struct gcn_term_pair q;
        bool k = false;
        if (i < n) {
            q = t[i];
            const gc_i32 w = wid[q.a0];
            k = w < 0 || w != wid[q.a1];
        }
        const unsigned m = __ballot_sync(0xffffffffu, k);
        unsigned long long base = 0;
        if (lane == 0 && m) base = atomicAdd(count, (unsigned long long)__popc(m));
        base = __shfl_sync(0xffffffffu, base, 0);
        if (k) keep[base + __popc(m & ((1u << lane) - 1u))] = q;
    }
}

/* The mixed mode's force-only steps skip excluded pairs inside a rigid water:
 * such a pair's force lies along a water edge, a direction SETTLE (VV1) or
 * RATTLE (VV2) removes exactly, and the constraint virial absorbs its virial.
 * Steps that publish energy and virial use the full list. Flexible waters,
 * waters not wholly on this rank, and FP64 runs keep every pair. */
/* The filter's launches and its count's readback into *kn (pinned), with
 * no synchronisation; *launched says whether it ran. */
static gc_status excl_keep_launch(struct gcn_device *d, cudaStream_t s,
                                  gc_i64 *kn, int *launched)
{
    *launched = 0;
    d->excl_keep_valid = 0;
    const bool checked = false;
    const gc_i64 n = d->term_count[GC_TERM_EXCL];
    if (checked || d->tab.nonbond_precision != GC_NONBOND_MIXED || d->num_water <= 0 ||
        n <= 0 || d->term_pair_n[GC_TERM_EXCL] != n ||
        d->tab.water_bond_calc || d->tab.water_angle_calc)
        return GC_OK;
    gc_status st;
    const gc_i64 wb = d->pitch * (gc_i64)sizeof(gc_i32);
    if (wb > d->excl_keep_wid_cap) {
        dev_release((void **)&d->excl_keep_wid);
        d->excl_keep_wid_cap = 0;
        st = dev_calloc((void **)&d->excl_keep_wid, wb + wb / 4);
        if (st != GC_OK) return st;
        d->excl_keep_wid_cap = wb + wb / 4;
    }
    const gc_i64 kb = n * (gc_i64)sizeof(struct gcn_term_pair) + 16;
    if (kb > d->excl_keep_cap) {
        dev_release((void **)&d->excl_keep);
        d->excl_keep_cap = 0;
        st = dev_calloc((void **)&d->excl_keep, kb + kb / 4);
        if (st != GC_OK) return st;
        d->excl_keep_cap = kb + kb / 4;
    }
    unsigned long long *cnt = (unsigned long long *)(d->excl_keep + n);
    if (cudaMemsetAsync(d->excl_keep_wid, 0xff, wb, s) != cudaSuccess ||
        cudaMemsetAsync(cnt, 0, sizeof(*cnt), s) != cudaSuccess)
        return GC_E_DEVICE;
    gcn_kern_excl_wid<<<(unsigned)grid_of(d->num_water, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
        d->water_slot, d->num_water, d->excl_keep_wid);
    gcn_kern_excl_keep<<<(unsigned)grid_of(n, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
        d->term_pair[GC_TERM_EXCL], n, d->excl_keep_wid, d->excl_keep, cnt);
    if (cudaMemcpyAsync(kn, cnt, sizeof(*kn), cudaMemcpyDeviceToHost, s) != cudaSuccess)
        return GC_E_DEVICE;
    *launched = 1;
    return dev_launched("excl_keep");
}

/* The filter's count, once its readback has landed. */
static gc_status excl_keep_finish(struct gcn_device *d, gc_i64 kn)
{
    if (kn < 0 || kn > d->term_count[GC_TERM_EXCL]) return GC_E_STATE;
    d->excl_keep_n = kn;
    d->excl_keep_valid = 1;
    return GC_OK;
}

/* Whether the rigid build counts its groups afresh: after a migration
 * (rigid_ready cleared). */
int native_rigid_recount(const struct gcn_device *d)
{
    return !d->rigid_ready;
}

/* The rigid build's flags, offsets and counts for the current groups, queued
 * in rigid_tmp; with `pin`, the water and hydrogen-group counts are read back
 * into pin[0], pin[1]. Grow-only storage, so a rebuild neither allocates nor
 * frees. */
static gc_status rigid_count(struct gcn_device *d, gc_i64 *pin)
{
    cudaStream_t s = d->stream;
    const gc_i64 ng = d->num_groups;
    gc_status st;
    const gc_i64 tmp = 2 * ng * (gc_i64)sizeof(gc_i32) +
                       2 * ng * (gc_i64)sizeof(gc_i64) + 4 * (gc_i64)sizeof(gc_i64);
    if (tmp > d->rigid_tmp_cap) {
        dev_release(&d->rigid_tmp);
        d->rigid_tmp_cap = 0;
        st = dev_calloc(&d->rigid_tmp, tmp + tmp / 4);
        if (st) return st;
        d->rigid_tmp_cap = tmp + tmp / 4;
    }
    gc_i64 *ow = (gc_i64 *)d->rigid_tmp, *oh = ow + ng;
    gc_i64 *tot = oh + ng;                   /* nw, bad, -, nh */
    gc_i32 *fw = (gc_i32 *)(tot + 4), *fh = fw + ng;
    if (cudaMemsetAsync(fw, 0, 2 * (size_t)ng * sizeof(gc_i32), s) != cudaSuccess ||
        cudaMemsetAsync(tot, 0, 4 * sizeof(gc_i64), s) != cudaSuccess)
        return GC_E_DEVICE;
    gcn_kern_rigid_flag<<<(unsigned)grid_of(ng, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
        d->group_kind, fw, fh, ng);
    st = native_scan(d, fw, ow, ng, tot);
    if (st == GC_OK) st = native_scan(d, fh, oh, ng, tot + 3);
    if (st != GC_OK) return st;
    if (pin &&
        (cudaMemcpyAsync(pin, tot, sizeof(gc_i64), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
         cudaMemcpyAsync(pin + 1, tot + 3, sizeof(gc_i64), cudaMemcpyDeviceToHost, s) !=
             cudaSuccess))
        return GC_E_DEVICE;
    return GC_OK;
}

/* The rebuild's early count (pinned words 58, 59), from which the rigid build
 * of the same groups starts. */
gc_status native_rigid_count_early(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;
    d->rigid_counted = 0;
    if (d->num_groups <= 0 || !native_rigid_recount(d)) return GC_OK;
    gc_i64 *pin = native_rebuild_pin(d);
    if (!pin) return GC_E_NOMEM;
    return rigid_count(d, pin + 58);
}

gc_status native_build_rigid(gc_context *ctx, struct gcn_rigid_pending *pend)
{
    pend->arity = 0;
    pend->excl = 0;
    struct gcn_device *d = ctx->native;
    cudaStream_t s = d->stream;
    const gc_i64 ng = d->num_groups;
    gc_status st;

    if (ng <= 0) { d->num_water = 0; d->num_hgroup = 0; return GC_OK; }

    const int counted = d->rigid_counted;
    d->rigid_counted = 0;
    if (!counted) {
        st = rigid_count(d, 0);
        if (st != GC_OK) return st;
    }
    gc_i64 *ow = (gc_i64 *)d->rigid_tmp, *oh = ow + ng;
    gc_i64 *tot = oh + ng;                   /* nw, bad, -, nh */
    gc_i32 *fw = (gc_i32 *)(tot + 4), *fh = fw + ng;
    const int same = !native_rigid_recount(d);
    gc_i64 cnt[4] = { d->num_water, 0, 0, d->num_hgroup };
    if (!same && counted) {
        const gc_i64 *pin = native_rebuild_pin(d);
        cnt[0] = pin[58];
        cnt[3] = pin[59];
    } else if (!same) {
        if (cudaMemcpyAsync(cnt, tot, sizeof(cnt), cudaMemcpyDeviceToHost, s)
                != cudaSuccess) return GC_E_DEVICE;
        if (dev_sync(s) != GC_OK) return GC_E_DEVICE;
    }
    const gc_i64 nw = cnt[0], nh = cnt[3];

    const gc_i64 ntick = 6 * (grid_of(ng, GCN_BLOCK) + grid_of(nw, GCN_BLOCK) +
                              grid_of(nh, GCN_BLOCK));
    if (ntick > d->tick_part_cap) {
        dev_release((void **)&d->tick_part);
        d->tick_part_cap = 0;
        st = dev_calloc((void **)&d->tick_part,
                        (ntick + ntick / 4) * (gc_i64)sizeof(gc_f64));
        if (st) return st;
        d->tick_part_cap = ntick + ntick / 4;
    }

    if (nw > d->water_cap || d->water_slot == 0) {
        const gc_i64 m = nw + nw / 4 + 16;
        dev_release((void **)&d->water_slot);
        dev_release((void **)&d->water_group);
        d->water_cap = 0;
        st = dev_calloc((void **)&d->water_slot, 3 * m * (gc_i64)sizeof(gc_i32));
        if (st) return st;
        st = dev_calloc((void **)&d->water_group, m * (gc_i64)sizeof(gc_i32));
        if (st) return st;
        d->water_cap = m;
    }
    if (nh > d->hgroup_cap || d->hgr_heavy == 0) {
        dev_release((void **)&d->hgr_heavy);
        dev_release((void **)&d->hgr_group);
        dev_release((void **)&d->hgr_h);
        dev_release((void **)&d->hgr_dist);
        dev_release((void **)&d->hgr_inv_mass_h);
        dev_release((void **)&d->hgr_inv_mass_c);
        dev_release((void **)&d->hgr_arity);
        d->hgroup_cap = 0;
        const gc_i64 m = nh + nh / 4 + 16;
        st = dev_calloc((void **)&d->hgr_heavy, m * (gc_i64)sizeof(gc_i32));
        if (st) return st;
        st = dev_calloc((void **)&d->hgr_group, m * (gc_i64)sizeof(gc_i32));
        if (st) return st;
        st = dev_calloc((void **)&d->hgr_arity, m * (gc_i64)sizeof(gc_i32));
        if (st) return st;
        st = dev_calloc((void **)&d->hgr_inv_mass_c, m * (gc_i64)sizeof(gc_f64));
        if (st) return st;
        st = dev_calloc((void **)&d->hgr_h,
                        GC_MAX_HGROUP_H * m * (gc_i64)sizeof(gc_i32));
        if (st) return st;
        st = dev_calloc((void **)&d->hgr_dist,
                        GC_MAX_HGROUP_H * m * (gc_i64)sizeof(gc_f64));
        if (st) return st;
        st = dev_calloc((void **)&d->hgr_inv_mass_h,
                        GC_MAX_HGROUP_H * m * (gc_i64)sizeof(gc_f64));
        if (st) return st;
        d->hgroup_cap = m;
    } else if (nh > 0) {
        const size_t hm = (size_t)(GC_MAX_HGROUP_H * nh);
        if (cudaMemsetAsync(d->hgr_h, 0, hm * sizeof(gc_i32), s) != cudaSuccess ||
            cudaMemsetAsync(d->hgr_dist, 0, hm * sizeof(gc_f64), s) != cudaSuccess ||
            cudaMemsetAsync(d->hgr_inv_mass_h, 0, hm * sizeof(gc_f64), s) != cudaSuccess)
            return GC_E_DEVICE;
    }
    d->num_water  = nw;
    d->num_hgroup = nh;

    gcn_kern_rigid_fill<<<(unsigned)grid_of(ng, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
        d->group_kind, d->group_offset, d->group_member, d->group_gid,
        fw, fh, ow, oh, d->inv_mass, d->rigid_gid, d->rigid_dist,
        d->rigid_count, d->water_slot, d->water_group, d->hgr_heavy,
        d->hgr_group, d->hgr_h, d->hgr_arity,
        d->hgr_dist, d->hgr_inv_mass_h, d->hgr_inv_mass_c, ng, nw, nh,
        same ? d->verdict + 3 : tot + 1);
    /* The arity check and the exclusion filter's count come back pinned with
     * the caller's next synchronisation (native_build_rigid_finish); the count
     * is published only after the check. */
    gc_i64 *pin = native_rebuild_pin(d);
    if (!pin) return GC_E_NOMEM;
    if (!same) {
        if (cudaMemcpyAsync(pin + 60, tot + 1, sizeof(gc_i64),
                            cudaMemcpyDeviceToHost, s) != cudaSuccess)
            return GC_E_DEVICE;
        pend->arity = 1;
    }
    st = excl_keep_launch(d, s, pin + 61, &pend->excl);
    if (st != GC_OK) return st;
    return dev_launched("rigid_fill");
}

gc_status native_build_rigid_finish(gc_context *ctx,
                                    const struct gcn_rigid_pending *pend)
{
    struct gcn_device *d = ctx->native;
    if (!pend->arity && !pend->excl) return GC_OK;
    gc_status st = dev_phase(d->stream, pend->arity ? "rigid_fill" : "excl_keep");
    if (st != GC_OK) return st;
    const gc_i64 *pin = native_rebuild_pin(d);
    if (pend->arity) {
        const gc_i64 bad = pin[60];
        d->rigid_ready = (bad == 0);
        if (bad != 0) return GC_E_ARITY;
    }
    return pend->excl ? excl_keep_finish(d, pin[61]) : GC_OK;
}

}  /* namespace gcn */

using namespace gcn;

extern "C" gc_status gpu_core_attach_tables(gc_context *ctx,
                                            const gc_table_desc *tables,
                                            const gc_constraint_desc *con,
                                            gc_i64 rigid_count,
                                            const gc_gid *rigid_gid,
                                            const gc_i32 *rigid_arity,
                                            const gc_f64 *rigid_dist,
                                            const gc_pme_desc *pme)
{
    if (ctx == 0 || tables == 0 || con == 0) return GC_E_ARG;
    if (rigid_count > 0 && (!rigid_gid || !rigid_arity || !rigid_dist))
        return GC_E_ARG;
    if (ctx->epoch == 0) return GC_E_STATE;
    if (ctx->native == 0) return GC_E_STATE;

    gc_status st = native_attach_tables(ctx, tables, con);
    if (st != GC_OK) return st;
    if (tables->nonbond_precision == GC_NONBOND_MIXED) {
        st = native_vf_narrow(ctx->native);
        if (st != GC_OK) return st;
    }

    struct gcn_device *d = ctx->native;
    if (step_words(d, ctx->nproc) != GC_OK) return GC_E_NOMEM;
    d->rigid_count = 0;
    if (rigid_count > 0) {
        st = dev_calloc((void **)&d->rigid_gid,
                        rigid_count * (gc_i64)sizeof(gc_gid));
        if (st) return st;
        st = dev_calloc((void **)&d->rigid_arity,
                        rigid_count * (gc_i64)sizeof(gc_i32));
        if (st) return st;
        st = dev_calloc((void **)&d->rigid_dist,
                        GC_MAX_HGROUP_H * rigid_count * (gc_i64)sizeof(gc_f64));
        if (st) return st;
        if (cudaMemcpy(d->rigid_gid, rigid_gid,
                       (size_t)rigid_count * sizeof(gc_gid),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d->rigid_arity, rigid_arity,
                       (size_t)rigid_count * sizeof(gc_i32),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return GC_E_DEVICE;
        if (cudaMemcpy(d->rigid_dist, rigid_dist,
                       (size_t)(GC_MAX_HGROUP_H * rigid_count) * sizeof(gc_f64),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return GC_E_DEVICE;
        d->rigid_count = rigid_count;
    }
    int wide = 0;
    for (gc_i64 i = 0; i < rigid_count; ++i)
        if (rigid_arity[i] > 4) wide = 1;
#ifdef HAVE_MPI_GENESIS
    if (ctx->nproc > 1) {
        int any = 0;
        if (MPI_Allreduce(&wide, &any, 1, MPI_INT, MPI_MAX,
                          MPI_Comm_f2c((MPI_Fint)ctx->comm)) != MPI_SUCCESS)
            return GC_E_DEVICE;
        wide = any;
    }
#endif
    d->rigid_wide = wide;

    if (pme == 0) {
        d->reciprocal_ready = 0;
        return GC_OK;
    }
    return native_pme_plan(ctx, pme);
}

/* A group that did not converge is an error with an identity, not a
 * silently wrong coordinate. */
static gc_status constraint_failed(const gc_i64 fc[2])
{
    std::fprintf(stderr,
                 "GPU_Core_Constraint> %lld group(s) did not converge; "
                 "first global id %lld\n",
                 (long long)fc[0], (long long)fc[1]);
    return GC_E_UNSUPPORTED;
}

/* The deferred checks of every step since the last read, once the caller has
 * synchronised d->stream past them. Counts are sticky on the device and any
 * count is fatal. */
gc_status gcn::native_verdict_check(gc_context *ctx, const gc_i64 *v,
                                    const gc_i64 *fc)
{
    gc_status st = GC_OK;
    if (v[0]) {
        std::fprintf(stderr, "GPU_Core_Error> phase=force_coordinates "
                     "invalid_groups_or_atoms=%lld\n", (long long)v[0]);
        st = GC_E_UNSUPPORTED;
    } else if (v[1] || v[2]) {
        std::fprintf(stderr, "GPU_Core_Error> phase=halo_verdict rank=%d "
                     "coord_bad=%lld force_bad=%lld\n", (int)ctx->rank,
                     (long long)v[1], (long long)v[2]);
        st = v[1] ? GC_E_ENDPOINT : GC_E_OWNER;
    } else if (v[3]) {
        std::fprintf(stderr, "GPU_Core_Error> phase=rigid_fill rank=%d "
                     "bad_arity=%lld\n", (int)ctx->rank, (long long)v[3]);
        st = GC_E_ARITY;
    } else if (v[4] || v[5] || v[6]) {
        std::fprintf(stderr, "GPU_Core_Error> phase=mask_exclusions rank=%d "
                     "unresolved_endpoints=%lld unplaced_terms=%lld "
                     "list_traps=%lld\n", (int)ctx->rank, (long long)v[4],
                     (long long)v[5], (long long)v[6]);
        st = v[4] ? GC_E_ENDPOINT : v[5] ? GC_E_UNSUPPORTED : GC_E_MISMATCH;
    }
    if (st == GC_OK && fc && fc[0] != 0) st = constraint_failed(fc);
    return st;
}

static gc_status verdict_read(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;
    gc_i64 v[GCN_VERDICT_W], fc[2] = { 0, 0 };
    if (cudaMemcpy(v, d->verdict, sizeof(v), cudaMemcpyDeviceToHost) !=
            cudaSuccess) return GC_E_DEVICE;
    if (d->con_fail &&
        cudaMemcpy(fc, d->con_fail, sizeof(fc), cudaMemcpyDeviceToHost) !=
            cudaSuccess) return GC_E_DEVICE;
    return native_verdict_check(ctx, v, d->con_fail ? fc : 0);
}

extern "C" gc_status gpu_core_step_setup(gc_context *ctx,
                                         const gc_step_plan *plan,
                                         const char **reason_out)
{
    static const char *reason = "eligible";
    if (ctx == 0 || plan == 0) { if (reason_out) *reason_out = reason;
                                 return GC_E_ARG; }
    if (ctx->native == 0 || ctx->native->tables_attached == 0)
        return GC_E_STATE;

    if (plan->ensemble != GC_ENSEMBLE_NVE && plan->ensemble != GC_ENSEMBLE_NVT &&
        plan->ensemble != GC_ENSEMBLE_NPT) {
        reason = "only NVE, NVT and NPT are native";
        if (reason_out) *reason_out = reason;
        return GC_E_UNSUPPORTED;
    }
    /* the MTK barostat is ported for the group-temperature path only
     * (mtk_barostat_vv1/_vv2 with compute_vv1_group/compute_vv2_group) */
    if (plan->ensemble == GC_ENSEMBLE_NPT &&
        (!plan->group_tp || !plan->rigid_bond)) {
        reason = "NPT is native only with group_tp = YES and rigid_bond = YES";
        if (reason_out) *reason_out = reason;
        return GC_E_UNSUPPORTED;
    }
    if (plan->ensemble != GC_ENSEMBLE_NVE &&
        plan->thermostat != GC_THERMOSTAT_BERENDSEN &&
        plan->thermostat != GC_THERMOSTAT_BUSSI &&
        plan->thermostat != GC_THERMOSTAT_NHC) {
        reason = "only the Berendsen, Bussi and NHC thermostats are native";
        if (reason_out) *reason_out = reason;
        return GC_E_UNSUPPORTED;
    }
    struct gcn_device *d = ctx->native;
    /* More than one rank runs on the halo native_dist_create admitted; the
     * reciprocal sum is the pencil PME (gpu_pme.cu). */
    if (ctx->nproc != 1 && d->dist == 0) {
        reason = "the distributed halo was not created";
        if (reason_out) *reason_out = reason;
        return GC_E_UNSUPPORTED;
    }
    if (!ctx->real_mask_ready) {
        reason = "the final stock real-space exclusion mask was not imported";
        if (reason_out) *reason_out = reason;
        return GC_E_STATE;
    }

    gc_i64 want = (gc_i64)ctx->geo.cell[0] * ctx->geo.cell[1] * ctx->geo.cell[2];
    if (ctx->nproc == 1 && want != ctx->ncell_local) {
        reason = "one rank must own every cell for the native step";
        if (reason_out) *reason_out = reason;
        return GC_E_UNSUPPORTED;
    }

    if (step_words(d, ctx->nproc) != GC_OK) return GC_E_NOMEM;
    {   /* resolves the join's overflow flag here, outside any capture */
        struct gcn_join jv;
        if (native_force_join_view(ctx, &jv) != GC_OK) return GC_E_DEVICE;
    }
    d->plan = *plan;
    d->next_scheduled_step = plan->nbupdate_period;
    d->step_ready = 1;
    d->list_builds++;           /* no graph outlives the plan it captured */
    reason = "eligible";
    if (reason_out) *reason_out = reason;
    return GC_OK;
}

/* step_begin's reference save, and nvt_half's half velocity when pending,
 * queued on d->stream before anything reads the references or VV1 writes
 * coord_ref again. Every VV1 variant and kinetic readback calls it first; a
 * thermostat tick takes both into its own pass. */
gc_status gcn::native_ref_join(struct gcn_device *d)
{
    const gc_i64 nb = grid_of(d->num_owned, GCN_BLOCK);
    vf_each(d->vf32, [&](auto z) {
        using S = decltype(z);
        if (d->ref_pending)
            gcn_kern_save_ref<<<(unsigned)nb, GCN_BLOCK, 0, d->stream>>>(
                d->coord_ref, d->vel_ref, d->coord, d->vel.as<S>(),
                d->num_owned, d->pitch);
        if (d->half_pending && !d->tick_prep)
            gcn_kern_nvt_half<<<(unsigned)nb, GCN_BLOCK, 0, d->stream>>>(
                d->vel_half, d->vel.as<S>(), d->num_owned, d->pitch);
    });
    if (!d->ref_pending && !d->half_pending) return GC_OK;
    d->ref_pending = d->half_pending = 0;
    return dev_launched("step_begin");
}

extern "C" gc_status gpu_core_step_begin(gc_context *ctx, gc_i64 istep,
                                         gc_i32 want_energy)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->step_ready || !d->list_valid) return GC_E_STATE;
    if (d->graph_mode == GCN_GRAPH_REPLAY) {
        d->native_steps++;
        return GC_OK;
    }

    d->ref_pending = 1;
    d->view_ready = 0;
    if (want_energy && d->plan.ensemble == GC_ENSEMBLE_NVE && d->plan.group_tp)
        vf_each(d->vf32, [&](auto z) {
            using S = decltype(z);
            gcn_kern_kin_prep<<<(unsigned)grid_of(d->num_owned, GCN_BLOCK),
                                GCN_BLOCK, 0, d->stream>>>(
                d->vel_half, d->force.as<S>(), d->inv_mass, d->plan.half_dt,
                d->num_owned, d->pitch);
        });

    d->native_steps++;
    (void)istep;
    return dev_launched("step_begin");
}

/* The plain step as one CUDA graph. Its launches take no per-step argument
 * between list builds: counters, verdicts, the guard slot and the fused
 * exchanges' epochs (gcx_seq) live on the device. The first plain step of a
 * kind after a build is captured, only when every halo pass and PME move is
 * fused on a steady epoch sequence; the graph is updated in place when the
 * build left its topology alone, and later plain steps replay it. Launches are
 * deferred: a run of plain steps of one kind is launched back to back when a
 * step of another kind arrives, when GCN_GRAPH_BLOCK steps are pending, or
 * before anything else is queued (graph_flush). */
#define GCN_GRAPH_BLOCK 16
static_assert(GCN_GRAPH_BLOCK <= (int)(sizeof(((gcn_device *)0)->graph_pending_slot) /
                                     sizeof(gc_i32)), "pending slots");

/* Put the pending plain steps on the device. */
static gc_status graph_flush(struct gcn_device *d)
{
    const int n = d->graph_pending, kind = d->graph_pending_kind;
    d->graph_pending = 0;
    if (n == 0) return GC_OK;
    cudaError_t e = cudaSuccess;
    for (int k = 0; k < n && e == cudaSuccess; ++k)
        e = cudaGraphLaunch(d->graph[kind][d->graph_pending_slot[k]], d->stream);
    if (e == cudaSuccess) d->graph_segments += n;
    return e == cudaSuccess ? GC_OK : dev_launched("graph_launch");
}

/* Admission of the copy-engine fused exchanges. A fused pass or move whose
 * peer stores are slower than the copy engines can pack locally and copy with
 * them between device signal words, on the step's stream; the transport's own
 * path drives the same copies from the host on its stream. The rule is fixed
 * by topology: host-driven when every peer link of the run is PCIe and some
 * GPU exchanges with two or more peers, fused otherwise. Native atomics over
 * the link (cudaDevP2PAttrNativeAtomicSupported) stand in for NVLink: NVLink
 * pairs report them and PCIe peer links do not. Measured with both paths
 * forced (median of 5 interleaved runs, 3000 steps, mixed precision, PCIe
 * Gen5 peer links): at 4 ranks fused took 5.0% (STMV) and 6.7% (cellulose)
 * longer per step, at 2 ranks 2.1% and 3.9% less; on NVLink-paired GPUs it
 * was as fast or faster.
 * Decided once, after the first list generations (when every plan exists);
 * plain-step graphs need the fused path. Both paths move the same bytes into
 * the same slots, so results are bitwise identical. */
static gc_status ce_admission(gc_context *ctx, struct gcn_device *d)
{
    if (d->ce_state == 2) return GC_OK;
    if (ctx->nproc == 1) {
        d->ce_state = 2; d->ce_graph = 1;
        d->graph_fused = d->graph_stores = 1;
        return GC_OK;
    }
#ifdef HAVE_MPI_GENESIS
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)ctx->comm);
    const gc_i64 nb = d->plan.nbupdate_period > 0 ? d->plan.nbupdate_period : 10;
    const gc_i64 w0 = nb * ((50 + nb - 1) / nb);       /* >= 50 steps */
    if (d->ce_count++ < w0) return GC_OK;
    const gc_status fs = graph_flush(d);
    if (fs != GC_OK) return fs;
    int fu = 0, so = 0, pf = 1, ps = 1, peers = 0, atomic = 0;
    gcn::native_dist_graph_capability(ctx, &fu, &so);
    if (d->reciprocal_ready) native_pme_graph_capability(ctx, &pf, &ps);
    gcx_peer_fanout(&peers, &atomic);
    int mine[5] = { !(fu && pf), !(so && ps), peers >= 2, atomic,
                    gcn::native_dist_ce_candidate(ctx) ||
                    (d->reciprocal_ready && native_pme_ce_candidate(ctx)) },
        any[5] = { 0, 0, 0, 0, 0 };
    if (MPI_Allreduce(mine, any, 5, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS)
        return GC_E_STATE;
    d->graph_fused = !any[0];
    d->graph_stores = !any[1];
    d->ce_fused = !(any[2] && !any[3]);
    d->ce_graph = 1;
    d->ce_state = 2;
    if (ctx->rank == 0 && any[4])
        std::printf("Native_Xchg> copy-engine fused exchange: %s (peer links %s, "
                    "%s)\n", d->ce_fused ? "admitted" : "not admitted",
                    any[3] ? "with native atomics" : "PCIe",
                    any[2] ? "two or more peers per GPU" : "one peer per GPU");
    native_pme_allow_ce(ctx, d->ce_fused);
    return GC_OK;
#else
    (void)ctx;
    d->ce_state = 2;
    d->ce_graph = 1;
    d->graph_fused = d->graph_stores = 1;
    return GC_OK;
#endif
}

extern "C" gc_status gpu_core_graph_begin(gc_context *ctx, gc_i32 kind)
{
    if (ctx == 0 || ctx->native == 0 || kind < 0 || kind >= GCN_GRAPH_KINDS)
        return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (d->graph_mode != GCN_GRAPH_EAGER) return GC_E_STATE;
    if (d->ce_state != 2) {
        const gc_status cs = ce_admission(ctx, d);
        if (cs != GC_OK) return cs;
    }
    if (d->ce_state == 2 && !d->ce_graph) kind = GC_GRAPH_NONE;
    if (ctx->nproc == 1 || !d->list_valid || !d->step_ready)
        kind = GC_GRAPH_NONE;
    if (kind == GC_GRAPH_TICK && !d->thermo_wsum) kind = GC_GRAPH_NONE;
    d->graph_kind = kind;
    const int slot = d->reciprocal_ready ? native_pme_graph_slot(ctx) : 0;
    d->graph_slot = slot;
    if (kind != GC_GRAPH_NONE && d->graph[kind][slot] &&
        d->graph_build[kind][slot] == d->list_builds) {
        if (d->graph_pending && d->graph_pending_kind != kind) {
            const gc_status st = graph_flush(d);
            if (st != GC_OK) return st;
        }
        d->graph_mode = GCN_GRAPH_REPLAY;
        return GC_OK;
    }
    gc_status st = graph_flush(d);
    if (st != GC_OK) return st;
    if (kind == GC_GRAPH_NONE) return GC_OK;
    if (!d->graph_fused || !(d->graph_stores || d->ce_fused) ||
        !native_dist_graph_ready(ctx) ||
        (d->reciprocal_ready && !native_pme_graph_ready(ctx))) {
        d->graph_kind = GC_GRAPH_NONE;
        return GC_OK;
    }
    if (cudaStreamBeginCapture(d->stream, cudaStreamCaptureModeRelaxed) !=
        cudaSuccess)
        return dev_launched("graph_capture");
    d->graph_mode = GCN_GRAPH_CAPTURE;
    return GC_OK;
}

extern "C" gc_status gpu_core_graph_end(gc_context *ctx, gc_i32 next_kind)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    const int mode = d->graph_mode, kind = d->graph_kind, slot = d->graph_slot;
    d->graph_mode = GCN_GRAPH_EAGER;
    d->graph_kind = GC_GRAPH_NONE;
    if (mode == GCN_GRAPH_EAGER) return GC_OK;
    cudaGraphExec_t &x = d->graph[kind][slot];
    if (mode == GCN_GRAPH_CAPTURE) {
        cudaGraph_t g = 0;
        cudaError_t e = cudaStreamEndCapture(d->stream, &g);
        if (e == cudaSuccess && x) {
            cudaGraphExecUpdateResultInfo info;
            if (cudaGraphExecUpdate(x, g, &info) != cudaSuccess) {
                (void)cudaGetLastError();
                cudaGraphExecDestroy(x);
                x = 0;
            }
        }
        if (e == cudaSuccess && !x)
            e = cudaGraphInstantiate(&x, g,
                                     cudaGraphInstantiateFlagUseNodePriority);
        if (g) cudaGraphDestroy(g);
        if (e != cudaSuccess) {
            x = 0;
            std::fprintf(stderr, "GPU_Core_Error> phase=graph_capture "
                         "rank=%d cuda=%s\n", (int)ctx->rank,
                         cudaGetErrorString(e));
            return GC_E_DEVICE;
        }
        d->graph_build[kind][slot] = d->list_builds;
    }
    d->graph_pending_kind = kind;
    d->graph_pending_slot[d->graph_pending] = slot;
    if (++d->graph_pending < GCN_GRAPH_BLOCK && next_kind == kind)
        return GC_OK;
    return graph_flush(d);
}

extern "C" gc_status gpu_core_nvt_half(gc_context *ctx)
{
    if (!ctx || !ctx->native) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid || !d->step_ready ||
        d->plan.ensemble == GC_ENSEMBLE_NVE || d->native_steps <= 0)
        return GC_E_STATE;
    if (d->graph_mode == GCN_GRAPH_REPLAY) return GC_OK;
    if (d->ref_pending) { d->half_pending = 1; return GC_OK; }
    if (d->tick_prep) return GC_OK;
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_nvt_half<<<(unsigned)grid_of(d->num_owned, GCN_BLOCK),
                            GCN_BLOCK, 0, d->stream>>>(
            d->vel_half, d->vel.as<decltype(z)>(), d->num_owned, d->pitch);
    });
    return dev_launched("nvt_half");
}

/* One kinetic tensor's per-block partials at `part`, on stream `s`;
 * returns the item count the reduction runs over. */
template <class S>
static gc_i64 kinetic_launch_v(struct gcn_device *d, gc_i32 which,
                               const S *v, gc_f64 *part, cudaStream_t s)
{
    gc_i64 nitem;
    if (which >= GC_KIN_GROUP_VEL) {
        nitem = d->num_groups;
        gcn_kern_kinetic_group<<<(unsigned)grid_of(nitem, GCN_RED_BLOCK),
                                 GCN_RED_BLOCK, 0, s>>>(
            v, d->mass, d->group_offset, d->group_member, d->group_kind,
            nitem, d->pitch, part, (int)grid_of(nitem, GCN_RED_BLOCK));
    } else {
        nitem = d->num_owned;
        gcn_kern_kinetic_flat<<<(unsigned)grid_of(nitem, GCN_RED_BLOCK),
                                GCN_RED_BLOCK, 0, s>>>(
            v, d->mass, nitem, d->pitch, part,
            (int)grid_of(nitem, GCN_RED_BLOCK));
    }
    return nitem;
}

static gc_i64 kinetic_launch(struct gcn_device *d, gc_i32 which, gc_f64 *part,
                             cudaStream_t s)
{
    if (which == GC_KIN_FLAT_VEL_REF  || which == GC_KIN_GROUP_VEL_REF)
        return kinetic_launch_v(d, which, (const gc_f64 *)d->vel_ref, part, s);
    if (which == GC_KIN_FLAT_VEL_HALF || which == GC_KIN_GROUP_VEL_HALF)
        return kinetic_launch_v(d, which, (const gc_f64 *)d->vel_half, part, s);
    gc_i64 nitem = 0;
    vf_each(d->vf32, [&](auto z) {
        nitem = kinetic_launch_v(d, which, d->vel.as<const decltype(z)>(),
                                 part, s);
    });
    return nitem;
}

extern "C" gc_status gpu_core_kinetic(const gc_context *ctx, gc_i32 which,
                                      gc_f64 *kin3, gc_f64 *ekin)
{
    if (ctx == 0 || ctx->native == 0 || kin3 == 0 || ekin == 0)
        return GC_E_ARG;
    gc_context *c = const_cast<gc_context *>(ctx);
    struct gcn_device *d = c->native;
    if (!d->list_valid) return GC_E_STATE;

    if (native_ref_join(d) != GC_OK) return GC_E_DEVICE;
    const gc_i64 nitem = kinetic_launch(d, which, d->reduce_partial,
                                        d->stream);
    gc_status st = reduce_n_global(c, nitem, kin3, 3);
    if (st != GC_OK) return st;
    *ekin = 0.5 * (kin3[0] + kin3[1] + kin3[2]);
    return GC_OK;
}

/* Both tensors' partials, a's three components then b's, so that one
 * six-component reduction finishes both.  The two launches read different
 * velocities and write different partials, so b runs on lane 0 beside a. */
static gc_status kinetic_pair_launch(struct gcn_device *d, gc_i32 which_a,
                                     gc_i32 which_b, gc_i64 *nitem)
{
    if (native_ref_join(d) != GC_OK ||
        native_lanes_fork(d, 1) != GC_OK) return GC_E_DEVICE;
    *nitem = kinetic_launch(d, which_a, d->reduce_partial, d->stream);
    kinetic_launch(d, which_b,
                   d->reduce_partial + 3 * grid_of(*nitem, GCN_RED_BLOCK),
                   d->lane[0]);
    return native_lanes_join(d, 1);
}

extern "C" gc_status gpu_core_kinetic_pair(const gc_context *ctx,
                                           gc_i32 which_a, gc_i32 which_b,
                                           gc_f64 *kin_a3, gc_f64 *ekin_a,
                                           gc_f64 *kin_b3, gc_f64 *ekin_b)
{
    if (ctx == 0 || ctx->native == 0 || kin_a3 == 0 || ekin_a == 0 ||
        kin_b3 == 0 || ekin_b == 0 ||
        (which_a >= GC_KIN_GROUP_VEL) != (which_b >= GC_KIN_GROUP_VEL))
        return GC_E_ARG;
    gc_context *c = const_cast<gc_context *>(ctx);
    struct gcn_device *d = c->native;
    if (!d->list_valid) return GC_E_STATE;

    gc_i64 nitem = 0;
    if (kinetic_pair_launch(d, which_a, which_b, &nitem) != GC_OK)
        return GC_E_DEVICE;
    gc_f64 k6[6];
    gc_status st = reduce_n_global(c, nitem, k6, 6);
    if (st != GC_OK) return st;
    for (int k = 0; k < 3; ++k) { kin_a3[k] = k6[k]; kin_b3[k] = k6[3 + k]; }
    *ekin_a = 0.5 * (kin_a3[0] + kin_a3[1] + kin_a3[2]);
    *ekin_b = 0.5 * (kin_b3[0] + kin_b3[1] + kin_b3[2]);
    return GC_OK;
}

/* The sum over the ranks of the tick's fixed-point words, for a
 * communicator the device cannot reduce over itself: one readback, one
 * integer MPI_SUM (exact, so the same bits in any order), one upload. */
static gc_status thermo_allreduce_host(gc_context *ctx)
{
#ifdef HAVE_MPI_GENESIS
    struct gcn_device *d = ctx->native;
    const size_t off = offsetof(struct gcn_thermo, words);
    unsigned long long *w = d->thermo_pin->words;
    char *dw = (char *)d->thermo + off;
    if (cudaMemcpyAsync(w, dw, sizeof(d->thermo_pin->words),
                        cudaMemcpyDeviceToHost, d->stream) != cudaSuccess ||
        dev_sync(d->stream) != GC_OK)
        return GC_E_DEVICE;
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)ctx->comm);
    if (MPI_Allreduce(MPI_IN_PLACE, w, 12, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                      comm) != MPI_SUCCESS)
        return GC_E_STATE;
    return cudaMemcpyAsync(dw, w, sizeof(d->thermo_pin->words),
                           cudaMemcpyHostToDevice, d->stream) == cudaSuccess
           ? GC_OK : GC_E_DEVICE;
#else
    (void)ctx;
    return GC_E_UNSUPPORTED;
#endif
}

extern "C" gc_status gpu_core_thermostat_init(
    gc_context *ctx, gc_i32 kind, gc_i32 nh_length, gc_i32 nh_step,
    gc_f64 degree, gc_f64 kboltz, gc_f64 temp0, gc_f64 dt_tau,
    gc_f64 factor, gc_f64 kbt, const gc_f64 *nh_dt, const gc_f64 *nh_mass,
    const gc_f64 *nh_vel, const gc_f64 *nh_force, const gc_f64 *nh_coef,
    const gc_f64 *kin6)
{
    if (!ctx || !ctx->native || !nh_dt || !nh_mass || !nh_vel ||
        !nh_force || !nh_coef || !kin6)
        return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->step_ready || !d->thermo) return GC_E_STATE;
    if (kind == GC_THERMOSTAT_NHC &&
        (nh_length < 2 || nh_length > GCN_NHC_MAX || nh_step < 1))
        return GC_E_UNSUPPORTED;
    if (kind != GC_THERMOSTAT_NHC) nh_length = 0;
    if (ctx->nproc > 1 && !d->thermo_wsum_tried) {
        d->thermo_wsum_tried = 1;
        gc_status st = gcx_wsum_create((gcx_comm)ctx->comm, 12, 1,
                                       &d->thermo_wsum);
        if (st != GC_OK) return st;
    }

    struct gcn_thermo_args a;
    std::memset(&a, 0, sizeof(a));
    a.kind = kind; a.nh_length = nh_length; a.nh_step = nh_step;
    a.degree = degree; a.kboltz = kboltz; a.temp0 = temp0;
    a.dt_tau = dt_tau; a.factor = factor; a.kbt = kbt;
    for (int k = 0; k < 12; ++k) a.nh_dt[k] = nh_dt[k];
    for (int k = 0; k < nh_length; ++k) a.nh_mass[k] = nh_mass[k];
    d->thermo_args = a;

    if (cudaEventSynchronize(d->thermo_up) != cudaSuccess)
        return GC_E_DEVICE;
    struct gcn_thermo *p = d->thermo_pin;
    std::memset(p, 0, sizeof(*p));
    p->scale = 1.0;
    for (int k = 0; k < 6; ++k) p->kin[k] = kin6[k];
    for (int k = 0; k < nh_length; ++k) {
        p->nh_vel[k] = nh_vel[k];
        p->nh_force[k] = nh_force[k];
        p->nh_coef[k] = nh_coef[k];
    }
    if (cudaMemcpyAsync(d->thermo, p, sizeof(*p), cudaMemcpyHostToDevice,
                        d->stream) != cudaSuccess ||
        cudaEventRecord(d->thermo_up, d->stream) != cudaSuccess)
        return GC_E_DEVICE;
    return GC_OK;
}

extern "C" gc_status gpu_core_thermostat_draws(gc_context *ctx, gc_i32 n,
                                               const gc_f64 *draws)
{
    if (!ctx || !ctx->native || !draws || n < 1 || n > GCN_THERMO_DRAWS)
        return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->thermo || d->graph_mode != GCN_GRAPH_EAGER) return GC_E_STATE;
    gc_status st = graph_flush(d);
    if (st != GC_OK) return st;
    /* the staging is reused: the previous upload has long left it, and the
     * device reads new draws in stream order */
    if (cudaEventSynchronize(d->thermo_up) != cudaSuccess)
        return GC_E_DEVICE;
    struct gcn_thermo *p = d->thermo_pin;
    std::memcpy(p->draw, draws, 2 * (size_t)n * sizeof(double));
    p->next = 0;
    p->count = n;
    const size_t off = offsetof(struct gcn_thermo, draw);
    if (cudaMemcpyAsync((char *)d->thermo + off, (char *)p + off,
                        sizeof(*p) - off, cudaMemcpyHostToDevice,
                        d->stream) != cudaSuccess ||
        cudaEventRecord(d->thermo_up, d->stream) != cudaSuccess)
        return GC_E_DEVICE;
    gcn_kern_tick_pending<<<1, 1, 0, d->stream>>>(d->thermo, d->thermo_args);
    return dev_launched("thermostat_draws");
}

extern "C" gc_status gpu_core_thermostat(gc_context *ctx, gc_i32 which_half,
                                         gc_i32 which_ref)
{
    if (!ctx || !ctx->native ||
        (which_half >= GC_KIN_GROUP_VEL) != (which_ref >= GC_KIN_GROUP_VEL))
        return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid || !d->step_ready || !d->thermo ||
        d->plan.ensemble != GC_ENSEMBLE_NVT)
        return GC_E_STATE;
    if (d->graph_mode == GCN_GRAPH_REPLAY) return GC_OK;
    const int group = which_half >= GC_KIN_GROUP_VEL;
    const int fused = d->ref_pending && d->half_pending &&
                      which_half == (group ? GC_KIN_GROUP_VEL_HALF
                                           : GC_KIN_FLAT_VEL_HALF) &&
                      which_ref == (group ? GC_KIN_GROUP_VEL_REF
                                          : GC_KIN_FLAT_VEL_REF);
    const int summed = ctx->nproc > 1;
    if (d->tick_prep && !fused) {
        if (d->tick_fold) return GC_E_STATE;
        d->tick_prep = 0;
    }
    if (fused && d->tick_fold) {
        d->ref_pending = d->half_pending = 0;
        d->tick_prep = d->tick_fold = 0;
        return GC_OK;
    }
    if (fused) {
        const gc_i64 nitem = group ? d->num_groups : d->num_owned;
        const int nb = (int)grid_of(nitem, GCN_RED_BLOCK);
        d->ref_pending = d->half_pending = 0;
        struct gcx_wsum_view ws;
        std::memset(&ws, 0, sizeof(ws));
        if (summed && d->thermo_wsum &&
            gcx_wsum_view_of(d->thermo_wsum, &ws) != GC_OK)
            return GC_E_STATE;
        if (d->tick_prep)
            gcn_kern_tick_finalize<<<1, GCN_RED_BLOCK, 0, d->stream>>>(
                d->tick_part, d->tick_nblk, d->thermo, d->thermo_args,
                summed, ws);
        else
            vf_each(d->vf32, [&](auto z) {
                using S = decltype(z);
                (group ? gcn_kern_tick_kinetic<1, S>
                       : gcn_kern_tick_kinetic<0, S>)
                    <<<(unsigned)nb, GCN_RED_BLOCK, 0, d->stream>>>(
                    d->vel.as<S>(), d->vel_ref, d->vel_half, d->mass,
                    d->group_offset, d->group_member, d->group_kind, nitem,
                    d->pitch, d->reduce_partial, nb, d->thermo,
                    d->thermo_args, summed, ws);
            });
        d->tick_prep = 0;
        if (!summed || d->thermo_wsum) return dev_launched("thermostat");
    } else {
        gc_i64 nitem = 0;
        if (kinetic_pair_launch(d, which_half, which_ref, &nitem) != GC_OK)
            return GC_E_DEVICE;
        finalize_launch(d, d->reduce_partial, nitem, 6,
                        d->thermo->kin_local);
        if (summed)
            gcn_kern_thermo_encode<<<1, 32, 0, d->stream>>>(d->thermo);
    }
    if (summed) {
        gc_status st = d->thermo_wsum
            ? gcx_wsum_launch(d->thermo_wsum, (gc_u64 *)d->thermo->words,
                              d->stream)
            : thermo_allreduce_host(ctx);
        if (st != GC_OK) return st;
    }
    gcn_kern_thermostat<<<1, 1, 0, d->stream>>>(d->thermo, d->thermo_args,
                                                 summed);
    return dev_launched("thermostat");
}

extern "C" gc_status gpu_core_thermostat_state(gc_context *ctx,
                                               gc_f64 *kin_half3,
                                               gc_f64 *ekin_half,
                                               gc_f64 *kin_full3,
                                               gc_f64 *ekin_full,
                                               gc_f64 *scale,
                                               gc_f64 *nh_vel,
                                               gc_f64 *nh_force,
                                               gc_f64 *nh_coef)
{
    if (!ctx || !ctx->native || !kin_half3 || !ekin_half || !kin_full3 ||
        !ekin_full || !scale || !nh_vel || !nh_force || !nh_coef)
        return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->thermo) return GC_E_STATE;
    struct gcn_thermo *p = d->thermo_pin;
    if (cudaMemcpyAsync(p, d->thermo, offsetof(struct gcn_thermo, kin_local),
                        cudaMemcpyDeviceToHost, d->stream) != cudaSuccess)
        return GC_E_DEVICE;
    gc_status st = dev_sync(d->stream);
    if (st != GC_OK) return st;
    for (int k = 0; k < 3; ++k) {
        kin_half3[k] = p->kin[k];
        kin_full3[k] = p->kin[3 + k];
    }
    *ekin_half = 0.5 * (kin_half3[0] + kin_half3[1] + kin_half3[2]);
    *ekin_full = 0.5 * (kin_full3[0] + kin_full3[1] + kin_full3[2]);
    *scale = p->scale;
    for (int k = 0; k < d->thermo_args.nh_length; ++k) {
        nh_vel[k] = p->nh_vel[k];
        nh_force[k] = p->nh_force[k];
        nh_coef[k] = p->nh_coef[k];
    }
    return GC_OK;
}

extern "C" gc_status gpu_core_dynvars_sums(const gc_context *ctx,
                                            gc_f64 *rmsg, gc_f64 *ekin_ref)
{
    if (!ctx || !ctx->native || !rmsg || !ekin_ref) return GC_E_ARG;
    gc_context *c = const_cast<gc_context *>(ctx);
    struct gcn_device *d = c->native;
    if (!d->list_valid || !d->step_ready) return GC_E_STATE;
    const gc_i64 nb = grid_of(d->num_owned, GCN_RED_BLOCK);
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_dynvars_sums<<<(unsigned)nb, GCN_RED_BLOCK, 0, d->stream>>>(
            d->force.as<decltype(z)>(), d->vel_ref, d->mass, d->num_owned,
            d->pitch, d->reduce_partial, (int)nb);
    });
    gc_f64 sums[2] = { 0.0, 0.0 };
    gc_status st = reduce_n(c, d->num_owned, sums, 2);
    if (st != GC_OK) return st;
    *rmsg = sums[0];
    *ekin_ref = 0.5 * sums[1];
    return GC_OK;
}

/* VV1's and VV2's kicks run inside gpu_core_constrain's solvers when every
 * atom is a single or in a group a solver takes; gpu_core_vv1 and gpu_core_vv2
 * are always followed by gpu_core_constrain of their half
 * (sp_gpu_core_step.fpp), which queues the pending kick. */
static int kick_in_solvers(const struct gcn_device *d)
{
    return d->plan.rigid_bond && (d->num_water == 0 || d->con.fast_water);
}

/* A plain step's force join is left to VV2's solvers when gpu_core_vv2
 * leaves them its kick (NVE and NVT; NPT's VV2 reads force itself).  Any
 * other VV2 queues the join first. */
static int join_in_solvers(const struct gcn_device *d)
{
    return kick_in_solvers(d) && (d->plan.ensemble == GC_ENSEMBLE_NVE ||
                                  d->plan.ensemble == GC_ENSEMBLE_NVT);
}

static gc_status join_flush(gc_context *ctx)
{
    if (!ctx->native->join_pending) return GC_OK;
    ctx->native->join_pending = 0;
    return native_force_join(ctx, 0, 0, 0, 0);
}

extern "C" gc_status gpu_core_vv1(gc_context *ctx, gc_f64 scale_vel,
                                  gc_i32 from_ref)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid) return GC_E_STATE;
    if (d->graph_mode == GCN_GRAPH_REPLAY) return GC_OK;
    if (native_ref_join(d) != GC_OK) return GC_E_DEVICE;
    const gc_i64 nb = grid_of(d->num_owned, GCN_BLOCK);

    if (from_ref == GC_VV1_FROM_REF) {
        vf_each(d->vf32, [&](auto z) {
            using S = decltype(z);
            gcn_kern_vv1_from_ref<<<(unsigned)nb, GCN_BLOCK, 0, d->stream>>>(
                d->vel.as<S>(), d->vel_ref, d->coord, d->coord_ref,
                d->force.as<S>(), d->inv_mass, d->num_owned, d->pitch,
                scale_vel, d->plan.dt, d->plan.half_dt);
        });
    } else {
        const int dev_scale = from_ref == GC_VV1_DEVICE_SCALE;
        const int fused = dev_scale && kick_in_solvers(d) &&
                          d->tab.nonbond_precision == GC_NONBOND_MIXED;
        if (!fused && (dev_scale || scale_vel != 1.0)) {
            gc_i64 nr = grid_of(d->plan.group_tp ? d->num_groups
                                                 : d->num_owned, GCN_BLOCK);
            vf_each(d->vf32, [&](auto z) {
                gcn_kern_rescale<<<(unsigned)nr, GCN_BLOCK, 0, d->stream>>>(
                    d->vel.as<decltype(z)>(), d->mass, d->group_offset,
                    d->group_member, d->group_kind, d->num_groups,
                    d->num_owned, d->pitch, scale_vel,
                    dev_scale ? &d->thermo->scale : 0, d->plan.group_tp);
            });
        }
        if (kick_in_solvers(d)) {
            d->vv1_pending = 1;
            d->vv1_scale = fused;
            return dev_launched("vv1");
        }
        vf_each(d->vf32, [&](auto z) {
            using S = decltype(z);
            gcn_kern_kick_drift<<<(unsigned)nb, GCN_BLOCK, 0, d->stream>>>(
                d->vel.as<S>(), d->coord, d->coord_ref, d->force.as<S>(),
                d->inv_mass, d->num_owned, d->pitch, d->plan.dt,
                d->plan.half_dt);
        });
    }
    return dev_launched("vv1");
}

extern "C" gc_status gpu_core_vv2(gc_context *ctx)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid) return GC_E_STATE;
    if (d->graph_mode == GCN_GRAPH_REPLAY) return GC_OK;
    if (kick_in_solvers(d)) { d->vv2_pending = 1; return GC_OK; }
    if (join_flush(ctx) != GC_OK) return GC_E_DEVICE;
    vf_each(d->vf32, [&](auto z) {
        using S = decltype(z);
        gcn_kern_vv2<<<(unsigned)grid_of(d->num_owned, GCN_BLOCK), GCN_BLOCK,
                       0, d->stream>>>(d->vel.as<S>(), d->vel_half,
                                       d->vel_full, d->force.as<S>(),
                                       d->inv_mass, d->plan.half_dt,
                                       d->num_owned, d->pitch,
                                       d->plan.rigid_bond);
    });
    return dev_launched("vv2");
}

extern "C" gc_status gpu_core_respa_half(gc_context *ctx)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid || !d->step_ready) return GC_E_STATE;
    vf_each(d->vf32, [&](auto z) {
        using S = decltype(z);
        gcn_kern_kin_prep<<<(unsigned)grid_of(d->num_owned, GCN_BLOCK),
                            GCN_BLOCK, 0, d->stream>>>(
            d->vel_half, d->force.as<S>(), d->inv_mass, d->plan.half_dt,
            d->num_owned, d->pitch);
    });
    return dev_launched("respa_half");
}

extern "C" gc_status gpu_core_respa_vv1(gc_context *ctx, gc_f64 scale_vel,
                                        gc_f64 half_dt_long)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid) return GC_E_STATE;
    if (native_ref_join(d) != GC_OK) return GC_E_DEVICE;
    if (scale_vel != 1.0) {
        gc_i64 nr = grid_of(d->plan.group_tp ? d->num_groups
                                             : d->num_owned, GCN_BLOCK);
        vf_each(d->vf32, [&](auto z) {
            gcn_kern_rescale<<<(unsigned)nr, GCN_BLOCK, 0, d->stream>>>(
                d->vel.as<decltype(z)>(), d->mass, d->group_offset,
                d->group_member, d->group_kind, d->num_groups, d->num_owned,
                d->pitch, scale_vel, (const double *)0, d->plan.group_tp);
        });
    }
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_respa_kick<<<(unsigned)grid_of(d->num_owned, GCN_BLOCK),
                              GCN_BLOCK, 0, d->stream>>>(
            d->vel.as<decltype(z)>(), d->coord, d->coord_ref, d->vel_full,
            d->force_real, d->force_bond, d->force_recip, d->inv_mass,
            d->num_owned, d->pitch, d->plan.dt, d->plan.half_dt,
            half_dt_long, 1, 0);
    });
    return dev_launched("respa_vv1");
}

extern "C" gc_status gpu_core_respa_vv2(gc_context *ctx, gc_f64 half_dt_long)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid) return GC_E_STATE;
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_respa_kick<<<(unsigned)grid_of(d->num_owned, GCN_BLOCK),
                              GCN_BLOCK, 0, d->stream>>>(
            d->vel.as<decltype(z)>(), d->coord, d->coord_ref, d->vel_full,
            d->force_real, d->force_bond, d->force_recip, d->inv_mass,
            d->num_owned, d->pitch, d->plan.dt, d->plan.half_dt,
            half_dt_long, 0, d->plan.rigid_bond);
    });
    return dev_launched("respa_vv2");
}

extern "C" gc_status gpu_core_constrain(gc_context *ctx, gc_i32 mode,
                                        gc_f64 dt, gc_f64 *viri3,
                                        gc_i64 *nfail)
{
    if (ctx == 0 || ctx->native == 0 || viri3 == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid) return GC_E_STATE;
    cudaStream_t s = d->stream;
    viri3[0] = viri3[1] = viri3[2] = 0.0;
    if (nfail) *nfail = 0;
    const int defer = (mode & GC_CONSTRAIN_DEFER) != 0;
    const int tick_next = (mode & GC_CONSTRAIN_TICK_NEXT) != 0;
    mode &= ~(GC_CONSTRAIN_DEFER | GC_CONSTRAIN_TICK_NEXT);
    if (mode != GC_CONSTRAIN_VV1 && mode != GC_CONSTRAIN_VV2) return GC_E_ARG;
    if (!d->plan.rigid_bond || d->graph_mode == GCN_GRAPH_REPLAY) return GC_OK;

    /* The water and hydrogen-group solvers move disjoint atoms, so hydrogen
     * groups run on a lane beside the waters (partials after the waters'), and
     * the singles' kick on a second lane. Convergence counters stay on the
     * device until read (verdict_read). */
    const int water = d->num_water > 0 && d->con.fast_water;
    const int hgrp = d->num_hgroup > 0;
    const int kick = mode == GC_CONSTRAIN_VV1 ? d->vv1_pending
                                              : d->vv2_pending;
    const int scale = mode == GC_CONSTRAIN_VV1 && d->vv1_scale;
    d->vv1_pending = d->vv2_pending = d->vv1_scale = 0;
    struct gcn_join jv;
    std::memset(&jv, 0, sizeof(jv));
    const int join = mode == GC_CONSTRAIN_VV2 && kick && d->join_pending;
    if (join) {
        d->join_pending = 0;
        if (native_force_join_view(ctx, &jv) != GC_OK) return GC_E_DEVICE;
    } else if (join_flush(ctx) != GC_OK) {
        return GC_E_DEVICE;
    }
    if (mode == GC_CONSTRAIN_VV1 && d->guard_ready) {
        d->guard_ready = 0;
        if (cudaMemsetAsync(d->guard_max, 0, 2 * sizeof(gc_f64), s) !=
                cudaSuccess)
            return GC_E_DEVICE;
    }
    const int nlane = kick ? 2 : (water && hgrp);
    cudaStream_t sh = nlane ? d->lane[0] : s;
    cudaStream_t ss = d->lane[1];
    if (nlane && native_lanes_fork(d, nlane) != GC_OK) return GC_E_DEVICE;

    if (mode == GC_CONSTRAIN_VV1) {
        const gc_i64 nbw = water ? grid_of(d->num_water, GCN_RED_BLOCK) : 0;
        /* Over several ranks the passes also write the step's force
         * coordinates for the atoms they move and take the list guard's
         * distances: two launches fewer on a short step. One rank keeps the
         * two passes, which run beside the reciprocal spread. */
        const int view = kick && ctx->nproc > 1 && d->force_move_valid &&
                         d->force_coord && d->group_force_move &&
                         d->list_ref;
        const struct gcn_view vw = {
            d->group_force_move, d->force_coord, d->verdict, d->list_ref,
            d->guard_max, { d->box[0], d->box[1], d->box[2] } };
        d->view_ready = view;
        d->guard_ready = view;
        const struct gcn_rescale rs = {
            scale ? &d->thermo->scale : 0, d->mass, d->plan.group_tp };
        /* hydrogen groups first: their SHAKE iterations are the longest per-
         * thread chain */
        /* mixed mode (vf32): the solvers run in FP32 (R = S) */
        vf_each(d->vf32, [&](auto z) {
        using S = decltype(z);
        if (hgrp) {
            gc_i64 nb = grid_of(d->num_hgroup, GCN_RED_BLOCK);
            if (kick)
                (view ? vv1_hgroup_kernel<1, S>(d->rigid_wide)
                      : vv1_hgroup_kernel<0, S>(d->rigid_wide))
                    <<<(unsigned)nb, GCN_RED_BLOCK, 0, sh>>>(
                    d->num_hgroup, d->hgr_heavy, d->hgr_h, d->hgr_arity,
                    d->hgr_dist, d->hgr_inv_mass_h, d->hgr_inv_mass_c,
                    d->vel.as<S>(), d->coord, d->coord_ref, d->force.as<S>(),
                    d->inv_mass, d->pitch, d->plan.dt, d->plan.half_dt, dt,
                    d->con.shake_iteration, d->con.shake_tolerance,
                    d->gid, d->con_fail, d->reduce_partial + 3 * nbw,
                    (int)nb, d->hgr_group, vw, rs);
            else
                gcn_kern_shake_vv1<<<(unsigned)nb, GCN_RED_BLOCK, 0, sh>>>(
                    d->num_hgroup, d->hgr_heavy, d->hgr_h, d->hgr_arity,
                    d->hgr_dist, d->hgr_inv_mass_h, d->hgr_inv_mass_c,
                    d->coord_ref, d->coord, d->vel.as<S>(), d->pitch, 1, dt,
                    d->con.shake_iteration, d->con.shake_tolerance,
                    d->gid, d->con_fail, d->reduce_partial + 3 * nbw,
                    (int)nb);
        }
        if (kick)
            (view ? gcn_kern_vv1_single<1, S> : gcn_kern_vv1_single<0, S>)
                <<<(unsigned)grid_of(d->num_groups, GCN_BLOCK),
                   GCN_BLOCK, 0, ss>>>(
                d->vel.as<S>(), d->coord, d->coord_ref, d->force.as<S>(),
                d->inv_mass, d->group_offset, d->group_member,
                d->group_kind, d->num_groups, d->pitch, d->plan.dt,
                d->plan.half_dt, vw, rs);
        if (water) {
            const double mO = d->con.water_mass_o, mH = d->con.water_mass_h;
            const double mtot = mO + 2.0 * mH;
            const double rOH = d->con.water_r_oh, rHH = d->con.water_r_hh;
            /* the SETTLE canonical triangle, from the two rigid distances:
               rc is half the H-H separation, ra the O-to-centroid distance
               and rb the H-to-centroid one. */
            const double rc = 0.5 * rHH;
            const double h  = sqrt(rOH * rOH - rc * rc);
            const double ra = h * (2.0 * mH) / mtot;
            const double rb = h - ra;
            if (kick)
                (view ? gcn_kern_vv1_water<1, S> : gcn_kern_vv1_water<0, S>)
                    <<<(unsigned)nbw, GCN_RED_BLOCK, 0, s>>>(
                    d->num_water, d->water_slot, d->vel.as<S>(), d->coord,
                    d->coord_ref, d->force.as<S>(), d->inv_mass, d->pitch,
                    d->plan.dt, d->plan.half_dt, mO, mH, mtot, ra, rb, rc, rHH,
                    1.0 / ra, 1.0 / dt, d->reduce_partial, (int)nbw,
                    d->water_group, vw, rs);
            else
                gcn_kern_settle_vv1<<<(unsigned)nbw, GCN_RED_BLOCK, 0, s>>>(
                    d->num_water, d->water_slot, d->coord_ref, d->coord,
                    d->vel.as<S>(), d->pitch, 1, mO, mH, mtot, ra, rb, rc,
                    rHH, 1.0 / ra, 1.0 / dt, d->reduce_partial, (int)nbw);
        }
        });
        if (nlane && native_lanes_join(d, nlane) != GC_OK) return GC_E_DEVICE;
        if (!defer && (water || hgrp)) {
            /* compute_constraints divides the assembled ConstraintModeLEAP
             * virial by dt*dt before nve_vv1 adds it (sp_constraints.fpp (compute_constraints),
             * :4007); both VV1 halves carry the divisor, VV2's does not. */
            gc_f64 wt[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            if (water)
                finalize_launch(d, d->reduce_partial, d->num_water, 3,
                                d->out_pin);
            if (hgrp)
                finalize_launch(d, d->reduce_partial + 3 * nbw,
                                d->num_hgroup, 3, d->out_pin + 3);
            gc_status st = read_out(d, wt, 6);
            if (st != GC_OK) return st;
            if (!water) wt[0] = wt[1] = wt[2] = 0.0;
            if (!hgrp)  wt[3] = wt[4] = wt[5] = 0.0;
            for (int c = 0; c < 3; ++c)
                viri3[c] = (wt[c] + wt[3 + c]) / (dt * dt);
        }
    } else {
        const int full = !defer;
        const unsigned gw = (unsigned)grid_of(d->num_water, GCN_BLOCK);
        const unsigned gh = (unsigned)grid_of(d->num_hgroup, GCN_BLOCK);
        const unsigned gs = (unsigned)grid_of(d->num_groups, GCN_BLOCK);
        const bool f32 = d->tab.nonbond_precision == GC_NONBOND_MIXED;
        /* Mixed mode: the next tick's pass is taken into these (each solver
         * thread holds whole groups); FP64 keeps the tick's own pass, whose
         * summation order differs. */
        const int prep = tick_next && kick && f32 && d->thermo &&
                         d->plan.ensemble == GC_ENSEMBLE_NVT &&
                         d->plan.thermo_period == 1;
        struct gcn_tick_prep tp;
        std::memset(&tp, 0, sizeof(tp));
        if (prep) {
            tp.vref = d->vel_ref;
            tp.part = d->tick_part;
            tp.mass = d->mass;
            tp.group_tp = d->plan.group_tp;
            tp.nblk = (int)(gs + (water ? gw : 0) + (hgrp ? gh : 0));
            tp.keep = d->graph_mode != GCN_GRAPH_CAPTURE;
            const int summed = ctx->nproc > 1;
            if (!summed || (d->thermo_wsum &&
                            gcx_wsum_view_of(d->thermo_wsum, &tp.ws) ==
                                GC_OK)) {
                tp.t = d->thermo;
                tp.a = d->thermo_args;
                tp.encode = summed;
            }
        }
        d->tick_prep = prep;
        d->tick_fold = tp.t != 0;
        d->tick_nblk = tp.nblk;
        /* The kinetic partials keep block order (singles, waters, hydrogen
         * groups); hydrogen groups launch first on the step's stream and
         * waters on the lane, since a water block holds a third of an SM's
         * registers and a waiting hydrogen-group pass would find no room. */
        cudaStream_t sv = nlane ? sh : s;
        const int off_w = (int)gs, off_h = (int)(gs + (water ? gw : 0));
        vf_each(d->vf32, [&](auto z) {
        using S = decltype(z);
        tp.off = off_h;
        if (hgrp && kick)
            (prep ? (join ? vv2_hgroup_kernel<1, S, 1>(d->rigid_wide)
                          : vv2_hgroup_kernel<0, S, 1>(d->rigid_wide))
                  : (join ? vv2_hgroup_kernel<1, S, 0>(d->rigid_wide)
                          : vv2_hgroup_kernel<0, S, 0>(d->rigid_wide)))
                <<<gh, GCN_BLOCK, 0, s>>>(
                d->num_hgroup, d->hgr_heavy, d->hgr_h, d->hgr_arity,
                d->hgr_dist, d->hgr_inv_mass_h, d->hgr_inv_mass_c,
                d->vel.as<S>(), d->vel_half, d->vel_full, d->coord,
                d->force.as<S>(), d->inv_mass, d->pitch, d->plan.half_dt,
                full, d->con.shake_iteration, d->con.shake_tolerance, d->gid,
                d->con_fail, jv, tp);
        else if (hgrp)
            gcn_kern_rattle_hgroup<<<gh, GCN_BLOCK, 0, s>>>(
                d->num_hgroup, d->hgr_heavy, d->hgr_h, d->hgr_arity,
                d->hgr_dist, d->hgr_inv_mass_h, d->hgr_inv_mass_c,
                d->coord, d->vel.as<S>(), d->pitch, d->con.shake_iteration,
                d->con.shake_tolerance, d->gid, d->con_fail);
        tp.off = off_w;
        if (water && kick)
            (prep ? (join ? gcn_kern_vv2_water<1, S, 1>
                          : gcn_kern_vv2_water<0, S, 1>)
                  : (join ? gcn_kern_vv2_water<1, S, 0>
                          : gcn_kern_vv2_water<0, S, 0>))
                <<<gw, GCN_BLOCK, 0, sv>>>(
                d->num_water, d->water_slot, d->vel.as<S>(), d->vel_half,
                d->vel_full, d->coord, d->force.as<S>(), d->inv_mass,
                d->pitch, d->plan.half_dt, full, d->con.water_mass_o,
                d->con.water_mass_h, jv, tp);
        else if (water)
            gcn_kern_rattle_water<<<gw, GCN_BLOCK, 0, sv>>>(
                d->num_water, d->water_slot, d->coord, d->vel.as<S>(),
                d->pitch, d->con.water_mass_o, d->con.water_mass_h);
        tp.off = 0;
        if (kick)
            (join ? (prep ? gcn_kern_vv2_single<1, 1, S>
                          : gcn_kern_vv2_single<1, 0, S>)
                  : (prep ? gcn_kern_vv2_single<0, 1, S>
                          : gcn_kern_vv2_single<0, 0, S>))
                <<<gs, GCN_BLOCK, 0, ss>>>(
                d->vel.as<S>(), d->vel_half, d->vel_full, d->force.as<S>(),
                d->inv_mass, d->group_offset, d->group_member, d->group_kind,
                d->num_groups, d->pitch, d->plan.half_dt, full, jv, tp);
        });
        if (nlane && native_lanes_join(d, nlane) != GC_OK) return GC_E_DEVICE;
    }

    if (defer)
        return dev_launched(mode == GC_CONSTRAIN_VV1 ? "constrain_vv1"
                                                     : "constrain_vv2");
    {
        gc_status cs = dev_phase(s, (mode == GC_CONSTRAIN_VV1)
                                    ? "constrain_vv1" : "constrain_vv2");
        if (cs != GC_OK) return cs;
    }
    gc_i64 fc[2] = { 0, 0 };
    if (cudaMemcpy(fc, d->con_fail, sizeof(fc), cudaMemcpyDeviceToHost)
            != cudaSuccess) return GC_E_DEVICE;
    if (nfail) *nfail = fc[0];
    gc_status vs = verdict_read(ctx);
    if (vs != GC_OK) return vs;
    if (mode == GC_CONSTRAIN_VV2) {
        const gc_i64 nb = grid_of(d->num_owned, GCN_RED_BLOCK);
        vf_each(d->vf32, [&](auto z) {
            gcn_kern_vv2_virial<<<(unsigned)nb, GCN_RED_BLOCK, 0, s>>>(
                d->vel.as<decltype(z)>(), d->vel_full, d->coord, d->mass,
                d->plan.half_dt, d->num_owned, d->pitch, d->reduce_partial,
                (int)nb);
        });
        gc_status st = reduce_n(ctx, d->num_owned, viri3, 3);
        if (st != GC_OK) return st;
    }
    return GC_OK;
}

extern "C" gc_status gpu_core_group_virial(const gc_context *ctx, gc_f64 *viri3)
{
    if (ctx == 0 || ctx->native == 0 || viri3 == 0) return GC_E_ARG;
    gc_context *c = const_cast<gc_context *>(ctx);
    struct gcn_device *d = c->native;
    if (!d->list_valid) return GC_E_STATE;
    gc_i64 nb = grid_of(d->num_groups, GCN_RED_BLOCK);
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_group_virial<<<(unsigned)nb, GCN_RED_BLOCK, 0, d->stream>>>(
            d->coord_ref, d->force.as<decltype(z)>(), d->mass,
            d->group_offset, d->group_member, d->group_kind, d->num_groups,
            d->pitch, d->reduce_partial, (int)nb);
    });
    return reduce_n(c, d->num_groups, viri3, 3);
}

extern "C" gc_status gpu_core_list_guard_defer(gc_context *ctx, gc_i32 armed)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid) return GC_E_STATE;
    if (d->guard_seq - d->guard_read >= GC_GUARD_RING) return GC_E_CAPACITY;
    if (d->graph_mode == GCN_GRAPH_REPLAY) {
        d->guard_seq++;
        return GC_OK;
    }
    /* runs on aux_stream beside the force's critical path and joins back
     * before anything on d->stream reads reduce_partial (aux_join) */
    if (cudaEventRecord(d->aux_fork, d->stream) != cudaSuccess ||
        cudaStreamWaitEvent(d->aux_stream, d->aux_fork, 0) != cudaSuccess)
        return GC_E_DEVICE;
    gc_status st = native_list_guard_launch(
        ctx, armed ? 2.0 * ctx->guard.half_skin : 0.0, d->aux_stream);
    if (st != GC_OK) return st;
    if (cudaEventRecord(d->aux_done, d->aux_stream) != cudaSuccess)
        return GC_E_DEVICE;
    d->guard_seq++;
    return GC_OK;
}

extern "C" gc_status gpu_core_list_guard_flush(gc_context *ctx, gc_f64 *d_max,
                                               gc_i32 *count, gc_i64 *over)
{
    if (ctx == 0 || ctx->native == 0 || d_max == 0 || count == 0 ||
        over == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    gc_status st = aux_join(d);
    if (st == GC_OK) st = dev_phase(d->stream, "list_guard");
    if (st == GC_OK) st = verdict_read(ctx);
    if (st == GC_OK &&
        cudaMemcpy(over, d->guard_idx + 1, sizeof(gc_i64),
                   cudaMemcpyDeviceToHost) != cudaSuccess) st = GC_E_DEVICE;
    if (st != GC_OK) return st;
    *count = (gc_i32)(d->guard_seq - d->guard_read);
    for (gc_i32 k = 0; k < *count; ++k) {
        const gc_f64 *slot = d->guard_pin + 2 * ((d->guard_read + k) % GC_GUARD_RING);
        d_max[k] = slot[0];
        if (std::isinf(slot[0]) || std::isinf(slot[1]))
            std::fprintf(stderr, "GPU_Core_Warning> phase=list_guard rank=%d "
                         "guard=%lld: non-finite displacement, the list "
                         "certificate is invalid\n", (int)ctx->rank,
                         (long long)(d->guard_read + k));
    }
    d->guard_read = d->guard_seq;
    return GC_OK;
}

/* The values of the guard queued `back` guards ago (1 = the latest), once the
 * device has written them: d_max and the step's largest move. The caller asks
 * for a guard at least one step old. */
extern "C" gc_status gpu_core_list_guard_read(gc_context *ctx, gc_i32 back,
                                              gc_f64 *vals)
{
    if (ctx == 0 || ctx->native == 0 || vals == 0 || back < 1 ||
        back > GC_GUARD_RING) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    const gc_i64 n = d->guard_seq - back;
    if (n < 0) return GC_E_STATE;
    gc_status st = graph_flush(d);
    if (st != GC_OK) return st;
    const volatile gc_i64 *done = d->guard_done;
    while (*done <= n) {
        const cudaError_t e = cudaStreamQuery(d->aux_stream);
        if (e != cudaSuccess && e != cudaErrorNotReady) return GC_E_DEVICE;
        if (e == cudaSuccess && cudaStreamQuery(d->stream) == cudaSuccess &&
            *done <= n) return GC_E_STATE;
    }
    const volatile gc_f64 *slot = d->guard_pin + 2 * (n % GC_GUARD_RING);
    vals[0] = slot[0];
    vals[1] = slot[1];
    return GC_OK;
}

extern "C" gc_status gpu_core_rebuild(gc_context *ctx, gc_i32 early)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    /* The deferred guard (aux_stream) reads the coordinates and the list
     * reference the rebuild rewrites.  No rebuild over a step whose
     * deferred checks failed: native_rebuild reads them with its first
     * synchronisation, before it changes anything. */
    gc_status st = ctx->native->aux_done ? aux_join(ctx->native) : GC_OK;
    if (st != GC_OK) return st;
    st = native_rebuild(ctx, early);
    /* The caller aborts every rank on a refusal, so no rank steps on while
     * a peer has invalidated or partially sorted. */
    st = native_dist_step_vote(ctx, st, "rebuild_list");
    if (st != GC_OK) return st;
    struct gcn_rigid_pending rigid;
    st = native_build_rigid(ctx, &rigid);
    st = native_dist_step_vote(ctx, st, "rebuild_rigid");
    if (st != GC_OK) return st;
    st = native_nbc_build(ctx);
    const gc_status rs = native_build_rigid_finish(ctx, &rigid);
    st = native_dist_step_vote(ctx, rs != GC_OK ? rs : st,
                               rs != GC_OK ? "rebuild_rigid" : "rebuild_nbcluster");
    if (st != GC_OK) return st;

    struct gcn_device *d = ctx->native;
    if (early) d->early_rebuilds++;
    else       d->scheduled_rebuilds++;
    ctx->epoch++;
    d->list_builds++;           /* the plain-step graphs are captured again */
    d->list_valid = 1;
    return GC_OK;
}

/* A force stage's refusal names itself: several stages return a bare
 * GC_E_DEVICE, and the Fortran caller only prints the status. */
static gc_status force_stage(const gc_context *ctx, gc_status st,
                             const char *phase)
{
    if (st != GC_OK)
        std::fprintf(stderr, "GPU_Core_Error> phase=gpu_core_force/%s "
                     "rank=%d status=%d\n", phase, (int)ctx->rank, (int)st);
    return st;
}

/* respa: 0 the one-step integrator; 1 an r-RESPA inner step (no
 * reciprocal sum); 2 an r-RESPA outer step */
static gc_status force_eval(gc_context *ctx, gc_i32 want_energy,
                            gc_i32 want_virial, gc_i32 respa,
                            gc_step_result *out)
{
    const gc_i32 with_recip = respa != 1;
    if (ctx == 0 || ctx->native == 0 || out == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid) return GC_E_STATE;

    gc_status st;
    if (d->graph_mode == GCN_GRAPH_REPLAY) {
        st = force_stage(ctx, gcn::native_dist_step_replay(ctx), "halo_replay");
        if (st == GC_OK && d->reciprocal_ready)
            st = force_stage(ctx, native_pme_step_replay(ctx), "pme_replay");
        if (st != GC_OK) return st;
        return GC_OK;
    }
    /* The reciprocal sum reads owned coordinates and writes owned forces only,
     * so it forks onto its own stream beside the halo, pairs, bonded terms and
     * force return, and joins before the force join. When the mesh spans nodes
     * a network move holds the host until it lands, so its launches follow the
     * pairs' and bonded terms'; on one node the sum is launched first. */
    const int run_pme = d->reciprocal_ready && with_recip;
    const int pme_late = run_pme && native_pme_crosses_nodes(ctx);
    const gc_i32 pme_scalars = want_energy || want_virial || respa;
    if (pme_late && cudaEventRecord(d->pme_fork, d->stream) != cudaSuccess)
        return force_stage(ctx, GC_E_DEVICE, "pme_fork");
    if (run_pme && !pme_late) {
        st = force_stage(ctx, native_pme_run(ctx, want_energy, pme_scalars),
                         "pme_run");                                         if (st != GC_OK) return st;
    }
    if (!d->view_ready) {
        st = force_stage(ctx, native_force_coordinates_step(ctx),
                         "force_coordinates");
        if (st != GC_OK) return st;
    }
    d->view_ready = 0;
    st = force_stage(ctx, native_force_real_clear(ctx), "force_real_clear");     if (st != GC_OK) return st;
    /* Interior pairs read no ghost and fork onto real_stream, filling the
     * device while d->stream runs the coordinate exchange, boundary pairs,
     * bonded terms and force return. They write only owned slots (atomic
     * adds); nothing reads force_real or acc_real before the join. */
    if (cudaEventRecord(d->real_done, d->stream) != cudaSuccess ||
        cudaStreamWaitEvent(d->real_stream, d->real_done, 0) != cudaSuccess)
        return force_stage(ctx, GC_E_DEVICE, "force_real_fork");
    st = force_stage(ctx, native_force_real(ctx, want_energy, d->real_stream,
                                            ctx->nproc > 1 ? GCN_REAL_INTERIOR
                                                           : GCN_REAL_ALL),
                     "force_real_interior");                                   if (st != GC_OK) return st;
    if (cudaEventRecord(d->real_done, d->real_stream) != cudaSuccess)
        return force_stage(ctx, GC_E_DEVICE, "force_real_fork");
    st = force_stage(ctx, gcn::native_dist_forward(ctx), "halo_forward");      if (st != GC_OK) return st;
    /* With peers the boundary pairs fork onto bnd_stream beside the bonded
     * terms: both need only the coordinate exchange and write different fixed-
     * point fields, so no sum changes. The force return waits for the join. */
    const int bnd_side = ctx->nproc > 1 && d->bnd_stream != 0;
    if (ctx->nproc > 1) {
        cudaStream_t bs = d->stream;
        if (bnd_side) {
            if (cudaEventRecord(d->bnd_fork, d->stream) != cudaSuccess ||
                cudaStreamWaitEvent(d->bnd_stream, d->bnd_fork, 0) != cudaSuccess)
                return force_stage(ctx, GC_E_DEVICE, "force_real_boundary_fork");
            bs = d->bnd_stream;
        }
        st = force_stage(ctx, native_force_real(ctx, want_energy, bs,
                                                GCN_REAL_BOUNDARY),
                         "force_real_boundary");                               if (st != GC_OK) return st;
        if (bnd_side && cudaEventRecord(d->bnd_done, d->bnd_stream) != cudaSuccess)
            return force_stage(ctx, GC_E_DEVICE, "force_real_boundary_fork");
    }
    const gc_i32 bonded_sums = want_energy || want_virial;
    st = force_stage(ctx, native_force_bonded(ctx, bonded_sums), "force_bonded"); if (st != GC_OK) return st;
    if (pme_late) {
        st = force_stage(ctx, native_pme_run(ctx, want_energy, pme_scalars, 1),
                         "pme_run");                                         if (st != GC_OK) return st;
    }
    gcn::native_dist_return_open(ctx);
    if (bnd_side &&
        cudaStreamWaitEvent(d->stream, d->bnd_done, 0) != cudaSuccess)
        return force_stage(ctx, GC_E_DEVICE, "force_real_boundary_join");
    /* No reciprocal plan: publish a zero contribution rather than the last
     * plan's accumulator (an r-RESPA inner step keeps the outer step's
     * force_recip and acc_recip, as stock's force_long does). */
    if (!d->reciprocal_ready &&
        cudaMemsetAsync(d->acc_recip, 0, GCN_PE_NSLOT * (gc_i64)sizeof(gc_f64),
                        d->stream) != cudaSuccess)
        return GC_E_DEVICE;
    st = force_stage(ctx, gcn::native_dist_reverse(ctx), "halo_reverse");     if (st != GC_OK) return st;
    if (cudaStreamWaitEvent(d->stream, d->real_done, 0) != cudaSuccess ||
        aux_join(d) != GC_OK)
        return force_stage(ctx, GC_E_DEVICE, "force_real_join");
    if (d->reciprocal_ready && with_recip) {
        st = force_stage(ctx, native_pme_join(ctx), "pme_join");             if (st != GC_OK) return st;
    }
    const gc_i32 publish = want_energy || want_virial;
    const gc_i32 keep = want_virial || respa;
    if (!publish && !keep && join_in_solvers(d)) {
        d->join_pending = 1;
    } else {
        st = force_stage(ctx, native_force_join(ctx, want_virial, publish,
                                                keep, out),
                         "force_join");
        if (st != GC_OK) return st;
    }
    if (publish) {
        st = verdict_read(ctx);
        if (st != GC_OK) return st;
    }

    return GC_OK;
}

extern "C" gc_status gpu_core_force(gc_context *ctx, gc_i32 want_energy,
                                    gc_i32 want_virial, gc_step_result *out)
{
    return force_eval(ctx, want_energy, want_virial, 0, out);
}

extern "C" gc_status gpu_core_force_respa(gc_context *ctx, gc_i32 want_energy,
                                          gc_i32 want_virial, gc_i32 outer,
                                          gc_step_result *out)
{
    return force_eval(ctx, want_energy, want_virial, outer ? 2 : 1, out);
}

extern "C" gc_status gpu_core_com_removal(gc_context *ctx, gc_i32 do_trans,
                                          gc_i32 do_rot)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    if (!ctx->native->list_valid) return GC_E_STATE;
    if (!do_trans && !do_rot) return GC_OK;
    struct gcn_device *d = ctx->native;
    cudaStream_t s = d->stream;
    gc_i64 nb = grid_of(d->num_owned, GCN_RED_BLOCK);

    double p[7];
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_com_partials<<<(unsigned)nb, GCN_RED_BLOCK, 0, s>>>(
            d->coord, d->vel.as<decltype(z)>(), d->mass, d->num_owned,
            d->pitch, d->reduce_partial, (int)nb);
    });
    gc_status st = reduce_n_global(ctx, d->num_owned, p, 7);
    if (st != GC_OK) return st;

    double par[9];
    std::memset(par, 0, sizeof(par));
    if (p[6] <= 0.0) return GC_OK;
    for (int c = 0; c < 3; ++c) {
        par[3 + c] = p[c] / p[6];       /* the coordinate centre  */
        par[c]     = p[3 + c] / p[6];   /* the velocity centre    */
    }

    if (do_rot) {
        double cv[6] = { par[3], par[4], par[5], par[0], par[1], par[2] };
        if (cudaMemcpyAsync(d->reduce_out, cv, sizeof(cv),
                            cudaMemcpyHostToDevice, s) != cudaSuccess)
            return GC_E_DEVICE;
        double q[9];
        vf_each(d->vf32, [&](auto z) {
            gcn_kern_com_rot_partials<<<(unsigned)nb, GCN_RED_BLOCK, 0, s>>>(
                d->coord, d->vel.as<decltype(z)>(), d->mass, d->reduce_out,
                d->num_owned, d->pitch, d->reduce_partial, (int)nb);
        });
        st = reduce_n_global(ctx, d->num_owned, q, 9);
        if (st != GC_OK) return st;

        /* omega = I^-1 L, a 3x3 symmetric solve by cofactors, as
         * stop_trans_rotation does */
        double I[3][3] = { { q[6] + q[8], -q[4],       -q[5]        },
                           { -q[4],        q[3] + q[8], -q[7]       },
                           { -q[5],       -q[7],        q[3] + q[6] } };
        double c00 = I[1][1]*I[2][2] - I[1][2]*I[2][1];
        double c01 = I[1][2]*I[2][0] - I[1][0]*I[2][2];
        double c02 = I[1][0]*I[2][1] - I[1][1]*I[2][0];
        double det = I[0][0]*c00 + I[0][1]*c01 + I[0][2]*c02;
        if (det != 0.0) {
            double inv[3][3];
            inv[0][0] = c00 / det;
            inv[0][1] = (I[0][2]*I[2][1] - I[0][1]*I[2][2]) / det;
            inv[0][2] = (I[0][1]*I[1][2] - I[0][2]*I[1][1]) / det;
            inv[1][0] = c01 / det;
            inv[1][1] = (I[0][0]*I[2][2] - I[0][2]*I[2][0]) / det;
            inv[1][2] = (I[0][2]*I[1][0] - I[0][0]*I[1][2]) / det;
            inv[2][0] = c02 / det;
            inv[2][1] = (I[0][1]*I[2][0] - I[0][0]*I[2][1]) / det;
            inv[2][2] = (I[0][0]*I[1][1] - I[0][1]*I[1][0]) / det;
            for (int i = 0; i < 3; ++i)
                par[6 + i] = inv[i][0]*q[0] + inv[i][1]*q[1] + inv[i][2]*q[2];
        }
    }

    if (cudaMemcpyAsync(d->reduce_out, par, sizeof(par),
                        cudaMemcpyHostToDevice, s) != cudaSuccess)
        return GC_E_DEVICE;
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_com_apply<<<(unsigned)grid_of(d->num_owned, GCN_BLOCK),
                             GCN_BLOCK, 0, s>>>(d->vel.as<decltype(z)>(),
                                                d->coord, d->reduce_out,
                                                d->num_owned, d->pitch,
                                                do_trans, do_rot);
    });
    return dev_sync(s);
}

__global__ void gcn_kern_vf_narrow(float *__restrict__ dst,
                                   const gc_f64 *__restrict__ src, gc_i64 n)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x)
        dst[i] = (float)src[i];
}

gc_status gcn::native_vf_get(const struct gcn_device *d, const gcn_vf &f,
                             gc_f64 *host)
{
    const size_t n = (size_t)(3 * d->pitch);
    if (!d->vf32)
        return cudaMemcpy(host, f.p, n * sizeof(gc_f64),
                          cudaMemcpyDeviceToHost) == cudaSuccess
                   ? GC_OK : GC_E_DEVICE;
    std::vector<float> t(n);
    if (n && cudaMemcpy(&t[0], f.p, n * sizeof(float),
                        cudaMemcpyDeviceToHost) != cudaSuccess)
        return GC_E_DEVICE;
    for (size_t i = 0; i < n; ++i) host[i] = (gc_f64)t[i];
    return GC_OK;
}

gc_status gcn::native_vf_put(const struct gcn_device *d, const gcn_vf &f,
                             const gc_f64 *host)
{
    const size_t n = (size_t)(3 * d->pitch);
    if (!d->vf32)
        return cudaMemcpy(f.p, host, n * sizeof(gc_f64),
                          cudaMemcpyHostToDevice) == cudaSuccess
                   ? GC_OK : GC_E_DEVICE;
    std::vector<float> t(n);
    for (size_t i = 0; i < n; ++i) t[i] = (float)host[i];
    if (n && cudaMemcpy(f.p, &t[0], n * sizeof(float),
                        cudaMemcpyHostToDevice) != cudaSuccess)
        return GC_E_DEVICE;
    return GC_OK;
}

/* Each field through scratch3 (3*pitch doubles), so that no thread
 * overwrites a double another has yet to read. */
gc_status gcn::native_vf_narrow(struct gcn_device *d)
{
    if (d->vf32) return GC_OK;
    const gc_i64 n3 = 3 * d->pitch;
    const gcn_vf *fields[2] = { &d->vel, &d->force };
    for (const gcn_vf *f : fields) {
        if (cudaMemcpyAsync(d->scratch3, f->p, (size_t)n3 * sizeof(gc_f64),
                            cudaMemcpyDeviceToDevice, d->stream) !=
                cudaSuccess)
            return GC_E_DEVICE;
        gcn_kern_vf_narrow<<<(unsigned)grid_of(n3, GCN_BLOCK), GCN_BLOCK, 0,
                             d->stream>>>(f->as<float>(), d->scratch3, n3);
    }
    gc_status st = dev_phase(d->stream, "vf_narrow");
    if (st != GC_OK) return st;
    d->vf32 = 1;
    return GC_OK;
}

/* The two declared boundary exceptions: they run at output, restart and
 * replica-exchange boundaries only, so no ordinary step moves host particle
 * bytes. */
extern "C" gc_status gpu_core_pull_state(const gc_context *ctx, gc_i64 count,
                                         gc_gid *gid, gc_f64 *coord,
                                         gc_f64 *vel)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    const struct gcn_device *d = ctx->native;
    if (count != d->num_owned) return GC_E_ARG;
    if ((coord || vel) && !d->list_valid) return GC_E_STATE;
    /* Even a GID-only pull must wait for the rebuild's slot permutation
     * before a boundary push can gather in the current compact order. */
    if ((gid || coord || vel) && dev_sync(d->stream) != GC_OK)
        return GC_E_DEVICE;

    if (gid && cudaMemcpy(gid, d->gid, (size_t)count * sizeof(gc_gid),
                          cudaMemcpyDeviceToHost) != cudaSuccess)
        return GC_E_DEVICE;

    std::vector<gc_f64> tmp((size_t)(3 * d->pitch));
    if (coord) {
        if (cudaMemcpy(&tmp[0], d->coord,
                       (size_t)(3 * d->pitch) * sizeof(gc_f64),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
            return GC_E_DEVICE;
        for (gc_i64 i = 0; i < count; ++i)
            for (int c = 0; c < 3; ++c)
                coord[3 * i + c] = tmp[(size_t)(c * d->pitch + i)];
    }
    if (vel) {
        if (native_vf_get(d, d->vel, &tmp[0]) != GC_OK)
            return GC_E_DEVICE;
        for (gc_i64 i = 0; i < count; ++i)
            for (int c = 0; c < 3; ++c)
                vel[3 * i + c] = tmp[(size_t)(c * d->pitch + i)];
    }
    return GC_OK;
}

/* The final trajectory/restart is already written when this is called. Stock
 * then performs one more VV1, which needs the current force and the half
 * velocity saved by the completed VV2. Never used in the ordinary loop. */
extern "C" gc_status gpu_core_pull_final_state(
    const gc_context *ctx, gc_i64 count, gc_gid *gid,
    gc_f64 *coord, gc_f64 *vel, gc_f64 *force, gc_f64 *vel_half)
{
    if (!ctx || !ctx->native || !gid || !coord || !vel ||
        !force || !vel_half) return GC_E_ARG;
    const gcn_device *d = ctx->native;
    if (count != d->num_owned || count <= 0 || !d->list_valid ||
        !d->step_ready || d->native_steps <= 0) return GC_E_STATE;
    gc_status st = gpu_core_pull_state(ctx, count, gid, coord, vel);
    if (st != GC_OK) return st;
    std::vector<gc_f64> tmp((size_t)(3 * d->pitch));
    gc_f64 *host[2] = { force, vel_half };
    for (int a = 0; a < 2; ++a) {
        const gc_status gs = a == 0
            ? native_vf_get(d, d->force, &tmp[0])
            : (cudaMemcpy(&tmp[0], d->vel_half, tmp.size() * sizeof(gc_f64),
                          cudaMemcpyDeviceToHost) == cudaSuccess
                   ? GC_OK : GC_E_DEVICE);
        if (gs != GC_OK) return GC_E_DEVICE;
        for (gc_i64 i = 0; i < count; ++i)
            for (int c = 0; c < 3; ++c)
                host[a][3 * i + c] = tmp[(size_t)((gc_i64)c * d->pitch + i)];
    }
    return GC_OK;
}

extern "C" gc_status gpu_core_step_summary(const gc_context *ctx,
                                           gc_i64 *scheduled,
                                           gc_i64 *early,
                                           gc_i64 *native_steps,
                                           gc_i64 *graph_segments)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    const struct gcn_device *d = ctx->native;
    if (scheduled) *scheduled = d->scheduled_rebuilds;
    if (early)     *early     = d->early_rebuilds;
    if (native_steps)   *native_steps   = d->native_steps;
    if (graph_segments) *graph_segments = d->graph_segments;
    native_dist_summary(ctx);
    return GC_OK;
}
