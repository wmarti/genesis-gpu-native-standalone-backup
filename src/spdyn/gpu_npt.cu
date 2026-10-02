/*
 * gpu_npt.cu : constant-pressure half of the native velocity-Verlet step,
 * GENESIS's MTK barostat with the group temperature convention.
 *
 *   npt_vv1, npt_vv2                 mtk_barostat_vv1/_vv2 (sp_md_vverlet.fpp)
 *   npt_group_scale, npt_respa_vv*   r-RESPA variants (sp_md_respa.fpp)
 *   box_scale                        box update and pme_pre for the new box
 *
 * The thermostat and barostat scalars (kinetic tensors, pressure, barostat
 * momentum, scale factors) are the host's, where GENESIS's random stream is;
 * only the four centre-of-mass sums cross when several ranks share a system.
 */

#include "gpu_core_native.h"
#include "gpu_nbcluster.h"

#include <cmath>
#include <cstdio>
#ifdef HAVE_MPI_GENESIS
#include <mpi.h>
#endif

namespace {

gc_i64 npt_grid(gc_i64 n, gc_i64 block)
{
    gc_i64 g = (n + block - 1) / block;
    if (g > GCN_MAX_BLOCKS) g = GCN_MAX_BLOCKS;
    if (g < 1) g = 1;
    return g;
}

static_assert(sizeof(struct gcn_pair) % sizeof(gc_f64) == 0,
              "gcn_pair is not a whole number of doubles");

}  /* anonymous namespace */

struct gcn_npt_vec3 { double v[3]; };

/* compute_vv1_group, then kick and drift.  A free atom's coordinate is
 * size_scale times its reference; a rigid group moves by (size_scale - 1)
 * times its reference centre of mass, so its geometry is untouched.  One
 * thread owns a group; every owned atom is in exactly one group. */
template <class S>
__global__ void gcn_kern_npt_vv1(S *__restrict__ v,
                                 gc_f64 *__restrict__ x,
                                 const gc_f64 *__restrict__ xref,
                                 const S *__restrict__ f,
                                 const gc_f64 *__restrict__ m,
                                 const gc_f64 *__restrict__ minv,
                                 const gc_i64 *__restrict__ goff,
                                 const gc_i32 *__restrict__ gmem,
                                 const gc_u8 *__restrict__ gkind,
                                 gc_i64 ngroup, gc_i64 pitch,
                                 gcn_npt_vec3 ss, gcn_npt_vec3 vs,
                                 double dt, double half_dt)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 b = goff[g], e = goff[g + 1];
        if (gkind[g] == GC_GROUP_SINGLE) {
            const int s = gmem[b];
            for (int c = 0; c < 3; ++c) {
                const gc_i64 k = (gc_i64)c * pitch + s;
                x[k] = ss.v[c] * xref[k];
                v[k] = vs.v[c] * v[k];
            }
        } else {
            double tm = 0.0, rcm[3] = { 0.0, 0.0, 0.0 };
            double vcm[3] = { 0.0, 0.0, 0.0 };
            for (gc_i64 i = b; i < e; ++i) {
                const int s = gmem[i];
                const double mm = m[s];
                tm += mm;
                for (int c = 0; c < 3; ++c) {
                    rcm[c] += mm * xref[(gc_i64)c * pitch + s];
                    vcm[c] += mm * v[(gc_i64)c * pitch + s];
                }
            }
            for (int c = 0; c < 3; ++c) { rcm[c] /= tm; vcm[c] /= tm; }
            for (gc_i64 i = b; i < e; ++i) {
                const int s = gmem[i];
                for (int c = 0; c < 3; ++c) {
                    const gc_i64 k = (gc_i64)c * pitch + s;
                    x[k] = xref[k] + (ss.v[c] - 1.0) * rcm[c];
                    v[k] = v[k] + (vs.v[c] - 1.0) * vcm[c];
                }
            }
        }
        for (gc_i64 i = b; i < e; ++i) {
            const int s = gmem[i];
            const double factor = half_dt * minv[s];
            for (int c = 0; c < 3; ++c) {
                const gc_i64 k = (gc_i64)c * pitch + s;
                const double vv = v[k] + factor * f[k];
                v[k] = vv;
                x[k] = x[k] + vv * dt;
            }
        }
    }
}

/* mtk_barostat_vv2's sums: momentum, mass, force and the atom count. */
template <class S, class F>
__global__ void gcn_kern_npt_com_partials(const S *__restrict__ v,
                                          const F *__restrict__ f,
                                          const gc_f64 *__restrict__ m,
                                          gc_i64 n, gc_i64 pitch,
                                          gc_f64 *__restrict__ part, int nblk)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double k[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        const double mm = m[s];
        for (int c = 0; c < 3; ++c) {
            k[c]     += mm * v[(gc_i64)c * pitch + s];
            k[4 + c] += f[(gc_i64)c * pitch + s];
        }
        k[3] += mm;
        k[7] += 1.0;
    }
    block_reduce_sum(k, 8, sh, part, nblk);
}

/* mtk_barostat_vv2 from the mean removal to the RATTLE: members lose the
 * mean velocity and force, keep velocity_half and take the half kick; then
 * compute_vv2_group scales the group's centre-of-mass velocity.  Under rigid
 * bonds the RATTLE constrains v + bmoment*x: the velocity is kept in vfull
 * and x_ref and v take the scaled-frame velocity (stock coord_ref/coord_deri). */
template <class S>
__global__ void gcn_kern_npt_vv2(S *__restrict__ v,
                                 gc_f64 *__restrict__ vhalf,
                                 gc_f64 *__restrict__ vfull,
                                 gc_f64 *__restrict__ xref,
                                 S *__restrict__ f,
                                 const gc_f64 *__restrict__ x,
                                 const gc_f64 *__restrict__ m,
                                 const gc_f64 *__restrict__ minv,
                                 const gc_i64 *__restrict__ goff,
                                 const gc_i32 *__restrict__ gmem,
                                 const gc_u8 *__restrict__ gkind,
                                 const gc_f64 *__restrict__ cm,
                                 gc_i64 ngroup, gc_i64 pitch,
                                 gcn_npt_vec3 vs, gcn_npt_vec3 bm,
                                 double half_dt, int rigid)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 b = goff[g], e = goff[g + 1];
        for (gc_i64 i = b; i < e; ++i) {
            const int s = gmem[i];
            const double factor = half_dt * minv[s];
            for (int c = 0; c < 3; ++c) {
                const gc_i64 k = (gc_i64)c * pitch + s;
                const double vv = v[k] - cm[c] / cm[3];
                const double ff = f[k] - cm[4 + c] / cm[7];
                f[k] = ff;
                vhalf[k] = vv;
                v[k] = vv + factor * ff;
            }
        }
        if (gkind[g] == GC_GROUP_SINGLE) {
            const int s = gmem[b];
            for (int c = 0; c < 3; ++c) v[(gc_i64)c * pitch + s] *= vs.v[c];
        } else {
            double tm = 0.0, vcm[3] = { 0.0, 0.0, 0.0 };
            for (gc_i64 i = b; i < e; ++i) {
                const int s = gmem[i];
                const double mm = m[s];
                tm += mm;
                for (int c = 0; c < 3; ++c)
                    vcm[c] += mm * v[(gc_i64)c * pitch + s];
            }
            for (int c = 0; c < 3; ++c) vcm[c] /= tm;
            for (gc_i64 i = b; i < e; ++i) {
                const int s = gmem[i];
                for (int c = 0; c < 3; ++c)
                    v[(gc_i64)c * pitch + s] += (vs.v[c] - 1.0) * vcm[c];
            }
        }
        if (!rigid) continue;
        for (gc_i64 i = b; i < e; ++i) {
            const int s = gmem[i];
            for (int c = 0; c < 3; ++c) {
                const gc_i64 k = (gc_i64)c * pitch + s;
                const double vv = v[k];
                const double cd = vv + bm.v[c] * x[k];
                vfull[k] = vv;
                xref[k] = cd;
                v[k] = cd;
            }
        }
    }
}

/* stock's vel = vel + (coord_deri - coord_ref) after the RATTLE */
template <class S>
__global__ void gcn_kern_npt_rattle_end(S *__restrict__ v,
                                        const gc_f64 *__restrict__ vfull,
                                        const gc_f64 *__restrict__ xref,
                                        gc_i64 n, gc_i64 pitch)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x)
        for (int c = 0; c < 3; ++c) {
            const gc_i64 k = (gc_i64)c * pitch + s;
            v[k] = vfull[k] + (v[k] - xref[k]);
        }
}

/* update_vel_group_3d: a rigid group's centre-of-mass velocity, a free
 * atom's velocity, times vs per axis. */
template <class S>
__device__ __forceinline__ void npt_group_vscale(S *__restrict__ v,
                                                 const gc_f64 *__restrict__ m,
                                                 const gc_i32 *__restrict__ gmem,
                                                 gc_i64 b, gc_i64 e, bool single,
                                                 gc_i64 pitch,
                                                 const gcn_npt_vec3 &vs)
{
    if (single) {
        const int s = gmem[b];
        for (int c = 0; c < 3; ++c) v[(gc_i64)c * pitch + s] *= vs.v[c];
        return;
    }
    double tm = 0.0, vcm[3] = { 0.0, 0.0, 0.0 };
    for (gc_i64 i = b; i < e; ++i) {
        const int s = gmem[i];
        const double mm = m[s];
        tm += mm;
        for (int c = 0; c < 3; ++c) vcm[c] += mm * v[(gc_i64)c * pitch + s];
    }
    for (int c = 0; c < 3; ++c) vcm[c] /= tm;
    for (gc_i64 i = b; i < e; ++i) {
        const int s = gmem[i];
        for (int c = 0; c < 3; ++c)
            v[(gc_i64)c * pitch + s] += (vs.v[c] - 1.0) * vcm[c];
    }
}

/* compute_vv1_coord_group's scaling: a rigid group moves by (ss - 1)
 * times the centre of mass of `from`, a free atom is ss times it. */
__device__ __forceinline__ void npt_group_xscale(gc_f64 *__restrict__ x,
                                                 const gc_f64 *__restrict__ from,
                                                 const gc_f64 *__restrict__ m,
                                                 const gc_i32 *__restrict__ gmem,
                                                 gc_i64 b, gc_i64 e, bool single,
                                                 gc_i64 pitch,
                                                 const gcn_npt_vec3 &ss)
{
    if (single) {
        const int s = gmem[b];
        for (int c = 0; c < 3; ++c) {
            const gc_i64 k = (gc_i64)c * pitch + s;
            x[k] = ss.v[c] * from[k];
        }
        return;
    }
    double tm = 0.0, rcm[3] = { 0.0, 0.0, 0.0 };
    for (gc_i64 i = b; i < e; ++i) {
        const int s = gmem[i];
        const double mm = m[s];
        tm += mm;
        for (int c = 0; c < 3; ++c) rcm[c] += mm * from[(gc_i64)c * pitch + s];
    }
    for (int c = 0; c < 3; ++c) rcm[c] /= tm;
    for (gc_i64 i = b; i < e; ++i) {
        const int s = gmem[i];
        for (int c = 0; c < 3; ++c) {
            const gc_i64 k = (gc_i64)c * pitch + s;
            x[k] = from[k] + (ss.v[c] - 1.0) * rcm[c];
        }
    }
}

template <class S>
__global__ void gcn_kern_npt_group_scale(S *__restrict__ v,
                                         const gc_f64 *__restrict__ m,
                                         const gc_i64 *__restrict__ goff,
                                         const gc_i32 *__restrict__ gmem,
                                         const gc_u8 *__restrict__ gkind,
                                         gc_i64 ngroup, gc_i64 pitch,
                                         gcn_npt_vec3 vs)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x)
        npt_group_vscale(v, m, gmem, goff[g], goff[g + 1],
                         gkind[g] == GC_GROUP_SINGLE, pitch, vs);
}

/* r-RESPA mtk_barostat_vv1: on an outer step (half_dt_long != 0) the group
 * scale vs and the long kick; the short kick; then compute_vv1_coord_group
 * (scale from x_ref, drift, scale about the drifted centre of mass). */
template <class S>
__global__ void gcn_kern_npt_respa_vv1(S *__restrict__ v,
                                       gc_f64 *__restrict__ x,
                                       const gc_f64 *__restrict__ xref,
                                       const gc_f64 *__restrict__ fr,
                                       const gc_f64 *__restrict__ fb,
                                       const gc_f64 *__restrict__ fp,
                                       const gc_f64 *__restrict__ m,
                                       const gc_f64 *__restrict__ minv,
                                       const gc_i64 *__restrict__ goff,
                                       const gc_i32 *__restrict__ gmem,
                                       const gc_u8 *__restrict__ gkind,
                                       gc_i64 ngroup, gc_i64 pitch,
                                       gcn_npt_vec3 vs, gcn_npt_vec3 ss,
                                       double dt, double half_dt,
                                       double half_dt_long)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 b = goff[g], e = goff[g + 1];
        const bool single = gkind[g] == GC_GROUP_SINGLE;
        if (half_dt_long != 0.0) {
            npt_group_vscale(v, m, gmem, b, e, single, pitch, vs);
            for (gc_i64 i = b; i < e; ++i) {
                const int s = gmem[i];
                const double factor = half_dt_long * minv[s];
                for (int c = 0; c < 3; ++c) {
                    const gc_i64 k = (gc_i64)c * pitch + s;
                    v[k] = v[k] + factor * fp[k];
                }
            }
        }
        for (gc_i64 i = b; i < e; ++i) {
            const int s = gmem[i];
            const double factor = half_dt * minv[s];
            for (int c = 0; c < 3; ++c) {
                const gc_i64 k = (gc_i64)c * pitch + s;
                v[k] = v[k] + factor * (fr[k] + fb[k]);
            }
        }
        npt_group_xscale(x, xref, m, gmem, b, e, single, pitch, ss);
        for (gc_i64 i = b; i < e; ++i) {
            const int s = gmem[i];
            for (int c = 0; c < 3; ++c) {
                const gc_i64 k = (gc_i64)c * pitch + s;
                x[k] = x[k] + v[k] * dt;
            }
        }
        npt_group_xscale(x, x, m, gmem, b, e, single, pitch, ss);
    }
}

/* r-RESPA mtk_barostat_vv2, per atom: on an outer step the mean velocity and
 * mean long force are removed (the joined force is refreshed) and the long
 * kick taken; then the short kick and, under rigid bonds, the frame change
 * before the RATTLE. */
template <class S>
__global__ void gcn_kern_npt_respa_vv2(S *__restrict__ v,
                                       gc_f64 *__restrict__ vfull,
                                       gc_f64 *__restrict__ xref,
                                       gc_f64 *__restrict__ fp,
                                       S *__restrict__ f,
                                       const gc_f64 *__restrict__ fr,
                                       const gc_f64 *__restrict__ fb,
                                       const gc_f64 *__restrict__ x,
                                       const gc_f64 *__restrict__ minv,
                                       const gc_f64 *__restrict__ cm,
                                       gc_i64 n, gc_i64 pitch,
                                       gcn_npt_vec3 bm, double half_dt,
                                       double half_dt_long, int rigid)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        const double fl = half_dt_long * minv[s];
        const double fs = half_dt * minv[s];
        for (int c = 0; c < 3; ++c) {
            const gc_i64 k = (gc_i64)c * pitch + s;
            double vv = v[k];
            if (half_dt_long != 0.0) {
                const double pp = fp[k] - cm[4 + c] / cm[7];
                vv = vv - cm[c] / cm[3];
                fp[k] = pp;
                f[k] = fr[k] + fb[k] + pp;
                vv = vv + fl * pp;
            }
            vv = vv + fs * (fr[k] + fb[k]);
            if (rigid) {
                const double cd = vv + bm.v[c] * x[k];
                vfull[k] = vv;
                xref[k] = cd;
                vv = cd;
            }
            v[k] = vv;
        }
    }
}

/* n image offsets of three components, `stride` doubles apart */
__global__ void gcn_kern_npt_scale_moves(gc_f64 *__restrict__ move,
                                         gc_i64 n, gc_i64 stride,
                                         gcn_npt_vec3 scale)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x)
        for (int c = 0; c < 3; ++c) move[i * stride + c] *= scale.v[c];
}

/* ABI entry points */

using namespace gcn;

static gc_status npt_ready(const gc_context *ctx)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    const struct gcn_device *d = ctx->native;
    if (!d->list_valid || !d->step_ready ||
        d->plan.ensemble != GC_ENSEMBLE_NPT) return GC_E_STATE;
    return GC_OK;
}

/* mtk_barostat_vv2's eight sums of velocity and force into reduce_out;
 * several ranks add theirs on the host first (stock's mpi_allreduce). */
template <class F>
static gc_status npt_com_sums(gc_context *ctx, const F *f)
{
    struct gcn_device *d = ctx->native;
    cudaStream_t s = d->stream;
    const gc_i64 nb = npt_grid(d->num_owned, GCN_RED_BLOCK);
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_npt_com_partials<<<(unsigned)nb, GCN_RED_BLOCK, 0, s>>>(
            d->vel.as<decltype(z)>(), f, d->mass, d->num_owned, d->pitch,
            d->reduce_partial, (int)nb);
    });
    gcn_kern_reduce_finalize<<<1, 32, 0, s>>>(d->reduce_partial, (int)nb, 8,
                                              d->reduce_out);
    if (ctx->nproc > 1) {
#ifdef HAVE_MPI_GENESIS
        gc_f64 cm[8];
        if (cudaMemcpyAsync(cm, d->reduce_out, sizeof(cm),
                            cudaMemcpyDeviceToHost, s) != cudaSuccess)
            return GC_E_DEVICE;
        gc_status st = dev_sync(s);
        if (st != GC_OK) return st;
        MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)ctx->comm);
        if (MPI_Allreduce(MPI_IN_PLACE, cm, 8, MPI_DOUBLE, MPI_SUM, comm)
                != MPI_SUCCESS) return GC_E_STATE;
        if (cudaMemcpyAsync(d->reduce_out, cm, sizeof(cm),
                            cudaMemcpyHostToDevice, s) != cudaSuccess)
            return GC_E_DEVICE;
        st = dev_sync(s);
        if (st != GC_OK) return st;
#else
        return GC_E_UNSUPPORTED;
#endif
    }
    return GC_OK;
}

extern "C" gc_status gpu_core_npt_vv1(gc_context *ctx, const gc_f64 *size_scale,
                                      const gc_f64 *vel_scale)
{
    gc_status st = npt_ready(ctx);
    if (st != GC_OK) return st;
    if (!size_scale || !vel_scale) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (native_ref_join(d) != GC_OK) return GC_E_DEVICE;
    gcn_npt_vec3 ss, vs;
    for (int c = 0; c < 3; ++c) { ss.v[c] = size_scale[c]; vs.v[c] = vel_scale[c]; }
    vf_each(d->vf32, [&](auto z) {
        using S = decltype(z);
        gcn_kern_npt_vv1<<<(unsigned)npt_grid(d->num_groups, GCN_BLOCK),
                           GCN_BLOCK, 0, d->stream>>>(
            d->vel.as<S>(), d->coord, d->coord_ref, d->force.as<S>(),
            d->mass, d->inv_mass, d->group_offset, d->group_member,
            d->group_kind, d->num_groups, d->pitch, ss, vs, d->plan.dt,
            d->plan.half_dt);
    });
    return dev_launched("npt_vv1");
}

extern "C" gc_status gpu_core_npt_vv2(gc_context *ctx, const gc_f64 *vel_scale,
                                      const gc_f64 *bmoment)
{
    gc_status st = npt_ready(ctx);
    if (st != GC_OK) return st;
    if (!vel_scale || !bmoment) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    cudaStream_t s = d->stream;

    vf_each(d->vf32, [&](auto z) {
        st = npt_com_sums(ctx, d->force.as<const decltype(z)>());
    });
    if (st != GC_OK) return st;

    gcn_npt_vec3 vs, bm;
    for (int c = 0; c < 3; ++c) { vs.v[c] = vel_scale[c]; bm.v[c] = bmoment[c]; }
    vf_each(d->vf32, [&](auto z) {
        using S = decltype(z);
        gcn_kern_npt_vv2<<<(unsigned)npt_grid(d->num_groups, GCN_BLOCK),
                           GCN_BLOCK, 0, s>>>(
            d->vel.as<S>(), d->vel_half, d->vel_full, d->coord_ref,
            d->force.as<S>(), d->coord, d->mass, d->inv_mass,
            d->group_offset, d->group_member, d->group_kind, d->reduce_out,
            d->num_groups, d->pitch, vs, bm, d->plan.half_dt,
            d->plan.rigid_bond);
    });
    return dev_launched("npt_vv2");
}

extern "C" gc_status gpu_core_npt_rattle_end(gc_context *ctx)
{
    gc_status st = npt_ready(ctx);
    if (st != GC_OK) return st;
    struct gcn_device *d = ctx->native;
    if (!d->plan.rigid_bond) return GC_OK;
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_npt_rattle_end<<<(unsigned)npt_grid(d->num_owned, GCN_BLOCK),
                                  GCN_BLOCK, 0, d->stream>>>(
            d->vel.as<decltype(z)>(), d->vel_full, d->coord_ref,
            d->num_owned, d->pitch);
    });
    return dev_launched("npt_rattle_end");
}

extern "C" gc_status gpu_core_npt_group_scale(gc_context *ctx,
                                              const gc_f64 *scale)
{
    gc_status st = npt_ready(ctx);
    if (st != GC_OK) return st;
    if (!scale) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (native_ref_join(d) != GC_OK) return GC_E_DEVICE;
    gcn_npt_vec3 vs;
    for (int c = 0; c < 3; ++c) vs.v[c] = scale[c];
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_npt_group_scale<<<(unsigned)npt_grid(d->num_groups,
                                                      GCN_BLOCK),
                                   GCN_BLOCK, 0, d->stream>>>(
            d->vel.as<decltype(z)>(), d->mass, d->group_offset,
            d->group_member, d->group_kind, d->num_groups, d->pitch, vs);
    });
    return dev_launched("npt_group_scale");
}

extern "C" gc_status gpu_core_npt_respa_vv1(gc_context *ctx,
                                            const gc_f64 *vel_scale,
                                            gc_f64 half_dt_long,
                                            const gc_f64 *size_scale)
{
    gc_status st = npt_ready(ctx);
    if (st != GC_OK) return st;
    if (!vel_scale || !size_scale) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (native_ref_join(d) != GC_OK) return GC_E_DEVICE;
    gcn_npt_vec3 vs, ss;
    for (int c = 0; c < 3; ++c) { vs.v[c] = vel_scale[c]; ss.v[c] = size_scale[c]; }
    vf_each(d->vf32, [&](auto z) {
    gcn_kern_npt_respa_vv1<<<(unsigned)npt_grid(d->num_groups, GCN_BLOCK),
                             GCN_BLOCK, 0, d->stream>>>(
        d->vel.as<decltype(z)>(), d->coord, d->coord_ref, d->force_real, d->force_bond,
        d->force_recip, d->mass, d->inv_mass, d->group_offset,
        d->group_member, d->group_kind, d->num_groups, d->pitch, vs, ss,
        d->plan.dt, d->plan.half_dt, half_dt_long);
    });
    return dev_launched("npt_respa_vv1");
}

extern "C" gc_status gpu_core_npt_respa_vv2(gc_context *ctx,
                                            gc_f64 half_dt_long,
                                            const gc_f64 *bmoment)
{
    gc_status st = npt_ready(ctx);
    if (st != GC_OK) return st;
    if (!bmoment) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (half_dt_long != 0.0) {
        st = npt_com_sums(ctx, d->force_recip);
        if (st != GC_OK) return st;
    }
    gcn_npt_vec3 bm;
    for (int c = 0; c < 3; ++c) bm.v[c] = bmoment[c];
    vf_each(d->vf32, [&](auto z) {
        using S = decltype(z);
        gcn_kern_npt_respa_vv2<<<(unsigned)npt_grid(d->num_owned, GCN_BLOCK),
                                 GCN_BLOCK, 0, d->stream>>>(
            d->vel.as<S>(), d->vel_full, d->coord_ref, d->force_recip,
            d->force.as<S>(), d->force_real, d->force_bond, d->coord,
            d->inv_mass, d->reduce_out, d->num_owned, d->pitch, bm,
            d->plan.half_dt, half_dt_long, d->plan.rigid_bond);
    });
    return dev_launched("npt_respa_vv2");
}

extern "C" gc_status gpu_core_box_scale(gc_context *ctx, const gc_f64 *scale,
                                        const gc_f64 *box, gc_i32 recip)
{
    gc_status st = npt_ready(ctx);
    if (st != GC_OK) return st;
    if (!scale || !box) return GC_E_ARG;
    struct gcn_device *d = ctx->native;

    /* The cell grid is fixed; the list radius must stay inside the two face
     * cells a split axis exchanges, or the halo would miss pairs. */
    double csize[3];
    for (int k = 0; k < 3; ++k) {
        if (!(box[k] > 0.0) || !std::isfinite(box[k]) ||
            !(scale[k] > 0.0) || !std::isfinite(scale[k])) return GC_E_ARG;
        csize[k] = box[k] / (double)ctx->geo.cell[k];
        const double need = ctx->geo.pairlistdist / csize[k];
        if (d->layout.nd[k] > 1 && need > 2.0 + 1.0e-12) {
            std::fprintf(stderr, "GPU_Core_Error> phase=box_scale rank=%d "
                         "axis=%d box=%.9g cell_size=%.9g: the list radius "
                         "now reaches past the two halo cells\n",
                         (int)ctx->rank, k, box[k], csize[k]);
            return GC_E_UNSUPPORTED;
        }
    }
    for (int k = 0; k < 3; ++k) {
        d->box[k] = box[k];
        ctx->geo.system_size[k] = box[k];
        ctx->geo.cell_size[k] = csize[k];
    }

    gcn_npt_vec3 sc;
    for (int c = 0; c < 3; ++c) sc.v[c] = scale[c];
    cudaStream_t s = d->stream;
    if (d->num_groups > 0)
        gcn_kern_npt_scale_moves<<<(unsigned)npt_grid(d->num_groups, GCN_BLOCK),
                                   GCN_BLOCK, 0, s>>>(
            d->group_force_move, d->num_groups, 3, sc);
    if (d->num_pairs > 0)
        gcn_kern_npt_scale_moves<<<(unsigned)npt_grid(d->num_pairs, GCN_BLOCK),
                                   GCN_BLOCK, 0, s>>>(
            &d->pair[0].move[0], d->num_pairs,
            (gc_i64)(sizeof(struct gcn_pair) / sizeof(gc_f64)), sc);
    st = native_nbc_rebox(ctx, scale);
    if (st != GC_OK) return st;
    gc_i64 nview = 0;
    gc_f64 *view = native_dist_view_move(ctx, &nview);
    if (view && nview > 0)
        gcn_kern_npt_scale_moves<<<(unsigned)npt_grid(nview, GCN_BLOCK),
                                   GCN_BLOCK, 0, s>>>(view, nview, 3, sc);
    st = dev_launched("box_scale");
    if (st != GC_OK) return st;
    if (recip && d->reciprocal_ready) st = native_pme_rebox(ctx);
    return st;
}
