/*
 * gpu_domain.h : this rank's place in the decomposition (private to the CUDA
 * units; nothing here crosses the C ABI).
 *
 * GENESIS lays the domains out as a regular grid and splits each axis by
 * quotient and remainder (sp_domain.fpp), so a rank's owned cell span, the
 * owner of any global cell and every neighbour are closed-form arithmetic on
 * three integers per axis; no rank or cell table is built.
 *
 *   rank = ip[0] + ip[1]*nd[0] + ip[2]*nd[0]*nd[1]
 *   the first (ncel % nd) domains of an axis own ncel/nd + 1 cells, the rest ncel/nd
 *
 * A rank's cell table is dense over the box of its owned and ghost cells in
 * the extended coordinates GENESIS's own map uses (a wrapped ghost keeps the
 * coordinate it reached):
 *
 *   lo[k] = (cell_start[k] - 1) - halo[k]  (zero based),  dim[k] = len[k] + 2*halo[k]
 *
 * `halo[k]` is zero on an axis this rank owns whole; at one rank lo is zero
 * and the box index is the global cell key.
 */

#ifndef GPU_DOMAIN_H
#define GPU_DOMAIN_H

#include "gpu_core_abi.h"

/* Host driver and kernels resolve a cell through the same code. */
#ifdef __CUDACC__
#define GCN_HD __host__ __device__ __forceinline__
#else
#define GCN_HD inline
#endif

struct gcn_layout {
    gc_i32 ncel[3];       /* global cell counts                           */
    gc_i32 nd[3];         /* domains per axis                             */
    gc_i32 ip[3];         /* this rank's domain coordinates               */
    gc_i32 start[3];      /* first owned cell, zero based                 */
    gc_i32 len[3];        /* owned cells on this axis                     */
    gc_i32 halo[3];       /* ghost cells beyond each face, 0 when nd == 1 */
    gc_i32 lo[3];         /* box origin in extended coordinates           */
    gc_i32 dim[3];        /* box extent                                   */
    gc_i32 rank;          /* communicator-local domain rank               */
    gc_i32 nproc;
    gc_i32 ncell_box;     /* dim[0]*dim[1]*dim[2]                         */
    gc_i32 ncell_owned;   /* len[0]*len[1]*len[2]                         */
};

namespace gcn {
gc_status layout_build(const gc_context *ctx, const gc_i32 *want_halo,
                       struct gcn_layout *out, const char **reason);
/* layout_build with the stencil the list radius needs (at least two). */
gc_status layout_native(const gc_context *ctx, struct gcn_layout *out,
                        const char **reason);
}

/* The span of domain index `p` on an axis of `n` cells split `d` ways
 * (sp_domain.fpp, zero based). */
GCN_HD void gcn_axis_span(gc_i32 n, gc_i32 d, gc_i32 p,
                          gc_i32 *start, gc_i32 *len)
{
    const gc_i32 q = n / d;
    const gc_i32 r = n - q * d;
    if (p < r) { *len = q + 1; *start = (q + 1) * p; }
    else       { *len = q;     *start = (q + 1) * r + q * (p - r); }
}

/* The domain index that owns zero-based global cell `g`: the inverse of
 * gcn_axis_span. */
GCN_HD gc_i32 gcn_axis_owner(gc_i32 n, gc_i32 d, gc_i32 g)
{
    const gc_i32 q = n / d;
    const gc_i32 r = n - q * d;
    const gc_i32 split = (q + 1) * r;
    if (g < split) return g / (q + 1);
    return r + (g - split) / q;
}

/* Wrap an extended coordinate into [0, n) and report the periods travelled.
 * The box never spans more than one period (layout_build refuses it), so one
 * step suffices. */
GCN_HD gc_i32 gcn_wrap_axis(gc_i32 g, gc_i32 n, gc_i32 *image)
{
    gc_i32 im = 0;
    if (g < 0)       { g += n; im = -1; }
    else if (g >= n) { g -= n; im = +1; }
    *image = im;
    return g;
}

/* The rank that owns an in-range global cell. */
GCN_HD gc_i32 gcn_rank_of_cell(const struct gcn_layout &L,
                               gc_i32 gx, gc_i32 gy, gc_i32 gz)
{
    const gc_i32 px = gcn_axis_owner(L.ncel[0], L.nd[0], gx);
    const gc_i32 py = gcn_axis_owner(L.ncel[1], L.nd[1], gy);
    const gc_i32 pz = gcn_axis_owner(L.ncel[2], L.nd[2], gz);
    return px + L.nd[0] * (py + L.nd[1] * pz);
}

/* The rank whose domain coordinates are this rank's shifted by (dx,dy,dz),
 * periodically; a shift that lands back on this rank gives this rank. */
GCN_HD gc_i32 gcn_rank_shift(const struct gcn_layout &L,
                             gc_i32 dx, gc_i32 dy, gc_i32 dz)
{
    gc_i32 px = (L.ip[0] + dx) % L.nd[0]; if (px < 0) px += L.nd[0];
    gc_i32 py = (L.ip[1] + dy) % L.nd[1]; if (py < 0) py += L.nd[1];
    gc_i32 pz = (L.ip[2] + dz) % L.nd[2]; if (pz < 0) pz += L.nd[2];
    return px + L.nd[0] * (py + L.nd[1] * pz);
}

/* Extended coordinates to the box index, or -1 outside this rank's box. On an
 * axis owned whole the coordinate is wrapped first (the box is the period). */
GCN_HD gc_i32 gcn_box_index(const struct gcn_layout &L,
                            gc_i32 ex, gc_i32 ey, gc_i32 ez)
{
    gc_i32 e[3] = { ex, ey, ez };
    gc_i32 c[3];
    for (int k = 0; k < 3; ++k) {
        gc_i32 v = e[k];
        if (L.halo[k] == 0) {
            while (v < 0)          v += L.ncel[k];
            while (v >= L.ncel[k]) v -= L.ncel[k];
        }
        c[k] = v - L.lo[k];
        if (c[k] < 0 || c[k] >= L.dim[k]) return -1;
    }
    return c[0] + L.dim[0] * (c[1] + L.dim[1] * c[2]);
}

/* The box index back to extended coordinates. */
GCN_HD void gcn_box_coords(const struct gcn_layout &L, gc_i32 c,
                           gc_i32 *ex, gc_i32 *ey, gc_i32 *ez)
{
    const gc_i32 cx = c % L.dim[0];
    const gc_i32 t  = c / L.dim[0];
    const gc_i32 cy = t % L.dim[1];
    const gc_i32 cz = t / L.dim[1];
    *ex = cx + L.lo[0];
    *ey = cy + L.lo[1];
    *ez = cz + L.lo[2];
}

/* Is that box cell one of this rank's owned cells? */
GCN_HD bool gcn_box_owned(const struct gcn_layout &L, gc_i32 c)
{
    gc_i32 e[3];
    gcn_box_coords(L, c, &e[0], &e[1], &e[2]);
    for (int k = 0; k < 3; ++k)
        if (e[k] < L.start[k] || e[k] >= L.start[k] + L.len[k]) return false;
    return true;
}

/* The displacement from one box cell to another, in the frame the pair
 * enumerator used when it built their pair: the offset it added to the home
 * cell's extended coordinate. On a split axis that is the plain difference of
 * the stored coordinates; on an axis owned whole the box is the period and
 * the raw difference is folded back by one period (the box spans at most one;
 * see gcn_wrap_axis). Pair storage, the near-pair test of the mask count and
 * the exclusion-mask lookup all use this frame. */
GCN_HD void gcn_box_cell_disp(const struct gcn_layout &L,
                              gc_i32 hx, gc_i32 hy, gc_i32 hz,
                              gc_i32 ox, gc_i32 oy, gc_i32 oz,
                              gc_i32 *d)
{
    const gc_i32 h[3] = { hx, hy, hz };
    const gc_i32 o[3] = { ox, oy, oz };
    for (int k = 0; k < 3; ++k) {
        gc_i32 t = o[k] - h[k];
        if (L.halo[k] == 0) {
            if (t >  L.ncel[k] / 2) t -= L.ncel[k];
            if (t < -L.ncel[k] / 2) t += L.ncel[k];
        }
        d[k] = t;
    }
}

#endif /* GPU_DOMAIN_H */
