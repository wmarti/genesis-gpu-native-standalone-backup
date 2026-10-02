/* gpu_domain.cu : the distributed state of the device-native core.
 *
 * Owns this rank's cell box (gpu_domain.h), the spatial halo that fills its
 * ghost cells, the additive return of the forces the ghosts accumulate, and
 * the migration of groups and their terms. Messages go through the exchange
 * primitive of gpu_core_xchg.h; this unit owns the schedules and the pack and
 * unpack kernels. */

#include "gpu_core_native.h"
#include "gpu_domain.h"
#include "gpu_nbcluster.h"
#include "gpu_fixed_sum.cuh"

#include <cstdio>
#include <cstring>
#include <climits>
#include <math.h>

namespace gcn {

/* Build this rank's layout, or decline the geometry and say why. `want_halo`
 * is the stencil half width derived from the list radius and cell size; an
 * axis this rank owns whole needs none (the box is the period there).
 *
 * Refused: fewer cells than domains on an axis; a halo wider than a
 * neighbour's whole domain (a group would travel more than one hop of the
 * three-pass schedule); a box that would wrap onto itself (its ghosts would
 * have no identity). */
gc_status layout_build(const gc_context *ctx, const gc_i32 *want_halo,
                       struct gcn_layout *out, const char **reason)
{
    static const char *ok = "eligible";
    if (!ctx || !want_halo || !out || !reason) return GC_E_ARG;
    const gc_geometry_desc &g = ctx->geo;
    struct gcn_layout L;

    L.rank  = ctx->rank;
    L.nproc = ctx->nproc;
    if (L.rank < 0 || L.nproc <= 0 || L.rank >= L.nproc) {
        *reason = "the domain rank is outside its communicator";
        return GC_E_ARG;
    }

    gc_i64 domains = 1;
    for (int k = 0; k < 3; ++k) {
        L.ncel[k] = g.cell[k];
        L.nd[k]   = g.num_domain[k];
        if (L.nd[k] <= 0 || L.ncel[k] <= 0) {
            *reason = "a non-positive cell or domain count";
            return GC_E_ARG;
        }
        if (L.ncel[k] < L.nd[k]) {
            *reason = "an axis has fewer cells than domains";
            return GC_E_UNSUPPORTED;
        }
        gc_i64 next;
        if (!mul_checked(domains, L.nd[k], &next)) {
            *reason = "the domain-grid product overflows";
            return GC_E_OVERFLOW;
        }
        domains = next;
    }
    if (domains != (gc_i64)L.nproc) {
        *reason = "the domain grid does not cover the communicator";
        return GC_E_MISMATCH;
    }

    L.ip[0] = L.rank % L.nd[0];
    L.ip[1] = (L.rank / L.nd[0]) % L.nd[1];
    L.ip[2] = L.rank / (L.nd[0] * L.nd[1]);

    for (int k = 0; k < 3; ++k) {
        gcn_axis_span(L.ncel[k], L.nd[k], L.ip[k], &L.start[k], &L.len[k]);
        /* GENESIS's own owned span is the authority; a layout that disagrees
         * would put the core and the bridge's import in different domains. */
        if (L.start[k] + 1 != g.cell_start[k] ||
            L.start[k] + L.len[k] != g.cell_end[k]) {
            *reason = "the derived owned span disagrees with GENESIS's";
            return GC_E_MISMATCH;
        }
        L.halo[k] = (L.nd[k] > 1) ? want_halo[k] : 0;
        if (L.halo[k] < 0) { *reason = "a negative halo width"; return GC_E_ARG; }
        /* Bound by the shortest adjacent span, so the three-pass schedule
         * never skips a second neighbour. */
        if (L.halo[k] > L.ncel[k] / L.nd[k]) {
            *reason = "the list radius reaches past a neighbour domain; "
                      "use fewer domains on that axis or a shorter list";
            return GC_E_UNSUPPORTED;
        }
        L.lo[k] = L.start[k] - L.halo[k];
        gc_i64 dim = (gc_i64)L.len[k] + 2 * (gc_i64)L.halo[k];
        if (!narrow(dim, &L.dim[k])) {
            *reason = "the cell-box extent does not fit an index";
            return GC_E_OVERFLOW;
        }
        if (L.halo[k] > 0 && L.dim[k] > L.ncel[k]) {
            *reason = "this rank's box would wrap onto itself";
            return GC_E_UNSUPPORTED;
        }
    }

    {
        gc_i64 box = 1, own = 1;
        for (int k = 0; k < 3; ++k) {
            if (!mul_checked(box, L.dim[k], &box) ||
                !mul_checked(own, L.len[k], &own)) {
                *reason = "the cell-box count overflows";
                return GC_E_OVERFLOW;
            }
        }
        gc_i32 n32;
        if (!narrow(box, &n32)) { *reason = "the cell box does not fit an index";
                                  return GC_E_OVERFLOW; }
        L.ncell_box   = (gc_i32)box;
        L.ncell_owned = (gc_i32)own;
        if (own != (gc_i64)ctx->ncell_local) {
            *reason = "the derived owned cell count disagrees with GENESIS's";
            return GC_E_MISMATCH;
        }
    }

    *out = L;
    *reason = ok;
    return GC_OK;
}

gc_status layout_native(const gc_context *ctx, struct gcn_layout *out,
                        const char **reason)
{
    if (!ctx || !out || !reason) return GC_E_ARG;
    gc_i32 want_halo[3];
    for (int k = 0; k < 3; ++k) {
        const double need = ctx->geo.pairlistdist / ctx->geo.cell_size[k];
        int nh = (int)ceil(need - 1.0e-12);
        if (nh < 2) nh = 2;
        if (nh > GCN_MAX_STENCIL) {
            *reason = "the list radius needs a wider stencil than the core has";
            return GC_E_UNSUPPORTED;
        }
        want_halo[k] = nh;
    }
    return layout_build(ctx, want_halo, out, reason);
}

}  /* namespace gcn */

/* Wire records, private to this unit. A forward record carries raw and force-
 * view coordinates (distinct numerical contracts); a reverse record keeps the
 * three force accumulators separate so the real-space coord*force virial is
 * not polluted by bonded or reciprocal terms. */

#include <vector>
#include <cstring>
#ifdef HAVE_MPI_GENESIS
#include <mpi.h>
#endif

#define GCN_HALO_RECORD_SCHEMA 1

/* The attach-time outer-layer count, on tags no plan owns: the halo plans
 * start at tag_base 1024, the bandwidth probe uses 0.  One tag per side. */
#define GCN_HALO_COUNT_TAG 1000

/* A record's identity. Fields every record of a message shares (schema,
 * sender, edge, epoch, generations) travel per record only in the checked
 * build; the transport verifies sender, destination, operation and epoch from
 * its header. */
struct gcn_halo_identity {
    gc_i32 owner_rank;
    gc_gid gid;
    gc_image image;
};

struct gcn_coord_wire {
    gcn_halo_identity id;
    gc_f64 coord[3];
    gc_f64 move[3];   /* the receiver forms force_coord = coord + move, the
                         owner's own sum (gcn_kern_force_coord), so the view
                         itself does not travel */
    gc_f64 charge;
    gc_i32 cls;
    gc_i32 cell;      /* the atom's home cell, global linear (x fastest):
                         its cell with the image taken off */
};

/* Registration record of a rebuild (native_dist_halo_refresh): gcn_coord_wire
 * in 64 bytes. The view move is carried as a per-axis period count
 * (gcn_group_move) that the pack must reproduce bit for bit; the image is
 * three int16 shifts. */
struct gcn_reg_wire {
    gc_f64 coord[3];
    gc_f64 charge;
    gc_gid gid;
    gc_i32 owner_rank;
    gc_i32 cell;      /* the home cell, as gcn_coord_wire::cell */
    gc_i32 cls;
    gc_i16 image[3];
    gc_i16 turns[3];
};
static_assert(sizeof(gcn_reg_wire) == 64, "registration record size");

/* A group's view move from its whole number of periods: the expression
 * gcn_kern_force_coord evaluates, out of line so pack and registration agree. */
__device__ __noinline__ gc_f64 gcn_group_move(gc_f64 box, gc_f64 turns)
{
    return box * 0.5 - box * turns;
}

/* A ghost's whole partial force, added by the owner to force_bond. With an
 * exact return (dist_exact_return) it is the sum of the ghost's two fixed-
 * point force words, added to force_bond_fx as integers so the owner's force
 * does not depend on which rank computed a term; else the FP64 `total`. */
struct gcn_force_wire {
    gcn_halo_identity id;
    union {
        gc_f64 total[3];
        unsigned long long word[3];
    };
};

/* Later steps of a generation send no record: coordinates forward and totals
 * back, three doubles per atom, in the order the generation's first step
 * fixed. */

/* One pass per split axis: coordinates x -> y -> z, forces back z -> y -> x.
 * Pass p exchanges the two axis-p bands (owned atoms and ghosts of earlier
 * passes) with the two axis-p neighbours, so a corner ghost reaches its
 * diagonal neighbour in up to three hops; the return relays forces back along
 * the same hops. */
struct gcn_dist_edges {
    gc_i32 key[3][2], partner[3][2], peer[3][2];   /* [axis][lower, upper] */
};

/* The pass and side each ghost slot of the last refresh arrived on. An owned
 * group that left the owned span keeps its owner until the next migration, so
 * relay and return follow the edge a record actually came on; before the
 * first refresh (`valid` 0) the cell label decides. */
struct gcn_dist_arrive {
    gc_i32 valid;
    gc_i64 first[3][2];
    gc_i64 n[3][2];
};

/* Counts of a device-counted refresh (dist_refresh_dev). */
struct gcn_dist_dcount {
    gc_i64 resident;
    gc_u64 over;
    gc_i64 sent[3][2];
    struct gcn_dist_arrive arrive;
};

struct gcn_dist_state {
    gcx_halo *halo;
    gcx_probe_report probe;
    cudaEvent_t producer;
    cudaEvent_t consumer;
    void *coord_send[2];                /* per side, reused by every pass */
    void *coord_recv[2];
    void *force_send[2];
    void *force_recv[2];
    gc_i64 capacity_records;            /* records per side and pass      */
    gc_i64 coord_capacity;
    gc_i64 force_capacity;
    gc_i64 *pack_count;                 /* [2] */
    gc_i64 *pack_pin;                   /* [3] pinned: a full pass's counts
                                           and its pack verdict */
    gc_i32 *pack_flag;
    gc_i64 *pack_pos;
    gc_i64 pack_cap;
    gc_i64 *bad;                        /* [3]: this pass's pack, the forward's
                                           and the reverse's step verdicts */
    /* The step's counts [reverse][axis][side], fixed by the generation's
     * first step. */
    gc_i64 step_sent[2][3][2];
    gc_i64 step_got[2][3][2];
    gc_i64 step_gen[2][2];              /* [reverse]: slot and plan generation */
    gc_i32 *route[2][3];                /* [2*capacity_records]: sent slots  */
    gc_i32 *land[2][3];                 /* [2*capacity_records]: landing slots */
    gc_f64 *view_move;                  /* [3*visit_capacity]: each slot's
                                           force_coord - coord, as added */
    gc_i32 *coord_visit;                /* [visit_capacity] per resident slot */
    gc_i32 *coord_reject;               /* [visit_capacity], CHECK: why not filled */
    gc_i32 *force_visit;                /* [visit_capacity], bit 2*axis+side */
    gc_i64 visit_capacity;              /* covers num_resident            */
    gc_gid *owner_key;
    gc_i32 *owner_val;
    gc_i64 owner_cap;
    gc_gid *ghost_key;
    gc_image *ghost_image;
    gc_i32 *ghost_edge;
    gc_i32 *ghost_val;
    gc_i64 ghost_cap;
    gc_i64 sequence;
    gc_i32 probe_live;
    gc_i32 map_ready;
    gc_i64 map_generation;
    gc_i64 map_plan_generation;
    struct gcn_dist_edges coord, force;
    struct gcn_dist_arrive arrive;
    /* [reverse][axis]: both edges are device-sequenced peer links (slim steps
     * use the transport's devsig protocol). */
    gcx_device_edge dev[2][3][2];
    gc_i32 fused[2][3];
    gc_i32 stores[2][3];
    /* [reverse][axis]: the open epoch held on the device (gcx_seq); the
     * pass's plan keeps the epochs (gcx_move_claim). */
    gcx_seq *seq;
    /* [reverse][axis]: a pass opened ahead (dist_open_all): epoch and slot.
     * `open_st` is the opening's status; `late_release` holds the slot
     * releases for after the last pass. */
    gc_i32 opened[2][3];
    gc_i64 open_epoch[2][3];
    gc_i32 open_slot[2][3];
    gc_status open_st[2];
    gc_i32 late_release[2];
    void *wire_out[2];
    const void *wire_in[2];
    /* Set by a halo refresh, taken by the next forward (native_dist_forward). */
    gc_i32 fresh;
    /* Device-counted refresh enabled: every forward edge is a device-
     * sequenced peer link and the ranks have a device sum (`devcount`, agreed
     * at attach). `dc_redo` counts refreshes that outgrew `dc_hist`. */
    gc_i32 devcount;
    gc_i32 dc_hist;
    gc_i64 dc_redo;
    gc_i64 dc_runs, dc_records, dc_margin;  /* this rank's sends: counted
                                               records, capacity unused */
    gcn_dist_dcount *dc;
    gcn_dist_dcount *dc_pin;
    gcx_wsum *wsum;
};

namespace {

__device__ __forceinline__ gc_u64 gcn_dist_hash(gc_u64 x)
{
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

/* A ghost is one atom at one image: (gid, image) names one slot, and the
 * arrival edge is an attribute the forward checks, not part of the key, so
 * the return finds a relayed ghost without knowing the edge. */
__device__ __forceinline__ gc_u64 gcn_dist_ghost_hash(gc_gid gid, gc_image image)
{
    return gcn_dist_hash((gc_u64)gid ^ gcn_dist_hash((gc_u64)image));
}

__host__ __device__ __forceinline__ gc_i32 gcn_dist_image_axis(gc_image im,
                                                               gc_i32 k)
{
    return (gc_i32)(gc_i16)((im >> (16 * k)) & 0xffffLL);
}

/* The image this rank adds to a record it sends to `side` of `axis`: the
 * period, at the seam only. */
__host__ __device__ __forceinline__ gc_image gcn_dist_sender_image(
    const gcn_layout &L, gc_i32 axis, gc_i32 side)
{
    gc_i32 shift = 0;
    if (side == 0 && L.start[axis] == 0) shift = 1;
    if (side == 1 && L.start[axis] + L.len[axis] == L.ncel[axis]) shift = -1;
    return (((gc_image)(gc_i16)shift & 0xffffLL) << (16 * axis));
}

/* The pass a box cell's atoms arrive in: the last axis on which the cell
 * lies outside the owned span, or -1 for an owned cell. */
__host__ __device__ __forceinline__ gc_i32 gcn_dist_arrival(
    const gcn_layout &L, const gc_i32 *e)
{
    gc_i32 a = -1;
    for (int k = 0; k < 3; ++k)
        if (e[k] < L.start[k] || e[k] >= L.start[k] + L.len[k]) a = k;
    return a;
}

/* The pass (-1 for an owned slot) and edge side a resident slot arrived
 * on; e is the slot's box coordinates. */
__host__ __device__ __forceinline__ gc_i32 gcn_dist_slot_arrival(
    const gcn_dist_arrive &A, const gcn_layout &L, gc_i64 s, gc_i64 nowned,
    const gc_i32 *e, gc_i32 *side)
{
    *side = 0;
    if (s < nowned) return -1;
    if (!A.valid) {
        const gc_i32 a = gcn_dist_arrival(L, e);
        if (a >= 0) *side = e[a] < L.start[a] ? 0 : 1;
        return a;
    }
    for (int p = 0; p < 3; ++p)
        for (int k = 0; k < 2; ++k)
            if (s >= A.first[p][k] && s < A.first[p][k] + A.n[p][k]) {
                *side = k;
                return p;
            }
    return -2;                           /* a slot no pass registered */
}

/* The axis-p bands a cell lies in: bit 0 the lower, bit 1 the upper.  The
 * two overlap when the span is shorter than two stencils. */
__host__ __device__ __forceinline__ gc_i32 gcn_dist_bands(
    const gcn_layout &L, const gc_i32 *e, gc_i32 p)
{
    return (e[p] < L.start[p] + L.halo[p] ? 1 : 0) |
           (e[p] >= L.start[p] + L.len[p] - L.halo[p] ? 2 : 0);
}

/* Does a cell GENESIS's import already covers -- the one-cell shell around
 * the owned span -- hold these coordinates? */
__host__ __device__ __forceinline__ bool gcn_dist_in_shell(
    const gcn_layout &L, const gc_i32 *e)
{
    for (int k = 0; k < 3; ++k)
        if (e[k] < L.start[k] - 1 || e[k] > L.start[k] + L.len[k]) return false;
    return true;
}

/* Will this band cell's copy sent to `side` of axis p land outside the
 * neighbour's imported shell?  The neighbour shares every coordinate but
 * axis p, where the cell lands one cell out from the first band layer. */
__host__ __device__ __forceinline__ bool gcn_dist_lands_outer(
    const gcn_layout &L, const gc_i32 *e, gc_i32 p, gc_i32 side)
{
    if (e[p] != (side == 0 ? L.start[p] : L.start[p] + L.len[p] - 1))
        return true;
    for (int k = 0; k < p; ++k)
        if (e[k] < L.start[k] - 1 || e[k] > L.start[k] + L.len[k]) return true;
    return false;
}

__global__ void gcn_dist_map_clear(gc_gid *key, gc_i32 *val, gc_i32 *edge,
                                   gc_image *image, gc_i64 cap)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < cap;
         i += (gc_i64)gridDim.x * blockDim.x) {
        key[i] = 0;
        val[i] = -1;
        if (edge) edge[i] = -1;
        if (image) image[i] = 0;
    }
}

__global__ void gcn_dist_owner_insert(const gc_gid *gid, gc_i64 n,
                                      gc_gid *key, gc_i32 *val, gc_i64 cap,
                                      gc_i64 *bad)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        const gc_gid g = gid[s];
        gc_u64 h = gcn_dist_hash((gc_u64)g) & (gc_u64)(cap - 1);
        int placed = 0;
        for (gc_i64 p = 0; p < cap; ++p) {
            gc_i64 i = (gc_i64)((h + (gc_u64)p) & (gc_u64)(cap - 1));
            unsigned long long old = atomicCAS(
                (unsigned long long *)&key[i], 0ull, (unsigned long long)g);
            if (old == 0ull || old == (unsigned long long)g) {
                if (old == (unsigned long long)g && val[i] != -1 && val[i] != s)
                    atomicAdd((unsigned long long *)bad, 1ull);
                val[i] = (gc_i32)s;
                placed = 1;
                break;
            }
        }
        if (!placed) atomicAdd((unsigned long long *)bad, 1ull);
    }
}

/* The edge ghost slot s arrived on, or -1 when no pass registered it. */
__device__ __forceinline__ gc_i32 gcn_dist_ghost_edge(
    const gcn_dist_arrive &A, const gcn_layout &L, gc_i64 s, gc_i64 nowned,
    gc_i32 cell, const gcn_dist_edges &edges)
{
    gc_i32 e[3], side;
    gcn_box_coords(L, cell, &e[0], &e[1], &e[2]);
    const gc_i32 a = gcn_dist_slot_arrival(A, L, s, nowned, e, &side);
    if (a < 0) return -1;
    return edges.key[a][side];
}

__global__ void gcn_dist_ghost_insert(const gc_gid *gid,
                                      const gc_image *slot_image,
                                      const gc_i32 *cell_of, gc_i64 first,
                                      gc_i64 n, struct gcn_layout L,
                                      struct gcn_dist_arrive A,
                                      struct gcn_dist_edges edges,
                                      gc_gid *key, gc_image *image,
                                      gc_i32 *edge, gc_i32 *val, gc_i64 cap,
                                      gc_i64 *bad)
{
    for (gc_i64 q = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; q < n;
         q += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 s = first + q;
        const gc_i32 ek = gcn_dist_ghost_edge(A, L, s, first, cell_of[s], edges);
        if (ek < 0) { atomicAdd((unsigned long long *)bad, 1ull); continue; }
        const gc_gid g = gid[s];
        const gc_image im = slot_image[s];
        const gc_u64 h = gcn_dist_ghost_hash(g, im) & (gc_u64)(cap - 1);
        int placed = 0;
        for (gc_i64 p = 0; p < cap; ++p) {
            const gc_i64 i = (gc_i64)((h + (gc_u64)p) & (gc_u64)(cap - 1));
            const unsigned long long old = atomicCAS(
                (unsigned long long *)&key[i], 0ull, (unsigned long long)g);
            if (old == 0ull) {
                image[i] = im; edge[i] = ek; val[i] = (gc_i32)s;
                placed = 1; break;
            }
            if (old == (unsigned long long)g && image[i] == im) {
                if (val[i] != -1 && val[i] != s)
                    atomicAdd((unsigned long long *)bad, 1ull);
                val[i] = (gc_i32)s; placed = 1; break;
            }
        }
        if (!placed) atomicAdd((unsigned long long *)bad, 1ull);
    }
}

/* A second stream-ordered pass detects duplicate exact keys even if two
 * insertions raced before a claimed slot's image and edge were published. */
__global__ void gcn_dist_ghost_verify(const gc_gid *gid,
                                      const gc_image *slot_image,
                                      const gc_i32 *cell_of, gc_i64 first,
                                      gc_i64 n, struct gcn_layout L,
                                      struct gcn_dist_arrive A,
                                      struct gcn_dist_edges edges,
                                      const gc_gid *key, const gc_image *image,
                                      const gc_i32 *edge, const gc_i32 *val,
                                      gc_i64 cap, gc_i64 *bad)
{
    for (gc_i64 q = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; q < n;
         q += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 s = first + q;
        const gc_i32 ek = gcn_dist_ghost_edge(A, L, s, first, cell_of[s], edges);
        const gc_gid g = gid[s];
        const gc_image im = slot_image[s];
        const gc_u64 h = gcn_dist_ghost_hash(g, im) & (gc_u64)(cap - 1);
        gc_i32 matches = 0, found = -1, found_edge = -1;
        for (gc_i64 p = 0; p < cap; ++p) {
            const gc_i64 i = (gc_i64)((h + (gc_u64)p) & (gc_u64)(cap - 1));
            if (key[i] == 0) break;
            if (key[i] == g && image[i] == im) {
                ++matches; found = val[i]; found_edge = edge[i];
            }
        }
        if (ek < 0 || matches != 1 || found != s || found_edge != ek)
            atomicAdd((unsigned long long *)bad, 1ull);
    }
}

__device__ __forceinline__ gc_i32 gcn_dist_find_owner(
    gc_gid gid, const gc_gid *key, const gc_i32 *val, gc_i64 cap)
{
    gc_u64 h = gcn_dist_hash((gc_u64)gid) & (gc_u64)(cap - 1);
    for (gc_i64 p = 0; p < cap; ++p) {
        gc_i64 i = (gc_i64)((h + (gc_u64)p) & (gc_u64)(cap - 1));
        gc_gid k = key[i];
        if (k == gid) return val[i];
        if (k == 0) return -1;
    }
    return -1;
}

/* The ghost slot of (gid, image), or -1.  `edge` >= 0 also requires the
 * ghost to have arrived on that edge. */
__device__ __forceinline__ gc_i32 gcn_dist_find_ghost(
    gc_gid gid, gc_image image, gc_i32 edge,
    const gc_gid *key, const gc_image *gimage, const gc_i32 *gedge,
    const gc_i32 *val, gc_i64 cap)
{
    const gc_u64 h = gcn_dist_ghost_hash(gid, image) & (gc_u64)(cap - 1);
    for (gc_i64 p = 0; p < cap; ++p) {
        const gc_i64 i = (gc_i64)((h + (gc_u64)p) & (gc_u64)(cap - 1));
        const gc_gid k = key[i];
        if (k == gid && gimage[i] == image)
            return (edge < 0 || gedge[i] == edge) ? val[i] : -1;
        if (k == 0) return -1;
    }
    return -1;
}

/* Why a record's metadata did not match the edge it arrived on, or OK. One
 * rule for the unpack kernel and the diagnostic; codes follow the kernel's
 * test order. The owner is checked against the ghost table or home cell (a
 * relayed ghost's owner is not the sender). */
enum {
    GCN_DIST_WHY_OK = 0,
    GCN_DIST_WHY_SCHEMA = 1,      /* id.schema is not this record schema     */
    GCN_DIST_WHY_EDGE = 2,        /* edge_instance is not this edge's key    */
    GCN_DIST_WHY_SENDER = 3,      /* sender_rank is not the peer             */
    GCN_DIST_WHY_EPOCH = 4,       /* a different import epoch                */
    GCN_DIST_WHY_SLOT_GEN = 5,    /* a different slot generation             */
    GCN_DIST_WHY_PLAN_GEN = 6,    /* a different plan generation             */
    GCN_DIST_WHY_GID = 7,         /* gid <= 0                                */
    GCN_DIST_WHY_IMAGE = 8,       /* image carries bits above its 48         */
    GCN_DIST_WHY_NOT_FOUND = 9,   /* the ghost table has no such ghost       */
    GCN_DIST_WHY_GHOST_IMAGE = 10,/* the ghost is held under another image   */
    GCN_DIST_WHY_GHOST_OWNER = 11,/* the ghost is held under another owner   */
    GCN_DIST_WHY_NAMED = 12,      /* a rejected record had named this ghost  */
    GCN_DIST_WHY_UNNAMED = 13,    /* no record named this ghost at all       */
    GCN_DIST_WHY_N = 14
};

__device__ __forceinline__ gc_i32
gcn_dist_meta_why(const struct gcn_halo_identity &h, gc_i32 expected_edge,
                  gc_i32 expected_peer, gc_i64 epoch, gc_i64 slot_generation,
                  gc_i64 plan_generation)
{
    (void)expected_edge; (void)expected_peer; (void)epoch;
    (void)slot_generation; (void)plan_generation;
    if (h.gid <= 0) return GCN_DIST_WHY_GID;
    if (((gc_u64)h.image >> 48) != 0) return GCN_DIST_WHY_IMAGE;
    return GCN_DIST_WHY_OK;
}

/* dist_compact's side totals: one scan over both sides' flags leaves the
 * lower side's count at pos[n] and the grand total in count[1]. */
__device__ __forceinline__ void gcn_dist_compact_counts(const gc_i64 *pos,
                                                        gc_i64 n, gc_i64 *count)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        count[1] -= pos[n];
        count[0] = pos[n];
    }
}

/* One record's fields as the pack found them. */
struct gcn_dist_put_args {
    gc_i32 owner;
    gc_gid gid;
    gc_image image;
    gc_f64 coord[3], move[3], charge;
    gc_i32 cls, cell;
    gc_i32 sender, edge;
    gc_i64 epoch, slot_generation, plan_generation;
};

__device__ __forceinline__ bool gcn_dist_put(gcn_coord_wire *r,
                                             const gcn_dist_put_args &a,
                                             const gc_f64 *)
{
    r->id.owner_rank = a.owner;
    r->id.gid = a.gid;
    r->id.image = a.image;
    for (int k = 0; k < 3; ++k) {
        r->coord[k] = a.coord[k];
        r->move[k] = a.move[k];
    }
    r->charge = a.charge; r->cls = a.cls; r->cell = a.cell;
    return true;
}

/* False when a move is not box/2 less a whole number of periods that
 * gcn_group_move gives back exactly. */
__device__ __forceinline__ bool gcn_dist_put(gcn_reg_wire *r,
                                             const gcn_dist_put_args &a,
                                             const gc_f64 *box)
{
    bool ok = true;
    for (int k = 0; k < 3; ++k) {
        r->coord[k] = a.coord[k];
        r->image[k] = (gc_i16)((a.image >> (16 * k)) & 0xffffLL);
        const gc_f64 t = round((box[k] * 0.5 - a.move[k]) / box[k]);
        const bool fits = t >= -32767.0 && t <= 32767.0;
        r->turns[k] = fits ? (gc_i16)t : (gc_i16)0;
        if (!fits || gcn_group_move(box[k], t) != a.move[k]) ok = false;
    }
    r->charge = a.charge;
    r->gid = a.gid;
    r->owner_rank = a.owner;
    r->cell = a.cell;
    r->cls = a.cls;
    return ok;
}

/* A record's identity and move as the receiver reads them. */
__device__ __forceinline__ struct gcn_halo_identity
gcn_dist_rec_id(const gcn_coord_wire &r) { return r.id; }

__device__ __forceinline__ struct gcn_halo_identity
gcn_dist_rec_id(const gcn_reg_wire &r)
{
    struct gcn_halo_identity h;
    h.owner_rank = r.owner_rank;
    h.gid = r.gid;
    h.image = 0;
    for (int k = 0; k < 3; ++k)
        h.image |= ((gc_image)r.image[k] & 0xffffLL) << (16 * k);
    return h;
}

__device__ __forceinline__ gc_f64 gcn_dist_rec_move(const gcn_coord_wire &r,
                                                    int k, const gc_f64 *)
{
    return r.move[k];
}

__device__ __forceinline__ gc_f64 gcn_dist_rec_move(const gcn_reg_wire &r,
                                                    int k, const gc_f64 *box)
{
    return gcn_group_move(box[k], (gc_f64)r.turns[k]);
}

/* The box a kernel's records measure moves in, by value. */
struct gcn_dist_box { gc_f64 v[3]; };

/* Pack pass p: every slot in [0, nscan) that is owned or arrived in an
 * earlier pass and lies in an axis-p band, in slot order (dist_compact:
 * with pos null the kernel only flags each side's slots, then writes each
 * at its scanned position).  The record keeps the slot's
 * image and original owner and adds this rank's seam image; its cell is
 * the atom's home cell.  `outer_only` keeps the records that land outside
 * the neighbour's imported shell (the attach-time outer layer).  R is the
 * record: gcn_coord_wire, or gcn_reg_wire for a rebuild's registration,
 * whose moves are read in box B. */
template <class R>
__global__ void gcn_dist_pack_coord(const gc_f64 *coord,
                                    const gc_f64 *charge,
                                    const gc_i32 *cls,
                                    const gc_gid *gid,
                                    const gc_image *image,
                                    const gc_i32 *owner,
                                    const gc_i32 *cell_of,
                                    gc_i64 nowned, gc_i64 nscan, gc_i64 pitch,
                                    struct gcn_layout L, gc_i32 p,
                                    gc_i32 outer_only,
                                    struct gcn_dist_arrive A,
                                    gc_i32 partner_lower, gc_i32 partner_upper,
                                    gc_i64 epoch, gc_i64 slot_generation,
                                    gc_i64 plan_generation,
                                    R *lower, R *upper,
                                    gc_i64 cap,
                                    gc_i64 *bad, const gc_f64 *view_move,
                                    gc_i32 *route, gc_i32 *flag,
                                    const gc_i64 *pos, gc_i64 *count,
                                    struct gcn_dist_box B,
                                    const gcn_dist_dcount *dc = 0)
{
    const gcn_dist_arrive &AA = dc ? dc->arrive : A;
    const gc_i64 live = dc ? dc->resident : nscan;
    if (pos) gcn_dist_compact_counts(pos, nscan, count);
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < nscan;
         s += (gc_i64)gridDim.x * blockDim.x) {
        if (!pos) { flag[s] = 0; flag[nscan + s] = 0; }
        if (s >= live) continue;
        gc_i32 e[3];
        gcn_box_coords(L, cell_of[s], &e[0], &e[1], &e[2]);
        const bool own = s < nowned;
        gc_i32 side_in;
        if (!own && gcn_dist_slot_arrival(AA, L, s, nowned, e, &side_in) >= p)
            continue;
        const gc_i32 bands = gcn_dist_bands(L, e, p);
        if (!bands) continue;
        gc_i32 imk[3];
        for (int k = 0; k < 3; ++k)
            imk[k] = own ? 0 : gcn_dist_image_axis(image[s], k);
        gc_i32 home[3];
        for (int k = 0; k < 3; ++k)
            home[k] = e[k] - imk[k] * L.ncel[k];
        for (int side = 0; side < 2; ++side) {
            if (!((bands >> side) & 1)) continue;
            if (outer_only && !gcn_dist_lands_outer(L, e, p, side)) continue;
            if (!pos) { flag[side * nscan + s] = 1; continue; }
            const gc_i64 at = pos[side * nscan + s] - (side ? pos[nscan] : 0);
            if (at >= cap) { atomicAdd((unsigned long long *)bad, 1ull); continue; }
            struct gcn_dist_put_args a;
            a.sender = L.rank;
            a.edge = side == 0 ? partner_lower : partner_upper;
            a.epoch = epoch;
            a.slot_generation = slot_generation;
            a.plan_generation = plan_generation;
            a.owner = own ? L.rank : owner[s];
            a.gid = gid[s];
            {
                const gc_i32 sh = gcn_dist_image_axis(
                    gcn_dist_sender_image(L, p, side), p);
                gc_image im = 0;
                for (int k = 0; k < 3; ++k)
                    im |= ((gc_image)(gc_i16)(imk[k] + (k == p ? sh : 0))
                           & 0xffffLL) << (16 * k);
                a.image = im;
            }
            for (int k = 0; k < 3; ++k) {
                a.coord[k] = coord[(gc_i64)k * pitch + s];
                a.move[k] = view_move ? view_move[3 * s + k] : 0.0;
            }
            a.charge = charge[s]; a.cls = cls[s];
            a.cell = (home[2] * L.ncel[1] + home[1]) * L.ncel[0] + home[0];
            if (!gcn_dist_put(side == 0 ? lower + at : upper + at, a, B.v))
                atomicAdd((unsigned long long *)bad, 1ull);
            if (route) route[side * cap + at] = (gc_i32)s;
        }
    }
}

__global__ void gcn_dist_unpack_coord(const gcn_coord_wire *rec, gc_i64 n,
                                      gc_i32 expected_edge, gc_i32 expected_peer,
                                      gc_i64 epoch, gc_i64 slot_generation,
                                      gc_i64 plan_generation,
                                      const gc_gid *gkey,
                                      const gc_image *gimage,
                                      const gc_i32 *gedge,
                                      const gc_i32 *gval, gc_i64 gcap,
                                      const gc_image *image,
                                      const gc_i32 *owner,
                                      gc_i32 *visit,
                                      gc_f64 *coord, gc_f64 *force_coord,
                                      gc_f64 *charge, gc_i32 *cls,
                                      gc_i64 pitch, gc_i64 *bad,
                                      gc_f64 *view_move, gc_i32 *land
                                      )
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gcn_halo_identity &h = rec[i].id;
        const gc_i32 why = gcn_dist_meta_why(h, expected_edge, expected_peer,
                                             epoch, slot_generation,
                                             plan_generation);
        if (why != GCN_DIST_WHY_OK) {
            atomicAdd((unsigned long long *)bad, 1ull);
            continue;
        }
        const gc_i32 s = gcn_dist_find_ghost(
            h.gid, h.image, expected_edge,   /* the record's, once checked */
            gkey, gimage, gedge, gval, gcap);
        if (s < 0 || image[s] != h.image || owner[s] != h.owner_rank) {
            atomicAdd((unsigned long long *)bad, 1ull);
            continue;
        }
        if (atomicCAS(&visit[s], 0, 1) != 0) {
            atomicAdd((unsigned long long *)bad, 1ull);
            continue;
        }
        /* The owner's raw coordinates go to the boundary cell as in GENESIS
         * communicate_coor; the cell-pair move applies the periodic offset
         * later. */
        for (int k = 0; k < 3; ++k) {
            coord[(gc_i64)k * pitch + s] = rec[i].coord[k];
            force_coord[(gc_i64)k * pitch + s] = rec[i].coord[k] + rec[i].move[k];
            view_move[3 * s + k] = rec[i].move[k];
        }
        charge[s] = rec[i].charge; cls[s] = rec[i].cls;
        land[i] = s;
    }
}

/* Each owned slot's view move is its group's, as gcn_kern_force_coord adds
 * it. */
__global__ void gcn_dist_view_move_owned(const gc_i64 *goff, const gc_f64 *gmove,
                                         gc_i64 ng, gc_f64 *view_move)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ng;
         g += (gc_i64)gridDim.x * blockDim.x)
        for (gc_i64 s = goff[g]; s < goff[g + 1]; ++s)
            for (int k = 0; k < 3; ++k)
                view_move[3 * s + k] = gmove[3 * g + k];
}

/* Later steps of a generation: n[0], n[1] entries of the two sides. Forward
 * sends each routed slot's coordinates; the return sends each ghost's total.
 * W is the wire's word: gc_f64, or float under nonbond_precision = MIXED
 * (dist_slim_word). */
template <typename W>
__global__ void gcn_dist_pack_coord_slim(const gc_i32 *route, gc_i64 cap,
                                         gc_i64 n0, gc_i64 n1,
                                         const gc_f64 *coord, gc_i64 pitch,
                                         W *lower, W *upper)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n0 + n1;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const int side = i >= n0;
        const gc_i64 j = side ? i - n0 : i;
        const gc_i64 s = route[side * cap + j];
        W *r = (side ? upper : lower) + 3 * j;
        for (int k = 0; k < 3; ++k) r[k] = (W)coord[(gc_i64)k * pitch + s];
    }
}

template <typename W>
__global__ void gcn_dist_unpack_coord_slim(const gc_i32 *land, gc_i64 cap,
                                           gc_i64 n0, gc_i64 n1,
                                           const W *lower,
                                           const W *upper,
                                           const gc_f64 *view_move,
                                           gc_f64 *coord, gc_f64 *force_coord,
                                           gc_i64 pitch)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n0 + n1;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const int side = i >= n0;
        const gc_i64 j = side ? i - n0 : i;
        const gc_i64 s = land[side * cap + j];
        const W *r = (side ? upper : lower) + 3 * j;
        for (int k = 0; k < 3; ++k) {
            const gc_f64 c = (gc_f64)r[k];
            coord[(gc_i64)k * pitch + s] = c;
            force_coord[(gc_i64)k * pitch + s] = c + view_move[3 * s + k];
        }
    }
}

/* The mixed mode's slim return (native_dist_mixed_words): the ghost's two
 * fixed-point words summed as integers and rounded once to the FP32 wire; it
 * is added to the owner's word as add1 adds a term. A force-unit overflow
 * reads as NaN, which the landing refuses. */
__global__ void gcn_dist_pack_force_fx32(const gc_i32 *route, gc_i64 cap,
                                         gc_i64 n0, gc_i64 n1,
                                         const unsigned long long *frx,
                                         const unsigned long long *fbx,
                                         const unsigned int *overflow,
                                         gc_i64 pitch, float *lower,
                                         float *upper)
{
    const unsigned int ovf = *overflow;
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n0 + n1;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const int side = i >= n0;
        const gc_i64 j = side ? i - n0 : i;
        const gc_i64 s = route[side * cap + j];
        float *r = (side ? upper : lower) + 3 * j;
        for (int k = 0; k < 3; ++k) {
            const gc_i64 at = (gc_i64)k * pitch + s;
            r[k] = (float)gcn_fx::value1_of<gcn_fx::kForce1>(frx[at] + fbx[at],
                                                             ovf);
        }
    }
}

__global__ void gcn_dist_unpack_force_fx32(const gc_i32 *land, gc_i64 cap,
                                           gc_i64 n0, gc_i64 n1,
                                           const float *lower,
                                           const float *upper,
                                           unsigned long long *fbx,
                                           gc_i64 pitch, gc_i64 *bad)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n0 + n1;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const int side = i >= n0;
        const gc_i64 j = side ? i - n0 : i;
        const gc_i64 s = land[side * cap + j];
        const float *r = (side ? upper : lower) + 3 * j;
        for (int k = 0; k < 3; ++k) {
            const float t = r[k] * (float)(1ull << gcn_fx::kForce1);
            if (!(fabsf(t) < 18014398509481984.0f)) {      /* 2^54 */
                atomicAdd((unsigned long long *)bad, 1ull);
                continue;
            }
            const long long q = __float2ll_rn(t);
            if (q != 0)
                atomicAdd(&fbx[(gc_i64)k * pitch + s], (unsigned long long)q);
        }
    }
}

/* The exact return's slim pass (dist_exact_return): the ghost's two fixed-
 * point words summed (gcn_force_wire::word) and added as integers where they
 * land. */
__global__ void gcn_dist_pack_force_fx(const gc_i32 *route, gc_i64 cap,
                                       gc_i64 n0, gc_i64 n1,
                                       const unsigned long long *frx,
                                       const unsigned long long *fbx,
                                       gc_i64 pitch, unsigned long long *lower,
                                       unsigned long long *upper)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n0 + n1;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const int side = i >= n0;
        const gc_i64 j = side ? i - n0 : i;
        const gc_i64 s = route[side * cap + j];
        unsigned long long *r = (side ? upper : lower) + 3 * j;
        for (int k = 0; k < 3; ++k) {
            const gc_i64 at = (gc_i64)k * pitch + s;
            r[k] = frx[at] + fbx[at];
        }
    }
}

__global__ void gcn_dist_unpack_force_fx(const gc_i32 *land, gc_i64 cap,
                                         gc_i64 n0, gc_i64 n1,
                                         const unsigned long long *lower,
                                         const unsigned long long *upper,
                                         unsigned long long *fbx, gc_i64 pitch)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n0 + n1;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const int side = i >= n0;
        const gc_i64 j = side ? i - n0 : i;
        const gc_i64 s = land[side * cap + j];
        const unsigned long long *r = (side ? upper : lower) + 3 * j;
        for (int k = 0; k < 3; ++k)
            atomicAdd(&fbx[(gc_i64)k * pitch + s], r[k]);
    }
}

/* The forward after a refresh, which sends nothing: each ghost slot in
 * [first, first + n) takes its coordinates in the slim word W and forms its
 * force view as gcn_dist_unpack_coord_slim would. */
template <typename W>
__global__ void gcn_dist_round_ghosts(gc_i64 first, gc_i64 n,
                                      const gc_f64 *view_move,
                                      gc_f64 *coord, gc_f64 *force_coord,
                                      gc_i64 pitch)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 s = first + i;
        for (int k = 0; k < 3; ++k) {
            const gc_i64 at = (gc_i64)k * pitch + s;
            const gc_f64 c = (gc_f64)(W)coord[at];
            coord[at] = c;
            force_coord[at] = c + view_move[3 * s + k];
        }
    }
}

/* A fused pass opens its epoch (gcx_seq_open), then waits until both
 * edges have acknowledged the slot's previous use. */
__global__ void gcn_dist_open(unsigned long long *a, unsigned long long *b,
                              gcx_seq *q, unsigned long long epoch,
                              unsigned long long prev,
                              unsigned long long stride)
{
    __shared__ unsigned long long wait_for;
    if (threadIdx.x == 0) wait_for = gcx_seq_open(q, epoch, prev, stride).prev;
    __syncthreads();
    gcx_sig_wait(threadIdx.x ? b : a, wait_for);
}

/* A fused pass's two signal words (one per edge): wait until both reach
 * the open epoch, or set both to it once this stream's earlier writes are
 * visible. */
__global__ void gcn_dist_sig(unsigned long long *a, unsigned long long *b,
                             const gcx_seq *q, int set)
{
    unsigned long long *f = threadIdx.x ? b : a;
    const unsigned long long v = q->epoch;
    if (set) { __threadfence_system(); *(volatile unsigned long long *)f = v; }
    else gcx_sig_wait(f, v);
}

/* The return's passes in one launch each: every pass's open
 * (gcn_dist_open), or every pass's release, two threads per pass. */
struct gcn_dist_seq_args {
    unsigned long long *a[3], *b[3];
    gcx_seq *q[3];
    unsigned long long epoch[3], prev[3], stride[3];
};

__global__ void gcn_dist_open_many(gcn_dist_seq_args g, int n)
{
    __shared__ unsigned long long wait_for[3];
    const int k = threadIdx.x >> 1;
    if (k < n && !(threadIdx.x & 1))
        wait_for[k] = gcx_seq_open(g.q[k], g.epoch[k], g.prev[k],
                                   g.stride[k]).prev;
    __syncthreads();
    if (k < n) gcx_sig_wait((threadIdx.x & 1) ? g.b[k] : g.a[k], wait_for[k]);
}

__global__ void gcn_dist_release_many(gcn_dist_seq_args g, int n)
{
    const int k = threadIdx.x >> 1;
    if (k >= n) return;
    unsigned long long *f = (threadIdx.x & 1) ? g.b[k] : g.a[k];
    const unsigned long long v = g.q[k]->epoch;
    __threadfence_system();
    *(volatile unsigned long long *)f = v;
}

__global__ void gcn_dist_verify_coord_visits(const gc_i32 *visit,
                                                gc_i64 first, gc_i64 n,
                                                gc_i64 *bad
                                                )
{
    for (gc_i64 q = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; q < n;
         q += (gc_i64)gridDim.x * blockDim.x) {
        if (visit[first + q] != 1) {
            atomicAdd((unsigned long long *)bad, 1ull);
        }
    }
}

/* Registration unpack: writes the accepted record into the slot the arrival
 * order assigns, with no ghost lookup, so a band whose identity changed is
 * published without a host round trip. The slot's cell is the record's home
 * cell moved by its image and must belong to this pass and edge. R is the
 * record, moves read in box B. */
template <class R>
__global__ void gcn_dist_register_coord(
    const R *rec, gc_i64 n, gc_i64 first, gc_i32 p,
    gc_i32 recv_side, gc_i32 expected_edge, gc_i32 expected_peer,
    gc_i64 epoch, gc_i64 slot_generation, gc_i64 plan_generation,
    struct gcn_layout L,
    gc_gid *gid_out, gc_image *image_out, gc_i32 *owner_out,
    gc_i32 *cell_out, gc_f64 *coord, gc_f64 *force_coord,
    gc_f64 *charge, gc_i32 *cls, gc_i64 pitch, gc_i64 resident,
    gc_i64 *bad, gc_f64 *view_move, gc_i32 *land,
    struct gcn_dist_box B, const gcn_dist_dcount *dc = 0)
{
    if (dc) {
        const gc_i64 m = dc->arrive.n[p][recv_side];
        first = dc->arrive.first[p][recv_side];
        if (m < n) n = m;
    }
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gcn_halo_identity h = gcn_dist_rec_id(rec[i]);
        const gc_i64 s = first + i;
        const gc_i32 g = rec[i].cell;
        if (gcn_dist_meta_why(h, expected_edge, expected_peer, epoch,
                              slot_generation, plan_generation) != GCN_DIST_WHY_OK ||
            s < 0 || s >= resident || g < 0 ||
            (gc_i64)g >= (gc_i64)L.ncel[0] * L.ncel[1] * L.ncel[2]) {
            atomicAdd((unsigned long long *)bad, 1ull);
            continue;
        }
        const gc_i32 home[3] = { g % L.ncel[0], (g / L.ncel[0]) % L.ncel[1],
                                 g / (L.ncel[0] * L.ncel[1]) };
        gc_i32 e[3];
        bool ok = true;
        for (int k = 0; k < 3; ++k) {
            const gc_i32 im = gcn_dist_image_axis(h.image, k);
            if (L.halo[k] == 0 && im != 0) ok = false;
            e[k] = home[k] + im * L.ncel[k];
        }
        const gc_i32 c = ok ? gcn_box_index(L, e[0], e[1], e[2]) : -1;
        if (c >= 0) gcn_box_coords(L, c, &e[0], &e[1], &e[2]);
        const gc_i32 lab = c >= 0 ? gcn_dist_arrival(L, e) : -1;
        const bool normal =
            lab == p && (e[p] < L.start[p] ? 0 : 1) == recv_side &&
            h.owner_rank == gcn_rank_of_cell(L, home[0], home[1], home[2]);
        if (c < 0 || !normal) {
            atomicAdd((unsigned long long *)bad, 1ull);
            continue;
        }
        gid_out[s]   = h.gid;
        image_out[s] = h.image;
        owner_out[s] = h.owner_rank;
        cell_out[s]  = c;
        charge[s]    = rec[i].charge;
        cls[s]       = rec[i].cls;
        gc_f64 mv[3];
        for (int k = 0; k < 3; ++k) {
            mv[k] = gcn_dist_rec_move(rec[i], k, B.v);
            coord[(gc_i64)k * pitch + s]       = rec[i].coord[k];
            force_coord[(gc_i64)k * pitch + s] = rec[i].coord[k] + mv[k];
        }
        if (view_move)
            for (int k = 0; k < 3; ++k) view_move[3 * s + k] = mv[k];
        if (land) land[i] = (gc_i32)s;
    }
}

/* The return routes of a generation: pass p sends back, on the side each
 * arrived from, the ghosts that arrived in it, in arrival order, so each
 * lands on the slot the sender's forward route names at the same position. */
__global__ void gcn_dist_route_arrivals(gc_i32 *route, gc_i64 first, gc_i64 n)
{
    for (gc_i64 j = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += (gc_i64)gridDim.x * blockDim.x)
        route[j] = (gc_i32)(first + j);
}

/* A device-counted refresh (dist_refresh_dev). Each side sends a fixed number
 * of records, cap[e], and the count as one word after them. A count over the
 * capacity or a pack refusal marks the refresh overflowed; the records are
 * still sent and the ranks agree on the redo after the passes. */
__global__ void gcn_dist_dc_init(gcn_dist_dcount *dc, gc_i64 owned)
{
    gcn_dist_dcount z;
    memset(&z, 0, sizeof z);
    z.resident = owned;
    z.arrive.valid = 1;
    *dc = z;
}

__global__ void gcn_dist_dc_sent(gcn_dist_dcount *dc, const gc_i64 *count,
                                 const gc_i64 *bad, char *lower, char *upper,
                                 gc_i64 cap0, gc_i64 cap1, gc_i64 rec, gc_i32 p)
{
    for (int e = 0; e < 2; ++e) {
        const gc_i64 n = count[e], c = e ? cap1 : cap0;
        dc->sent[p][e] = n;
        *(gc_i64 *)((e ? upper : lower) + c * rec) = n;
        if (n > c) dc->over += 1;
    }
    if (*bad) dc->over += 1;
}

/* The arrivals of pass p: each side's count from after its records, and
 * the slots they take, lower side first. */
__global__ void gcn_dist_dc_arrive(gcn_dist_dcount *dc, const char *lower,
                                   const char *upper, gc_i64 cap0,
                                   gc_i64 cap1, gc_i64 rec, gc_i32 p)
{
    for (int e = 0; e < 2; ++e) {
        const gc_i64 c = e ? cap1 : cap0;
        gc_i64 n = *(const gc_i64 *)((e ? upper : lower) + c * rec);
        if (n < 0 || n > c) { dc->over += 1; n = 0; }
        dc->arrive.first[p][e] = dc->resident;
        dc->arrive.n[p][e] = n;
        dc->resident += n;
    }
}

/* Return pass p: every ghost that arrived in pass p goes back to the edge
 * it came from, carrying what it accumulated -- its own partial forces and
 * those relayed into it by the later passes' returns. */
__global__ void gcn_dist_pack_force(const gc_gid *gid,
                                    const gc_image *image,
                                    const gc_i32 *owner,
                                    const gc_i32 *cell_of,
                                    const gc_f64 *fr, const gc_f64 *fb,
                                    const unsigned long long *frx,
                                    const unsigned long long *fbx,
                                    gc_i64 first, gc_i64 nghost, gc_i64 pitch,
                                    struct gcn_layout L,
                                    struct gcn_dist_arrive A, gc_i32 p,
                                    gc_i32 partner_lower, gc_i32 partner_upper,
                                    gc_i64 epoch, gc_i64 slot_generation,
                                    gc_i64 plan_generation,
                                    gcn_force_wire *lower,
                                    gcn_force_wire *upper,
                                    gc_i64 cap, gc_i64 *bad,
                                    gc_i32 *route, gc_i32 *flag,
                                    const gc_i64 *pos, gc_i64 *count)
{
    if (pos) gcn_dist_compact_counts(pos, nghost, count);
    for (gc_i64 q = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; q < nghost;
         q += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 s = first + q;
        if (!pos) { flag[q] = 0; flag[nghost + q] = 0; }
        gc_i32 e[3], side;
        gcn_box_coords(L, cell_of[s], &e[0], &e[1], &e[2]);
        const gc_i32 a = gcn_dist_slot_arrival(A, L, s, first, e, &side);
        if (a < 0) { atomicAdd((unsigned long long *)bad, 1ull); continue; }
        if (a != p) continue;
        if (!pos) { flag[side * nghost + q] = 1; continue; }
        const gc_i64 at = pos[side * nghost + q] - (side ? pos[nghost] : 0);
        if (at >= cap) { atomicAdd((unsigned long long *)bad, 1ull); continue; }
        gcn_force_wire *r = side == 0 ? lower + at : upper + at;
        r->id.owner_rank = owner[s];
        r->id.gid = gid[s];
        r->id.image = image[s];
        for (int k = 0; k < 3; ++k) {
            const gc_i64 f = (gc_i64)k * pitch + s;
            if (frx) r->word[k] = frx[f] + fbx[f];
            else     r->total[k] = fr[f] + fb[f];
        }
        route[side * cap + at] = (gc_i32)s;
    }
}

/* A returned record names the slot this rank sent on that edge: its image
 * less this rank's seam image `shift`; an owned atom when this rank is the
 * owner, else a relayed ghost. */
__global__ void gcn_dist_unpack_force(const gcn_force_wire *rec, gc_i64 n,
                                      gc_i32 expected_edge, gc_i32 expected_peer,
                                      gc_i32 p, gc_i32 side, gc_image shift,
                                      gc_i64 epoch, gc_i64 slot_generation,
                                      gc_i64 plan_generation,
                                      struct gcn_layout L,
                                      const gc_gid *okey, const gc_i32 *oval,
                                      gc_i64 ocap,
                                      const gc_gid *gkey, const gc_image *gimage,
                                      const gc_i32 *gedge, const gc_i32 *gval,
                                      gc_i64 gcap, const gc_i32 *owner,
                                      gc_i32 *visit, gc_f64 *fb,
                                      unsigned long long *fbx,
                                      gc_i64 pitch, gc_i64 *bad, gc_i32 *land)
{
    const gc_image field = (gc_image)0xffffLL << (16 * p);
    const gc_i32 sh = gcn_dist_image_axis(shift, p);
    const gc_i32 bit = 1 << (2 * p + side);
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gcn_halo_identity &h = rec[i].id;
        if (gcn_dist_meta_why(h, expected_edge, expected_peer, epoch,
                              slot_generation, plan_generation) !=
            GCN_DIST_WHY_OK) {
            atomicAdd((unsigned long long *)bad, 1ull); continue;
        }
        const gc_image im = (h.image & ~field) |
            (((gc_image)(gc_i16)(gcn_dist_image_axis(h.image, p) - sh)
              & 0xffffLL) << (16 * p));
        gc_i32 s = -1;
        if (h.owner_rank == L.rank) {
            s = gcn_dist_find_owner(h.gid, okey, oval, ocap);
            if (s >= 0 && im != 0) s = -1;
        } else {
            s = gcn_dist_find_ghost(h.gid, im, -1, gkey, gimage, gedge, gval,
                                    gcap);
            if (s >= 0 && owner[s] != h.owner_rank) s = -1;
        }
        if (s < 0) { atomicAdd((unsigned long long *)bad, 1ull); continue; }
        if (atomicOr(&visit[s], bit) & bit) {
            atomicAdd((unsigned long long *)bad, 1ull); continue;
        }
        for (int k = 0; k < 3; ++k)
            if (fbx) atomicAdd(&fbx[(gc_i64)k * pitch + s], rec[i].word[k]);
            else     atomicAdd(&fb[(gc_i64)k * pitch + s], rec[i].total[k]);
        land[i] = s;
    }
}

/* Every slot the forward sent came back exactly once per pass and side. */
__global__ void gcn_dist_verify_force_visits(
    const gc_i32 *cell_of, const gc_i32 *visit, gc_i64 nowned,
    gc_i64 nresident, struct gcn_layout L, struct gcn_dist_arrive A,
    gc_i64 *bad)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
         s < nresident; s += (gc_i64)gridDim.x * blockDim.x) {
        gc_i32 e[3];
        gcn_box_coords(L, cell_of[s], &e[0], &e[1], &e[2]);
        gc_i32 side;
        const gc_i32 a = gcn_dist_slot_arrival(A, L, s, nowned, e, &side);
        gc_i32 want = 0;
        for (int p = a + 1; p < 3; ++p) want |= gcn_dist_bands(L, e, p) << (2 * p);
        if (visit[s] != want) atomicAdd((unsigned long long *)bad, 1ull);
    }
}

static gc_i64 dist_grid(gc_i64 n)
{
    gc_i64 g = (n + GCN_BLOCK - 1) / GCN_BLOCK;
    if (g < 1) g = 1;
    if (g > GCN_MAX_BLOCKS) g = GCN_MAX_BLOCKS;
    return g;
}

static gc_i64 pow2_cap(gc_i64 n)
{
    gc_i64 c = 1;
    gc_i64 want = n > 0 ? 2 * n : 2;
    while (c < want && c <= ((gc_i64)1 << 61)) c <<= 1;
    return c;
}

} /* anonymous namespace */

namespace gcn {

gc_status native_dist_vote(gc_context *ctx, gc_status local, const char *phase)
{
    if (!ctx) return GC_E_ARG;
    if (ctx->nproc <= 1) return local;
#ifdef HAVE_MPI_GENESIS
    int in = (int)local, out = 0;
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)ctx->comm);
    if (MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS)
        return GC_E_STATE;
    if (out != 0 && ctx->rank == 0)
        std::fprintf(stderr, "GPU_Core_Error> phase=%s collective_status=%d\n",
                     phase ? phase : "distributed_vote", out);
    return (gc_status)out;
#else
    (void)phase;
    return GC_E_UNSUPPORTED;
#endif
}

void native_dist_max_start(gc_context *ctx, gc_i64 local,
                           struct gcn_dist_max_req *r)
{
    r->in = (long long)local;
    r->out = (long long)local;
    r->active = 0;
    if (!ctx || ctx->nproc <= 1) return;
#ifdef HAVE_MPI_GENESIS
    static_assert(sizeof(MPI_Request) <= sizeof(r->req), "request storage");
    MPI_Request q;
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)ctx->comm);
    if (MPI_Iallreduce(&r->in, &r->out, 1, MPI_LONG_LONG, MPI_MAX, comm, &q) !=
        MPI_SUCCESS) {
        r->out = 1;
        return;
    }
    std::memcpy(r->req, &q, sizeof(q));
    r->active = 1;
#endif
}

gcx_wsum *native_dist_wsum(gc_context *ctx)
{
    const gcn_dist_state *x = ctx && ctx->native ? ctx->native->dist : 0;
    return x ? x->wsum : 0;
}

gc_i64 native_dist_max_wait(struct gcn_dist_max_req *r)
{
#ifdef HAVE_MPI_GENESIS
    if (r->active) {
        MPI_Request q;
        std::memcpy(&q, r->req, sizeof(q));
        r->active = 0;
        if (MPI_Wait(&q, MPI_STATUS_IGNORE) != MPI_SUCCESS) return 1;
    }
#endif
    return (gc_i64)r->out;
}

gc_status native_dist_gather(gc_context *ctx, const std::vector<gc_i64> &mine,
                             std::vector<gc_i64> *all)
{
    if (!ctx || !all) return GC_E_ARG;
    if (ctx->nproc <= 1) { *all = mine; return GC_OK; }
#ifdef HAVE_MPI_GENESIS
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)ctx->comm);
    int n = (int)mine.size();
    if ((size_t)n != mine.size()) return GC_E_CAPACITY;
    std::vector<int> count((size_t)ctx->nproc, 0), displ((size_t)ctx->nproc, 0);
    if (MPI_Allgather(&n, 1, MPI_INT, &count[0], 1, MPI_INT, comm) != MPI_SUCCESS)
        return GC_E_STATE;
    gc_i64 total = 0;
    for (int r = 0; r < ctx->nproc; ++r) {
        displ[(size_t)r] = (int)total;
        total += count[(size_t)r];
        if (total > (gc_i64)INT_MAX) return GC_E_CAPACITY;
    }
    all->assign((size_t)total, 0);
    if (total > 0 &&
        MPI_Allgatherv(mine.empty() ? 0 : (void *)&mine[0], n, MPI_LONG_LONG,
                       &(*all)[0], &count[0], &displ[0], MPI_LONG_LONG,
                       comm) != MPI_SUCCESS)
        return GC_E_STATE;
    return GC_OK;
#else
    return GC_E_UNSUPPORTED;
#endif
}

/* Status check of the per-step halo and the rebuild's stages. The checked
 * build votes collectively after every stage; the product build avoids whole-
 * communicator collectives: a failing stage prints its phase and rank and
 * returns the status, and every caller ends the run with gpu_core_abort. */
gc_status native_dist_step_vote(gc_context *ctx, gc_status local,
                                const char *phase)
{
    if (local != GC_OK)
        std::fprintf(stderr, "GPU_Core_Error> phase=%s rank=%d status=%d\n",
                     phase ? phase : "halo", (int)ctx->rank, (int)local);
    return local;
}

/* Admit the second face layer before resident arrays are allocated, and count
 * it. GENESIS imports a one-cell shell around the owned cells; the native
 * stencil needs two. The count runs the halo's schedule on atom counts per
 * box cell, with neighbour messages only. */
gc_status native_dist_prepare_outer(gc_context *ctx)
{
    if (!ctx) return GC_E_ARG;
    std::memset(ctx->native_outer_send, 0, sizeof(ctx->native_outer_send));
    std::memset(ctx->native_outer_recv, 0, sizeof(ctx->native_outer_recv));
    if (ctx->nproc == 1) return GC_OK;

    struct gcn_layout L;
    const char *why = "eligible";
    gc_status st = layout_native(ctx, &L, &why);
    if (st == GC_OK &&
        (ctx->num_owned < 0 || ctx->num_ghost < 0 || ctx->ncell < 0 ||
         ctx->atom_cell.size() != (size_t)ctx->num_owned ||
         ctx->ghost_cell.size() < (size_t)ctx->num_ghost ||
         ctx->cell_gx.size() < (size_t)ctx->ncell ||
         ctx->cell_gy.size() < (size_t)ctx->ncell ||
         ctx->cell_gz.size() < (size_t)ctx->ncell)) {
        st = GC_E_UNSUPPORTED; why = "the imported cell tables are incomplete";
    }
    for (int k = 0; k < 3 && st == GC_OK; ++k) {
        if (L.nd[k] == 1) continue;
        /* Exactly the second face: a list that needs one face takes the
         * stock path, a wider one needs more layers than are admitted. */
        const double need = ctx->geo.pairlistdist / ctx->geo.cell_size[k];
        if (L.halo[k] != 2 || !isfinite(need) || need <= 1.0 + 1.0e-12 ||
            need > 2.0 + 1.0e-12) {
            st = GC_E_UNSUPPORTED;
            why = "a split axis needs other than two face cells";
        }
    }
    std::vector<gc_i64> cnt;
    if (st == GC_OK) cnt.assign((size_t)L.ncell_box, 0);
    for (gc_i64 i = 0; st == GC_OK && i < ctx->num_owned + ctx->num_ghost; ++i) {
        const bool own = i < ctx->num_owned;
        const gc_i32 hc = own ? ctx->atom_cell[(size_t)i]
                              : ctx->ghost_cell[(size_t)(i - ctx->num_owned)];
        if (hc < 0 || (gc_i64)hc >= (own ? ctx->ncell_local : ctx->ncell)) {
            st = GC_E_ENDPOINT; why = "an imported atom names no cell"; break;
        }
        const gc_i32 b = gcn_box_index(L, ctx->cell_gx[(size_t)hc] - 1,
                                       ctx->cell_gy[(size_t)hc] - 1,
                                       ctx->cell_gz[(size_t)hc] - 1);
        gc_i32 e[3] = {0, 0, 0};
        if (b >= 0) gcn_box_coords(L, b, &e[0], &e[1], &e[2]);
        if (b < 0 || gcn_box_owned(L, b) != own ||
            (!own && !gcn_dist_in_shell(L, e))) {
            st = GC_E_MISMATCH;
            why = "an imported cell is outside its owned span or ghost shell";
            break;
        }
        ++cnt[(size_t)b];
    }
    if (st != GC_OK) {
        std::fprintf(stderr, "GPU_Core_Decline> phase=dist_outer_count_admit "
                     "rank=%d status=%d\n", (int)ctx->rank, (int)st);
    }
    st = native_dist_vote(ctx, st, "dist_outer_count_admit");
    if (st != GC_OK) return st;

#ifdef HAVE_MPI_GENESIS
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)ctx->comm);
    gc_status agree = GC_OK;          /* a count the import contradicts */
    for (int p = 0; p < 3 && st == GC_OK; ++p) {
        if (L.nd[p] == 1) continue;
        /* The band: axis p over the stencil width, earlier axes over the
         * whole box (their ghosts relay), later axes over the owned span. */
        gc_i32 lo[3], n[3];
        for (int k = 0; k < 3; ++k) {
            lo[k] = k < p ? L.lo[k] : L.start[k];
            n[k]  = k < p ? L.dim[k] : (k > p ? L.len[k] : L.halo[p]);
        }
        const size_t m = (size_t)n[0] * n[1] * n[2];
        std::vector<gc_i64> out[2], in[2];
        for (int side = 0; side < 2; ++side) {
            out[side].resize(m); in[side].resize(m);
            for (size_t j = 0; j < m; ++j) {
                gc_i32 e[3] = { lo[0] + (gc_i32)(j % n[0]),
                                lo[1] + (gc_i32)(j / n[0] % n[1]),
                                lo[2] + (gc_i32)(j / n[0] / n[1]) };
                if (side == 1) e[p] += L.len[p] - L.halo[p];
                const gc_i64 v = cnt[(size_t)gcn_box_index(L, e[0], e[1], e[2])];
                out[side][j] = v;
                if (gcn_dist_lands_outer(L, e, p, side))
                    ctx->native_outer_send[p][side] += v;
            }
        }
        const gc_i32 nb[2] = {
            gcn_rank_shift(L, p == 0 ? -1 : 0, p == 1 ? -1 : 0, p == 2 ? -1 : 0),
            gcn_rank_shift(L, p == 0 ? +1 : 0, p == 1 ? +1 : 0, p == 2 ? +1 : 0) };
        for (int side = 0; side < 2 && st == GC_OK; ++side)
            if (MPI_Sendrecv(out[side].data(), (int)m, MPI_LONG_LONG, nb[side],
                             GCN_HALO_COUNT_TAG + side,
                             in[1 - side].data(), (int)m, MPI_LONG_LONG,
                             nb[1 - side], GCN_HALO_COUNT_TAG + side, comm,
                             MPI_STATUS_IGNORE) != MPI_SUCCESS)
                st = GC_E_STATE;
        for (int edge = 0; edge < 2 && st == GC_OK; ++edge) {
            for (size_t j = 0; j < m; ++j) {
                gc_i32 e[3] = { lo[0] + (gc_i32)(j % n[0]),
                                lo[1] + (gc_i32)(j / n[0] % n[1]),
                                lo[2] + (gc_i32)(j / n[0] / n[1]) };
                e[p] += edge == 0 ? -L.halo[p] : L.len[p];
                const gc_i32 b = gcn_box_index(L, e[0], e[1], e[2]);
                const gc_i64 v = in[edge][j];
                if (gcn_dist_in_shell(L, e)) {
                    if (cnt[(size_t)b] != v) agree = GC_E_MISMATCH;
                } else {
                    cnt[(size_t)b] = v;
                    ctx->native_outer_recv[p][edge] += v;
                }
            }
        }
    }
    if (st == GC_OK) st = agree;
    gc_i64 resident = ctx->num_owned + ctx->num_ghost;
    for (int k = 0; k < 6 && st == GC_OK; ++k) {
        const gc_i64 v = ctx->native_outer_recv[k / 2][k % 2];
        if (v < 0 || !add_checked(resident, v, &resident) ||
            resident > GC_MAX_LOCAL_INDEX)
            st = GC_E_CAPACITY;
    }
#else
    st = GC_E_UNSUPPORTED;
#endif
    st = native_dist_vote(ctx, st, "dist_outer_count_agree");
    return st;
}

static gc_status dist_realloc(void **p, gc_i64 bytes)
{
    void *n = 0;
    const gc_status st = dev_calloc(&n, bytes);
    if (st != GC_OK) return st;
    dev_release(p);
    *p = n;
    return GC_OK;
}

/* Size the receipts and the identity maps for the resident population. */
static gc_status gcn_dist_fit(gcn_dist_state *x, const gcn_device *d)
{
    gc_status st = GC_OK;
    if (x->visit_capacity < d->num_resident) {
        const gc_i64 bytes = (d->num_resident + 64) * (gc_i64)sizeof(gc_i32);
        st = dist_realloc((void **)&x->coord_visit, bytes);
        if (st == GC_OK) st = dist_realloc((void **)&x->force_visit, bytes);
        if (st == GC_OK)
            st = dist_realloc((void **)&x->view_move,
                              3 * (d->num_resident + 64) * (gc_i64)sizeof(gc_f64));
        if (st != GC_OK) return st;
        x->visit_capacity = d->num_resident + 64;
    }
    if (x->owner_cap == 0 || d->num_owned > x->owner_cap / 2) {
        const gc_i64 cap = pow2_cap(d->num_owned);
        st = dist_realloc((void **)&x->owner_key, cap * (gc_i64)sizeof(gc_gid));
        if (st == GC_OK)
            st = dist_realloc((void **)&x->owner_val, cap * (gc_i64)sizeof(gc_i32));
        if (st != GC_OK) return st;
        x->owner_cap = cap;
    }
    const gc_i64 ng = d->num_resident - d->num_owned;
    if (x->ghost_cap == 0 || ng > x->ghost_cap / 2) {
        const gc_i64 cap = pow2_cap(ng);
        st = dist_realloc((void **)&x->ghost_key, cap * (gc_i64)sizeof(gc_gid));
        if (st == GC_OK)
            st = dist_realloc((void **)&x->ghost_image, cap * (gc_i64)sizeof(gc_image));
        if (st == GC_OK)
            st = dist_realloc((void **)&x->ghost_edge, cap * (gc_i64)sizeof(gc_i32));
        if (st == GC_OK)
            st = dist_realloc((void **)&x->ghost_val, cap * (gc_i64)sizeof(gc_i32));
        if (st != GC_OK) return st;
        x->ghost_cap = cap;
    }
    return GC_OK;
}

/* The per-slot arrays of gcn_dist_fit for `need` residents while a refresh
 * registers them; the view moves of the first `keep` slots are kept. */
static gc_status gcn_dist_fit_moves(gcn_dist_state *x, gc_i64 need,
                                    gc_i64 keep, cudaStream_t s)
{
    if (need <= x->visit_capacity) return GC_OK;
    const gc_i64 cap = need + need / 4 + 64;
    const gc_i64 bytes = cap * (gc_i64)sizeof(gc_i32);
    gc_f64 *mv = 0;
    gc_status st = dev_calloc((void **)&mv, 3 * cap * (gc_i64)sizeof(gc_f64));
    if (st == GC_OK && keep > 0 &&
        cudaMemcpyAsync(mv, x->view_move, 3 * keep * sizeof(gc_f64),
                        cudaMemcpyDeviceToDevice, s) != cudaSuccess)
        st = GC_E_DEVICE;
    if (st != GC_OK) { dev_release((void **)&mv); return st; }
    dev_release((void **)&x->view_move);
    x->view_move = mv;
    st = dist_realloc((void **)&x->coord_visit, bytes);
    if (st == GC_OK) st = dist_realloc((void **)&x->force_visit, bytes);
    if (st != GC_OK) return st;
    x->visit_capacity = cap;
    return GC_OK;
}

/* A pack's stable compaction over n scanned slots: `launch(flag, pos)`
 * launches the pack kernel, first with pos null to flag each side's slots,
 * then with the scanned positions; side totals land in pack_count. Records
 * are in slot order, so a pack is the same at every run. */
template <class Launch>
static gc_status dist_compact(gcn_device *d, gcn_dist_state *x, gc_i64 n,
                              cudaStream_t s, Launch launch)
{
    if (n <= 0) return GC_OK;
    if (2 * n > x->pack_cap) {
        const gc_i64 cap = 2 * n + n / 2 + 256;
        gc_status st = dist_realloc((void **)&x->pack_flag, cap * (gc_i64)sizeof(gc_i32));
        if (st == GC_OK)
            st = dist_realloc((void **)&x->pack_pos, cap * (gc_i64)sizeof(gc_i64));
        if (st != GC_OK) { x->pack_cap = 0; return st; }
        x->pack_cap = cap;
    }
    launch(x->pack_flag, (const gc_i64 *)0);
    /* one scan over both sides; the upper side's positions are offset by the
     * lower count (gcn_dist_compact_counts) */
    const gc_status st = native_scan(d, x->pack_flag, x->pack_pos, 2 * n,
                                     x->pack_count + 1);
    if (st != GC_OK) return st;
    launch((gc_i32 *)0, (const gc_i64 *)x->pack_pos);
    return GC_OK;
}

/* One pass of the halo on `axis`: `pack(stream)` launches the pack into the
 * send buffers, the transport exchanges with the two axis neighbours, and
 * `unpack(stream, records)` launches the consumers. `sent` and `got` are the
 * pass's record counts per edge. `collective` votes every stage over the
 * communicator (attach); the step and the rebuild use native_dist_step_vote.
 *
 * A full pass reads its pack counts back. A `slim` pass (a step after its
 * generation's first) knows them on both ends, sends three doubles per atom
 * and does not wait for the pack; a peer edge is sequenced on the device with
 * no host round trip. When both edges are device-sequenced the pass is fused:
 * the pack writes the peers' slots, the unpack reads this rank's, with signal
 * words on the stream between them.
 *
 * Four votes a pass, fused or not: pack and sizes (with `carry`, a caller's
 * status owed to the same vote), transport begin, its consume, unpack with
 * release. `rec_bytes` names another record on the coordinate channel;
 * `step_counts` makes the pass set the step's counts (the registration
 * record). */
/* The slim records' word: FP32 under nonbond_precision = MIXED (a ghost's
 * coordinate and returned force are rounded once on the wire and widened on
 * arrival), FP64 otherwise. */
static inline gcn_dist_box dist_box(const gcn_device *d)
{
    gcn_dist_box b;
    for (int k = 0; k < 3; ++k) b.v[k] = d->box[k];
    return b;
}

static inline gc_i64 dist_slim_word(const gcn_device *d)
{
    return d->tab.nonbond_precision == GC_NONBOND_MIXED
        ? (gc_i64)sizeof(float) : (gc_i64)sizeof(gc_f64);
}

/* The FP64 mode returns ghost forces as fixed-point words, so an owned atom's
 * force is the same bits however its terms are spread over the ranks. The
 * mixed mode keeps its wire. */
static inline int dist_exact_return(const gcn_device *d)
{
    return dist_slim_word(d) == (gc_i64)sizeof(gc_f64);
}

/* The mixed mode keeps its FP32 wire but packs it from the ghosts' fixed-
 * point words and adds what lands to the owners' words, so no double fold
 * precedes the return. */
int native_dist_mixed_words(const gcn_device *d)
{
    return d->layout.nproc > 1 &&
           dist_slim_word(d) == (gc_i64)sizeof(float);
}

/* The plan of pass `axis` of a direction. */
static gcx_plan *dist_plan(gcn_dist_state *x, gc_i32 reverse, gc_i32 axis)
{
    gcx_plan *plan = 0;
    gcx_halo_plan(x->halo, reverse ? 2 - axis : axis, reverse, &plan);
    return plan;
}

/* A fused pass's epoch and slot claim (gcx_move_claim), and its open's
 * arguments (entry k of `o`). */
static gc_status dist_claim(gcn_dist_state *x, gc_i32 axis, gc_i32 reverse,
                            cudaStream_t s, gc_i64 *epoch, gc_i32 *slot,
                            gcn_dist_seq_args *o, int k)
{
    const gcx_device_edge *g = x->dev[reverse][axis];
    gcx_move_open m = {0, 0, 0, 0};
    *epoch = ++x->sequence;
    const gc_status st = gcx_move_claim(dist_plan(x, reverse, axis), *epoch,
                                        (void *)s, &m);
    *slot = m.slot;
    o->a[k] = g[0].my_sig + 1;
    o->b[k] = g[1].my_sig + 1;
    o->q[k] = x->seq + 3 * reverse + axis;
    o->epoch[k] = (unsigned long long)*epoch;
    o->prev[k] = (unsigned long long)m.prev;
    o->stride[k] = m.stride;
    return st;
}

static void dist_open_all(gcn_device *d, gcn_dist_state *x, int reverse);
static gc_status dist_release_all(gcn_device *d, gcn_dist_state *x,
                                  int reverse, cudaStream_t s);

template <class Pack, class Unpack>
static gc_status dist_pass(gc_context *ctx, gc_i32 axis, gc_i32 reverse,
                           const char *tag, int collective, int slim,
                           gc_i64 sent[2], gc_i64 got[2],
                           Pack pack, Unpack unpack, gc_status carry = GC_OK,
                           gc_i64 rec_bytes = 0, int step_counts = 0)
{
    const int counts = !rec_bytes || step_counts;
    gcn_device *d = ctx->native;
    gcn_dist_state *x = d->dist;
    cudaStream_t s = d->stream;
    char ph[64];
    auto vote = [&](gc_status st, const char *stage) {
        std::snprintf(ph, sizeof ph, "%s_%s", tag, stage);
        return collective ? native_dist_vote(ctx, st, ph)
                          : native_dist_step_vote(ctx, st, ph);
    };
    const gc_i64 rec = rec_bytes ? rec_bytes
                     : slim ? 3 * dist_slim_word(d)
                     : reverse ? (gc_i64)sizeof(gcn_force_wire)
                               : (gc_i64)sizeof(gcn_coord_wire);
    const gc_i64 bytes = reverse ? x->force_capacity : x->coord_capacity;
    const gc_i64 max_records = counts ? x->capacity_records : bytes / rec;
    void *const *send = reverse ? x->force_send : x->coord_send;
    void *const *recv = reverse ? x->force_recv : x->coord_recv;
    gc_i64 bad = 0;
    sent[0] = sent[1] = got[0] = got[1] = 0;

    gc_status st = GC_OK;
    gc_i64 *known = x->step_sent[reverse][axis];
    gc_i64 *landed = x->step_got[reverse][axis];
    const int fused = slim && x->fused[reverse][axis] &&
                      (x->stores[reverse][axis] || d->ce_fused);
    const gcx_device_edge *g = x->dev[reverse][axis];
    gcx_seq *q = x->seq + 3 * reverse + axis;
    gc_i64 epoch = 0;
    gc_i32 slot = 0;
    if (slim) {
        sent[0] = known[0]; sent[1] = known[1];
        for (int e = 0; e < 2; ++e) {
            x->wire_out[e] = send[e];
            x->wire_in[e] = recv[e];
        }
        if (fused && x->opened[reverse][axis]) {
            epoch = x->open_epoch[reverse][axis];
            slot = x->open_slot[reverse][axis];
            x->opened[reverse][axis] = 0;
        } else if (fused) {
            gcn_dist_seq_args o;
            st = dist_claim(x, axis, reverse, s, &epoch, &slot, &o, 0);
            if (st == GC_OK)
                gcn_dist_open<<<1, 2, 0, s>>>(o.a[0], o.b[0], o.q[0], o.epoch[0],
                                              o.prev[0], o.stride[0]);
        }
        for (int e = 0; e < 2 && st == GC_OK && fused; ++e) {
            if (x->stores[reverse][axis])
                x->wire_out[e] = g[e].peer_slots + slot * g[e].send_capacity;
            x->wire_in[e] = g[e].my_slots + slot * g[e].recv_capacity;
        }
        if (st == GC_OK) st = pack(s);
        for (int e = 0; e < 2 && st == GC_OK && fused &&
                        !x->stores[reverse][axis]; ++e) {
            gc_i64 nb = 0;
            if (sent[e] < 0 || !mul_checked(sent[e], rec, &nb) ||
                nb > g[e].send_capacity || nb > bytes)
                st = GC_E_CAPACITY;
            else if (nb > 0 &&
                     cudaMemcpyAsync(g[e].peer_slots + slot * g[e].send_capacity,
                                     send[e], (size_t)nb, cudaMemcpyDefault, s) !=
                         cudaSuccess)
                st = GC_E_DEVICE;
        }
        if (st == GC_OK && fused)
            gcn_dist_sig<<<1, 2, 0, s>>>(g[0].peer_sig, g[1].peer_sig, q, 1);
        if (st == GC_OK && !fused && cudaEventRecord(x->producer, s) != cudaSuccess)
            st = GC_E_DEVICE;
    } else {
        if (cudaMemsetAsync(x->pack_count, 0, 2 * sizeof(gc_i64), s) != cudaSuccess ||
            cudaMemsetAsync(x->bad, 0, sizeof(gc_i64), s) != cudaSuccess)
            st = GC_E_DEVICE;
        if (st == GC_OK) st = pack(s);
        /* the counts and the verdict land in pinned words: both copies
         * are asynchronous and one synchronisation reads them */
        if (st == GC_OK &&
            (cudaEventRecord(x->producer, s) != cudaSuccess ||
             cudaMemcpyAsync(x->pack_pin, x->pack_count, 2 * sizeof(gc_i64),
                             cudaMemcpyDeviceToHost, s) != cudaSuccess ||
             cudaMemcpyAsync(x->pack_pin + 2, x->bad, sizeof(gc_i64),
                             cudaMemcpyDeviceToHost, s) != cudaSuccess))
            st = GC_E_DEVICE;
        if (st == GC_OK) {
            std::snprintf(ph, sizeof ph, "%s_pack", tag);
            st = host_run();
            if (st == GC_OK) st = dev_phase(s, ph);
        }
        if (st == GC_OK) {
            sent[0] = x->pack_pin[0];
            sent[1] = x->pack_pin[1];
            bad = x->pack_pin[2];
        }
        if (st == GC_OK && bad) st = GC_E_CAPACITY;
        if (counts) { known[0] = sent[0]; known[1] = sent[1]; }
    }
    if (st == GC_OK) st = carry;

    gcx_buffer sb[2], rb[2];
    gc_i64 send_bytes[2] = {0,0}, recv_bytes[2] = {0,0};
    for (int e = 0; e < 2; ++e) {
        if (st == GC_OK &&
            (sent[e] < 0 || sent[e] > max_records ||
             !mul_checked(sent[e], rec, &send_bytes[e]) || send_bytes[e] > bytes))
            st = GC_E_CAPACITY;
        sb[e].ptr = send[e]; sb[e].bytes = bytes;
        rb[e].ptr = recv[e]; rb[e].bytes = slim ? landed[e] * rec : bytes;
    }
    st = vote(st, "pack");
    if (st != GC_OK) return st;

    if (fused) {
        /* Whether a pass fuses depends on its own edges, so it can differ
         * between ranks while the votes span the communicator. A fused pass
         * has no transport begin or consume but casts their votes, so every
         * rank calls the same sequence of collectives. */
        st = vote(GC_OK, "begin");
        if (st == GC_OK) st = vote(GC_OK, "consume");
        if (st != GC_OK) return st;
        gcn_dist_sig<<<1, 2, 0, s>>>(g[0].my_sig, g[1].my_sig, q, 0);
        got[0] = landed[0]; got[1] = landed[1];
        st = unpack(s, (const gc_i64 *)got);
        if (st == GC_OK && !x->late_release[reverse])
            gcn_dist_sig<<<1, 2, 0, s>>>(g[0].peer_sig + 1, g[1].peer_sig + 1,
                                         q, 1);
        if (st == GC_OK && cudaGetLastError() != cudaSuccess) st = GC_E_DEVICE;
        return vote(st, "release");
    }

    gcx_op_desc od; std::memset(&od, 0, sizeof(od));
    od.epoch = ++x->sequence;
    od.op = reverse ? GCX_OP_HALO_FORCE : GCX_OP_HALO_COORD; od.variable = !slim;
    od.send = sb; od.recv = rb; od.send_bytes = send_bytes;
    od.send_records = sent; od.recv_bytes = recv_bytes;
    od.recv_records = got; od.producer_event = (void *)x->producer;
    gcx_token *tok = 0;
    st = reverse ? gcx_halo_reverse(x->halo, 2 - axis, &od, &tok)
                 : gcx_halo_forward(x->halo, axis, &od, &tok);
    st = vote(st, "begin");
    if (st != GC_OK) { if (tok) gcx_abort(tok); return st; }
    st = gcx_consume(tok, (void *)s);
    st = vote(st, "consume");
    if (st != GC_OK) { gcx_abort(tok); return st; }

    for (int e = 0; e < 2; ++e) {
        if (slim && recv_bytes[e] == landed[e] * rec) got[e] = landed[e];
        if (recv_bytes[e] < 0 || recv_bytes[e] > bytes || recv_bytes[e] % rec != 0 ||
            got[e] != recv_bytes[e] / rec || (slim && got[e] != landed[e]))
            st = GC_E_MISMATCH;
        if (!slim && counts) landed[e] = got[e];
    }
    /* Unpack only records this rank admitted; the token is released or
     * aborted here either way. */
    if (st == GC_OK) st = unpack(s, (const gc_i64 *)got);
    if (st == GC_OK && cudaEventRecord(x->consumer, s) != cudaSuccess)
        st = GC_E_DEVICE;
    if (st == GC_OK) st = gcx_release(tok, (void *)x->consumer);
    else gcx_abort(tok);
    return vote(st, "release");
}

/* The outer layer at attach: the passes carry the band atoms that land
 * outside the neighbour's imported shell, each registered into the next outer
 * slot after the imported ghosts; later passes relay earlier ones. Counts are
 * prepare_outer's. */
static gc_status gcn_dist_bootstrap_outer(gc_context *ctx)
{
    gcn_device *d = ctx->native;
    gcn_dist_state *x = d->dist;
    const gc_i64 first = d->num_owned + ctx->num_ghost;
    gc_i64 filled = 0, bad = 0;
    gc_status st = cudaMemsetAsync(x->bad + 1, 0, sizeof(gc_i64), d->stream)
                   == cudaSuccess ? GC_OK : GC_E_DEVICE;
    st = native_dist_vote(ctx, st, "dist_outer_clear");
    for (int p = 0; p < 3 && st == GC_OK; ++p) {
        if (d->layout.nd[p] == 1) continue;
        const gc_i64 nscan = first + filled;
        gc_i64 sent[2], got[2];
        st = dist_pass(ctx, p, 0, "dist_outer", 1, 0, sent, got,
            [&](cudaStream_t s) {
                return dist_compact(d, x, nscan, s,
                    [&](gc_i32 *flag, const gc_i64 *pos) {
                    gcn_dist_pack_coord<<<(unsigned)dist_grid(nscan), GCN_BLOCK, 0, s>>>(
                        d->coord, d->charge, d->cls, d->gid,
                        d->image, d->owner, d->cell_of, d->num_owned, nscan,
                        d->pitch, d->layout, p, 1, x->arrive,
                        x->coord.partner[p][0], x->coord.partner[p][1],
                        ctx->epoch, d->slot_generation, d->plan_generation,
                        (gcn_coord_wire *)x->coord_send[0],
                        (gcn_coord_wire *)x->coord_send[1],
                        x->capacity_records, x->bad, 0, 0, flag, pos, x->pack_count,
                        dist_box(d));
                });
            },
            [&](cudaStream_t s, const gc_i64 *n) -> gc_status {
                gc_i64 at = nscan;
                for (int e = 0; e < 2; ++e)
                    if (n[e] != ctx->native_outer_recv[p][e]) return GC_E_MISMATCH;
                for (int e = 0; e < 2; ++e) {
                    if (n[e] > 0)
                        gcn_dist_register_coord<<<(unsigned)dist_grid(n[e]),
                                                  GCN_BLOCK, 0, s>>>(
                            (const gcn_coord_wire *)x->coord_recv[e], n[e], at,
                            p, e, x->coord.key[p][e], x->coord.peer[p][e],
                            ctx->epoch, d->slot_generation, d->plan_generation,
                            d->layout, d->gid, d->image, d->owner, d->cell_of,
                            d->coord, d->force_coord, d->charge, d->cls,
                            d->pitch, d->num_resident, x->bad + 1, 0, 0,
                            dist_box(d));
                    at += n[e];
                }
                return GC_OK;
            });
        if (st != GC_OK) return st;
        if (sent[0] != ctx->native_outer_send[p][0] ||
            sent[1] != ctx->native_outer_send[p][1]) st = GC_E_MISMATCH;
        filled += got[0] + got[1];
        st = native_dist_vote(ctx, st, "dist_outer_counts");
    }
    if (st == GC_OK) st = dev_phase(d->stream, "dist_outer_unpack");
    if (st == GC_OK &&
        cudaMemcpy(&bad, x->bad + 1, sizeof(bad), cudaMemcpyDeviceToHost) != cudaSuccess)
        st = GC_E_DEVICE;
    if (st == GC_OK && (bad || first + filled != d->num_resident)) st = GC_E_ENDPOINT;
    return native_dist_vote(ctx, st, "dist_outer_publish");
}

gc_status native_dist_create(gc_context *ctx)
{
    if (!ctx || !ctx->native) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (ctx->nproc == 1) return GC_OK;
    const struct gcn_layout &L = d->layout;
    gc_status st = d->dist ? GC_E_STATE : GC_OK;
    for (int k = 0; k < 3; ++k)
        if (L.nd[k] > 1 && L.halo[k] != 2) st = GC_E_UNSUPPORTED;
    st = native_dist_vote(ctx, st, "dist_create_geometry");
    if (st != GC_OK) return st;

    gcn_dist_state *x = new gcn_dist_state;
    std::memset(x, 0, sizeof(*x));
    d->dist = x;
    /* A band never holds more records than its sender has resident slots;
     * both ends of an edge declare one capacity per axis, so the bound is the
     * largest resident count, agreed once at attach. */
    gc_i64 mine = d->num_resident > 0 ? d->num_resident : 1;
    x->capacity_records = mine;
#ifdef HAVE_MPI_GENESIS
    if (MPI_Allreduce(&mine, &x->capacity_records, 1, MPI_LONG_LONG, MPI_MAX,
                      MPI_Comm_f2c((MPI_Fint)ctx->comm)) != MPI_SUCCESS)
        st = GC_E_STATE;
#endif
    if (st == GC_OK &&
        (!mul_checked(x->capacity_records, (gc_i64)sizeof(gcn_coord_wire),
                      &x->coord_capacity) ||
         !mul_checked(x->capacity_records, (gc_i64)sizeof(gcn_force_wire),
                      &x->force_capacity)))
        st = GC_E_OVERFLOW;
    if (st == GC_OK) st = dev_calloc((void **)&x->pack_count, 2 * (gc_i64)sizeof(gc_i64));
    if (st == GC_OK &&
        cudaHostAlloc((void **)&x->pack_pin, 3 * sizeof(gc_i64),
                      cudaHostAllocDefault) != cudaSuccess) {
        x->pack_pin = 0;
        st = GC_E_NOMEM;
    }
    if (st == GC_OK) st = dev_calloc((void **)&x->bad, 3 * (gc_i64)sizeof(gc_i64));
    if (st == GC_OK) st = dev_calloc((void **)&x->seq, 6 * (gc_i64)sizeof(gcx_seq));
    for (int r = 0; r < 2; ++r)
        for (int p = 0; p < 3 && st == GC_OK; ++p) {
            if (L.nd[p] == 1) continue;
            const gc_i64 b = 2 * x->capacity_records * (gc_i64)sizeof(gc_i32);
            st = dev_calloc((void **)&x->route[r][p], b);
            if (st == GC_OK) st = dev_calloc((void **)&x->land[r][p], b);
        }
    x->step_gen[0][0] = x->step_gen[1][0] = -1;
    if (st == GC_OK) st = gcn_dist_fit(x, d);
    for (int e = 0; e < 2 && st == GC_OK; ++e) {
        st = dev_calloc(&x->coord_send[e], x->coord_capacity);
        if (st == GC_OK) st = dev_calloc(&x->coord_recv[e], x->coord_capacity);
        if (st == GC_OK) st = dev_calloc(&x->force_send[e], x->force_capacity);
        if (st == GC_OK) st = dev_calloc(&x->force_recv[e], x->force_capacity);
    }
    if (st == GC_OK &&
        (cudaEventCreateWithFlags(&x->producer, cudaEventDisableTiming) != cudaSuccess ||
         cudaEventCreateWithFlags(&x->consumer, cudaEventDisableTiming) != cudaSuccess))
        st = GC_E_DEVICE;
    st = native_dist_vote(ctx, st, "dist_create_alloc");
    if (st != GC_OK) return st;

    std::memset(&x->probe, 0, sizeof(x->probe));
    st = gcx_probe_run((gcx_comm)ctx->comm, &x->probe);
    if (st == GC_OK) x->probe_live = 1;
    st = native_dist_vote(ctx, st, "dist_create_probe");
    if (st != GC_OK) return st;

    gcx_halo_desc hd;
    std::memset(&hd, 0, sizeof(hd));
    hd.tag_base = 1024;
    hd.deadline_ms = GCX_DEADLINE_MS;
    for (int k = 0; k < 3; ++k) {
        hd.num_domain[k] = L.nd[k];
        hd.neighbour_lower[k] = gcn_rank_shift(L,
            k == 0 ? -1 : 0, k == 1 ? -1 : 0, k == 2 ? -1 : 0);
        hd.neighbour_upper[k] = gcn_rank_shift(L,
            k == 0 ? +1 : 0, k == 1 ? +1 : 0, k == 2 ? +1 : 0);
        hd.coord_capacity[k] = L.nd[k] > 1 ? x->coord_capacity : 0;
        hd.force_capacity[k] = L.nd[k] > 1 ? x->force_capacity : 0;
    }
    st = gcx_halo_create(&hd, (gcx_comm)ctx->comm, &x->halo);
    st = native_dist_vote(ctx, st, "dist_create_halo");
    if (st != GC_OK) return st;

    for (int p = 0; p < 3 && st == GC_OK; ++p) {
        if (L.nd[p] == 1) continue;
        gcx_plan *fwd = 0, *rev = 0;
        st = gcx_halo_plan(x->halo, p, 0, &fwd);
        if (st == GC_OK) st = gcx_halo_plan(x->halo, 2 - p, 1, &rev);
        for (int e = 0; e < 2 && st == GC_OK; ++e) {
            st = gcx_plan_edge_identity(fwd, e, &x->coord.key[p][e],
                                        &x->coord.partner[p][e],
                                        &x->coord.peer[p][e]);
            if (st == GC_OK)
                st = gcx_plan_edge_identity(rev, e, &x->force.key[p][e],
                                            &x->force.partner[p][e],
                                            &x->force.peer[p][e]);
        }
        for (int r = 0; r < 2 && st == GC_OK; ++r)
            st = gcx_move_edges(r ? rev : fwd, x->dev[r][p], &x->fused[r][p],
                                &x->stores[r][p]);
        for (int e = 0; e < 2 && st == GC_OK; ++e) {
            const int peer = e == 0 ? hd.neighbour_lower[p] : hd.neighbour_upper[p];
            if (x->coord.peer[p][e] != peer || x->force.peer[p][e] != peer ||
                x->coord.partner[p][e] != x->coord.key[p][1-e] ||
                x->force.partner[p][e] != x->force.key[p][1-e])
                st = GC_E_MISMATCH;
        }
    }
    st = native_dist_vote(ctx, st, "dist_create_edges");
    if (st != GC_OK) return st;
    /* The device-counted refresh needs every rank's forward edges device-
     * sequenced and a device sum; agreed once, here. */
    {
        gc_i64 no = 0;
        for (int p = 0; p < 3; ++p)
            for (int e = 0; e < 2 && L.nd[p] > 1; ++e)
                if (!x->dev[0][p][e].peer_slots) no = 1;
        st = gcx_wsum_create((gcx_comm)ctx->comm, 1, 0, &x->wsum);
        if (st == GC_OK && !x->wsum) no = 1;
        if (st == GC_OK && !no)
            st = dev_calloc((void **)&x->dc, (gc_i64)sizeof(gcn_dist_dcount));
        if (st == GC_OK && !no &&
            cudaHostAlloc((void **)&x->dc_pin, sizeof(gcn_dist_dcount),
                          cudaHostAllocDefault) != cudaSuccess) {
            x->dc_pin = 0;
            st = GC_E_NOMEM;
        }
        struct gcn_dist_max_req r;
        native_dist_max_start(ctx, st != GC_OK ? 1 : no, &r);
        x->devcount = native_dist_max_wait(&r) == 0;
        st = native_dist_vote(ctx, st, "dist_create_devcount");
        if (st != GC_OK) return st;
    }
    ++d->plan_generation;
    st = gcn_dist_bootstrap_outer(ctx);
    if (st != GC_OK) return st;
    st = native_dist_refresh_maps(ctx, 0);
    return native_dist_vote(ctx, st, "dist_create_maps");
}

/* The device-counted refreshes of this rank, for the run's summary. */
void native_dist_summary(const gc_context *ctx)
{
    const gcn_dist_state *x = ctx && ctx->native ? ctx->native->dist : 0;
    if (!x || !x->devcount) return;
    std::fprintf(stderr,
        "GPU_Core_Summary> rank=%d halo_refresh_device_counted=%lld "
        "redone=%lld records=%lld margin_records=%lld\n", (int)ctx->rank,
        (long long)x->dc_runs, (long long)x->dc_redo,
        (long long)x->dc_records, (long long)x->dc_margin);
}

gc_status native_dist_destroy(gc_context *ctx)
{
    if (!ctx || !ctx->native) return GC_OK;
    struct gcn_device *d = ctx->native;
    gcn_dist_state *x = d->dist;
    if (!x) return GC_OK;
    if (x->halo) gcx_halo_destroy(x->halo);
    if (x->wsum) gcx_wsum_destroy(x->wsum);
    dev_release((void **)&x->dc);
    if (x->dc_pin) { cudaFreeHost(x->dc_pin); x->dc_pin = 0; }
    if (x->producer) cudaEventDestroy(x->producer);
    if (x->consumer) cudaEventDestroy(x->consumer);
    for (int e = 0; e < 2; ++e) {
        dev_release(&x->coord_send[e]); dev_release(&x->coord_recv[e]);
        dev_release(&x->force_send[e]); dev_release(&x->force_recv[e]);
    }
    dev_release((void **)&x->pack_count); dev_release((void **)&x->bad);
    if (x->pack_pin) { cudaFreeHost(x->pack_pin); x->pack_pin = 0; }
    dev_release((void **)&x->seq);
    dev_release((void **)&x->pack_flag); dev_release((void **)&x->pack_pos);
    for (int r = 0; r < 2; ++r)
        for (int p = 0; p < 3; ++p) {
            dev_release((void **)&x->route[r][p]);
            dev_release((void **)&x->land[r][p]);
        }
    dev_release((void **)&x->view_move);
    dev_release((void **)&x->coord_visit); dev_release((void **)&x->force_visit);
    dev_release((void **)&x->owner_key); dev_release((void **)&x->owner_val);
    dev_release((void **)&x->ghost_key); dev_release((void **)&x->ghost_image);
    dev_release((void **)&x->ghost_edge);
    dev_release((void **)&x->ghost_val);
    if (x->probe_live) gcx_probe_release();
    delete x;
    d->dist = 0;
    return GC_OK;
}

gc_status native_dist_refresh_maps(gc_context *ctx, gc_i32 defer)
{
    if (!ctx || !ctx->native) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (ctx->nproc == 1) return GC_OK;
    gcn_dist_state *x = d->dist;
    if (!x) return GC_E_STATE;
    x->map_ready = 0;
    gc_status st = gcn_dist_fit(x, d);
    if (st != GC_OK) return st;
    cudaStream_t s = d->stream;
    /* deferred: the failures add to the halo refresh's register count,
     * which the rebuild reads with its next synchronisation */
    gc_i64 *bad = defer ? x->bad + 1 : x->bad;
    if (!defer) cudaMemsetAsync(bad, 0, sizeof(gc_i64), s);
    gcn_dist_map_clear<<<(unsigned)dist_grid(x->owner_cap), GCN_BLOCK, 0, s>>>(
        x->owner_key, x->owner_val, 0, 0, x->owner_cap);
    gcn_dist_map_clear<<<(unsigned)dist_grid(x->ghost_cap), GCN_BLOCK, 0, s>>>(
        x->ghost_key, x->ghost_val, x->ghost_edge, x->ghost_image,
        x->ghost_cap);
    gcn_dist_owner_insert<<<(unsigned)dist_grid(d->num_owned), GCN_BLOCK, 0, s>>>(
        d->gid, d->num_owned, x->owner_key, x->owner_val, x->owner_cap, bad);
    if (d->num_ghost > 0)
        gcn_dist_ghost_insert<<<(unsigned)dist_grid(d->num_ghost), GCN_BLOCK, 0, s>>>(
            d->gid, d->image, d->cell_of, d->num_owned, d->num_ghost,
            d->layout, x->arrive, x->coord, x->ghost_key, x->ghost_image, x->ghost_edge,
            x->ghost_val, x->ghost_cap, bad);
    if (d->num_ghost > 0)
        gcn_dist_ghost_verify<<<(unsigned)dist_grid(d->num_ghost), GCN_BLOCK, 0, s>>>(
            d->gid, d->image, d->cell_of, d->num_owned, d->num_ghost,
            d->layout, x->arrive, x->coord, x->ghost_key, x->ghost_image, x->ghost_edge,
            x->ghost_val, x->ghost_cap, bad);
    x->map_generation = d->slot_generation;
    x->map_plan_generation = d->plan_generation;
    if (defer) {
        x->map_ready = 1;
        return dev_launched("dist_identity_map");
    }
    gc_i64 *pin = native_rebuild_pin(d);
    if (!pin) return GC_E_NOMEM;
    if (cudaMemcpyAsync(pin + 50, x->bad, sizeof(gc_i64), cudaMemcpyDeviceToHost,
                        s) != cudaSuccess)
        return GC_E_DEVICE;
    st = dev_phase(s, "dist_identity_map");
    if (st != GC_OK) return st;
    if (pin[50]) return GC_E_MISMATCH;
    x->map_ready = 1;
    return GC_OK;
}

gc_i64 *native_dist_verdict(gc_context *ctx)
{
    gcn_dist_state *x = ctx && ctx->native ? ctx->native->dist : 0;
    return x ? x->bad + 1 : 0;
}

gc_status native_dist_forward(gc_context *ctx)
{
    if (!ctx) return GC_E_ARG;
    if (ctx->nproc == 1) return GC_OK;
    struct gcn_device *d = ctx->native;
    gcn_dist_state *x = d ? d->dist : 0;
    gc_status st = (!d || !x || !x->halo || !d->list_valid || !x->map_ready ||
        x->map_generation != d->slot_generation ||
        x->map_plan_generation != d->plan_generation) ? GC_E_STATE : GC_OK;
    st = native_dist_step_vote(ctx, st, "halo_coord_admit");
    if (st != GC_OK) return st;

    cudaStream_t s = d->stream;
    const int slim = x->step_gen[0][0] == d->slot_generation &&
                     x->step_gen[0][1] == d->plan_generation;
    /* The step's failures add to the sticky d->verdict[1]. */
    gc_i64 *bad = d->verdict + 1;
    /* First forward after a halo refresh: the refresh registered every ghost
     * with its owner's coordinates and view move and nothing has written a
     * coordinate since, so the forward would deliver the same values. Every
     * rank skips the same passes, so the transport's sequence stays paired. */
    const int fresh = slim && x->fresh;
    x->fresh = 0;
    const int wide = dist_slim_word(d) == (gc_i64)sizeof(gc_f64);
    if (fresh) {
        if (!wide && d->num_ghost > 0)
            gcn_dist_round_ghosts<float><<<(unsigned)dist_grid(d->num_ghost),
                                           GCN_BLOCK, 0, s>>>(
                d->num_owned, d->num_ghost, x->view_move, d->coord,
                d->force_coord, d->pitch);
        return native_dist_step_vote(ctx, dev_launched("halo_coord_fresh"),
                                     "halo_coord_publish");
    }
    if ((!slim && (cudaMemsetAsync(x->coord_visit, 0,
                                   (size_t)x->visit_capacity * sizeof(gc_i32),
                                   s) != cudaSuccess
                   )))
        st = GC_E_DEVICE;
    if (!slim && d->num_groups > 0)
        gcn_dist_view_move_owned<<<(unsigned)dist_grid(d->num_groups),
                                   GCN_BLOCK, 0, s>>>(
            d->group_offset, d->group_force_move, d->num_groups, x->view_move);
    st = native_dist_step_vote(ctx, st, "halo_coord_clear");
    if (st != GC_OK) return st;

    gc_i64 sent[2] = {0,0}, got[2] = {0,0};
    gc_i64 nscan = d->num_owned;       /* owned only until a pass has run */
    dist_open_all(d, x, 0);
    gc_status carry = x->open_st[0];
    x->open_st[0] = GC_OK;
    for (int p = 0; p < 3; ++p) {
        if (d->layout.nd[p] == 1) continue;
        gc_i64 ps[2], pg[2];
        const gc_i64 cap = x->capacity_records;
        const gc_i64 *known = x->step_sent[0][p];
        const gc_status pc = carry;
        carry = GC_OK;
        st = dist_pass(ctx, p, 0, "halo_coord", 0, slim, ps, pg,
            [&](cudaStream_t s) -> gc_status {
                if (slim) {
                    if (known[0] + known[1] > 0) {
                        const unsigned gb = (unsigned)dist_grid(known[0] + known[1]);
                        if (dist_slim_word(d) == (gc_i64)sizeof(float))
                            gcn_dist_pack_coord_slim<float><<<gb, GCN_BLOCK, 0, s>>>(
                                x->route[0][p], cap, known[0], known[1], d->coord,
                                d->pitch, (float *)x->wire_out[0], (float *)x->wire_out[1]);
                        else
                            gcn_dist_pack_coord_slim<gc_f64><<<gb, GCN_BLOCK, 0, s>>>(
                                x->route[0][p], cap, known[0], known[1], d->coord,
                                d->pitch, (gc_f64 *)x->wire_out[0], (gc_f64 *)x->wire_out[1]);
                    }
                    return GC_OK;
                }
                return dist_compact(d, x, nscan, s,
                    [&](gc_i32 *flag, const gc_i64 *pos) {
                    gcn_dist_pack_coord<<<(unsigned)dist_grid(nscan), GCN_BLOCK, 0, s>>>(
                        d->coord, d->charge, d->cls, d->gid,
                        d->image, d->owner, d->cell_of, d->num_owned, nscan,
                        d->pitch, d->layout, p, 0, x->arrive,
                        x->coord.partner[p][0], x->coord.partner[p][1],
                        ctx->epoch, d->slot_generation, d->plan_generation,
                        (gcn_coord_wire *)x->coord_send[0],
                        (gcn_coord_wire *)x->coord_send[1],
                        cap, x->bad, x->view_move, x->route[0][p], flag, pos, x->pack_count,
                        dist_box(d));
                });
            },
            [&](cudaStream_t s, const gc_i64 *n) -> gc_status {
                if (slim) {
                    if (n[0] + n[1] > 0) {
                        const unsigned gb = (unsigned)dist_grid(n[0] + n[1]);
                        if (dist_slim_word(d) == (gc_i64)sizeof(float))
                            gcn_dist_unpack_coord_slim<float><<<gb, GCN_BLOCK, 0, s>>>(
                                x->land[0][p], cap, n[0], n[1],
                                (const float *)x->wire_in[0], (const float *)x->wire_in[1],
                                x->view_move, d->coord, d->force_coord, d->pitch);
                        else
                            gcn_dist_unpack_coord_slim<gc_f64><<<gb, GCN_BLOCK, 0, s>>>(
                                x->land[0][p], cap, n[0], n[1],
                                (const gc_f64 *)x->wire_in[0], (const gc_f64 *)x->wire_in[1],
                                x->view_move, d->coord, d->force_coord, d->pitch);
                    }
                    return GC_OK;
                }
                for (int e = 0; e < 2; ++e)
                    if (n[e] > 0)
                        gcn_dist_unpack_coord<<<(unsigned)dist_grid(n[e]),
                                                GCN_BLOCK, 0, s>>>(
                            (const gcn_coord_wire *)x->coord_recv[e], n[e],
                            x->coord.key[p][e], x->coord.peer[p][e],
                            ctx->epoch, d->slot_generation, d->plan_generation,
                            x->ghost_key, x->ghost_image, x->ghost_edge,
                            x->ghost_val, x->ghost_cap, d->image, d->owner,
                            x->coord_visit, d->coord, d->force_coord,
                            d->charge, d->cls, d->pitch, bad,
                            x->view_move, x->land[0][p] + e * cap
                            );
                return GC_OK;
            }, pc);
        if (st != GC_OK) return st;
        for (int e = 0; e < 2; ++e) { sent[e] += ps[e]; got[e] += pg[e]; }
        nscan = d->num_resident;
    }
    st = dist_release_all(d, x, 0, s);

    if (!slim && d->num_ghost > 0)
        gcn_dist_verify_coord_visits<<<(unsigned)dist_grid(d->num_ghost),
                                       GCN_BLOCK, 0, s>>>(
            x->coord_visit, d->num_owned, d->num_ghost, bad
            );
    /* The verdict is read at the step driver's next synchronising call, not
     * here. */
    if (st == GC_OK) {
        x->step_gen[0][0] = d->slot_generation;
        x->step_gen[0][1] = d->plan_generation;
    }
    return native_dist_step_vote(ctx, st, "halo_coord_publish");
}

/* The force view's per-slot offsets, which a constant-pressure step scales
 * with the box (gpu_npt.cu).  None without a halo. */
gc_f64 *native_dist_view_move(gc_context *ctx, gc_i64 *n)
{
    *n = 0;
    struct gcn_device *d = ctx ? ctx->native : 0;
    gcn_dist_state *x = d ? d->dist : 0;
    if (!x || !x->view_move) return 0;
    *n = d->num_resident < x->visit_capacity ? d->num_resident
                                             : x->visit_capacity;
    return x->view_move;
}

/* A direction's passes opened ahead, in one launch: each fused pass's open
 * waits only for its slot's previous use. Only when every pass of the
 * direction fuses, so epochs are taken in the same order as pass by pass; the
 * passes then release their slots in one launch after the last
 * (dist_release_all). A failed claim is reported by the first pass's vote. */
static void dist_open_all(gcn_device *d, gcn_dist_state *x, int reverse)
{
    for (int p = 0; p < 3; ++p) x->opened[reverse][p] = 0;
    x->late_release[reverse] = 0;
    x->open_st[reverse] = GC_OK;
    if (!x->halo || !d->list_valid || !x->map_ready ||
        x->map_generation != d->slot_generation ||
        x->map_plan_generation != d->plan_generation ||
        x->step_gen[reverse][0] != d->slot_generation ||
        x->step_gen[reverse][1] != d->plan_generation)
        return;
    for (int p = 0; p < 3; ++p)
        if (d->layout.nd[p] > 1 &&
            !(x->fused[reverse][p] && (x->stores[reverse][p] || d->ce_fused)))
            return;
    gcn_dist_seq_args o;
    int k = 0;
    gc_status st = GC_OK;
    for (int i = 0; i < 3 && st == GC_OK; ++i) {
        const int p = reverse ? 2 - i : i;
        if (d->layout.nd[p] == 1) continue;
        st = dist_claim(x, p, reverse, d->stream, &x->open_epoch[reverse][p],
                        &x->open_slot[reverse][p], &o, k++);
        x->opened[reverse][p] = 1;
    }
    if (st == GC_OK && k > 0) {
        gcn_dist_open_many<<<1, 2 * k, 0, d->stream>>>(o, k);
        x->late_release[reverse] = 1;
    }
    x->open_st[reverse] = st;
}

/* The slot releases dist_open_all held back, after the direction's last
 * pass. */
static gc_status dist_release_all(gcn_device *d, gcn_dist_state *x,
                                  int reverse, cudaStream_t s)
{
    if (!x->late_release[reverse]) return GC_OK;
    x->late_release[reverse] = 0;
    gcn_dist_seq_args o;
    int k = 0;
    for (int p = 0; p < 3; ++p) {
        if (d->layout.nd[p] == 1) continue;
        const gcx_device_edge *g = x->dev[reverse][p];
        o.a[k] = g[0].peer_sig + 1;
        o.b[k] = g[1].peer_sig + 1;
        o.q[k] = x->seq + 3 * reverse + p;
        ++k;
    }
    gcn_dist_release_many<<<1, 2 * k, 0, s>>>(o, k);
    return cudaGetLastError() == cudaSuccess ? GC_OK : GC_E_DEVICE;
}

void native_dist_return_open(gc_context *ctx)
{
    gcn_device *d = ctx && ctx->nproc > 1 ? ctx->native : 0;
    if (d && d->dist) dist_open_all(d, d->dist, 1);
}

/* A ghost's real-space and bonded force return into its owner's
 * force_bond.  The reciprocal force has no ghost part: each rank gathers
 * its owned atoms' whole force (gpu_pme.cu). */
gc_status native_dist_reverse(gc_context *ctx)
{
    if (!ctx) return GC_E_ARG;
    if (ctx->nproc == 1) return GC_OK;
    struct gcn_device *d = ctx->native;
    gcn_dist_state *x = d ? d->dist : 0;
    gc_status st = (!d || !x || !x->halo || !d->list_valid || !x->map_ready ||
        x->map_generation != d->slot_generation ||
        x->map_plan_generation != d->plan_generation) ? GC_E_STATE : GC_OK;
    if (st != GC_OK)
        std::fprintf(stderr,
            "GPU_Core_Error> phase=halo_force_admit rank=%d d=%d x=%d "
            "halo=%d list_valid=%d map_ready=%d mapgen=%lld slotgen=%lld "
            "map_plangen=%lld plangen=%lld\n",
            (int)ctx->rank, d != 0, x != 0, x ? (x->halo != 0) : 0,
            d ? (int)d->list_valid : -1, x ? (int)x->map_ready : -1,
            x ? (long long)x->map_generation : -1,
            d ? (long long)d->slot_generation : -1,
            x ? (long long)x->map_plan_generation : -1,
            d ? (long long)d->plan_generation : -1);
    st = native_dist_step_vote(ctx, st, "halo_force_admit");
    if (st != GC_OK) return st;

    cudaStream_t s = d->stream;
    const int slim = x->step_gen[1][0] == d->slot_generation &&
                     x->step_gen[1][1] == d->plan_generation;
    /* The step's failures add to the sticky d->verdict[2]. */
    gc_i64 *bad = d->verdict + 2;
    const gc_f64 *pr = d->force_real;
    gc_f64 *pb = d->force_bond;
    const int fx = dist_exact_return(d);
    const int mw = native_dist_mixed_words(d);
    const unsigned long long *prx = fx || mw ? d->force_real_fx : 0;
    unsigned long long *pbx = fx || mw ? d->force_bond_fx : 0;
    struct gcn_join jv;
    if (mw && native_force_join_view(ctx, &jv) != GC_OK) st = GC_E_DEVICE;
    if (!slim && cudaMemsetAsync(x->force_visit, 0,
                                 (size_t)x->visit_capacity * sizeof(gc_i32),
                                 s) != cudaSuccess)
        st = GC_E_DEVICE;
    st = native_dist_step_vote(ctx, st, "halo_force_clear");
    if (st != GC_OK) return st;

    gc_status carry = x->open_st[1];
    x->open_st[1] = GC_OK;
    for (int p = 2; p >= 0; --p) {
        if (d->layout.nd[p] == 1) continue;
        gc_i64 ps[2], pg[2];
        const gc_i64 cap = x->capacity_records;
        const gc_i64 *known = x->step_sent[1][p];
        const gc_status pc = carry;
        carry = GC_OK;
        st = dist_pass(ctx, p, 1, "halo_force", 0, slim, ps, pg,
            [&](cudaStream_t s) -> gc_status {
                if (slim) {
                    if (known[0] + known[1] > 0) {
                        const unsigned gb = (unsigned)dist_grid(known[0] + known[1]);
                        /* nproc > 1: one of the two word returns holds */
                        if (fx)
                            gcn_dist_pack_force_fx<<<gb, GCN_BLOCK, 0, s>>>(
                                x->route[1][p], cap, known[0], known[1], prx, pbx,
                                d->pitch, (unsigned long long *)x->wire_out[0],
                                (unsigned long long *)x->wire_out[1]);
                        else
                            gcn_dist_pack_force_fx32<<<gb, GCN_BLOCK, 0, s>>>(
                                x->route[1][p], cap, known[0], known[1], prx, pbx,
                                jv.overflow, d->pitch, (float *)x->wire_out[0],
                                (float *)x->wire_out[1]);
                    }
                    return GC_OK;
                }
                return dist_compact(d, x, d->num_ghost, s,
                    [&](gc_i32 *flag, const gc_i64 *pos) {
                    gcn_dist_pack_force<<<(unsigned)dist_grid(d->num_ghost),
                                          GCN_BLOCK, 0, s>>>(
                        d->gid, d->image, d->owner, d->cell_of,
                        pr, pb, prx, pbx,
                        d->num_owned, d->num_ghost, d->pitch, d->layout,
                        x->arrive, p, x->force.partner[p][0], x->force.partner[p][1],
                        ctx->epoch, d->slot_generation, d->plan_generation,
                        (gcn_force_wire *)x->force_send[0],
                        (gcn_force_wire *)x->force_send[1],
                        cap, x->bad, x->route[1][p], flag, pos, x->pack_count);
                });
            },
            [&](cudaStream_t s, const gc_i64 *n) -> gc_status {
                if (slim) {
                    if (n[0] + n[1] > 0) {
                        const unsigned gb = (unsigned)dist_grid(n[0] + n[1]);
                        if (fx)
                            gcn_dist_unpack_force_fx<<<gb, GCN_BLOCK, 0, s>>>(
                                x->land[1][p], cap, n[0], n[1],
                                (const unsigned long long *)x->wire_in[0],
                                (const unsigned long long *)x->wire_in[1],
                                pbx, d->pitch);
                        else
                            gcn_dist_unpack_force_fx32<<<gb, GCN_BLOCK, 0, s>>>(
                                x->land[1][p], cap, n[0], n[1],
                                (const float *)x->wire_in[0],
                                (const float *)x->wire_in[1],
                                pbx, d->pitch, bad);
                    }
                    return GC_OK;
                }
                for (int e = 0; e < 2; ++e)
                    if (n[e] > 0)
                        gcn_dist_unpack_force<<<(unsigned)dist_grid(n[e]),
                                                GCN_BLOCK, 0, s>>>(
                            (const gcn_force_wire *)x->force_recv[e], n[e],
                            x->force.key[p][e], x->force.peer[p][e], p, e,
                            gcn_dist_sender_image(d->layout, p, e),
                            ctx->epoch, d->slot_generation, d->plan_generation,
                            d->layout, x->owner_key, x->owner_val, x->owner_cap,
                            x->ghost_key, x->ghost_image, x->ghost_edge,
                            x->ghost_val, x->ghost_cap, d->owner,
                            x->force_visit, pb, pbx, d->pitch,
                            bad, x->land[1][p] + e * cap);
                return GC_OK;
            }, pc);
        if (st != GC_OK) return st;
    }
    if (st == GC_OK) st = dist_release_all(d, x, 1, s);

    if (!slim && d->num_resident > 0)
        gcn_dist_verify_force_visits<<<(unsigned)dist_grid(d->num_resident),
                                       GCN_BLOCK, 0, s>>>(
            d->cell_of, x->force_visit, d->num_owned, d->num_resident,
            d->layout, x->arrive, bad);
    /* The exact return landed in force_bond_fx; the mixed mode's readers take
     * the words (native_dist_mixed_words). */
    if (st == GC_OK && fx) native_force_bond_fold_owned(d);
    if (st == GC_OK) {
        x->step_gen[1][0] = d->slot_generation;
        x->step_gen[1][1] = d->plan_generation;
    }
    return native_dist_step_vote(ctx, st, "halo_force_publish");
}

/* A step captured in a CUDA graph: allowed when both halves are slim, every
 * pass is fused, and each pass's next epoch lies an even stride after its
 * last (the device advances the epoch: gcx_seq). A replay runs each pass's
 * host side: the epoch and the plan's slot claim, checked against that
 * stride. */
template <class Visit>
static gc_status dist_each_pass(gcn_device *d, Visit visit)
{
    for (int k = 0; k < 6; ++k) {
        const int reverse = k >= 3, axis = reverse ? 5 - k : k;
        if (d->layout.nd[axis] == 1) continue;
        const gc_status st = visit(reverse, axis);
        if (st != GC_OK) return st;
    }
    return GC_OK;
}

int native_dist_graph_ready(const gc_context *ctx)
{
    if (ctx->nproc == 1) return 1;
    gcn_device *d = ctx->native;
    gcn_dist_state *x = d ? d->dist : 0;
    if (!x || !x->halo) return 0;
    for (int r = 0; r < 2; ++r)
        if (x->step_gen[r][0] != d->slot_generation ||
            x->step_gen[r][1] != d->plan_generation) return 0;
    gc_i64 next = x->sequence;
    return dist_each_pass(d, [&](int reverse, int axis) -> gc_status {
        ++next;
        return x->fused[reverse][axis] &&
               (x->stores[reverse][axis] || d->ce_fused) &&
               gcx_move_steady(dist_plan(x, reverse, axis), next, 1)
               ? GC_OK : GC_E_STATE;
    }) == GC_OK;
}

void native_dist_graph_capability(const gc_context *ctx, int *fused,
                                  int *stores)
{
    const gcn_device *d = ctx->native;
    const gcn_dist_state *x = d ? d->dist : 0;
    *fused = *stores = ctx->nproc == 1 || (x != 0 && x->halo != 0);
    if (ctx->nproc == 1 || !*fused) return;
    for (int r = 0; r < 2; ++r)
        for (int p = 0; p < 3; ++p) {
            if (d->layout.nd[p] == 1) continue;
            if (!x->fused[r][p]) *fused = 0;
            if (!x->stores[r][p]) *stores = 0;
        }
}

int native_dist_ce_candidate(const gc_context *ctx)
{
    const gcn_device *d = ctx->native;
    const gcn_dist_state *x = d ? d->dist : 0;
    if (ctx->nproc == 1 || !x) return 0;
    for (int r = 0; r < 2; ++r)
        for (int p = 0; p < 3; ++p)
            if (d->layout.nd[p] > 1 && x->fused[r][p] && !x->stores[r][p]) return 1;
    return 0;
}

gc_status native_dist_step_replay(gc_context *ctx)
{
    if (ctx->nproc == 1) return GC_OK;
    gcn_device *d = ctx->native;
    gcn_dist_state *x = d->dist;
    const gc_status st = dist_each_pass(d, [&](int reverse, int axis) {
        return gcx_move_replay(dist_plan(x, reverse, axis), ++x->sequence);
    });
    return native_dist_step_vote(ctx, st, "halo_replay");
}

/* The records a device-counted refresh sends on an edge whose last
 * refresh carried n: n and a tenth, and a round 64 for a small band. */
static gc_i64 dist_dc_cap(const gcn_dist_state *x, gc_i64 n)
{
    gc_i64 c = n + n / 10 + 64;
    const gc_i64 most = x->coord_capacity / (gc_i64)sizeof(gcn_reg_wire) - 1;
    const gc_i64 lim = x->capacity_records - 1 < most ? x->capacity_records - 1 : most;
    return c < lim ? c : lim;
}

/* The refresh with its counts on the device. Every pass sends a fixed number
 * of records per edge (dist_dc_cap) with the true count after them, so a
 * device-sequenced edge takes no host round trip. One synchronisation after
 * the passes reads the counts, and a device sum of the overflow words decides
 * whether every rank redoes the refresh on the counted path (`*done` 0). */
static gc_status dist_refresh_dev(gc_context *ctx, gc_status admit, int *done)
{
    *done = 0;
    gcn_device *d = ctx->native;
    gcn_dist_state *x = d->dist;
    cudaStream_t s = d->stream;
    const gc_i64 rec = (gc_i64)sizeof(gcn_reg_wire);
    const gc_i64 cap = x->capacity_records;
    gc_i64 cs[3][2], cr[3][2], bound[4];
    bound[0] = d->num_owned;
    for (int p = 0; p < 3; ++p) {
        for (int e = 0; e < 2; ++e) {
            const int on = d->layout.nd[p] > 1;
            cs[p][e] = on ? dist_dc_cap(x, x->step_sent[0][p][e]) : 0;
            cr[p][e] = on ? dist_dc_cap(x, x->step_got[0][p][e]) : 0;
        }
        bound[p + 1] = bound[p] + cr[p][0] + cr[p][1];
    }
    gc_status st = admit;
    if (st == GC_OK)
        st = native_migration_reserve(ctx, d->num_owned, d->num_groups,
                                      d->num_members, bound[3]);
    if (st == GC_OK) st = gcn_dist_fit_moves(x, bound[3], d->num_owned, s);
    if (st == GC_OK) gcn_dist_dc_init<<<1, 1, 0, s>>>(x->dc, d->num_owned);
    for (int p = 0; p < 3; ++p) {
        if (d->layout.nd[p] == 1) continue;
        const gc_i64 nscan = bound[p];
        if (st == GC_OK &&
            (cudaMemsetAsync(x->pack_count, 0, 2 * sizeof(gc_i64), s) != cudaSuccess ||
             cudaMemsetAsync(x->bad, 0, sizeof(gc_i64), s) != cudaSuccess))
            st = GC_E_DEVICE;
        if (st == GC_OK)
            st = dist_compact(d, x, nscan, s,
                [&](gc_i32 *flag, const gc_i64 *pos) {
                gcn_dist_pack_coord<<<(unsigned)dist_grid(nscan), GCN_BLOCK,
                                      0, s>>>(
                    d->coord, d->charge, d->cls, d->gid, d->image, d->owner,
                    d->cell_of, d->num_owned, nscan, d->pitch, d->layout, p,
                    0, x->arrive, x->coord.partner[p][0],
                    x->coord.partner[p][1], ctx->epoch, d->slot_generation,
                    d->plan_generation, (gcn_reg_wire *)x->coord_send[0],
                    (gcn_reg_wire *)x->coord_send[1], cap, x->bad,
                    x->view_move, x->route[0][p], flag, pos, x->pack_count,
                    dist_box(d), x->dc);
            });
        if (st == GC_OK) {
            gcn_dist_dc_sent<<<1, 1, 0, s>>>(x->dc, x->pack_count, x->bad,
                (char *)x->coord_send[0], (char *)x->coord_send[1],
                cs[p][0], cs[p][1], rec, p);
            if (cudaEventRecord(x->producer, s) != cudaSuccess)
                st = GC_E_DEVICE;
        }
        st = native_dist_step_vote(ctx, st, "halo_refresh_dev_pack");
        if (st != GC_OK) return st;

        gcx_buffer sb[2], rb[2];
        gc_i64 sent[2], got[2] = {0, 0};
        gc_i64 send_bytes[2], recv_bytes[2] = {0, 0};
        for (int e = 0; e < 2; ++e) {
            sent[e] = cs[p][e] + 1;
            send_bytes[e] = sent[e] * rec;
            sb[e].ptr = x->coord_send[e]; sb[e].bytes = x->coord_capacity;
            rb[e].ptr = x->coord_recv[e]; rb[e].bytes = (cr[p][e] + 1) * rec;
        }
        gcx_op_desc od; std::memset(&od, 0, sizeof(od));
        od.epoch = ++x->sequence;
        od.op = GCX_OP_HALO_COORD; od.variable = 0;
        od.send = sb; od.recv = rb; od.send_bytes = send_bytes;
        od.send_records = sent; od.recv_bytes = recv_bytes;
        od.recv_records = got; od.producer_event = (void *)x->producer;
        gcx_token *tok = 0;
        st = gcx_halo_forward(x->halo, p, &od, &tok);
        st = native_dist_step_vote(ctx, st, "halo_refresh_dev_begin");
        if (st != GC_OK) { if (tok) gcx_abort(tok); return st; }
        st = gcx_consume(tok, (void *)s);
        for (int e = 0; e < 2 && st == GC_OK; ++e)
            if (recv_bytes[e] != rb[e].bytes) st = GC_E_MISMATCH;
        if (st == GC_OK) {
            gcn_dist_dc_arrive<<<1, 1, 0, s>>>(x->dc,
                (const char *)x->coord_recv[0], (const char *)x->coord_recv[1],
                cr[p][0], cr[p][1], rec, p);
            for (int e = 0; e < 2; ++e)
                if (cr[p][e] > 0)
                    gcn_dist_register_coord<<<(unsigned)dist_grid(cr[p][e]),
                                              GCN_BLOCK, 0, s>>>(
                        (const gcn_reg_wire *)x->coord_recv[e], cr[p][e], 0,
                        p, e, x->coord.key[p][e], x->coord.peer[p][e],
                        ctx->epoch, d->slot_generation, d->plan_generation,
                        d->layout, d->gid, d->image, d->owner, d->cell_of,
                        d->coord, d->force_coord, d->charge, d->cls,
                        d->pitch, bound[3], x->bad + 1, x->view_move, x->land[0][p] + e * cap, dist_box(d),
                        x->dc);
            if (cudaEventRecord(x->consumer, s) != cudaSuccess)
                st = GC_E_DEVICE;
        }
        if (st == GC_OK) st = gcx_release(tok, (void *)x->consumer);
        else gcx_abort(tok);
        st = native_dist_step_vote(ctx, st, "halo_refresh_dev_release");
        if (st != GC_OK) return st;
    }
    st = gcx_wsum_launch(x->wsum, &x->dc->over, (void *)s);
    if (st == GC_OK &&
        cudaMemcpyAsync(x->dc_pin, x->dc, sizeof(gcn_dist_dcount),
                        cudaMemcpyDeviceToHost, s) != cudaSuccess)
        st = GC_E_DEVICE;
    if (st == GC_OK) st = host_run();
    if (st == GC_OK) st = dev_phase(s, "halo_refresh_count");
    st = native_dist_step_vote(ctx, st, "halo_refresh_dev_count");
    if (st != GC_OK) return st;
    const gcn_dist_dcount &c = *x->dc_pin;
    if (c.over) { ++x->dc_redo; return GC_OK; }
    x->arrive = c.arrive;
    ++x->dc_runs;
    for (int p = 0; p < 3; ++p)
        for (int e = 0; e < 2; ++e) {
            x->dc_records += c.sent[p][e];
            x->dc_margin += cs[p][e] - c.sent[p][e];
            x->step_sent[0][p][e] = d->layout.nd[p] > 1 ? c.sent[p][e] : 0;
            x->step_got[0][p][e] = d->layout.nd[p] > 1 ? c.arrive.n[p][e] : 0;
        }
    d->num_resident = c.resident;
    d->num_ghost = c.resident - d->num_owned;
    *done = 1;
    return GC_OK;
}

/* After a migration the ghost identities changed, so the band is replaced
 * pass by pass from the neighbours' packs: each arriving record is registered
 * into the next ghost slot and later passes relay the earlier ones. These are
 * the generation's first passes, so the first step after the rebuild is
 * already slim. Records are gcn_reg_wire; a move that is not box/2 less whole
 * periods fails the pack (GC_E_CAPACITY). */
gc_status native_dist_halo_refresh(gc_context *ctx)
{
    if (!ctx) return GC_E_ARG;
    if (ctx->nproc == 1) return GC_OK;
    struct gcn_device *d = ctx->native;
    gcn_dist_state *x = d ? d->dist : 0;
    if (!d || !x || !x->halo) return native_dist_step_vote(ctx, GC_E_STATE,
                                                           "halo_refresh_admit");
    gc_status admit =
        cudaMemsetAsync(x->bad + 1, 0, sizeof(gc_i64), d->stream) == cudaSuccess
        ? GC_OK : GC_E_DEVICE;
    gc_status st = GC_OK;
    const gc_i64 cap = x->capacity_records;

    if (admit == GC_OK)
        admit = gcn_dist_fit_moves(x, d->num_owned, 0, d->stream);
    if (admit == GC_OK && d->num_groups > 0)
        gcn_dist_view_move_owned<<<(unsigned)dist_grid(d->num_groups),
                                   GCN_BLOCK, 0, d->stream>>>(
            d->group_offset, d->group_force_move, d->num_groups, x->view_move);
    x->step_gen[0][0] = x->step_gen[1][0] = -1;
    x->fresh = 0;

    d->num_ghost = 0;
    d->num_resident = d->num_owned;
    std::memset(&x->arrive, 0, sizeof(x->arrive));
    x->arrive.valid = 1;
    int done = 0;
    if (x->devcount && x->dc_hist) {
        st = dist_refresh_dev(ctx, admit, &done);
        if (st != GC_OK) return st;
        admit = GC_OK;
        if (!done &&
            cudaMemsetAsync(x->bad + 1, 0, sizeof(gc_i64), d->stream) != cudaSuccess)
            admit = GC_E_DEVICE;
    }
    for (int p = 0; p < 3 && !done; ++p) {
        if (d->layout.nd[p] == 1) continue;
        gc_i64 ps[2], pg[2];
        st = dist_pass(ctx, p, 0, "halo_refresh", 0, 0, ps, pg,
            [&](cudaStream_t s) {
                const gc_i64 nscan = d->num_resident;
                return dist_compact(d, x, nscan, s,
                    [&](gc_i32 *flag, const gc_i64 *pos) {
                    gcn_dist_pack_coord<<<(unsigned)dist_grid(nscan),
                                          GCN_BLOCK, 0, s>>>(
                        d->coord, d->charge, d->cls, d->gid,
                        d->image, d->owner, d->cell_of, d->num_owned,
                        nscan, d->pitch, d->layout, p, 0, x->arrive,
                        x->coord.partner[p][0], x->coord.partner[p][1],
                        ctx->epoch, d->slot_generation, d->plan_generation,
                        (gcn_reg_wire *)x->coord_send[0],
                        (gcn_reg_wire *)x->coord_send[1],
                        cap, x->bad, x->view_move, x->route[0][p], flag, pos,
                        x->pack_count, dist_box(d));
                });
            },
            [&](cudaStream_t s, const gc_i64 *n) -> gc_status {
                const gc_i64 total = n[0] + n[1];
                gc_status r = native_migration_reserve(
                    ctx, d->num_owned, d->num_groups, d->num_members,
                    d->num_resident + total);
                if (r == GC_OK)
                    r = gcn_dist_fit_moves(x, d->num_resident + total,
                                           d->num_resident, s);
                if (r != GC_OK) return r;
                gc_i64 at = d->num_resident;
                for (int e = 0; e < 2; ++e) {
                    if (n[e] > 0)
                        gcn_dist_register_coord<<<(unsigned)dist_grid(n[e]),
                                                  GCN_BLOCK, 0, s>>>(
                            (const gcn_reg_wire *)x->coord_recv[e], n[e], at,
                            p, e, x->coord.key[p][e], x->coord.peer[p][e],
                            ctx->epoch, d->slot_generation, d->plan_generation,
                            d->layout, d->gid, d->image, d->owner, d->cell_of,
                            d->coord, d->force_coord, d->charge, d->cls,
                            d->pitch, d->num_resident + total, x->bad + 1,
                            x->view_move,
                            x->land[0][p] + e * cap, dist_box(d));
                    x->arrive.first[p][e] = at;
                    x->arrive.n[p][e] = n[e];
                    at += n[e];
                }
                d->num_ghost += total;
                d->num_resident += total;
                return GC_OK;
            }, admit, (gc_i64)sizeof(gcn_reg_wire), 1);
        admit = GC_OK;
        if (st != GC_OK) return st;
    }

    /* The return: pass p sends back the ghosts that arrived in it; each lands
     * on the slot this rank's forward route sent on that side at the same
     * position. */
    for (int p = 0; p < 3 && st == GC_OK; ++p) {
        if (d->layout.nd[p] == 1) continue;
        for (int e = 0; e < 2; ++e) {
            const gc_i64 n = x->arrive.n[p][e], m = x->step_sent[0][p][e];
            if (n > 0)
                gcn_dist_route_arrivals<<<(unsigned)dist_grid(n), GCN_BLOCK, 0,
                                          d->stream>>>(
                    x->route[1][p] + e * cap, x->arrive.first[p][e], n);
            if (m > 0 &&
                cudaMemcpyAsync(x->land[1][p] + e * cap, x->route[0][p] + e * cap,
                                (size_t)m * sizeof(gc_i32),
                                cudaMemcpyDeviceToDevice, d->stream) != cudaSuccess)
                st = GC_E_DEVICE;
            x->step_sent[1][p][e] = n;
            x->step_got[1][p][e] = m;
        }
    }
    /* The register count stays on the device (native_dist_verdict); the
     * caller reads it with its next synchronisation. */
    if (st == GC_OK) st = dev_launched("halo_refresh_register");
    if (st == GC_OK) st = admit;           /* no split axis ran a pass */
    if (st == GC_OK) {
        x->step_gen[0][0] = x->step_gen[1][0] = d->slot_generation;
        x->step_gen[0][1] = x->step_gen[1][1] = d->plan_generation;
        x->fresh = 1;
        x->dc_hist = 1;
    }
    return native_dist_step_vote(ctx, st, "halo_refresh_publish");
}

} /* namespace gcn */

