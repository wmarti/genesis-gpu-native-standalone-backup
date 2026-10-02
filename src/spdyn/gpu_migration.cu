/*
 * gpu_migration.cu : atom migration between ranks on the device.
 *
 * native_migration_reserve / _apply: grow slot capacities, compact departing
 * groups out of the CSR, append arriving groups and install their atom
 * records device-to-device, then publish the new counts.  Runs only after
 * the commit vote.
 *
 * migration_txn_native_run: one exchange with every neighbour rank (corners
 * included), each moving group sent straight to its owner.  The receiver
 * validates every parcel on the device before the vote; a refusal leaves the
 * epoch, the sender's state and the CSR as they were.
 */

#include "gpu_core_abi.h"
#include "gpu_core_internal.h"
#include "gpu_core_native.h"
#include "gpu_core_xchg.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <climits>
#include <iterator>
#include <unordered_map>
#include <memory>
#include <math.h>

#ifdef HAVE_MPI_GENESIS
#include <mpi.h>
#endif

#define GCN_MIG_BUMP(p) atomicAdd((unsigned long long *)(p), 1ull)

namespace gcn {

static gc_status mig_alloc(void **p, gc_i64 bytes)
{
    *p = 0;
    if (bytes <= 0) return GC_OK;
    if (cudaMalloc(p, (size_t)bytes) != cudaSuccess) return GC_E_NOMEM;
    return cudaMemset(*p, 0, (size_t)bytes) == cudaSuccess ? GC_OK
                                                           : GC_E_DEVICE;
}

static void mig_release(void **p)
{
    if (*p) { cudaFree(*p); *p = 0; }
}

static gc_status mig_d2h(void *dst, const void *src, gc_i64 bytes)
{
    if (bytes <= 0) return GC_OK;
    return cudaMemcpy(dst, src, (size_t)bytes, cudaMemcpyDeviceToHost) ==
                   cudaSuccess ? GC_OK : GC_E_DEVICE;
}

static gc_status mig_h2d(void *dst, const void *src, gc_i64 bytes)
{
    if (bytes <= 0) return GC_OK;
    return cudaMemcpy(dst, src, (size_t)bytes, cudaMemcpyHostToDevice) ==
                   cudaSuccess ? GC_OK : GC_E_DEVICE;
}

/* On the rebuild stream: a pageable source is staged before return. */
static gc_status mig_h2d_async(const struct gcn_device *d, void *dst,
                               const void *src, gc_i64 bytes)
{
    if (bytes <= 0) return GC_OK;
    return cudaMemcpyAsync(dst, src, (size_t)bytes, cudaMemcpyHostToDevice,
                           d->stream) == cudaSuccess ? GC_OK : GC_E_DEVICE;
}

/* A 64K-bit filter in front of an exact lookup of moving atoms. */
struct GidFilter {
    std::vector<gc_u64> w = std::vector<gc_u64>(1024, 0);
    static unsigned bit(gc_gid x)
    {
        return (unsigned)(((gc_u64)x * 0x9E3779B97F4A7C15ull) >> 48);
    }
    void add(gc_gid x) { w[bit(x) >> 6] |= 1ull << (bit(x) & 63); }
    bool maybe(gc_gid x) const { return (w[bit(x) >> 6] >> (bit(x) & 63)) & 1; }
};

/* One group leaving this rank: its GID, its destination, and its member
 * GIDs in the group's listed order (member 0 is the representative). */
struct MigMember {
    gc_gid group_gid;
    gc_i32 destination_rank;
    gc_i64 count;
    const gc_gid *gid;
};

/* A device (MigBuf) or pinned host (MigHost) buffer kept across rebuilds. */
struct MigBuf {
    void *p = 0;
    gc_i64 cap = 0;
};
typedef MigBuf MigHost;

/* Device buffers, events and the edge plan, kept across rebuilds. */
struct MigCache {
    MigBuf send, recv, stage, gt, rg, fa, fc, compact;
    MigBuf query, nhit;
    MigBuf keep, cnt, gat, mat, n_off, n_kind, n_gid, n_cell, arr;
    MigBuf layout;          /* the device copy of d->layout the validator reads */
    /* The sorted owned GIDs (the receiver's duplicate-owner table), kept
     * on the device and edited at each commit: owned_tab[owned_cur] holds
     * owned_n of them, -1 when it must be read afresh. */
    MigBuf owned_tab[2], oq, oflag, ogone;
    int owned_cur = 0;
    gc_i64 owned_n = -1;
    MigBuf rrow, rgid, rarity, rdist, rsend;
    MigHost hsend, hval, hrecv;
    MigHost howned, hrigid;
    const void *rigid_owner = 0;
    gc_i64 rigid_cap = 0;
    gcx_migration *plan = 0;
    gc_i64 plan_cap = 0;
    std::vector<gc_i32> plan_peers;
    /* the transaction's vote words (2 + nproc), summed on the device over
     * the ranks when they share a node (gcx_wsum), else null */
    gcx_wsum *vote = 0;
    gc_u64 *vote_words = 0;
    bool vote_tried = false;
    cudaEvent_t producer = 0, consumer = 0;
    gc_i64 owned_gone = 0, owned_next = 0;
    gc_i64 mask_rows = -1;
    gc_i64 send_g = 0, send_a = 0;
};

static std::unordered_map<const struct gcn_device *, MigCache> g_mig_cache;

/* b's buffer, at least `bytes` long; zeroed over `bytes` when `zero`. */
static gc_status mig_buf(MigBuf &b, gc_i64 bytes, int zero, void **out)
{
    *out = 0;
    if (bytes <= 0) return GC_OK;
    if (bytes > b.cap) {
        mig_release(&b.p);
        b.cap = 0;
        const gc_i64 want = bytes + bytes / 4 + 256;
        if (cudaMalloc(&b.p, (size_t)want) != cudaSuccess) {
            b.p = 0;
            return GC_E_NOMEM;
        }
        b.cap = want;
    }
    if (zero && cudaMemset(b.p, 0, (size_t)bytes) != cudaSuccess)
        return GC_E_DEVICE;
    *out = b.p;
    return GC_OK;
}

/* h's pinned buffer, at least `bytes` long; the contents are not kept. */
static gc_status mig_host(MigHost &h, gc_i64 bytes, char **out)
{
    *out = 0;
    if (bytes <= 0) bytes = 8;
    if (bytes > h.cap) {
        if (h.p) cudaFreeHost(h.p);
        h.p = 0;
        h.cap = 0;
        const gc_i64 want = bytes + bytes / 2 + 4096;
        if (cudaHostAlloc(&h.p, (size_t)want, cudaHostAllocDefault) != cudaSuccess) {
            h.p = 0;
            return GC_E_NOMEM;
        }
        h.cap = want;
    }
    *out = (char *)h.p;
    return GC_OK;
}

static void mig_plan_drop(MigCache &c)
{
    if (c.plan) gcx_migration_destroy(c.plan);
    c.plan = 0;
    c.plan_cap = 0;
}

void native_migration_release(const gc_context *ctx)
{
    const auto it = ctx ? g_mig_cache.find(ctx->native) : g_mig_cache.end();
    if (it == g_mig_cache.end()) return;
    MigCache &c = it->second;
    mig_plan_drop(c);
    MigBuf *b[] = { &c.owned_tab[0], &c.owned_tab[1], &c.oq, &c.oflag,
                    &c.ogone, &c.rrow, &c.rgid, &c.rarity, &c.rdist, &c.rsend,
                    &c.send, &c.recv, &c.stage,
                    &c.gt, &c.rg, &c.fa, &c.fc,
                    &c.compact, &c.query, &c.nhit, &c.keep, &c.cnt,
                    &c.gat, &c.mat, &c.n_off, &c.n_kind, &c.n_gid, &c.n_cell,
                    &c.arr, &c.layout };
    for (MigBuf *x : b) mig_release(&x->p);
    MigHost *h[] = { &c.hsend, &c.hval, &c.hrecv, &c.howned, &c.hrigid };
    for (MigHost *x : h) if (x->p) cudaFreeHost(x->p);
    if (c.vote) gcx_wsum_destroy(c.vote);
    if (c.vote_words) mig_release((void **)&c.vote_words);
    if (c.producer) cudaEventDestroy(c.producer);
    if (c.consumer) cudaEventDestroy(c.consumer);
    g_mig_cache.erase(it);
}

/* Blocks of GCN_BLOCK threads for n items, at least 1, at most GCN_MAX_BLOCKS. */
static unsigned mig_grid(gc_i64 n)
{
    gc_i64 g = (n + GCN_BLOCK - 1) / GCN_BLOCK;
    if (g > GCN_MAX_BLOCKS) g = GCN_MAX_BLOCKS;
    if (g < 1) g = 1;
    return (unsigned)g;
}

template <class T>
__global__ void gcn_kern_mig_copy_strided(T *__restrict__ dst,
                                          gc_i64 dpitch,
                                          const T *__restrict__ src,
                                          gc_i64 spitch, gc_i64 slots)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
         i < 3 * slots; i += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 c = i / slots;
        const gc_i64 s = i - c * slots;
        dst[c * dpitch + s] = src[c * spitch + s];
    }
}

namespace {

/* Reallocate one slot-indexed array into `dpitch`; the old storage joins
 * `old`, freed once the stream has drained every copy (one
 * synchronisation for the whole reserve).  `comps` is 1 for a per-slot
 * array and 3 for a component-major one of E elements (f64, or the f32
 * of a gcn_vf field under vf32) in T-sized storage. */
template <typename T, typename E = gc_f64>
gc_status mig_grow_slots(const gc_context *ctx, T **field, gc_i64 dpitch,
                         gc_i64 spitch, gc_i64 slots, gc_i64 comps,
                         std::vector<void *> &old)
{
    void *np = 0;
    gc_status st = mig_alloc(&np, comps * dpitch * (gc_i64)sizeof(T));
    if (st != GC_OK) return st;
    if (*field != 0 && slots > 0) {
        if (comps == 3) {
            gcn_kern_mig_copy_strided<E><<<mig_grid(slots), GCN_BLOCK, 0,
                                           ctx->native->stream>>>(
                (E *)np, dpitch, (const E *)*field, spitch, slots);
        } else if (cudaMemcpyAsync(np, (const void *)*field,
                                   (size_t)(slots * (gc_i64)sizeof(T)),
                                   cudaMemcpyDeviceToDevice,
                                   ctx->native->stream) != cudaSuccess) {
            mig_release(&np);
            return GC_E_DEVICE;
        }
    }
    if (*field) old.push_back((void *)*field);
    *field = (T *)np;
    return GC_OK;
}

/* A velocity or force field (gcn_vf), storage sized for doubles */
gc_status mig_grow_vf(const gc_context *ctx, gcn_vf *field, gc_i64 dpitch,
                      gc_i64 spitch, gc_i64 slots, std::vector<void *> &old)
{
    gc_f64 *p = field->as<gc_f64>();
    const gc_status st = ctx->native->vf32
        ? mig_grow_slots<gc_f64, float>(ctx, &p, dpitch, spitch, slots, 3, old)
        : mig_grow_slots<gc_f64, gc_f64>(ctx, &p, dpitch, spitch, slots, 3,
                                         old);
    field->p = p;
    return st;
}

/* A flat array regrown to `count` elements, its first `held` kept. */
template <typename T>
gc_status mig_grow_flat(const gc_context *ctx, T **field, gc_i64 count,
                        gc_i64 held, std::vector<void *> &old)
{
    return mig_grow_slots(ctx, field, count, 0, *field ? held : 0, 1, old);
}

/* The growth native_migration_reserve asks for; old storage joins `old`. */
gc_status mig_reserve_grow(gc_context *ctx, struct gcn_device *d,
                           gc_i64 owned, gc_i64 groups, gc_i64 members,
                           gc_i64 resident, gc_i64 old_pitch, gc_i64 slots,
                           std::vector<void *> &old)
{
    if (resident > d->pitch) {
        gc_i64 np = old_pitch > 0 ? old_pitch : 128;
        while (np < resident) np = np * 2 + 128;
        gc_status st;
#define MIG_GROW3(field)                                                     \
        do {                                                                 \
            st = mig_grow_slots(ctx, &d->field, np, old_pitch, slots, 3, old);\
            if (st != GC_OK) return st;                                      \
        } while (0)
#define MIG_GROW1(field)                                                     \
        do {                                                                 \
            st = mig_grow_slots(ctx, &d->field, np, old_pitch, slots, 1, old);\
            if (st != GC_OK) return st;                                      \
        } while (0)
        MIG_GROW3(coord);
        MIG_GROW3(force_coord);
        MIG_GROW3(coord_ref);
        MIG_GROW3(list_ref);
        st = mig_grow_vf(ctx, &d->vel, np, old_pitch, slots, old);
        if (st != GC_OK) return st;
        MIG_GROW3(vel_ref);
        MIG_GROW3(vel_half);
        MIG_GROW3(vel_full);
        st = mig_grow_vf(ctx, &d->force, np, old_pitch, slots, old);
        if (st != GC_OK) return st;
        MIG_GROW3(force_real);
        MIG_GROW3(force_bond);
        MIG_GROW3(force_recip);
        MIG_GROW3(scratch3);
        st = mig_grow_slots(ctx, &d->force_real_fx, np, old_pitch, 0, 3, old);
        if (st != GC_OK) return st;
        st = mig_grow_slots(ctx, &d->force_bond_fx, np, old_pitch, 0, 3, old);
        if (st != GC_OK) return st;
        d->fx_clear_pitch[0] = d->fx_clear_pitch[1] = 0;
        for (gc_f64 **f : { &d->coord2, &d->coord_ref2, &d->vel_ref2,
                            &d->vel_half2, &d->vel_full2 })
            if ((st = mig_grow_slots(ctx, f, np, old_pitch, 0, 3, old)) != GC_OK)
                return st;
        if ((st = mig_grow_vf(ctx, &d->vel2, np, old_pitch, 0, old)) != GC_OK)
            return st;
        for (gc_f64 **f : { &d->charge2, &d->mass2, &d->inv_mass2 })
            if ((st = mig_grow_slots(ctx, f, np, old_pitch, 0, 1, old)) != GC_OK)
                return st;
        if ((st = mig_grow_slots(ctx, &d->cls2, np, old_pitch, 0, 1, old)) != GC_OK ||
            (st = mig_grow_slots(ctx, &d->gid2, np, old_pitch, 0, 1, old)) != GC_OK)
            return st;
        MIG_GROW1(charge);
        MIG_GROW1(mass);
        MIG_GROW1(inv_mass);
        MIG_GROW1(cls);
        MIG_GROW1(gid);
        MIG_GROW1(image);
        MIG_GROW1(owner);
        MIG_GROW1(cell_of);
        MIG_GROW1(perm);
#undef MIG_GROW1
#undef MIG_GROW3
        d->pitch = np;
    }

    if (resident > d->resident_cap) {
        gc_i64 rcap = d->resident_cap > 0 ? d->resident_cap : 128;
        while (rcap < resident) rcap = rcap * 2 + 128;
        /* A run index: the publication path refills it from cell_of. */
        gc_status st = mig_grow_slots(ctx, &d->resident_slot, rcap,
                                      d->resident_cap, 0, 1, old);
        if (st != GC_OK) return st;
        d->resident_cap = rcap;
    }

    if (owned > d->owned_cap) {
        gc_i64 oc = d->owned_cap > 0 ? d->owned_cap : 128;
        while (oc < owned) oc = oc * 2 + 128;
        gc_status gst = mig_grow_slots(ctx, &d->migration_atom_send, oc,
                                       d->owned_cap, 0, 1, old);
        if (gst != GC_OK) return gst;
        d->owned_cap = oc;
    }

    if (groups + 1 > d->group_cap) {
        gc_i64 gc_cap = d->group_cap > 0 ? d->group_cap : 8;
        while (gc_cap < groups + 1) gc_cap = gc_cap * 2 + 8;
        gc_status st;
        /* `held` is how many entries of the old array are live: the two
         * offset arrays publish a sentinel at [group_cap], and dropping it
         * would make the next classification read a run that ends at 0. */
#define MIG_GROUP_FLAT(field, type, count, factor, held)                     \
        do {                                                                 \
            st = mig_grow_flat(ctx, &d->field, (gc_i64)(count),              \
                               d->group_cap > 0 ? (gc_i64)(held) * (factor)  \
                                                : 0, old);                   \
            if (st != GC_OK) return st;                                      \
        } while (0)
        MIG_GROUP_FLAT(group_offset, gc_i64, gc_cap + 1, 1, d->group_cap + 1);
        MIG_GROUP_FLAT(group_offset2, gc_i64, gc_cap + 1, 1, d->group_cap + 1);
        MIG_GROUP_FLAT(group_kind, gc_u8, gc_cap, 1, d->group_cap);
        MIG_GROUP_FLAT(group_kind2, gc_u8, gc_cap, 1, d->group_cap);
        MIG_GROUP_FLAT(group_gid, gc_gid, gc_cap, 1, d->group_cap);
        MIG_GROUP_FLAT(group_gid2, gc_gid, gc_cap, 1, d->group_cap);
        MIG_GROUP_FLAT(group_cell, gc_i32, gc_cap, 1, d->group_cap);
        MIG_GROUP_FLAT(group_cell2, gc_i32, gc_cap, 1, d->group_cap);
        MIG_GROUP_FLAT(group_dest_rank, gc_i32, gc_cap, 1, d->group_cap);
        MIG_GROUP_FLAT(group_dest_coord, gc_i32, 3 * gc_cap, 3, d->group_cap);
        MIG_GROUP_FLAT(migration_group_offset, gc_i64, gc_cap, 1, d->group_cap);
        MIG_GROUP_FLAT(migration_atom_offset, gc_i64, gc_cap, 1, d->group_cap);
        MIG_GROUP_FLAT(migration_group_send, gcn_group_migration_wire,
                       gc_cap, 1, d->group_cap);
        MIG_GROUP_FLAT(group_force_move, gc_f64, 3 * gc_cap, 3, d->group_cap);
#undef MIG_GROUP_FLAT
        d->group_cap = gc_cap;
    }

    if (members > d->member_cap) {
        gc_i64 mc = d->member_cap > 0 ? d->member_cap : 8;
        while (mc < members) mc = mc * 2 + 8;
        gc_status st = mig_grow_flat(ctx, &d->group_member, mc,
                                     d->member_cap, old);
        if (st == GC_OK)
            st = mig_grow_flat(ctx, &d->group_member2, mc, d->member_cap, old);
        if (st != GC_OK) return st;
        d->member_cap = mc;
    }

    {
        const gc_i64 want = groups > resident ? groups : resident;
        if (want > d->sort_cap) {
            gc_i64 sc = d->sort_cap > 0 ? d->sort_cap : 128;
            while (sc < want) sc = sc * 2 + 128;
            gc_status st;
#define MIG_SORT_FLAT(field, type)                                           \
            do {                                                             \
                st = mig_grow_flat(ctx, &d->field, sc, d->sort_cap, old);    \
                if (st != GC_OK) return st;                                  \
            } while (0)
            MIG_SORT_FLAT(sort_key_hi, gc_u64);
            MIG_SORT_FLAT(sort_key_lo, gc_u64);
            MIG_SORT_FLAT(sort_val, gc_i32);
            MIG_SORT_FLAT(sort_key_hi2, gc_u64);
            MIG_SORT_FLAT(sort_key_lo2, gc_u64);
            MIG_SORT_FLAT(sort_val2, gc_i32);
#undef MIG_SORT_FLAT
            d->sort_cap = sc;
            d->sort_capacity = sc;
        }
    }

    if (d->num_owned > d->pitch || d->num_resident > d->resident_cap)
        return GC_E_STATE;
    return GC_OK;
}

}  /* anonymous namespace */

gc_status native_migration_reserve(gc_context *ctx, gc_i64 owned,
                                   gc_i64 groups, gc_i64 members,
                                   gc_i64 resident)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (!ctx || !d) return GC_E_ARG;
    if (owned < 0 || groups < 0 || members < 0 || resident < 0 ||
        resident < owned)
        return GC_E_ARG;
    if (groups + 1 > (gc_i64)GC_MAX_LOCAL_INDEX ||
        members > (gc_i64)GC_MAX_LOCAL_INDEX ||
        resident > (gc_i64)GC_MAX_LOCAL_INDEX)
        return GC_E_CAPACITY;

    const gc_i64 old_pitch = d->pitch;
    const gc_i64 slots = d->num_resident;   /* live slots to preserve */
    std::vector<void *> old;
    const gc_status st = mig_reserve_grow(ctx, d, owned, groups, members,
                                          resident, old_pitch, slots, old);
    const gc_status ds = old.empty() ? GC_OK :
                         dev_phase(d->stream, "migration_reserve");
    for (void *p : old) mig_release(&p);
    if (st != GC_OK) return st;
    return ds;
}

/* Compaction: every live slot moves to its published index.  The new CSR is
 * in published order, so member entry m names the old slot that becomes slot
 * m; slot-indexed fields are gathered as columns, bit for bit. */
#define GCN_MIG_COLS 16
struct gcn_mig_cols {
    void *col[GCN_MIG_COLS];
    gc_i32 wide[GCN_MIG_COLS];      /* 8-byte elements, else 4-byte */
    gc_i32 n;
};

__global__ void gcn_kern_mig_gather_cols(struct gcn_mig_cols c,
                                         const gc_i32 *__restrict__ perm,
                                         gc_i64 n, gc_u64 *__restrict__ tmp)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 o = perm[s];
        for (int j = 0; j < c.n; ++j)
            tmp[(gc_i64)j * n + s] =
                c.wide[j] ? ((const gc_u64 *)c.col[j])[o]
                          : (gc_u64)((const uint32_t *)c.col[j])[o];
    }
}

__global__ void gcn_kern_mig_commit_cols(struct gcn_mig_cols c, gc_i64 n,
                                         const gc_u64 *__restrict__ tmp)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x)
        for (int j = 0; j < c.n; ++j) {
            const gc_u64 v = tmp[(gc_i64)j * n + s];
            if (c.wide[j]) ((gc_u64 *)c.col[j])[s] = v;
            else ((uint32_t *)c.col[j])[s] = (uint32_t)v;
        }
}

template <class S>
__global__ void gcn_kern_mig_install_atoms(
    const gcn_atom_migration_wire *__restrict__ w, gc_i64 n, gc_i64 base,
    const gc_i32 *__restrict__ cell_in, gc_i64 pitch, gc_i32 owner_rank,
    gc_f64 *__restrict__ coord, gc_f64 *__restrict__ coord_ref,
    gc_f64 *__restrict__ list_ref, S *__restrict__ vel,
    gc_f64 *__restrict__ vel_ref, gc_f64 *__restrict__ vel_half,
    gc_f64 *__restrict__ vel_full, S *__restrict__ force,
    gc_f64 *__restrict__ charge, gc_f64 *__restrict__ mass,
    gc_f64 *__restrict__ inv_mass, gc_i32 *__restrict__ cls,
    gc_gid *__restrict__ gid, gc_image *__restrict__ image,
    gc_i32 *__restrict__ owner, gc_i32 *__restrict__ cell_of)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 s = base + i;
        const gcn_atom_migration_wire &a = w[i];
        for (int k = 0; k < 3; ++k) {
            const gc_i64 at = (gc_i64)k * pitch + s;
            coord[at]     = a.coord[k];
            coord_ref[at] = a.coord_ref[k];
            list_ref[at]  = a.list_ref[k];
            vel[at]       = a.vel[k];
            vel_ref[at]   = a.vel_ref[k];
            vel_half[at]  = a.vel_half[k];
            vel_full[at]  = a.vel_full[k];
            force[at]     = a.force[k];
        }
        charge[s]   = a.charge;
        mass[s]     = a.mass;
        inv_mass[s] = a.inv_mass;
        cls[s]      = a.cls;
        gid[s]      = a.gid;
        image[s]    = 0;
        owner[s]    = owner_rank;
        cell_of[s]  = cell_in[i];
    }
}

/* The survivors of the published groups: a group whose owner is still
 * this rank keeps its members, a departing one keeps none. */
__global__ void gcn_kern_mig_keep(const gc_i32 *__restrict__ dest,
                                  const gc_i64 *__restrict__ off, gc_i64 ng,
                                  gc_i32 self, gc_i32 *__restrict__ keep,
                                  gc_i64 *__restrict__ cnt)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ng;
         g += (gc_i64)gridDim.x * blockDim.x) {
        keep[g] = dest[g] == self ? 1 : 0;
        cnt[g] = keep[g] ? off[g + 1] - off[g] : 0;
    }
}

/* Published groups whose GID is in the sorted arrival list q. */
__global__ void gcn_kern_mig_collide(const gc_gid *__restrict__ ggid,
                                     gc_i64 ng, const gc_gid *__restrict__ q,
                                     gc_i64 nq, gc_i64 *__restrict__ hits)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ng;
         g += (gc_i64)gridDim.x * blockDim.x) {
        gc_i64 lo = 0, hi = nq;
        while (lo < hi) {
            const gc_i64 mid = (lo + hi) / 2;
            if (q[mid] < ggid[g]) lo = mid + 1; else hi = mid;
        }
        if (lo < nq && q[lo] == ggid[g]) GCN_MIG_BUMP(hits);
    }
}

/* The new CSR's survivors, in published order at their scan index, and
 * perm: the old slot of each new member slot. */
__global__ void gcn_kern_mig_survivors(
    const gc_i32 *__restrict__ keep, const gc_i64 *__restrict__ gat,
    const gc_i64 *__restrict__ mat, const gc_i64 *__restrict__ off,
    const gc_i32 *__restrict__ mem, const gc_u8 *__restrict__ kind,
    const gc_gid *__restrict__ gid, const gc_i32 *__restrict__ cell,
    gc_i64 ng, gc_i64 *__restrict__ n_off, gc_u8 *__restrict__ n_kind,
    gc_gid *__restrict__ n_gid, gc_i32 *__restrict__ n_cell,
    gc_i32 *__restrict__ perm)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ng;
         g += (gc_i64)gridDim.x * blockDim.x) {
        if (!keep[g]) continue;
        const gc_i64 o = gat[g], m = mat[g];
        n_off[o] = m;
        n_kind[o] = kind[g];
        n_gid[o] = gid[g];
        n_cell[o] = cell[g];
        for (gc_i64 j = off[g]; j < off[g + 1]; ++j) perm[m + j - off[g]] = mem[j];
    }
}

/* One arriving group: its members are the appended slots [base, base +
 * count), listed after the survivors' at member offset msurv + moff. */
struct gcn_mig_arrival {
    gc_gid gid;
    gc_i64 base, moff;
    gc_i32 count, cell, kind, pad;
};

__global__ void gcn_kern_mig_arrivals(const gcn_mig_arrival *__restrict__ a,
                                      gc_i64 na, gc_i64 nsurv, gc_i64 msurv,
                                      gc_i64 members,
                                      gc_i64 *__restrict__ n_off,
                                      gc_u8 *__restrict__ n_kind,
                                      gc_gid *__restrict__ n_gid,
                                      gc_i32 *__restrict__ n_cell,
                                      gc_i32 *__restrict__ perm)
{
    const gc_i64 t0 = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t0 == 0) n_off[nsurv + na] = members;
    for (gc_i64 i = t0; i < na; i += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 o = nsurv + i, m = msurv + a[i].moff;
        n_off[o] = m;
        n_kind[o] = (gc_u8)a[i].kind;
        n_gid[o] = a[i].gid;
        n_cell[o] = a[i].cell;
        for (gc_i32 j = 0; j < a[i].count; ++j) perm[m + j] = (gc_i32)(a[i].base + j);
    }
}

__global__ void gcn_kern_mig_iota(gc_i32 *__restrict__ p, gc_i64 n)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x)
        p[s] = (gc_i32)s;
}

/* One piece of a parcel: `words` 8-byte words from src to dst. */
struct gcn_mig_piece {
    const gc_u64 *src;
    gc_u64 *dst;
    gc_i64 words;
};

/* Every piece of every parcel in one launch: piece blockIdx.y, its words
 * over the blocks in x. */
__global__ void gcn_kern_mig_gather(const gcn_mig_piece *__restrict__ piece)
{
    const gcn_mig_piece pc = piece[blockIdx.y];
    for (gc_i64 w = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; w < pc.words;
         w += (gc_i64)gridDim.x * blockDim.x)
        pc.dst[w] = pc.src[w];
}

/* The approved publication on the device.  The departing groups are this
 * rank's own movers -- exactly the published groups whose classified owner
 * is another rank -- and the survivors keep their order and slots; the
 * arrivals append.  Every check runs before anything is mutated. */
gc_status native_migration_apply(gc_context *ctx,
                                 const gc_gid *depart_gid, gc_i64 ndepart,
                                 const gcn_group_migration_wire *arrive_group,
                                 gc_i64 narrive_group,
                                 const gcn_atom_migration_wire *d_arrive_atom,
                                 gc_i64 narrive_atom,
                                 const gc_i32 *arrive_atom_cell,
                                 gc_i64 *moved_atoms_out)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (!ctx || !d) return GC_E_ARG;
    if (moved_atoms_out) *moved_atoms_out = 0;
    if (ndepart < 0 || narrive_group < 0 || narrive_atom < 0)
        return GC_E_ARG;
    if (ndepart > 0 && !depart_gid) return GC_E_ARG;
    if (narrive_group > 0 && !arrive_group) return GC_E_ARG;
    if (narrive_atom > 0 && (!d_arrive_atom || !arrive_atom_cell))
        return GC_E_ARG;

    const gc_i64 owned_old = d->num_owned;
    const gc_i64 groups_old = d->num_groups;
    const gc_i64 members_old = d->num_members;
    if (groups_old <= 0 || members_old != owned_old) return GC_E_STATE;
    MigCache &mc = g_mig_cache[d];
    cudaStream_t s = d->stream;

    /* The arrivals (host, small): distinct positive GIDs, a kind, a run
     * inside the shipment, a destination cell in this rank's box. */
    std::vector<gcn_mig_arrival> arr((size_t)narrive_group);
    std::vector<gc_gid> agid((size_t)narrive_group);
    gc_i64 amem = 0;
    for (gc_i64 i = 0; i < narrive_group; ++i) {
        const gcn_group_migration_wire &g = arrive_group[i];
        if (g.schema != GCN_MIGRATION_RECORD_SCHEMA ||
            g.destination_rank != ctx->rank || g.source_rank == ctx->rank ||
            g.group_gid <= 0 || g.atom_count <= 0 || g.atom_offset < 0 ||
            g.atom_offset + g.atom_count > narrive_atom ||
            g.kind < 0 || g.kind > GC_GROUP_HGROUP)
            return GC_E_MISMATCH;
        const gc_i32 box = gcn_box_index(d->layout, g.destination_cell[0],
                                        g.destination_cell[1],
                                        g.destination_cell[2]);
        if (box < 0 || !gcn_box_owned(d->layout, box)) return GC_E_OWNER;
        gcn_mig_arrival &a = arr[(size_t)i];
        std::memset(&a, 0, sizeof(a));
        a.gid = g.group_gid;
        a.base = owned_old + g.atom_offset;
        a.moff = amem;
        a.count = (gc_i32)g.atom_count;
        a.cell = box;
        a.kind = g.kind;
        amem += g.atom_count;
        agid[(size_t)i] = g.group_gid;
    }
    if (amem != narrive_atom) return GC_E_MISMATCH;
    std::sort(agid.begin(), agid.end());
    if (std::adjacent_find(agid.begin(), agid.end()) != agid.end())
        return GC_E_MISMATCH;

    /* Survivors and their member offsets (scans), and any arrival that
     * would republish a group this rank holds. */
    const gc_i64 ng = groups_old;
    void *keep = 0, *cnt = 0, *gat = 0, *mat = 0, *dq = 0, *da = 0;
    gc_status st = mig_buf(mc.keep, ng * (gc_i64)sizeof(gc_i32), 0, &keep);
    if (st == GC_OK) st = mig_buf(mc.cnt, ng * (gc_i64)sizeof(gc_i64), 0, &cnt);
    if (st == GC_OK) st = mig_buf(mc.gat, (ng + 1) * (gc_i64)sizeof(gc_i64), 0, &gat);
    if (st == GC_OK) st = mig_buf(mc.mat, (ng + 1) * (gc_i64)sizeof(gc_i64), 0, &mat);
    if (st == GC_OK) st = mig_buf(mc.nhit, sizeof(gc_i64), 1, &dq);
    if (st == GC_OK && narrive_group > 0)
        st = mig_buf(mc.query, narrive_group * (gc_i64)sizeof(gc_gid), 0, &da);
    if (st == GC_OK && narrive_group > 0 &&
        cudaMemcpyAsync(da, agid.data(), agid.size() * sizeof(gc_gid),
                        cudaMemcpyHostToDevice, s) != cudaSuccess)
        st = GC_E_DEVICE;
    if (st != GC_OK) return st;
    gcn_kern_mig_keep<<<mig_grid(ng), GCN_BLOCK, 0, s>>>(
        d->group_dest_rank, d->group_offset, ng, ctx->rank, (gc_i32 *)keep,
        (gc_i64 *)cnt);
    st = native_scan(d, (const gc_i32 *)keep, (gc_i64 *)gat, ng,
                     (gc_i64 *)gat + ng);
    if (st == GC_OK)
        st = native_scan64(d, (const gc_i64 *)cnt, (gc_i64 *)mat, ng,
                           (gc_i64 *)mat + ng);
    if (st != GC_OK) return st;
    if (narrive_group > 0)
        gcn_kern_mig_collide<<<mig_grid(ng), GCN_BLOCK, 0, s>>>(
            d->group_gid, ng, (const gc_gid *)da, narrive_group, (gc_i64 *)dq);
    gc_i64 *pin = native_rebuild_pin(d);
    if (!pin) return GC_E_NOMEM;
    if (cudaMemcpyAsync(pin + 0, (gc_i64 *)gat + ng, sizeof(gc_i64),
                        cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaMemcpyAsync(pin + 1, (gc_i64 *)mat + ng, sizeof(gc_i64),
                        cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaMemcpyAsync(pin + 2, dq, sizeof(gc_i64),
                        cudaMemcpyDeviceToHost, s) != cudaSuccess)
        return GC_E_DEVICE;
    st = dev_phase(s, "migration_plan");
    if (st != GC_OK) return st;
    const gc_i64 nsurv = pin[0], msurv = pin[1];
    if (pin[2] != 0 || nsurv != ng - ndepart || msurv < 0 ||
        msurv > members_old)
        return GC_E_MISMATCH;
    const gc_i64 new_groups = nsurv + narrive_group;
    const gc_i64 new_members = msurv + narrive_atom;
    const gc_i64 new_owned = new_members;
    const gc_i64 departed_members = owned_old - msurv;

    /* Capacity, then mutate: the new CSR is built in staging, then
     * published; perm names each new slot's old slot. */
    st = native_migration_reserve(ctx, new_owned, new_groups, new_members,
                                  new_owned + d->num_ghost);
    if (st != GC_OK) return st;
    void *n_off = 0, *n_kind = 0, *n_gid = 0, *n_cell = 0, *d_arr = 0;
    st = mig_buf(mc.n_off, (new_groups + 1) * (gc_i64)sizeof(gc_i64), 0, &n_off);
    if (st == GC_OK) st = mig_buf(mc.n_kind, new_groups * (gc_i64)sizeof(gc_u8), 0, &n_kind);
    if (st == GC_OK) st = mig_buf(mc.n_gid, new_groups * (gc_i64)sizeof(gc_gid), 0, &n_gid);
    if (st == GC_OK) st = mig_buf(mc.n_cell, new_groups * (gc_i64)sizeof(gc_i32), 0, &n_cell);
    if (st == GC_OK && narrive_group > 0)
        st = mig_buf(mc.arr, narrive_group * (gc_i64)sizeof(gcn_mig_arrival), 0, &d_arr);
    if (st == GC_OK && narrive_group > 0 &&
        cudaMemcpyAsync(d_arr, arr.data(), arr.size() * sizeof(gcn_mig_arrival),
                        cudaMemcpyHostToDevice, s) != cudaSuccess)
        st = GC_E_DEVICE;
    if (st != GC_OK) return st;
    gcn_kern_mig_survivors<<<mig_grid(ng), GCN_BLOCK, 0, s>>>(
        (const gc_i32 *)keep, (const gc_i64 *)gat, (const gc_i64 *)mat,
        d->group_offset, d->group_member, d->group_kind, d->group_gid,
        d->group_cell, ng, (gc_i64 *)n_off, (gc_u8 *)n_kind, (gc_gid *)n_gid,
        (gc_i32 *)n_cell, d->perm);
    gcn_kern_mig_arrivals<<<mig_grid(narrive_group), GCN_BLOCK, 0, s>>>(
        (const gcn_mig_arrival *)d_arr, narrive_group, nsurv, msurv,
        new_members, (gc_i64 *)n_off, (gc_u8 *)n_kind, (gc_gid *)n_gid,
        (gc_i32 *)n_cell, d->perm);
    if (cudaMemcpyAsync(d->group_offset, n_off,
                        (size_t)(new_groups + 1) * sizeof(gc_i64),
                        cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(d->group_kind, n_kind, (size_t)new_groups * sizeof(gc_u8),
                        cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(d->group_gid, n_gid, (size_t)new_groups * sizeof(gc_gid),
                        cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(d->group_cell, n_cell, (size_t)new_groups * sizeof(gc_i32),
                        cudaMemcpyDeviceToDevice, s) != cudaSuccess)
        return GC_E_DEVICE;

    if (narrive_atom > 0) {
        vf_each(d->vf32, [&](auto z) {
            using S = decltype(z);
            gcn_kern_mig_install_atoms<S><<<mig_grid(narrive_atom), GCN_BLOCK,
                                            0, d->stream>>>(
                d_arrive_atom, narrive_atom, owned_old, arrive_atom_cell,
                d->pitch, ctx->rank, d->coord, d->coord_ref, d->list_ref,
                d->vel.as<S>(), d->vel_ref, d->vel_half, d->vel_full,
                d->force.as<S>(), d->charge, d->mass, d->inv_mass, d->cls,
                d->gid, d->image, d->owner, d->cell_of);
        });
        st = dev_launched("migration_install");
        if (st != GC_OK) return st;
    }

    /* Compact: gather every slot-indexed field through the CSR's member
     * permutation, then rewrite the members as the identity (idempotent for
     * the publication's re-sort). */
    if (new_owned > 0) {
        const gc_i64 p = d->pitch;
        const int vw = d->vf32 ? 0 : 1;
        struct { void *p; int wide; } const vec[] = {
            { d->coord, 1 }, { d->force_coord, 1 }, { d->coord_ref, 1 },
            { d->list_ref, 1 }, { d->vel.p, vw }, { d->vel_ref, 1 },
            { d->vel_half, 1 }, { d->vel_full, 1 }, { d->force.p, vw },
            { d->force_real, 1 }, { d->force_bond, 1 }, { d->force_recip, 1 },
        };
        struct { void *p; int wide; } const one[] = {
            { d->charge, 1 }, { d->mass, 1 }, { d->inv_mass, 1 },
            { d->cls, 0 }, { d->gid, 1 }, { d->image, 1 }, { d->owner, 0 },
            { d->cell_of, 0 },
        };
        static_assert(sizeof(gc_f64) == 8 && sizeof(gc_gid) == 8 &&
                      sizeof(gc_image) == 8 && sizeof(gc_i32) == 4,
                      "the column widths");
        const int nvec = (int)(sizeof(vec) / sizeof(vec[0]));
        const int ncol = 3 * nvec + (int)(sizeof(one) / sizeof(one[0]));
        void *tmp = 0;
        st = mig_buf(g_mig_cache[d].compact,
                     GCN_MIG_COLS * new_owned * (gc_i64)sizeof(gc_u64), 0, &tmp);
        if (st != GC_OK) return st;
        const unsigned g1 = mig_grid(new_owned);
        struct gcn_mig_cols c;
        c.n = 0;
        for (int j = 0; j < ncol; ++j) {
            if (j < 3 * nvec) {
                const int w = vec[j / 3].wide;
                c.col[c.n] = (char *)vec[j / 3].p +
                             (gc_i64)(j % 3) * p * (w ? 8 : 4);
                c.wide[c.n] = w;
            } else {
                c.col[c.n] = one[j - 3 * nvec].p;
                c.wide[c.n] = one[j - 3 * nvec].wide;
            }
            if (++c.n < GCN_MIG_COLS && j + 1 < ncol) continue;
            gcn_kern_mig_gather_cols<<<g1, GCN_BLOCK, 0, s>>>(c,
                d->perm, new_owned, (gc_u64 *)tmp);
            gcn_kern_mig_commit_cols<<<g1, GCN_BLOCK, 0, s>>>(c,
                new_owned, (const gc_u64 *)tmp);
            c.n = 0;
        }
        st = dev_launched("migration_compact");
        if (st != GC_OK) return st;
        gcn_kern_mig_iota<<<g1, GCN_BLOCK, 0, s>>>(d->group_member,
            new_owned);
        st = dev_launched("migration_members");
        if (st != GC_OK) return st;
    }

    d->num_groups = new_groups;
    d->num_members = new_members;
    d->num_owned = new_owned;
    d->num_resident = new_owned + d->num_ghost;
    if (moved_atoms_out) *moved_atoms_out = narrive_atom - departed_members;
    return GC_OK;
}

/* The first index of the sorted v[0..n) not below x. */
static __device__ __forceinline__ gc_i64 dev_lower(const gc_gid *v, gc_i64 n, gc_gid x)
{
    gc_i64 lo = 0, hi = n;
    while (lo < hi) {
        const gc_i64 mid = (lo + hi) / 2;
        if (v[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* The sorted table after a commit: the kept entries of `old` (less the
 * sorted `gone`, each found counted in *removed) and the sorted `came`,
 * each written at its merged position. */
__global__ void gcn_kern_mig_owned_keep(const gc_gid *__restrict__ old,
                                        gc_i64 n,
                                        const gc_gid *__restrict__ gone,
                                        gc_i64 ng,
                                        const gc_gid *__restrict__ came,
                                        gc_i64 nc, gc_gid *__restrict__ out,
                                        gc_i64 *__restrict__ removed)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gc_gid x = old[i];
        const gc_i64 lg = dev_lower(gone, ng, x);
        if (lg < ng && gone[lg] == x) { GCN_MIG_BUMP(removed); continue; }
        out[i - lg + dev_lower(came, nc, x)] = x;
    }
}

__global__ void gcn_kern_mig_owned_add(const gc_gid *__restrict__ old,
                                       gc_i64 n,
                                       const gc_gid *__restrict__ gone,
                                       gc_i64 ng,
                                       const gc_gid *__restrict__ came,
                                       gc_i64 nc, gc_gid *__restrict__ out)
{
    for (gc_i64 j = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; j < nc;
         j += (gc_i64)gridDim.x * blockDim.x) {
        const gc_gid y = came[j];
        out[j + dev_lower(old, n, y) - dev_lower(gone, ng, y)] = y;
    }
}

/* flag[i] = 1 when q[i] is in the sorted table. */
__global__ void gcn_kern_mig_member(const gc_gid *__restrict__ tab, gc_i64 n,
                                    const gc_gid *__restrict__ q, gc_i64 nq,
                                    gc_u8 *__restrict__ flag)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < nq;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 at = dev_lower(tab, n, q[i]);
        flag[i] = at < n && tab[at] == q[i];
    }
}

/* Which of the sorted GIDs q this rank owned before the transaction. */
static gc_status owned_flags(struct gcn_device *d, MigCache &mc,
                             const std::vector<gc_gid> &q,
                             std::vector<gc_u8> *flag)
{
    flag->assign(q.size(), 0);
    const gc_i64 nq = (gc_i64)q.size();
    if (nq == 0 || mc.owned_n <= 0) return GC_OK;
    void *dq = 0, *df = 0;
    char *hf = 0;
    gc_status st = mig_buf(mc.oq, nq * (gc_i64)sizeof(gc_gid), 0, &dq);
    if (st == GC_OK) st = mig_buf(mc.oflag, nq, 0, &df);
    if (st == GC_OK) st = mig_host(mc.hval, nq, &hf);
    if (st != GC_OK) return st;
    if (cudaMemcpyAsync(dq, q.data(), (size_t)nq * sizeof(gc_gid),
                        cudaMemcpyHostToDevice, d->stream) != cudaSuccess)
        return GC_E_DEVICE;
    gcn_kern_mig_member<<<mig_grid(nq), GCN_BLOCK, 0, d->stream>>>(
        (const gc_gid *)mc.owned_tab[mc.owned_cur].p, mc.owned_n,
        (const gc_gid *)dq, nq, (gc_u8 *)df);
    if (cudaMemcpyAsync(hf, df, (size_t)nq, cudaMemcpyDeviceToHost,
                        d->stream) != cudaSuccess)
        return GC_E_DEVICE;
    st = dev_phase(d->stream, "migration_owned_flags");
    if (st == GC_OK) std::memcpy(flag->data(), hf, (size_t)nq);
    return st;
}

/* The owned table of the published epoch: sorted departed and arrived GIDs
 * applied on the device, queued only; owned_verify adopts the result. */
static gc_status owned_commit(struct gcn_device *d, MigCache &mc,
                              const std::vector<gc_gid> &gone,
                              const std::vector<gc_gid> &came)
{
    const gc_i64 n = mc.owned_n, ng = (gc_i64)gone.size(),
                 nc = (gc_i64)came.size(), m = n - ng + nc;
    if (n < 0 || m != d->num_owned) return GC_E_MISMATCH;
    const gc_i64 G = (gc_i64)sizeof(gc_gid), in_b = 8 + (ng + nc) * G;
    void *din = 0, *out = 0;
    char *h = 0;
    gc_status st = mig_buf(mc.ogone, in_b, 0, &din);
    if (st == GC_OK) st = mig_buf(mc.owned_tab[1 - mc.owned_cur],
                                  m * (gc_i64)sizeof(gc_gid), 0, &out);
    if (st == GC_OK) st = mig_host(mc.howned, in_b, &h);
    if (st != GC_OK) return st;
    gc_i64 *pin = native_rebuild_pin(d);
    if (!pin) return GC_E_NOMEM;
    std::memset(h, 0, 8);
    if (ng > 0) std::memcpy(h + 8, gone.data(), (size_t)(ng * G));
    if (nc > 0) std::memcpy(h + 8 + ng * G, came.data(), (size_t)(nc * G));
    void *dr = din, *dg = (char *)din + 8, *dc = (char *)din + 8 + ng * G;
    if (cudaMemcpyAsync(din, h, (size_t)in_b, cudaMemcpyHostToDevice,
                        d->stream) != cudaSuccess)
        return GC_E_DEVICE;
    const gc_gid *old = (const gc_gid *)mc.owned_tab[mc.owned_cur].p;
    gcn_kern_mig_owned_keep<<<mig_grid(n), GCN_BLOCK, 0, d->stream>>>(
        old, n, (const gc_gid *)dg, ng, (const gc_gid *)dc, nc, (gc_gid *)out,
        (gc_i64 *)dr);
    gcn_kern_mig_owned_add<<<mig_grid(nc), GCN_BLOCK, 0, d->stream>>>(
        old, n, (const gc_gid *)dg, ng, (const gc_gid *)dc, nc, (gc_gid *)out);
    if (cudaMemcpyAsync(pin + 55, dr, sizeof(gc_i64), cudaMemcpyDeviceToHost,
                        d->stream) != cudaSuccess)
        return GC_E_DEVICE;
    mc.owned_gone = ng;
    mc.owned_next = m;
    return dev_launched("migration_owned_commit");
}

/* After owned_commit and a synchronisation of the stream (`synced`, else
 * one here): the new table is adopted when the device removed exactly the
 * departed GIDs. */
static gc_status owned_verify(struct gcn_device *d, MigCache &mc, int synced)
{
    const gc_i64 *pin = native_rebuild_pin(d);
    gc_status st = pin ? GC_OK : GC_E_NOMEM;
    if (st == GC_OK && !synced) st = dev_phase(d->stream, "migration_owned_commit");
    if (st == GC_OK && pin[55] != mc.owned_gone) st = GC_E_MISMATCH;
    if (st != GC_OK) return st;
    mc.owned_n = mc.owned_next;
    mc.owned_cur = 1 - mc.owned_cur;
    return GC_OK;
}

#define MIG_TAG_BASE     4096
#define MIG_TAG_SWAP     (MIG_TAG_BASE - 6)   /* two tags, below the plan's */
#define MIG_DEADLINE_MS  GCX_DEADLINE_MS

/* Remaining-room policy for admission: one edge may bring as many groups and
 * atoms as this rank holds (at least these floors). */
#define MIG_ROOM_GROUPS 1024
#define MIG_ROOM_ATOMS  4096

/* A rank's zero-bit rows are every stock zero bit naming one of its owned
 * atoms, the owned endpoint at the identity image (gpu_core_import_real_mask).
 * Rows travel with the atoms they name. */
typedef gcn_mask_row MaskRow;

/* Unordered identity of a row: the endpoints by GID and their relative
 * image, which no re-imaging changes. */
static std::pair<std::pair<gc_gid, gc_gid>, gc_image> mask_key(const MaskRow &r)
{
    return r.a < r.b ?
        std::make_pair(std::make_pair(r.a, r.b), image_sub(r.ib, r.ia)) :
        std::make_pair(std::make_pair(r.b, r.a), image_sub(r.ia, r.ib));
}

/* Rows naming a moved atom (moved: 1 = left, 2 = arrived), re-imaged so the
 * first owned endpoint is the identity copy, or dropped when none is owned;
 * then arrived rows, less duplicates.  owned(x): owned before the transaction. */
template <class Owned>
static void mask_follow_rows(const std::vector<MaskRow> &rows,
                             const KeyMap &moved, Owned owned,
                             const std::vector<MaskRow> &carried,
                             std::vector<MaskRow> *out)
{
    const auto owned_after = [&](gc_gid x) {
        const gc_i32 m = moved.find(x);
        return m == 2 || (m != 1 && owned(x));
    };
    const auto arriving = [&](const MaskRow &r) {
        return moved.find(r.a) == 2 || moved.find(r.b) == 2;
    };
    const auto norm = [&](MaskRow &r) {
        const bool oa = owned_after(r.a);
        if (!oa && !owned_after(r.b)) return false;
        if (oa) { r.ib = image_sub(r.ib, r.ia); r.ia = 0; }
        else    { r.ia = image_sub(r.ia, r.ib); r.ib = 0; }
        return true;
    };
    typedef std::pair<std::pair<gc_gid, gc_gid>, gc_image> Key;
    std::vector<Key> near;   /* kept rows naming an arriving atom */
    for (MaskRow r : rows) {
        if (!norm(r)) continue;
        if (arriving(r)) near.push_back(mask_key(r));
        out->push_back(r);
    }
    std::sort(near.begin(), near.end());
    for (MaskRow r : carried) {
        if (!arriving(r) || !norm(r)) continue;
        const Key k = mask_key(r);
        const auto at = std::lower_bound(near.begin(), near.end(), k);
        if (at != near.end() && *at == k) continue;
        near.insert(at, k);
        out->push_back(r);
    }
}

/* The host list after an approved transaction.  A list this function wrote
 * has no duplicates and is re-imaged: rows naming no moved atom stay, rows
 * naming one are replaced by mask_follow_rows' (appended), as on the device.
 * Any other list is normalized and deduplicated.  `owned` is the sorted
 * owned set before the transaction. */
static void real_mask_follow(gc_context *ctx, const std::vector<gc_gid> &owned,
                             const KeyMap &moved, const GidFilter &seen,
                             const std::vector<MaskRow> &carried)
{
    const auto owned_before = [&](gc_gid x) {
        return std::binary_search(owned.begin(), owned.end(), x);
    };
    std::vector<gc_gid> &g = ctx->real_mask_gid;
    std::vector<gc_image> &im = ctx->real_mask_image;
    im.resize(g.size(), 0);
    MigCache &mc = g_mig_cache[ctx->native];
    size_t w = 0;
    const auto put = [&](const MaskRow &r) {
        g[2 * w] = r.a;     im[2 * w] = r.ia;
        g[2 * w + 1] = r.b; im[2 * w + 1] = r.ib;
        ++w;
    };
    std::vector<MaskRow> named, out;
    if (mc.mask_rows == (gc_i64)g.size()) {
        for (size_t i = 0; i + 1 < g.size(); i += 2) {
            const MaskRow r = { g[i], im[i], g[i + 1], im[i + 1] };
            if ((seen.maybe(r.a) || seen.maybe(r.b)) &&
                (moved.find(r.a) >= 0 || moved.find(r.b) >= 0))
                named.push_back(r);
            else
                put(r);
        }
        mask_follow_rows(named, moved, owned_before, carried, &out);
        g.resize(2 * (w + out.size()));
        im.resize(g.size());
        for (const MaskRow &r : out) put(r);
        mc.mask_rows = (gc_i64)g.size();
        return;
    }
    for (size_t i = 0; i + 1 < g.size(); i += 2)
        named.push_back(MaskRow{ g[i], im[i], g[i + 1], im[i + 1] });
    const auto owned_after = [&](gc_gid x) {
        const gc_i32 m = moved.find(x);
        return m == 2 || (m != 1 && owned_before(x));
    };
    for (MaskRow r : named) {
        const bool oa = owned_after(r.a);
        if (!oa && !owned_after(r.b)) continue;
        if (oa) { r.ib = image_sub(r.ib, r.ia); r.ia = 0; }
        else    { r.ia = image_sub(r.ia, r.ib); r.ib = 0; }
        out.push_back(r);
    }
    for (MaskRow r : carried) {
        if (moved.find(r.a) != 2 && moved.find(r.b) != 2) continue;
        const bool oa = owned_after(r.a);
        if (oa) { r.ib = image_sub(r.ib, r.ia); r.ia = 0; }
        else    { r.ia = image_sub(r.ia, r.ib); r.ib = 0; }
        out.push_back(r);
    }
    typedef std::pair<std::pair<gc_gid, gc_gid>, gc_image> Key;
    std::vector<std::pair<Key, size_t> > key(out.size());
    for (size_t i = 0; i < out.size(); ++i) key[i] = std::make_pair(mask_key(out[i]), i);
    std::sort(key.begin(), key.end());
    std::vector<gc_u8> dup(out.size(), 0);
    for (size_t i = 1; i < key.size(); ++i)
        if (key[i].first == key[i - 1].first) dup[key[i].second] = 1;
    g.resize(2 * out.size());
    im.resize(g.size());
    for (size_t i = 0; i < out.size(); ++i) if (!dup[i]) put(out[i]);
    g.resize(2 * w);
    im.resize(2 * w);
    mc.mask_rows = (gc_i64)g.size();
}

/* A hydrogen group's bond distances travel with it, so the destination's
 * rigid table (sorted by representative GID) names every group it owns.  The
 * table is edited on the device at each commit; the checked build keeps a
 * host mirror (gc_context::rigid_rows) and compares. */
static bool rigid_less(const RigidRow &l, const RigidRow &r)
{
    return l.gid < r.gid;
}

static const RigidRow *rigid_find(const std::vector<RigidRow> &rows, gc_gid g)
{
    RigidRow key;
    key.gid = g;
    const auto it = std::lower_bound(rows.begin(), rows.end(), key, rigid_less);
    return it != rows.end() && it->gid == g ? &*it : 0;
}

/* The row of each moving hydrogen group in this rank's shipment, from the
 * sorted device table; any other record gets an empty row (arity 0). */
__global__ void gcn_kern_mig_rigid_rows(const gcn_group_migration_wire *__restrict__ w,
                                        gc_i64 room,
                                        const gc_i64 *__restrict__ peer_counts,
                                        gc_i32 nproc,
                                        const gc_gid *__restrict__ gid,
                                        const gc_i32 *__restrict__ arity,
                                        const gc_f64 *__restrict__ dist,
                                        gc_i64 n, RigidRow *__restrict__ out)
{
    gc_i64 ng = 0;
    for (gc_i32 p = 0; p < nproc; ++p) ng += peer_counts[3 * p];
    if (ng > room) ng = room;
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < ng;
         i += (gc_i64)gridDim.x * blockDim.x) {
        RigidRow r;
        r.gid = w[i].group_gid;
        r.arity = 0;
        r.pad0 = 0;
        for (int h = 0; h < GC_MAX_HGROUP_H; ++h) r.dist[h] = 0.0;
        if (w[i].kind == (gc_i32)GC_GROUP_HGROUP) {
            const gc_i64 at = dev_lower(gid, n, r.gid);
            if (at < n && gid[at] == r.gid) {
                r.arity = arity[at];
                for (int h = 0; h < GC_MAX_HGROUP_H; ++h)
                    r.dist[h] = dist[(gc_i64)GC_MAX_HGROUP_H * at + h];
            }
        }
        out[i] = r;
    }
}

/* This rank's vote words from its classification: [0] the groups and atoms
 * it ships, [1] 1 if a count is negative, [2 + rank] the largest parcel it
 * ships to one rank, every other word 0.  Summed over the ranks, [0] says
 * whether anything moves, [1] whether any rank refuses, and words 2.. give
 * each rank's parcel bound. */
__global__ void gcn_kern_mig_vote(const gc_i64 *__restrict__ peer_counts,
                                  gc_i32 nproc, gc_i32 rank,
                                  gc_u64 *__restrict__ words)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    gc_i64 moving = 0, bound = 0, bad = 0;
    for (gc_i32 p = 0; p < nproc; ++p) {
        const gc_i64 g = peer_counts[3 * p], a = peer_counts[3 * p + 1];
        if (g < 0 || a < 0) { bad = 1; continue; }
        moving += g + a;
        if (p != rank) {
            const gc_i64 b = g * (gc_i64)sizeof(gcn_group_migration_wire) +
                             a * (gc_i64)sizeof(gcn_atom_migration_wire);
            if (b > bound) bound = b;
        }
    }
    words[0] = (gc_u64)moving;
    words[1] = (gc_u64)bad;
    for (gc_i32 p = 0; p < nproc; ++p) words[2 + p] = p == rank ? (gc_u64)bound : 0;
}

/* The device table after a commit, merged like the owned-GID table: the
 * kept rows of the old table (less the sorted departed GIDs, each found
 * counted in bad[0]) and the sorted arriving rows, each at its merged
 * position; an arriving GID the old table keeps counts in bad[1]. */
__global__ void gcn_kern_mig_rigid_keep(const gc_gid *__restrict__ gid,
                                        const gc_i32 *__restrict__ arity,
                                        const gc_f64 *__restrict__ dist,
                                        gc_i64 n,
                                        const gc_gid *__restrict__ gone,
                                        gc_i64 ng,
                                        const gc_gid *__restrict__ came,
                                        gc_i64 nc,
                                        gc_gid *__restrict__ o_gid,
                                        gc_i32 *__restrict__ o_arity,
                                        gc_f64 *__restrict__ o_dist,
                                        gc_i64 *__restrict__ bad)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gc_gid x = gid[i];
        const gc_i64 lg = dev_lower(gone, ng, x);
        if (lg < ng && gone[lg] == x) { GCN_MIG_BUMP(bad); continue; }
        const gc_i64 lc = dev_lower(came, nc, x);
        if (lc < nc && came[lc] == x) GCN_MIG_BUMP(bad + 1);
        const gc_i64 o = i - lg + lc;
        o_gid[o] = x;
        o_arity[o] = arity[i];
        for (int h = 0; h < GC_MAX_HGROUP_H; ++h)
            o_dist[o * GC_MAX_HGROUP_H + h] = dist[i * GC_MAX_HGROUP_H + h];
    }
}

__global__ void gcn_kern_mig_rigid_add(const RigidRow *__restrict__ row,
                                       gc_i64 nc,
                                       const gc_gid *__restrict__ came,
                                       const gc_gid *__restrict__ gid,
                                       gc_i64 n,
                                       const gc_gid *__restrict__ gone,
                                       gc_i64 ng,
                                       gc_gid *__restrict__ o_gid,
                                       gc_i32 *__restrict__ o_arity,
                                       gc_f64 *__restrict__ o_dist)
{
    for (gc_i64 j = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; j < nc;
         j += (gc_i64)gridDim.x * blockDim.x) {
        const gc_gid y = came[j];
        const gc_i64 o = j + dev_lower(gid, n, y) - dev_lower(gone, ng, y);
        o_gid[o] = y;
        o_arity[o] = row[j].arity;
        for (int h = 0; h < GC_MAX_HGROUP_H; ++h)
            o_dist[o * GC_MAX_HGROUP_H + h] = row[j].dist[h];
    }
}

/* The destination's rigid table after a commit: departed HGROUP GIDs out,
 * arrived rows in, merged on the device; d->rigid_* keep their allocation. */
static gc_status rigid_rows_commit(struct gcn_device *d, MigCache &mc,
                                   const std::vector<gc_gid> &gone,
                                   const std::vector<RigidRow> &came_rows)
{
    const gc_i64 n = d->rigid_count, ng = (gc_i64)gone.size(),
                 nc = (gc_i64)came_rows.size(), m = n - ng + nc;
    const gc_i64 H = GC_MAX_HGROUP_H;
    if (m < 0) return GC_E_MISMATCH;
    std::vector<gc_gid> came((size_t)nc);
    for (gc_i64 j = 0; j < nc; ++j) came[(size_t)j] = came_rows[(size_t)j].gid;
    const gc_i64 G = (gc_i64)sizeof(gc_gid), R = (gc_i64)sizeof(RigidRow);
    const gc_i64 in_b = 16 + (ng + nc) * G + nc * R;
    static_assert(sizeof(RigidRow) % 8 == 0, "the rows follow 8-byte words");
    void *din = 0, *og = 0, *oa = 0, *od = 0;
    char *h = 0;
    gc_status st = mig_buf(mc.rrow, in_b, 0, &din);
    if (st == GC_OK) st = mig_buf(mc.rgid, m * (gc_i64)sizeof(gc_gid), 0, &og);
    if (st == GC_OK) st = mig_buf(mc.rarity, m * (gc_i64)sizeof(gc_i32), 0, &oa);
    if (st == GC_OK) st = mig_buf(mc.rdist, H * m * (gc_i64)sizeof(gc_f64), 0, &od);
    if (st == GC_OK) st = mig_host(mc.hrigid, in_b, &h);
    gc_i64 *pin = native_rebuild_pin(d);
    if (st == GC_OK && !pin) st = GC_E_NOMEM;
    if (st != GC_OK) return st;
    std::memset(h, 0, 16);
    if (ng > 0) std::memcpy(h + 16, gone.data(), (size_t)(ng * G));
    if (nc > 0) {
        std::memcpy(h + 16 + ng * G, came.data(), (size_t)(nc * G));
        std::memcpy(h + 16 + (ng + nc) * G, came_rows.data(), (size_t)(nc * R));
    }
    void *db = din, *dg = (char *)din + 16, *dc = (char *)din + 16 + ng * G,
         *dr = (char *)din + 16 + (ng + nc) * G;
    cudaStream_t s = d->stream;
    if (cudaMemcpyAsync(din, h, (size_t)in_b, cudaMemcpyHostToDevice, s) !=
            cudaSuccess)
        return GC_E_DEVICE;
    gcn_kern_mig_rigid_keep<<<mig_grid(n), GCN_BLOCK, 0, d->stream>>>(
        d->rigid_gid, d->rigid_arity, d->rigid_dist, n, (const gc_gid *)dg, ng,
        (const gc_gid *)dc, nc, (gc_gid *)og, (gc_i32 *)oa, (gc_f64 *)od,
        (gc_i64 *)db);
    gcn_kern_mig_rigid_add<<<mig_grid(nc), GCN_BLOCK, 0, d->stream>>>(
        (const RigidRow *)dr, nc, (const gc_gid *)dc, d->rigid_gid, n,
        (const gc_gid *)dg, ng, (gc_gid *)og, (gc_i32 *)oa, (gc_f64 *)od);
    if (cudaMemcpyAsync(pin + 52, db, 2 * sizeof(gc_i64), cudaMemcpyDeviceToHost,
                        s) != cudaSuccess)
        return GC_E_DEVICE;
    st = dev_phase(s, "migration_rigid_merge");
    if (st == GC_OK && (pin[52] != ng || pin[53] != 0)) st = GC_E_MISMATCH;
    if (st != GC_OK) return st;
    if (m > (d->rigid_gid == mc.rigid_owner ? mc.rigid_cap : n)) {
        const gc_i64 cap = m + m / 4 + 64;
        mig_release((void **)&d->rigid_gid);
        mig_release((void **)&d->rigid_arity);
        mig_release((void **)&d->rigid_dist);
        d->rigid_count = 0;
        st = mig_alloc((void **)&d->rigid_gid, cap * (gc_i64)sizeof(gc_gid));
        if (st == GC_OK)
            st = mig_alloc((void **)&d->rigid_arity, cap * (gc_i64)sizeof(gc_i32));
        if (st == GC_OK)
            st = mig_alloc((void **)&d->rigid_dist, H * cap * (gc_i64)sizeof(gc_f64));
        if (st != GC_OK) return st;
        mc.rigid_owner = d->rigid_gid;
        mc.rigid_cap = cap;
    }
    if (m > 0 &&
        (cudaMemcpyAsync(d->rigid_gid, og, (size_t)m * sizeof(gc_gid),
                         cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
         cudaMemcpyAsync(d->rigid_arity, oa, (size_t)m * sizeof(gc_i32),
                         cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
         cudaMemcpyAsync(d->rigid_dist, od, (size_t)(H * m) * sizeof(gc_f64),
                         cudaMemcpyDeviceToDevice, s) != cudaSuccess))
        return GC_E_DEVICE;
    d->rigid_count = m;
    return GC_OK;
}

#ifdef HAVE_MPI_GENESIS
/* Each edge's parcel to its peer and what that peer sent here, in one round
 * trip when they fit.  A parcel's first 8 bytes are its length; a first
 * message carries up to MIG_SWAP_HEAD bytes and only a larger parcel sends
 * the rest (tag + 1).  Every receive is posted before its send, and nothing
 * is waited on until all are posted.  Receives are sized by the bound and the
 * length, not by a matched probe.  A refusing rank sends length 0.  in[i]
 * keeps its room across calls; len[i] is what arrived. */
#define MIG_SWAP_HEAD 65536
static gc_status edge_swap(MPI_Comm comm, const gc_i32 *peer, int ne, int tag,
                           gc_status local, std::vector<char> *out,
                           std::vector<char> *in, size_t *len)
{
    const size_t hl = sizeof(long long);
    gc_status st = local;
    std::vector<MPI_Request> rq((size_t)ne, MPI_REQUEST_NULL);
    std::vector<MPI_Request> sq((size_t)(2 * ne), MPI_REQUEST_NULL);
    std::vector<size_t> osz((size_t)ne, 0);
    for (int i = 0; i < ne; ++i) {
        len[i] = 0;
        const bool send = local == GC_OK && out[i].size() > hl &&
                          out[i].size() <= (size_t)INT_MAX;
        if (local == GC_OK && !send && st == GC_OK) st = GC_E_CAPACITY;
        if (out[i].size() < hl) out[i].resize(hl);
        const long long n = send ? (long long)(out[i].size() - hl) : 0;
        std::memcpy(out[i].data(), &n, hl);
        osz[i] = send ? out[i].size() : hl;
        if (in[i].size() < MIG_SWAP_HEAD) in[i].resize(MIG_SWAP_HEAD);
        if (MPI_Irecv(in[i].data(), MIG_SWAP_HEAD, MPI_BYTE, peer[i], tag,
                      comm, &rq[(size_t)i]) != MPI_SUCCESS && st == GC_OK)
            st = GC_E_STATE;
    }
    for (int i = 0; i < ne; ++i) {
        const size_t n1 = std::min(osz[i], (size_t)MIG_SWAP_HEAD);
        if (MPI_Isend(out[i].data(), (int)n1, MPI_BYTE, peer[i], tag, comm,
                      &sq[(size_t)(2 * i)]) != MPI_SUCCESS && st == GC_OK)
            st = GC_E_STATE;
        if (osz[i] > MIG_SWAP_HEAD &&
            MPI_Isend(out[i].data() + MIG_SWAP_HEAD,
                      (int)(osz[i] - MIG_SWAP_HEAD), MPI_BYTE, peer[i],
                      tag + 1, comm, &sq[(size_t)(2 * i + 1)]) != MPI_SUCCESS &&
            st == GC_OK)
            st = GC_E_STATE;
    }
    std::vector<MPI_Status> rs((size_t)ne);
    if (MPI_Waitall(ne, rq.data(), rs.data()) != MPI_SUCCESS && st == GC_OK)
        st = GC_E_STATE;
    for (int i = 0; i < ne; ++i) {
        rq[(size_t)i] = MPI_REQUEST_NULL;
        int got = 0;
        long long isz = 0;
        if (MPI_Get_count(&rs[(size_t)i], MPI_BYTE, &got) != MPI_SUCCESS ||
            got < (int)hl) {
            if (st == GC_OK) st = GC_E_STATE;
            continue;
        }
        std::memcpy(&isz, in[i].data(), hl);
        const size_t whole = hl + (size_t)(isz > 0 ? isz : 0);
        if (isz <= 0 || isz > (long long)INT_MAX - (long long)hl ||
            (size_t)got != std::min(whole, (size_t)MIG_SWAP_HEAD)) {
            if (st == GC_OK) st = GC_E_STATE;
            continue;
        }
        if (whole > in[i].size()) in[i].resize(whole);
        len[i] = whole;
        if (whole > MIG_SWAP_HEAD &&
            MPI_Irecv(in[i].data() + MIG_SWAP_HEAD, (int)(whole - MIG_SWAP_HEAD),
                      MPI_BYTE, peer[i], tag + 1, comm,
                      &rq[(size_t)i]) != MPI_SUCCESS && st == GC_OK)
            st = GC_E_STATE;
    }
    if (MPI_Waitall(ne, rq.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS &&
        st == GC_OK)
        st = GC_E_STATE;
    if (MPI_Waitall(2 * ne, sq.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS &&
        st == GC_OK)
        st = GC_E_STATE;
    return st;
}

/* Per edge in one exchange: two counts (`cnt`: the payload's group and atom
 * records) and two record lists (constraint rows, zero-bit rows).  `cin`
 * receives the peer's counts, -1 when its parcel did not arrive whole; a
 * malformed arrival is this rank's refusal. */
template <class A, class B>
static gc_status edge_swap2(MPI_Comm comm, const gc_i32 *peer, int ne, int tag,
                            gc_status local, const gc_i64 *cnt,
                            const std::vector<A> *a, const std::vector<B> *b,
                            gc_i64 *cin, std::vector<A> *ai,
                            std::vector<B> *bi)
{
    static std::vector<std::vector<char> > out, in;
    if ((int)out.size() < ne) { out.resize((size_t)ne); in.resize((size_t)ne); }
    std::vector<size_t> len((size_t)ne, 0);
    const size_t hl = sizeof(long long);
    for (int i = 0; i < 2 * ne; ++i) cin[i] = -1;
    for (int i = 0; i < ne; ++i) {
        const gc_i64 n[2] = { (gc_i64)a[i].size(), (gc_i64)b[i].size() };
        const size_t sz = hl + 4 * sizeof(gc_i64) + (size_t)n[0] * sizeof(A) +
                          (size_t)n[1] * sizeof(B);
        std::vector<char> &o = out[(size_t)i];
        o.resize(sz);
        char *q = o.data() + hl;
        std::memcpy(q, cnt + 2 * i, 2 * sizeof(gc_i64));
        std::memcpy(q + 2 * sizeof(gc_i64), n, 2 * sizeof(gc_i64));
        q += 4 * sizeof(gc_i64);
        if (n[0]) std::memcpy(q, a[i].data(), (size_t)n[0] * sizeof(A));
        q += (size_t)n[0] * sizeof(A);
        if (n[1]) std::memcpy(q, b[i].data(), (size_t)n[1] * sizeof(B));
    }
    gc_status st = edge_swap(comm, peer, ne, tag, local, out.data(), in.data(),
                             len.data());
    for (int i = 0; i < ne; ++i) {
        ai[i].clear(); bi[i].clear();
        if (st != GC_OK) continue;
        const char *p = in[(size_t)i].data() + hl;
        const size_t have = len[(size_t)i] >= hl ? len[(size_t)i] - hl : 0;
        gc_i64 k2[2] = { -1, -1 }, n[2] = { -1, -1 };
        if (have >= sizeof(k2) + sizeof(n)) {
            std::memcpy(k2, p, sizeof(k2));
            std::memcpy(n, p + sizeof(k2), sizeof(n));
        }
        if (n[0] < 0 || n[1] < 0 ||
            have != sizeof(k2) + sizeof(n) + (size_t)n[0] * sizeof(A) +
                    (size_t)n[1] * sizeof(B)) {
            st = GC_E_MISMATCH;
            continue;
        }
        for (int k = 0; k < 2; ++k) cin[2 * i + k] = k2[k];
        p += sizeof(k2) + sizeof(n);
        ai[i].assign((const A *)p, (const A *)p + n[0]);
        p += (size_t)n[0] * sizeof(A);
        bi[i].assign((const B *)p, (const B *)p + n[1]);
    }
    return st;
}

/* Every rank's imported term pools, gathered into the replicated topology
 * (native_terms_topology): per kind, terms in (rank, pool index) order,
 * stably sorted by anchor GID, parameter ids offset into the concatenated
 * payloads.  Collective; every rank computes the same arrays. */
static gc_status terms_replicate(gc_context *ctx, MPI_Comm comm)
{
    const int np = ctx->nproc;
    const int K = GC_TERM_NKIND;
    std::vector<long long> mine((size_t)(3 * K)), all((size_t)(3 * K * np));
    for (int k = 0; k < K; ++k) {
        const TermPool &p = ctx->term[k];
        mine[(size_t)(3 * k)] = p.count;
        mine[(size_t)(3 * k + 1)] = p.param_count;
        mine[(size_t)(3 * k + 2)] = p.param_count > 0 ? p.param_stride : 0;
    }
    if (MPI_Allgather(mine.data(), 3 * K, MPI_LONG_LONG, all.data(), 3 * K,
                      MPI_LONG_LONG, comm) != MPI_SUCCESS)
        return GC_E_STATE;
    /* One Allgatherv of `each[r]` elements of `size` bytes per rank; st takes
     * only values every rank computes alike, a rank's own refusal waits in `bad`. */
    gc_status st = GC_OK, bad = GC_OK;
    const auto gatherv = [&](const void *src, const std::vector<gc_i64> &each,
                             gc_i64 size, void *dst) {
        std::vector<int> cnt((size_t)np), dsp((size_t)np);
        gc_i64 off = 0;
        for (int r = 0; r < np; ++r) {
            const gc_i64 b = each[(size_t)r] * size;
            if (b < 0 || off + b > (gc_i64)INT_MAX) st = GC_E_CAPACITY;
            cnt[(size_t)r] = (int)b;
            dsp[(size_t)r] = (int)off;
            off += b;
        }
        if (st != GC_OK) return;   /* every rank sees the same sizes */
        if (MPI_Allgatherv(src, cnt[(size_t)ctx->rank], MPI_BYTE, dst, cnt.data(),
                           dsp.data(), MPI_BYTE, comm) != MPI_SUCCESS)
            st = GC_E_STATE;
    };
    std::vector<gc_gid> ep[GC_TERM_NKIND];
    std::vector<gc_image> epi[GC_TERM_NKIND];
    std::vector<gc_i32> pid[GC_TERM_NKIND], img[GC_TERM_NKIND],
                        pbc[GC_TERM_NKIND], first[GC_TERM_NKIND];
    std::vector<gc_f64> pay[GC_TERM_NKIND];
    gc_i64 stride[GC_TERM_NKIND] = {}, pcount[GC_TERM_NKIND] = {};
    gc_gid ngid = 0;
    for (int k = 0; k < K && st == GC_OK; ++k) {
        const TermPool &p = ctx->term[k];
        const gc_i64 ar = gc_term_arity[k], nc = gc_term_pbc_codes[k];
        std::vector<gc_i64> n((size_t)np), pc((size_t)np), poff((size_t)np + 1, 0),
                            toff((size_t)np + 1, 0);
        gc_i64 ps = 0;
        for (int r = 0; r < np; ++r) {
            n[(size_t)r] = all[(size_t)(3 * K * r + 3 * k)];
            pc[(size_t)r] = all[(size_t)(3 * K * r + 3 * k + 1)];
            const gc_i64 s = all[(size_t)(3 * K * r + 3 * k + 2)];
            if (pc[(size_t)r] > 0 && ps > 0 && s != ps) st = GC_E_MISMATCH;
            if (pc[(size_t)r] > 0) ps = s;
            toff[(size_t)r + 1] = toff[(size_t)r] + n[(size_t)r];
            poff[(size_t)r + 1] = poff[(size_t)r] + pc[(size_t)r];
        }
        if (st != GC_OK) break;
        const gc_i64 N = toff[(size_t)np];
        std::vector<gc_gid> gep((size_t)(N * ar));
        std::vector<gc_image> gepi((size_t)(N * ar));
        std::vector<gc_i32> gpid((size_t)N), gimg((size_t)N), gpbc((size_t)(N * nc));
        pay[k].resize((size_t)(poff[(size_t)np] * ps));
        std::vector<gc_i64> ne(n), nq(n), nr(pc);
        for (int r = 0; r < np; ++r) { ne[(size_t)r] *= ar; nq[(size_t)r] *= nc; nr[(size_t)r] *= ps; }
        const gc_i64 mc = p.count > 0 ? p.count : 0;
        gatherv(mc ? p.endpoint.data() : 0, ne, sizeof(gc_gid), gep.data());
        gatherv(mc ? p.endpoint_image.data() : 0, ne, sizeof(gc_image), gepi.data());
        gatherv(mc ? p.param_id.data() : 0, n, sizeof(gc_i32), gpid.data());
        gatherv(mc ? p.image.data() : 0, n, sizeof(gc_i32), gimg.data());
        if (nc > 0) gatherv(mc ? p.pbc.data() : 0, nq, sizeof(gc_i32), gpbc.data());
        if (ps > 0)
            gatherv(p.param_count > 0 ? p.param_payload.data() : 0, nr,
                    sizeof(gc_f64), pay[k].data());
        if (st != GC_OK) break;
        for (int r = 0; r < np; ++r)
            for (gc_i64 t = toff[(size_t)r]; t < toff[(size_t)r + 1]; ++t) {
                const gc_i64 q = gpid[(size_t)t];
                if (q < 0 || (pc[(size_t)r] > 0 ? q >= pc[(size_t)r] : q != 0))
                    bad = GC_E_MISMATCH;
                gpid[(size_t)t] = (gc_i32)(poff[(size_t)r] + q);
            }
        std::vector<gc_i32> order((size_t)N);
        for (gc_i64 t = 0; t < N; ++t) order[(size_t)t] = (gc_i32)t;
        std::stable_sort(order.begin(), order.end(), [&](gc_i32 x, gc_i32 y) {
            return gep[(size_t)(x * ar)] < gep[(size_t)(y * ar)];
        });
        ep[k].resize(gep.size()); epi[k].resize(gepi.size());
        pid[k].resize((size_t)N); img[k].resize((size_t)N); pbc[k].resize(gpbc.size());
        for (gc_i64 o = 0; o < N; ++o) {
            const gc_i64 t = order[(size_t)o];
            for (gc_i64 e = 0; e < ar; ++e) {
                ep[k][(size_t)(o * ar + e)] = gep[(size_t)(t * ar + e)];
                epi[k][(size_t)(o * ar + e)] = gepi[(size_t)(t * ar + e)];
            }
            pid[k][(size_t)o] = gpid[(size_t)t];
            img[k][(size_t)o] = gimg[(size_t)t];
            for (gc_i64 c = 0; c < nc; ++c)
                pbc[k][(size_t)(o * nc + c)] = gpbc[(size_t)(t * nc + c)];
        }
        if (N > 0) {
            if (ep[k][0] < 0) st = GC_E_MISMATCH;   /* the same everywhere */
            ngid = std::max(ngid, ep[k][(size_t)((N - 1) * ar)] + 1);
        }
        stride[k] = ps;
        pcount[k] = poff[(size_t)np];
    }
    if (st == GC_OK && ngid >= (gc_gid)INT_MAX) st = GC_E_CAPACITY;
    gcn_term_topo_in in[GC_TERM_NKIND];
    for (int k = 0; k < K && st == GC_OK; ++k) {
        const gc_i64 ar = gc_term_arity[k], N = (gc_i64)pid[k].size();
        first[k].assign((size_t)(ngid + 1), 0);
        for (gc_i64 t = 0; t < N; ++t) first[k][(size_t)(ep[k][(size_t)(t * ar)] + 1)]++;
        for (gc_gid g = 0; g < ngid; ++g) first[k][(size_t)(g + 1)] += first[k][(size_t)g];
        in[k].n = N;
        in[k].ep = ep[k].data();
        in[k].epi = epi[k].data();
        in[k].pid = pid[k].data();
        in[k].img = img[k].data();
        in[k].pbc = pbc[k].data();
        in[k].first = first[k].data();
        in[k].param_stride = stride[k];
        in[k].param_count = pcount[k];
        in[k].payload = pay[k].data();
    }
    if (st == GC_OK) st = bad;
    if (st == GC_OK) st = native_terms_topology(ctx, in, ngid);
    /* every rank keeps the topology, or none does */
    int ok = st == GC_OK, all_ok = 0;
    if (MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS)
        all_ok = 0;
    if (!all_ok) {
        native_terms_topology_reset(ctx);
        if (st == GC_OK) st = GC_E_STATE;
    }
    if (st != GC_OK)
        std::fprintf(stderr, "GPU_Core_Error> phase=term_topology rank=%d "
                     "status=%d\n", (int)ctx->rank, (int)st);
    return st;
}

/* Receiver validation.  One edge's parcel is [groups | atoms] (group wires,
 * then atom wires), announced by the count pair [groups, atoms]; nothing a
 * parcel carries is used before the receiver has checked it in place on the
 * device and every rank has voted. */
static_assert(sizeof(gcn_group_migration_wire) == 64,
              "group migration wire size changed");
static_assert(sizeof(gcn_atom_migration_wire) == 264,
              "atom migration wire size changed");

/* One edge's admitted counts and byte sizes, and its GID hash tables'
 * capacities (a power of two, at least 8, when the side is live; else 0). */
struct RecvPlan {
    gc_i64 groups, atoms, group_bytes, atom_bytes;
    gc_u64 gtable_cap, atable_cap;
};

/* The only validation output read back: code is a gc_status, bad_group /
 * bad_atom the reported record (-1 when not record-specific), reported the
 * first-writer flag.  Initialised to {0, 0, GC_OK, -1, -1}. */
struct RecvResult {
    gc_i32 reported;
    gc_i32 pad0;
    gc_i64 code;
    gc_i64 bad_group;
    gc_i64 bad_atom;
};

/* The destination's context: the epoch and ranks every record must carry,
 * and its rigid descriptor (representative GIDs ascending, with arities). */
struct RecvParams {
    gc_i64 expect_epoch;
    gc_i32 self_rank;
    gc_i32 peer_rank;
    const gc_gid *rigid_gid;
    const gc_i32 *rigid_arity;
    gc_i64 rigid_count;
};

/* a * b when it fits in 64 bits (a, b >= 0). */
static bool checked_mul(gc_i64 a, gc_i64 b, gc_i64 *out)
{
    if (a < 0 || b < 0) return false;
    if (a == 0 || b == 0) { *out = 0; return true; }
    if (a > (gc_i64)0x7FFFFFFFFFFFFFFFLL / b) return false;
    *out = a * b;
    return true;
}

/* Smallest power of two >= max(8, 2n); 0 for an empty side. */
static gc_u64 recv_table_cap(gc_i64 n)
{
    if (n <= 0) return 0;
    gc_u64 cap = 8;
    while (cap < (gc_u64)2 * (gc_u64)n) cap *= 2;
    return cap;
}

/* Admit one edge's announced counts against this rank's room before any
 * payload is read.  Refusals in order: ARG (negative count or epoch), EPOCH
 * (no successor), MISMATCH (a group without atoms or atoms without a
 * group), OVERFLOW, CAPACITY.  *out is written only on GC_OK. */
static gc_status recv_admit(const gc_i64 cnt[2], gc_i64 self_epoch,
                            gc_i64 group_room, gc_i64 atom_room,
                            RecvPlan *out)
{
    const gc_i64 ng = cnt[0], na = cnt[1];
    if (ng < 0 || na < 0 || self_epoch < 0) return GC_E_ARG;
    if (self_epoch == (gc_i64)0x7FFFFFFFFFFFFFFFLL) return GC_E_EPOCH;
    if (na < ng || (ng == 0) != (na == 0)) return GC_E_MISMATCH;
    gc_i64 gb = 0, ab = 0;
    if (!checked_mul(ng, (gc_i64)sizeof(gcn_group_migration_wire), &gb) ||
        !checked_mul(na, (gc_i64)sizeof(gcn_atom_migration_wire), &ab))
        return GC_E_OVERFLOW;
    if (ng > (gc_i64)GC_MAX_LOCAL_INDEX || na > (gc_i64)GC_MAX_LOCAL_INDEX ||
        ng > group_room || na > atom_room)
        return GC_E_CAPACITY;
    out->groups = ng;
    out->atoms = na;
    out->group_bytes = gb;
    out->atom_bytes = ab;
    out->gtable_cap = recv_table_cap(ng);
    out->atable_cap = recv_table_cap(na);
    return GC_OK;
}

/* First writer wins: only the thread whose CAS flips reported 0 -> 1 stores
 * the code and indices, so a report is never a mix of two defects.  Which
 * defect is reported is scheduling-dependent; every one is genuine. */
static __device__ inline void recv_report(RecvResult *res, gc_status code,
                                          gc_i64 bad_group, gc_i64 bad_atom)
{
    if (atomicCAS(&res->reported, 0, 1) == 0) {
        res->code = (gc_i64)code;
        res->bad_group = bad_group;
        res->bad_atom = bad_atom;
    }
}

static __device__ __forceinline__ bool recv_finite(gc_f64 v)
{
    return isfinite(v) != 0;
}

/* Open-addressing insert (0 marks empty, key > 0): 1 on a fresh insert, 0
 * when present, -1 when the probe bound is exhausted. */
static __device__ inline int recv_table_insert(gc_u64 *table, gc_u64 mask,
                                               gc_u64 key)
{
    gc_u64 h = (key * (gc_u64)0x9E3779B97F4A7C15ULL) & mask;
    for (gc_u64 step = 0; step <= mask + 1; ++step) {
        const gc_u64 slot = (h + step) & mask;
        const gc_u64 seen =
            (gc_u64)atomicCAS((unsigned long long *)&table[slot], 0ULL,
                              (unsigned long long)key);
        if (seen == 0ULL) return 1;
        if (seen == key) return 0;
    }
    return -1;
}

/* One arriving group: schema, kind and arity, its run inside the atoms, a
 * positive GID, epoch and ranks, a destination cell owned by this rank, and
 * for a hydrogen group the destination's rigid row (constraint statics are
 * rebuilt, not carried). */
static __device__ __forceinline__ gc_status
recv_check_group(const gcn_group_migration_wire &gr, gc_i64 na,
                 const RecvParams &dp, const struct gcn_layout *layout)
{
    if (gr.schema != GCN_MIGRATION_RECORD_SCHEMA) return GC_E_MISMATCH;
    if (gr.kind < (gc_i32)GC_GROUP_SINGLE ||
        gr.kind > (gc_i32)GC_GROUP_HGROUP)
        return GC_E_MISMATCH;
    if (gr.atom_count <= 0 || gr.atom_offset < 0) return GC_E_MISMATCH;
    if (gr.atom_offset > na) return GC_E_MISMATCH;
    if (gr.atom_count > na - gr.atom_offset) return GC_E_MISMATCH;
    if ((gr.kind == (gc_i32)GC_GROUP_SINGLE && gr.atom_count != 1) ||
        (gr.kind == (gc_i32)GC_GROUP_WATER && gr.atom_count != 3) ||
        (gr.kind == (gc_i32)GC_GROUP_HGROUP &&
         (gr.atom_count < 2 || gr.atom_count > (gc_i64)GC_MAX_HGROUP_H + 1)))
        return GC_E_MISMATCH;
    if (gr.group_gid <= (gc_gid)0) return GC_E_MISMATCH;
    if (gr.epoch != dp.expect_epoch) return GC_E_EPOCH;
    if (gr.source_rank != dp.peer_rank || gr.destination_rank != dp.self_rank)
        return GC_E_MISMATCH;
    for (int k = 0; k < 3; ++k)
        if (gr.destination_cell[k] < 0 ||
            gr.destination_cell[k] >= layout->ncel[k])
            return GC_E_MISMATCH;
    if ((gc_i64)gcn_rank_of_cell(*layout, gr.destination_cell[0],
                                 gr.destination_cell[1],
                                 gr.destination_cell[2]) !=
        (gc_i64)gr.destination_rank)
        return GC_E_MISMATCH;
    if (gr.kind == (gc_i32)GC_GROUP_HGROUP) {
        if (dp.rigid_count <= 0) return GC_E_UNSUPPORTED;
        const gc_i64 at = dev_lower(dp.rigid_gid, dp.rigid_count, gr.group_gid);
        if (at >= dp.rigid_count || dp.rigid_gid[at] != gr.group_gid)
            return GC_E_UNSUPPORTED;
        if ((gc_i64)dp.rigid_arity[at] != gr.atom_count - 1)
            return GC_E_MISMATCH;
    }
    return GC_OK;
}

/* One arriving atom on its own: schema, a positive GID, finite FP64 fields,
 * mass, inverse mass and class, epoch and ranks. */
static __device__ __forceinline__ gc_status
recv_check_atom(const gcn_atom_migration_wire &a, const RecvParams &dp)
{
    if (a.schema != GCN_MIGRATION_RECORD_SCHEMA) return GC_E_MISMATCH;
    if (a.gid <= (gc_gid)0) return GC_E_MISMATCH;
    const gc_f64 *vec[8] = { a.coord,   a.coord_ref, a.list_ref, a.vel,
                             a.vel_ref, a.vel_half,  a.vel_full, a.force };
    for (int f = 0; f < 8; ++f)
        for (int k = 0; k < 3; ++k)
            if (!recv_finite(vec[f][k])) return GC_E_MISMATCH;
    if (!recv_finite(a.charge) || !recv_finite(a.mass) ||
        !recv_finite(a.inv_mass))
        return GC_E_MISMATCH;
    if (!(a.mass > 0.0) || !(a.inv_mass >= 0.0) || a.cls < 1)
        return GC_E_MISMATCH;
    if (a.epoch != dp.expect_epoch) return GC_E_EPOCH;
    if (a.source_rank != dp.peer_rank || a.destination_rank != dp.self_rank)
        return GC_E_MISMATCH;
    return GC_OK;
}

/* Every group and atom record on its own; grid-stride loops. */
__global__ void gcn_kern_mig_recv_records(
    const gcn_group_migration_wire *groups, gc_i64 ng,
    const gcn_atom_migration_wire *atoms, gc_i64 na, RecvParams dp,
    const struct gcn_layout *layout, RecvResult *res)
{
    const gc_i64 tid =
        (gc_i64)blockIdx.x * (gc_i64)blockDim.x + (gc_i64)threadIdx.x;
    const gc_i64 stride = (gc_i64)blockDim.x * (gc_i64)gridDim.x;
    for (gc_i64 g = tid; g < ng; g += stride) {
        const gc_status s = recv_check_group(groups[g], na, dp, layout);
        if (s != GC_OK) recv_report(res, s, g, -1);
    }
    for (gc_i64 j = tid; j < na; j += stride) {
        const gc_status s = recv_check_atom(atoms[j], dp);
        if (s != GC_OK) recv_report(res, s, -1, j);
    }
}

/* Membership and uniqueness, after the records kernel on the same stream (a
 * no-op once it refused): distinct group GIDs; each atom claimed by exactly
 * one group, matching it (epoch, ranks, group GID, member ordinal, ordinal 0
 * the representative); distinct atom GIDs, none already owned here.
 * gtable / atable are zeroed device scratch; owned is sorted ascending. */
__global__ void gcn_kern_mig_recv_cross(
    const gcn_group_migration_wire *groups, gc_i64 ng,
    const gcn_atom_migration_wire *atoms, gc_i64 na, const gc_gid *owned,
    gc_i64 nowned, gc_u64 *gtable, gc_u64 gmask, gc_u64 *atable,
    gc_u64 amask, RecvResult *res)
{
    if (res->code != (gc_i64)GC_OK) return;
    const gc_i64 tid =
        (gc_i64)blockIdx.x * (gc_i64)blockDim.x + (gc_i64)threadIdx.x;
    const gc_i64 stride = (gc_i64)blockDim.x * (gc_i64)gridDim.x;
    for (gc_i64 g = tid; g < ng; g += stride) {
        const gc_gid gid = groups[g].group_gid;
        if (gid <= (gc_gid)0 ||
            recv_table_insert(gtable, gmask, (gc_u64)gid) <= 0)
            recv_report(res, GC_E_MISMATCH, g, -1);
    }
    for (gc_i64 j = tid; j < na; j += stride) {
        const gcn_atom_migration_wire &a = atoms[j];
        gc_i64 owner = -1, owners = 0;
        for (gc_i64 g = 0; g < ng; ++g) {
            const gcn_group_migration_wire &gr = groups[g];
            if (gr.atom_offset < 0 || gr.atom_count <= 0) continue;
            if (j >= gr.atom_offset && j - gr.atom_offset < gr.atom_count) {
                owner = g;
                ++owners;
            }
        }
        if (owners != 1) {
            recv_report(res, GC_E_MISMATCH, owners ? owner : -1, j);
            continue;
        }
        const gcn_group_migration_wire &gr = groups[owner];
        const gc_i64 ordinal = j - gr.atom_offset;
        if (a.epoch != gr.epoch) {
            recv_report(res, GC_E_EPOCH, owner, j);
            continue;
        }
        if (a.source_rank != gr.source_rank ||
            a.destination_rank != gr.destination_rank ||
            a.group_gid != gr.group_gid ||
            a.member_ordinal != (gc_i32)ordinal ||
            (ordinal == 0 && a.gid != gr.group_gid) || a.gid <= (gc_gid)0 ||
            recv_table_insert(atable, amask, (gc_u64)a.gid) <= 0) {
            recv_report(res, GC_E_MISMATCH, owner, j);
            continue;
        }
        const gc_i64 at = dev_lower(owned, nowned, a.gid);
        if (at < nowned && owned[at] == a.gid)
            recv_report(res, GC_E_MISMATCH, owner, j);
    }
}

/* Validate one edge's arrived parcel in place on stream s, without waiting.
 * The delivered bytes and records must be the admitted ones (a truncated or
 * overlong delivery refuses here); then the records and cross kernels.  The
 * verdict is *res once the stream reaches it.  `tables` is zeroed scratch:
 * the group table, then the atom table. */
static gc_status recv_validate(const RecvPlan &pl, const void *payload,
                               gc_i64 got_bytes, gc_i64 got_records,
                               const gc_gid *owned, gc_i64 nowned,
                               gc_u64 *tables, const RecvParams &dp,
                               const struct gcn_layout *layout,
                               RecvResult *res, cudaStream_t s)
{
    const gc_i64 ng = pl.groups, na = pl.atoms;
    if (pl.group_bytes > got_bytes ||
        pl.atom_bytes != got_bytes - pl.group_bytes || ng > got_records ||
        na != got_records - ng)
        return GC_E_MISMATCH;
    const gcn_group_migration_wire *groups =
        (const gcn_group_migration_wire *)payload;
    const gcn_atom_migration_wire *atoms =
        (const gcn_atom_migration_wire *)((const char *)payload +
                                          pl.group_bytes);
    gc_u64 *gtable = tables;
    gc_u64 *atable = tables + (pl.gtable_cap ? pl.gtable_cap : 8);
    const unsigned blocks = (unsigned)((ng + na + 127) / 128);
    gcn_kern_mig_recv_records<<<blocks, 128, 0, s>>>(groups, ng, atoms, na,
                                                    dp, layout, res);
    if (cudaGetLastError() != cudaSuccess) return GC_E_DEVICE;
    gcn_kern_mig_recv_cross<<<blocks, 128, 0, s>>>(
        groups, ng, atoms, na, owned, nowned, gtable,
        pl.gtable_cap ? pl.gtable_cap - 1 : 0, atable,
        pl.atable_cap ? pl.atable_cap - 1 : 0, res);
    return cudaGetLastError() == cudaSuccess ? GC_OK : GC_E_DEVICE;
}

/* The zero-bit rows' follow on the host, deferred past the publication:
 * its inputs and state, owned by the deferred work. */
struct MaskFollow {
    std::vector<MaskRow> named, carried;
    std::vector<gc_u8> rf, cf, uf;
    std::vector<gc_gid> cu, u, gone, came;
};

/* One transaction: one exchange with every neighbour rank (owners of the
 * domains one step away on the split axes, corners included), each moving
 * group sent straight to its owner.  Nothing is applied until every edge has
 * validated; then one scalar vote commits.  Its phases share this state. */
struct MigTxn {
    gc_context *const ctx;
    struct gcn_device *const d;
    MigCache &mc;
    const gc_i32 rank, nproc;
    const gcn_layout &L;
    const gc_i64 epoch;
    const MPI_Comm comm;

    /* the shipment: per peer [groups, atoms, -] and segment bases, the
     * packed groups, their atoms' GIDs and rigid rows, the vote words */
    std::vector<gc_i64> pc, gbase, abase;
    gc_i64 n_send_g = 0, n_send_a = 0;
    std::vector<gcn_group_migration_wire> g_w;
    std::vector<gc_gid> a_gid;
    std::vector<RigidRow> g_row;
    std::vector<gc_u64> vote_sum;
    /* the movers; mover_hop is (atom GID, destination), sorted */
    std::vector<MigMember> moving;
    std::vector<const RigidRow *> moving_row;
    std::vector<gc_gid> moving_hgroups;
    std::vector<std::pair<gc_gid, gc_i32> > mover_hop;
    KeyMap hop_of;
    GidFilter hop_seen;
    /* the edges: neighbour ranks, each once, and each edge's room */
    std::vector<gc_i32> peers, edge_of;
    int ne = 0;
    gc_i64 edge_bound = 0, cap = 0;
    /* the receiver's owned table, device layout and zero-bit rows */
    gc_i64 nowned = 0;
    std::vector<gc_gid> owned_host;           /* sorted, when read */
    void *d_owned = 0, *d_layout = 0;
    int mask_dev = 0, mask_host = 0;
    std::vector<std::pair<gc_i64, MaskRow> > mine;
    /* per edge: counts and rows sent and received, admission, delivery */
    std::vector<gc_i64> own_g, own_a, bytes, sc, rc;
    std::vector<std::vector<RigidRow> > rows, rows_in;
    std::vector<std::vector<MaskRow> > mrow, mrow_in;
    std::vector<RecvPlan> plan;
    std::vector<gc_i64> gbytes, grec;
    void *d_recv = 0;
    /* the validated arrivals */
    std::vector<gcn_group_migration_wire> fin_g;
    std::vector<gcn_atom_migration_wire> fin_a;
    std::vector<RigidRow> fin_r;
    std::vector<MaskRow> carried;
    std::vector<gc_gid> arr_gid;
    std::vector<gc_i32> arr_cell;
    std::shared_ptr<MaskFollow> mf;
    gc_status local = GC_OK;

    explicit MigTxn(gc_context *c)
        : ctx(c), d(c->native), mc(g_mig_cache[c->native]), rank(c->rank),
          nproc(c->nproc), L(c->native->layout), epoch(c->epoch + 1),
          comm(MPI_Comm_f2c((MPI_Fint)c->comm)) {}

    /* This rank's owned GIDs, sorted. */
    gc_status read_owned()
    {
        owned_host.resize((size_t)nowned);
        const gc_status rs = mig_d2h(owned_host.data(), d->gid,
                                     nowned * (gc_i64)sizeof(gc_gid));
        std::sort(owned_host.begin(), owned_host.end());
        return rs;
    }

    /* This rank's classification in one pinned readback: the per-peer
     * arrays, the packed groups, their atoms' GIDs, their rigid rows and the
     * vote words, sized by the earlier transactions' largest plus a margin
     * and re-read only when it outgrew that.  Then the movers.  The first
     * transaction replicates the term topology; later ones derive the pools
     * from it.  A failure is this rank's refusal, carried into the vote. */
    gc_status classify(const struct gcn_pack_check *pack)
    {
        pc.assign((size_t)(3 * nproc), 0);
        gbase.assign((size_t)nproc, 0);
        abase.assign((size_t)nproc, 0);
        const gc_i64 pbytes = (gc_i64)nproc * (gc_i64)sizeof(gc_i64);
        gc_i64 room_g = std::min(d->num_groups, mc.send_g + mc.send_g / 4 + 32);
        gc_i64 room_a = std::min(d->num_owned, mc.send_a + mc.send_a / 4 + 128);
        gc_status st = native_terms_topology_ready(ctx) ? GC_OK :
                       terms_replicate(ctx, comm);
        /* On one node the ranks add the vote words on the device (gcx_wsum);
         * creating the sum is collective, so it is entered whatever st says. */
        const gc_i64 nvote = 2 + (gc_i64)nproc;
        gc_status sv = GC_OK;
        if (!mc.vote_tried) {
            mc.vote_tried = true;
            sv = gcx_wsum_create((gcx_comm)ctx->comm, (gc_i32)nvote, 0, &mc.vote);
            if (sv == GC_OK && mc.vote &&
                (sv = dev_calloc((void **)&mc.vote_words,
                                 nvote * (gc_i64)sizeof(gc_u64))) != GC_OK) {
                gcx_wsum_destroy(mc.vote);
                mc.vote = 0;
            }
        }
        if (sv == GC_OK && mc.vote) {
            gcn_kern_mig_vote<<<1, 32, 0, d->stream>>>(
                d->migration_peer_counts, nproc, rank, mc.vote_words);
            sv = gcx_wsum_launch(mc.vote, mc.vote_words, (void *)d->stream);
        }
        if (st == GC_OK) st = sv;
        vote_sum.assign((size_t)nvote, 0);
        for (int pass = 0; st == GC_OK && pass < 2; ++pass) {
            const gc_i64 gb = room_g * (gc_i64)sizeof(gcn_group_migration_wire);
            const gc_i64 ab = room_a * (gc_i64)sizeof(gc_gid);
            const gc_i64 rb = room_g * (gc_i64)sizeof(RigidRow);
            char *hp = 0;
            void *drow = 0;
            const gc_i64 vb =
                mc.vote && pass == 0 ? nvote * (gc_i64)sizeof(gc_u64) : 0;
            st = mig_host(mc.hval, 5 * pbytes + gb + ab + rb + vb, &hp);
            if (st == GC_OK && vb > 0 &&
                cudaMemcpyAsync(hp + 5 * pbytes + gb + ab + rb, mc.vote_words,
                                (size_t)vb, cudaMemcpyDeviceToHost,
                                d->stream) != cudaSuccess)
                st = GC_E_DEVICE;
            if (st == GC_OK) st = mig_buf(mc.rsend, rb, 0, &drow);
            if (st == GC_OK && room_g > 0)
                gcn_kern_mig_rigid_rows<<<mig_grid(room_g), GCN_BLOCK, 0,
                                          d->stream>>>(
                    d->migration_group_send, room_g, d->migration_peer_counts,
                    nproc, d->rigid_gid, d->rigid_arity, d->rigid_dist,
                    d->rigid_count, (RigidRow *)drow);
            if (st == GC_OK &&
                ((rb > 0 && cudaMemcpyAsync(hp + 5 * pbytes + gb + ab, drow,
                                            (size_t)rb, cudaMemcpyDeviceToHost,
                                            d->stream) != cudaSuccess) ||
                 cudaMemcpyAsync(hp, d->migration_peer_counts,
                                 (size_t)(3 * pbytes), cudaMemcpyDeviceToHost,
                                 d->stream) != cudaSuccess ||
                 cudaMemcpyAsync(hp + 3 * pbytes, d->migration_peer_group_base,
                                 (size_t)pbytes, cudaMemcpyDeviceToHost,
                                 d->stream) != cudaSuccess ||
                 cudaMemcpyAsync(hp + 4 * pbytes, d->migration_peer_atom_base,
                                 (size_t)pbytes, cudaMemcpyDeviceToHost,
                                 d->stream) != cudaSuccess ||
                 (gb > 0 && cudaMemcpyAsync(hp + 5 * pbytes,
                                            d->migration_group_send, (size_t)gb,
                                            cudaMemcpyDeviceToHost,
                                            d->stream) != cudaSuccess) ||
                 (ab > 0 && cudaMemcpy2DAsync(
                                hp + 5 * pbytes + gb, sizeof(gc_gid),
                                (const char *)d->migration_atom_send +
                                    offsetof(gcn_atom_migration_wire, gid),
                                sizeof(gcn_atom_migration_wire), sizeof(gc_gid),
                                (size_t)room_a, cudaMemcpyDeviceToHost,
                                d->stream) != cudaSuccess)))
                st = GC_E_DEVICE;
            if (st == GC_OK) st = dev_phase(d->stream, "migration_movers");
            if (st == GC_OK && pass == 0 && pack)
                st = native_pack_verdict(ctx, pack);
            if (st != GC_OK) break;
            std::memcpy(&pc[0], hp, (size_t)(3 * pbytes));
            if (vb > 0)
                std::memcpy(vote_sum.data(), hp + 5 * pbytes + gb + ab + rb,
                            (size_t)vb);
            std::memcpy(&gbase[0], hp + 3 * pbytes, (size_t)pbytes);
            std::memcpy(&abase[0], hp + 4 * pbytes, (size_t)pbytes);
            n_send_g = n_send_a = 0;
            for (gc_i32 p = 0; st == GC_OK && p < nproc; ++p) {
                if (pc[(size_t)(3 * p)] < 0 || pc[(size_t)(3 * p + 1)] < 0)
                    st = GC_E_MISMATCH;
                n_send_g += pc[(size_t)(3 * p)];
                n_send_a += pc[(size_t)(3 * p + 1)];
            }
            if (st != GC_OK) break;
            if (n_send_g <= room_g && n_send_a <= room_a) {
                const char *q = hp + 5 * pbytes;
                g_w.assign((const gcn_group_migration_wire *)q,
                           (const gcn_group_migration_wire *)q + n_send_g);
                a_gid.assign((const gc_gid *)(q + gb),
                             (const gc_gid *)(q + gb) + n_send_a);
                g_row.assign((const RigidRow *)(q + gb + ab),
                             (const RigidRow *)(q + gb + ab) + n_send_g);
                break;
            }
            if (pass > 0 || n_send_g > d->num_groups || n_send_a > d->num_owned)
                st = GC_E_MISMATCH;
            room_g = n_send_g;
            room_a = n_send_a;
        }
        if (st == GC_OK) {
            mc.send_g = std::max(mc.send_g, n_send_g);
            mc.send_a = std::max(mc.send_a, n_send_a);
        }
        g_w.resize((size_t)(n_send_g > 0 ? n_send_g : 1));
        a_gid.resize((size_t)(n_send_a > 0 ? n_send_a : 1), 0);

        for (gc_i64 g = 0; g < n_send_g && st == GC_OK; ++g) {
            const gcn_group_migration_wire &w = g_w[(size_t)g];
            const gc_i32 P = w.destination_rank;
            const RigidRow *row = 0;
            if (P < 0 || P >= nproc || P == rank || w.atom_count <= 0 ||
                w.atom_offset < 0 ||
                abase[(size_t)P] + w.atom_offset + w.atom_count > n_send_a) {
                st = GC_E_MISMATCH;
                break;
            }
            if (w.kind == (gc_i32)GC_GROUP_HGROUP) {
                moving_hgroups.push_back(w.group_gid);
                row = &g_row[(size_t)g];
                if (row->gid != w.group_gid || row->arity <= 0) {
                    st = GC_E_MISMATCH;
                    break;
                }
            }
            MigMember mm;
            mm.group_gid = w.group_gid;
            mm.destination_rank = P;
            mm.count = w.atom_count;
            mm.gid = &a_gid[(size_t)(abase[(size_t)P] + w.atom_offset)];
            moving.push_back(mm);
            moving_row.push_back(row);
        }
        if (st == GC_OK && (gc_i64)moving.size() != n_send_g)
            st = GC_E_MISMATCH;
        for (const MigMember &m : moving)
            for (gc_i64 j = 0; j < m.count; ++j)
                mover_hop.push_back(std::make_pair(m.gid[j], m.destination_rank));
        std::sort(mover_hop.begin(), mover_hop.end());

        /* Zero-bit rows naming an own mover; terms do not travel. */
        if (st == GC_OK && !mover_hop.empty() &&
            native_mask_device(ctx, mc.mask_rows ==
                                        (gc_i64)ctx->real_mask_gid.size())) {
            std::vector<gc_gid> mg(mover_hop.size());
            for (size_t i = 0; i < mover_hop.size(); ++i)
                mg[i] = mover_hop[i].first;
            st = native_mask_select_prefetch(ctx, mg.data(), (gc_i64)mg.size());
        }
        return st;
    }

    /* The neighbour ranks, each once; the classification refuses a move of
     * more than one domain on an axis. */
    gc_status find_edges(gc_status st)
    {
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx && L.nd[0] == 1) || (dy && L.nd[1] == 1) ||
                        (dz && L.nd[2] == 1))
                        continue;
                    const gc_i32 p = gcn_rank_shift(L, dx, dy, dz);
                    if (p != rank &&
                        std::find(peers.begin(), peers.end(), p) == peers.end())
                        peers.push_back(p);
                }
        edge_of.assign((size_t)nproc, -1);
        for (size_t i = 0; i < peers.size(); ++i)
            edge_of[(size_t)peers[i]] = (gc_i32)i;
        ne = (int)peers.size();
        for (const MigMember &m : moving)
            if (st == GC_OK && edge_of[(size_t)m.destination_rank] < 0) {
                std::fprintf(stderr, "GPU_Core_Error> phase=migration_route "
                             "rank=%d group=%lld destination=%d\n", (int)rank,
                             (long long)m.group_gid, (int)m.destination_rank);
                st = GC_E_OWNER;
            }
        return st;
    }

    /* The vote before the exchange: any refusal stops every rank, no mover
     * ends it, and the largest parcel bounds every edge's plan.  False when
     * the transaction ends here with *ret. */
    bool vote(gc_status st, gc_status *ret)
    {
        gc_i64 own = 0;
        for (int i = 0; i < ne; ++i) {
            const gc_i32 P = peers[(size_t)i];
            own = std::max(own,
                pc[(size_t)(3 * P)] * (gc_i64)sizeof(gcn_group_migration_wire) +
                pc[(size_t)(3 * P + 1)] * (gc_i64)sizeof(gcn_atom_migration_wire));
        }
        gc_i64 in[3] = { st != GC_OK ? 1 : 0, n_send_g + n_send_a, own };
        gc_i64 out[3] = { 0, 0, 0 };
        gc_i64 dev[3] = { 0, 0, 0 };
        for (gc_i64 r = 0; r < nproc; ++r)
            dev[2] = std::max(dev[2], (gc_i64)vote_sum[(size_t)(2 + r)]);
        dev[0] = vote_sum[1] != 0;
        dev[1] = (gc_i64)vote_sum[0];
        if (!mc.vote) {
            if (MPI_Allreduce(in, out, 3, MPI_LONG_LONG, MPI_MAX, comm) !=
                MPI_SUCCESS) {
                *ret = GC_E_STATE;
                return false;
            }
        } else {
            /* a local refusal returns; the caller stops every rank */
            if (st != GC_OK) {
                *ret = st;
                return false;
            }
            std::memcpy(out, dev, sizeof out);
        }
        *ret = out[0] ? (st != GC_OK ? st : GC_E_MISMATCH) : GC_OK;
        if (out[0] || out[1] == 0) return false;
        edge_bound = out[2] + 64;
        return true;
    }

    /* The receiver's duplicate-owner table (this rank's owned GIDs, kept
     * sorted on the device across rebuilds and edited at each commit), the
     * validator's device layout and the selected zero-bit rows; then each
     * edge's counts, constraint rows and zero-bit rows. */
    void prepare(gc_status st)
    {
        nowned = d->num_owned;
        if (st == GC_OK && mc.owned_n != nowned) {
            void *tab = 0;
            st = read_owned();
            if (st == GC_OK)
                st = mig_buf(mc.owned_tab[mc.owned_cur],
                             nowned * (gc_i64)sizeof(gc_gid), 0, &tab);
            if (st == GC_OK)
                st = mig_h2d(tab, owned_host.data(),
                             nowned * (gc_i64)sizeof(gc_gid));
            mc.owned_n = st == GC_OK ? nowned : -1;
        }
        d_owned = mc.owned_tab[mc.owned_cur].p;
        if (st == GC_OK)
            st = mig_buf(mc.layout, (gc_i64)sizeof(gcn_layout), 0, &d_layout);
        if (st == GC_OK)
            st = mig_h2d_async(d, d_layout, &d->layout,
                               (gc_i64)sizeof(gcn_layout));

        /* Zero-bit rows live on the device once the list is normalized. */
        mask_dev = native_mask_device(
            ctx, mc.mask_rows == (gc_i64)ctx->real_mask_gid.size());
        mask_host = !mask_dev;
        if (st == GC_OK && mask_dev && !mover_hop.empty()) {
            std::vector<gc_gid> mg(mover_hop.size());
            for (size_t i = 0; i < mover_hop.size(); ++i)
                mg[i] = mover_hop[i].first;
            const MaskRow *r = 0;
            const gc_i64 *at = 0;
            gc_i64 n = 0;
            st = native_mask_select(ctx, mg.data(), (gc_i64)mg.size(), &r, &at,
                                    &n);
            for (gc_i64 i = 0; st == GC_OK && i < n; ++i)
                mine.push_back(std::make_pair(at[i], r[i]));
        }
        hop_of.reset((gc_i64)mover_hop.size());
        for (const auto &m : mover_hop) {
            hop_of.insert(m.first, m.second);
            hop_seen.add(m.first);
        }
        local = st;

        const gc_i64 gw = (gc_i64)sizeof(gcn_group_migration_wire);
        const gc_i64 aw = (gc_i64)sizeof(gcn_atom_migration_wire);
        rows.resize((size_t)ne);
        rows_in.resize((size_t)ne);
        mrow.resize((size_t)ne);
        mrow_in.resize((size_t)ne);
        own_g.resize((size_t)ne);
        own_a.resize((size_t)ne);
        bytes.resize((size_t)ne);
        sc.resize((size_t)(2 * ne));
        rc.resize((size_t)(2 * ne));
        for (int i = 0; i < ne; ++i) {
            const gc_i32 P = peers[(size_t)i];
            own_g[(size_t)i] = pc[(size_t)(3 * P)];
            own_a[(size_t)i] = pc[(size_t)(3 * P + 1)];
            bytes[(size_t)i] = own_g[(size_t)i] * gw + own_a[(size_t)i] * aw;
            /* the bound every rank sized its plan by: refuse, never overrun */
            if (bytes[(size_t)i] > edge_bound && local == GC_OK)
                local = GC_E_CAPACITY;
            sc[(size_t)(2 * i)] = own_g[(size_t)i];
            sc[(size_t)(2 * i + 1)] = own_a[(size_t)i];
        }
        for (size_t m = 0; m < moving.size(); ++m)
            if (moving_row[m])
                rows[(size_t)edge_of[(size_t)moving[m].destination_rank]]
                    .push_back(*moving_row[m]);
        /* Zero-bit rows naming an own mover go to its owner (two movers:
         * both). */
        const auto route_row = [&](const MaskRow &r,
                                   std::vector<std::vector<MaskRow> > &to) {
            const gc_i32 pa = hop_seen.maybe(r.a) ? hop_of.find(r.a) : -1;
            const gc_i32 pb = hop_seen.maybe(r.b) ? hop_of.find(r.b) : -1;
            if (pa >= 0) to[(size_t)edge_of[(size_t)pa]].push_back(r);
            if (pb >= 0 && pb != pa) to[(size_t)edge_of[(size_t)pb]].push_back(r);
        };
        if (mask_host) {
            const std::vector<gc_gid> &mg = ctx->real_mask_gid;
            const std::vector<gc_image> &mi = ctx->real_mask_image;
            for (size_t j = 0; j + 1 < mg.size(); j += 2) {
                const MaskRow r = { mg[j], j < mi.size() ? mi[j] : 0,
                                    mg[j + 1], j + 1 < mi.size() ? mi[j + 1] : 0 };
                route_row(r, mrow);
            }
        }
        if (mask_dev && local == GC_OK) {
            std::vector<std::vector<MaskRow> > dev((size_t)ne);
            for (const auto &c : mine) route_row(c.second, dev);
            for (int i = 0; i < ne; ++i) {
                std::vector<MaskRow> &h = mrow[(size_t)i], &v = dev[(size_t)i];
                if (!mask_host) h.swap(v);
                else if (v.size() != h.size() ||
                         (!v.empty() && std::memcmp(v.data(), h.data(),
                                                    v.size() * sizeof(MaskRow)))) {
                    std::fprintf(stderr, "GPU_Core_Error> phase=migration_rows_"
                                 "device rank=%d peer=%d host=%lld device=%lld\n",
                                 (int)rank, (int)peers[(size_t)i],
                                 (long long)h.size(), (long long)v.size());
                    local = GC_E_MISMATCH;
                }
            }
        }
    }

    /* The plan, regrown only when the bound outgrows it or the neighbours
     * change.  False when the transaction ends here with *ret. */
    bool make_plan(gc_status *ret)
    {
        cap = mc.plan && mc.plan_peers == peers ? mc.plan_cap : 0;
        if (edge_bound > cap) {
            /* half again over the need, at least twice the old room, so
             * creeping payloads do not rebuild the plan at every rebuild */
            cap = std::max(edge_bound + edge_bound / 2, 2 * cap);
            cap = (cap + 255) / 256 * 256;     /* the parcels sit side by side */
            mig_plan_drop(mc);
        }
        if (mc.plan) return true;
        std::vector<gcx_edge_desc> ed((size_t)ne);
        std::memset(ed.data(), 0, ed.size() * sizeof(gcx_edge_desc));
        for (int i = 0; i < ne; ++i) {
            ed[(size_t)i].peer = peers[(size_t)i];
            ed[(size_t)i].axis = -1;
            ed[(size_t)i].dir = 0;
            ed[(size_t)i].key = -1;
            ed[(size_t)i].partner = -1;
            ed[(size_t)i].send_capacity = cap;
            ed[(size_t)i].recv_capacity = cap;
        }
        gcx_migration_desc md;
        std::memset(&md, 0, sizeof(md));
        md.edge = ne ? ed.data() : 0;
        md.num_edges = ne;
        md.tag_base = MIG_TAG_BASE;
        md.deadline_ms = MIG_DEADLINE_MS;
        /* Every rank decided alike whether this transaction builds a plan. */
        gc_status st = gcx_migration_create(&md, (gcx_comm)ctx->comm, &mc.plan);
        mc.plan_cap = st == GC_OK ? cap : 0;
        mc.plan_peers = peers;
        st = native_dist_vote(ctx, st, "migration_plan");
        if (st == GC_OK) return true;
        mig_plan_drop(mc);
        *ret = st;
        return false;
    }

    /* Per edge: the record counts, the moving hydrogen groups' constraint
     * rows and the zero-bit rows; then admission, staging and the payload,
     * cap bytes each.  A failure is this rank's refusal; it still takes
     * part in every exchange. */
    void exchange()
    {
        local = edge_swap2(comm, peers.data(), ne, MIG_TAG_SWAP, local,
                           sc.data(), rows.data(), mrow.data(), rc.data(),
                           rows_in.data(), mrow_in.data());
        for (int i = 0; i < ne; ++i)
            carried.insert(carried.end(), mrow_in[(size_t)i].begin(),
                           mrow_in[(size_t)i].end());

        plan.assign((size_t)ne, RecvPlan());
        std::vector<gcx_buffer> sbuf((size_t)ne), rbuf((size_t)ne);
        std::vector<gc_i64> sbytes((size_t)ne, 0), srec((size_t)ne, 0);
        gbytes.assign((size_t)ne, 0);
        grec.assign((size_t)ne, 0);
        void *d_send = 0;
        bool staged = mig_buf(mc.send, (gc_i64)ne * cap, 0, &d_send) == GC_OK &&
                      mig_buf(mc.recv, (gc_i64)ne * cap, 0, &d_recv) == GC_OK;
        /* Every parcel is [groups | atoms] from the device's packed segments;
         * one launch moves every piece, and the transport's consume waits
         * for it. */
        const gc_i64 hbytes = 2 * (gc_i64)ne * (gc_i64)sizeof(gcn_mig_piece);
        const gc_i64 gw = (gc_i64)sizeof(gcn_group_migration_wire);
        const gc_i64 aw = (gc_i64)sizeof(gcn_atom_migration_wire);
        char *hs = 0;
        void *d_stage = 0;
        if (mig_host(mc.hsend, hbytes, &hs) != GC_OK ||
            mig_buf(mc.stage, hbytes, 0, &d_stage) != GC_OK)
            staged = false;
        gcn_mig_piece *pieces = (gcn_mig_piece *)hs;
        gc_i64 np = 0, maxw = 0;
        for (int i = 0; i < ne; ++i) {
            const gc_i32 P = peers[(size_t)i];
            const gc_status adm = recv_admit(
                &rc[(size_t)(2 * i)], ctx->epoch,
                std::max((gc_i64)MIG_ROOM_GROUPS, d->num_groups),
                std::max((gc_i64)MIG_ROOM_ATOMS, d->num_owned),
                &plan[(size_t)i]);
            if (adm != GC_OK) {
                std::fprintf(stderr,
                    "GPU_Core_Error> phase=migration_admit rank=%d peer=%d "
                    "status=%d recv_counts=%lld,%lld\n", (int)rank,
                    (int)P, (int)adm, (long long)rc[(size_t)(2 * i)],
                    (long long)rc[(size_t)(2 * i + 1)]);
                if (local == GC_OK) local = adm;
            }
            if (!staged) continue;
            char *p = (char *)d_send + (size_t)i * (size_t)cap;
            const auto put = [&](const void *src, gc_i64 nb) {
                if (nb > 0) {
                    pieces[np].src = (const gc_u64 *)src;
                    pieces[np].dst = (gc_u64 *)p;
                    pieces[np].words = nb / 8;
                    maxw = std::max(maxw, nb / 8);
                    ++np;
                }
                p += nb > 0 ? nb : 0;
            };
            put(d->migration_group_send + gbase[(size_t)P], own_g[(size_t)i] * gw);
            put(d->migration_atom_send + abase[(size_t)P], own_a[(size_t)i] * aw);
            sbuf[(size_t)i].ptr = (char *)d_send + (size_t)i * (size_t)cap;
            sbuf[(size_t)i].bytes = cap;
            rbuf[(size_t)i].ptr = (char *)d_recv + (size_t)i * (size_t)cap;
            rbuf[(size_t)i].bytes = cap;
            /* a refusing rank sends an empty parcel so its peer never waits */
            sbytes[(size_t)i] = local == GC_OK ? bytes[(size_t)i] : 0;
            srec[(size_t)i] = local == GC_OK ?
                sc[(size_t)(2 * i)] + sc[(size_t)(2 * i + 1)] : 0;
        }
        if (staged && np > 0) {
            if (cudaMemcpyAsync(d_stage, hs, (size_t)hbytes,
                                cudaMemcpyHostToDevice, d->stream) != cudaSuccess)
                staged = false;
            const unsigned gx =
                (unsigned)std::min<gc_i64>((maxw + GCN_BLOCK - 1) / GCN_BLOCK, 64);
            if (staged) {
                gcn_kern_mig_gather<<<dim3(gx, (unsigned)np), GCN_BLOCK, 0,
                                      d->stream>>>((const gcn_mig_piece *)d_stage);
                if (cudaGetLastError() != cudaSuccess) staged = false;
            }
        }
        if (!mc.producer &&
            cudaEventCreateWithFlags(&mc.producer, cudaEventDisableTiming) !=
                cudaSuccess)
            mc.producer = 0;
        if (!mc.consumer &&
            cudaEventCreateWithFlags(&mc.consumer, cudaEventDisableTiming) !=
                cudaSuccess)
            mc.consumer = 0;
        if (!staged || !mc.producer || !mc.consumer ||
            cudaEventRecord(mc.producer, d->stream) != cudaSuccess) {
            if (local == GC_OK) local = staged ? GC_E_DEVICE : GC_E_NOMEM;
            staged = false;
        }
        /* Collective on the plan: a rank that refused admission still takes
         * part. */
        if (!staged) return;
        gcx_op_desc od;
        std::memset(&od, 0, sizeof(od));
        od.epoch = epoch;
        od.op = GCX_OP_MIGRATE_PAYLOAD;
        od.variable = 1;
        od.send = sbuf.data();
        od.recv = rbuf.data();
        od.send_bytes = sbytes.data();
        od.send_records = srec.data();
        od.recv_bytes = gbytes.data();
        od.recv_records = grec.data();
        od.producer_event = (void *)mc.producer;
        gcx_token *tok = 0;
        gc_status ps = gcx_migration_payload(mc.plan, &od, &tok);
        if (ps == GC_OK) ps = gcx_consume(tok, (void *)d->stream);
        if (ps == GC_OK) {
            if (cudaEventRecord(mc.consumer, d->stream) == cudaSuccess)
                ps = gcx_release(tok, (void *)mc.consumer);
            else { gcx_abort(tok); ps = GC_E_DEVICE; }
        } else if (tok) {
            gcx_abort(tok);
        }
        if (ps != GC_OK && local == GC_OK) local = ps;
    }

    /* Receiver validation of every edge on the rebuild stream, with the
     * records' readback; one synchronisation reads verdicts and records.
     * The validated arrivals are then listed with their atoms in arrival
     * order. */
    void validate()
    {
        const gc_i64 rsb = (gc_i64)sizeof(RecvResult);
        std::vector<gc_i64> tab_at((size_t)ne + 1, 0), in_at((size_t)ne + 1, 0),
            rec_at((size_t)ne + 1, 0);
        for (int i = 0; i < ne; ++i) {
            const RecvPlan &pl = plan[(size_t)i];
            const bool any = pl.groups + pl.atoms > 0;
            std::sort(rows_in[(size_t)i].begin(), rows_in[(size_t)i].end(),
                      rigid_less);
            const gc_i64 nr = (gc_i64)rows_in[(size_t)i].size();
            tab_at[(size_t)i + 1] = tab_at[(size_t)i] + (!any ? 0 :
                ((pl.gtable_cap ? (gc_i64)pl.gtable_cap : 8) +
                 (pl.atable_cap ? (gc_i64)pl.atable_cap : 8)) *
                    (gc_i64)sizeof(gc_u64));
            /* [results of all edges | per edge: row gids, row arities] */
            in_at[(size_t)i + 1] = in_at[(size_t)i] + (!any ? 0 :
                (nr * (gc_i64)(sizeof(gc_gid) + sizeof(gc_i32)) + 7) / 8 * 8);
            rec_at[(size_t)i + 1] = rec_at[(size_t)i] + (!any ? 0 :
                (gc_i64)(pl.group_bytes + pl.atom_bytes));
        }
        const gc_i64 rs_all = (gc_i64)ne * rsb;
        char *hv = 0, *hr = 0;
        void *d_tab = 0, *d_in = 0;
        if (local == GC_OK &&
            (mig_buf(mc.gt, tab_at[(size_t)ne], 0, &d_tab) != GC_OK ||
             mig_buf(mc.rg, rs_all + in_at[(size_t)ne], 0, &d_in) != GC_OK ||
             mig_host(mc.hval, rs_all + in_at[(size_t)ne], &hv) != GC_OK ||
             mig_host(mc.hrecv, rec_at[(size_t)ne], &hr) != GC_OK))
            local = GC_E_NOMEM;
        if (local == GC_OK) {
            const RecvResult init = { 0, 0, (gc_i64)GC_OK, -1, -1 };
            for (int i = 0; i < ne; ++i) {
                std::memcpy(hv + (size_t)i * (size_t)rsb, &init, (size_t)rsb);
                char *q = hv + rs_all + in_at[(size_t)i];
                const gc_i64 nr = in_at[(size_t)i + 1] > in_at[(size_t)i] ?
                                  (gc_i64)rows_in[(size_t)i].size() : 0;
                for (gc_i64 r = 0; r < nr; ++r)
                    std::memcpy(q + (size_t)r * sizeof(gc_gid),
                                &rows_in[(size_t)i][(size_t)r].gid,
                                sizeof(gc_gid));
                for (gc_i64 r = 0; r < nr; ++r)
                    std::memcpy(q + (size_t)nr * sizeof(gc_gid) +
                                    (size_t)r * sizeof(gc_i32),
                                &rows_in[(size_t)i][(size_t)r].arity,
                                sizeof(gc_i32));
            }
            if ((tab_at[(size_t)ne] > 0 &&
                 cudaMemsetAsync(d_tab, 0, (size_t)tab_at[(size_t)ne],
                                 d->stream) != cudaSuccess) ||
                cudaMemcpyAsync(d_in, hv, (size_t)(rs_all + in_at[(size_t)ne]),
                                cudaMemcpyHostToDevice, d->stream) != cudaSuccess)
                local = GC_E_DEVICE;
        }
        /* An error another library left behind is named here, not the
         * validators'. */
        const cudaError_t prior = cudaGetLastError();
        if (prior != cudaSuccess)
            std::fprintf(stderr, "GPU_Core_Error> phase=migration_recv_"
                         "validate rank=%d prior_cuda=%s\n", (int)rank,
                         cudaGetErrorString(prior));
        for (int i = 0; i < ne && local == GC_OK; ++i) {
            const RecvPlan &pl = plan[(size_t)i];
            if (pl.groups + pl.atoms == 0) continue;
            const gc_i64 rb = rec_at[(size_t)i + 1] - rec_at[(size_t)i];
            const char *d_rin = (const char *)d_recv + (size_t)i * (size_t)cap;
            if (rb > 0 &&
                cudaMemcpyAsync(hr + rec_at[(size_t)i], d_rin, (size_t)rb,
                                cudaMemcpyDeviceToHost, d->stream) != cudaSuccess) {
                local = GC_E_DEVICE;
                break;
            }
            const gc_i64 nr = (gc_i64)rows_in[(size_t)i].size();
            const char *din = (const char *)d_in + rs_all + in_at[(size_t)i];
            RecvParams dp;
            dp.expect_epoch = epoch;
            dp.self_rank = rank;
            dp.peer_rank = peers[(size_t)i];
            dp.rigid_gid = (const gc_gid *)din;
            dp.rigid_arity = (const gc_i32 *)(din + (size_t)nr * sizeof(gc_gid));
            dp.rigid_count = nr;
            const gc_status r = recv_validate(
                pl, d_rin, gbytes[(size_t)i], grec[(size_t)i],
                (const gc_gid *)d_owned, nowned,
                (gc_u64 *)((char *)d_tab + tab_at[(size_t)i]), dp,
                (const gcn_layout *)d_layout,
                (RecvResult *)((char *)d_in + (size_t)i * (size_t)rsb),
                d->stream);
            if (r != GC_OK) {
                local = r;
                std::fprintf(stderr, "GPU_Core_Error> phase=migration_recv_"
                             "validate rank=%d peer=%d status=%d\n", (int)rank,
                             (int)peers[(size_t)i], (int)r);
            }
        }
        if (local == GC_OK &&
            (cudaMemcpyAsync(hv, d_in, (size_t)rs_all, cudaMemcpyDeviceToHost,
                             d->stream) != cudaSuccess ||
             cudaStreamSynchronize(d->stream) != cudaSuccess))
            local = GC_E_DEVICE;
        for (int i = 0; i < ne && local == GC_OK; ++i) {
            const RecvPlan &pl = plan[(size_t)i];
            if (pl.groups + pl.atoms == 0) continue;
            RecvResult result;
            std::memcpy(&result, hv + (size_t)i * (size_t)rsb, sizeof(result));
            if (result.code != GC_OK) {
                local = (gc_status)result.code;
                std::fprintf(stderr, "GPU_Core_Error> phase=migration_recv_"
                             "validate rank=%d peer=%d status=%d "
                             "bad_group=%lld bad_atom=%lld\n", (int)rank,
                             (int)peers[(size_t)i], (int)result.code,
                             (long long)result.bad_group,
                             (long long)result.bad_atom);
                break;
            }
            const char *h = hr + rec_at[(size_t)i];
            const gcn_group_migration_wire *g_recv =
                (const gcn_group_migration_wire *)h;
            const gcn_atom_migration_wire *a_recv =
                (const gcn_atom_migration_wire *)(h + pl.group_bytes);
            for (gc_i64 g = 0; g < pl.groups; ++g) {
                const gcn_group_migration_wire &gr = g_recv[g];
                const RigidRow *row =
                    gr.kind == (gc_i32)GC_GROUP_HGROUP ?
                        rigid_find(rows_in[(size_t)i], gr.group_gid) : 0;
                if (gr.kind == (gc_i32)GC_GROUP_HGROUP && !row) {
                    std::fprintf(stderr, "GPU_Core_Error> phase=migration_route "
                                 "rank=%d group=%lld rigid_row=missing\n",
                                 (int)rank, (long long)gr.group_gid);
                    local = GC_E_OWNER;
                    break;
                }
                gcn_group_migration_wire f = gr;
                f.atom_offset = (gc_i64)fin_a.size();
                fin_g.push_back(f);
                fin_a.insert(fin_a.end(), a_recv + gr.atom_offset,
                             a_recv + gr.atom_offset + gr.atom_count);
                if (row) fin_r.push_back(*row);
            }
        }
        if (local != GC_OK) mig_plan_drop(mc);
    }

    /* Destination resolution: arriving atoms append at the owned count in
     * arrival order, each at its group's validated destination cell. */
    void resolve()
    {
        const gc_i64 nfg = (gc_i64)fin_g.size(), nfa = (gc_i64)fin_a.size();
        arr_gid.assign((size_t)(nfa > 0 ? nfa : 1), 0);
        arr_cell.assign(arr_gid.size(), -1);
        const gc_i64 kx = (gc_i64)ctx->geo.cell[0] + 2;
        const gc_i64 ky = (gc_i64)ctx->geo.cell[1] + 2;
        for (gc_i64 i = 0; local == GC_OK && i < nfg; ++i) {
            const gcn_group_migration_wire &g = fin_g[(size_t)i];
            const gc_i32 c = ctx->cell_map.find(
                ((gc_i64)(g.destination_cell[2] + 1) * ky +
                 (g.destination_cell[1] + 1)) * kx + (g.destination_cell[0] + 1));
            if (c < 0 || (gc_i64)c >= ctx->ncell_local) {
                local = GC_E_MISMATCH;
                break;
            }
            for (gc_i64 j = 0; j < g.atom_count; ++j) {
                const gc_i64 idx = g.atom_offset + j;
                arr_gid[(size_t)idx] = fin_a[(size_t)idx].gid;
                arr_cell[(size_t)idx] = gcn_box_index(
                    L, g.destination_cell[0], g.destination_cell[1],
                    g.destination_cell[2]);
            }
        }
    }

    /* The zero-bit rows naming a moved atom and the pre-migration ownership
     * of their endpoints, selected on the device before the commit; the
     * follow runs on the host while the publication's kernels run. */
    void mask_select()
    {
        mf = std::make_shared<MaskFollow>();
        if (local != GC_OK || !mask_dev) return;
        const gc_i64 nfa = (gc_i64)fin_a.size();
        gc_status st = GC_OK;
        std::vector<gc_gid> gone;
        for (const auto &m : mover_hop) gone.push_back(m.first);
        std::vector<gc_gid> came(arr_gid.begin(), arr_gid.begin() + nfa);
        if (!radix_sort(came, [](gc_gid x) { return x; }))
            std::sort(came.begin(), came.end());
        std::vector<gc_gid> q;
        std::merge(gone.begin(), gone.end(), came.begin(), came.end(),
                   std::back_inserter(q));
        const MaskRow *r = 0;
        const gc_i64 *at = 0;
        gc_i64 n = 0;
        std::vector<gc_gid> cu;
        for (const MaskRow &x : carried) { cu.push_back(x.a); cu.push_back(x.b); }
        if (!radix_sort(cu, [](gc_gid x) { return x; }))
            std::sort(cu.begin(), cu.end());
        cu.erase(std::unique(cu.begin(), cu.end()), cu.end());
        gcn_mask_member mm;
        mm.table = (const gc_gid *)mc.owned_tab[mc.owned_cur].p;
        mm.ntable = mc.owned_n;
        mm.extra = cu.data();
        mm.nextra = (gc_i64)cu.size();
        st = native_mask_select(ctx, q.data(), (gc_i64)q.size(), &r, &at, &n,
                                mc.owned_n > 0 ? &mm : 0);
        if (st == GC_OK) mf->named.assign(r, r + n);
        if (st == GC_OK && mm.row) mf->rf.assign(mm.row, mm.row + 2 * n);
        if (st == GC_OK && mm.extra_flag)
            mf->cf.assign(mm.extra_flag, mm.extra_flag + cu.size());
        else if (st == GC_OK && mc.owned_n > 0)
            st = owned_flags(d, mc, cu, &mf->cf);
        mf->cu.swap(cu);
        if (st != GC_OK) {
            std::fprintf(stderr, "GPU_Core_Error> phase=migration_mask_select "
                         "rank=%d status=%d\n", (int)rank, (int)st);
            local = st;
        }
    }

    /* Approved: the device publication, the zero-bit rows following the
     * atoms, the owned and rigid tables, and every kind's term pool of the
     * published epoch, derived on the device from the replicated topology
     * and the owned table. */
    gc_status publish()
    {
        const gc_i64 nfg = (gc_i64)fin_g.size(), nfa = (gc_i64)fin_a.size();
        std::vector<gc_gid> depart((size_t)(n_send_g > 0 ? n_send_g : 1), 0);
        for (gc_i64 i = 0; i < n_send_g; ++i)
            depart[(size_t)i] = moving[(size_t)i].group_gid;
        void *d_fa = 0, *d_fc = 0;
        gc_status st = GC_OK;
        if (nfa > 0) {
            st = mig_buf(mc.fa, nfa * (gc_i64)sizeof(gcn_atom_migration_wire),
                         0, &d_fa);
            if (st == GC_OK)
                st = mig_buf(mc.fc, nfa * (gc_i64)sizeof(gc_i32), 0, &d_fc);
            if (st == GC_OK)
                st = mig_h2d_async(d, d_fa, fin_a.data(),
                                   nfa * (gc_i64)sizeof(gcn_atom_migration_wire));
            if (st == GC_OK)
                st = mig_h2d_async(d, d_fc, arr_cell.data(),
                                   nfa * (gc_i64)sizeof(gc_i32));
        }
        gc_i64 moved = 0;
        if (st == GC_OK)
            st = native_migration_apply(ctx, depart.data(), n_send_g,
                                        nfg > 0 ? fin_g.data() : 0, nfg,
                                        (const gcn_atom_migration_wire *)d_fa,
                                        nfa, (const gc_i32 *)d_fc, &moved);
        /* The zero-bit rows follow the atoms; the real-mask stage reads them. */
        if (st == GC_OK) {
            std::vector<gc_gid> gone;
            for (const auto &m : mover_hop) gone.push_back(m.first);
            std::vector<gc_gid> came(arr_gid.begin(), arr_gid.begin() + nfa);
            std::sort(came.begin(), came.end());
            if (mask_host) {
                KeyMap moved;       /* 1 = left this rank, 2 = arrived */
                GidFilter seen;
                moved.reset((gc_i64)(gone.size() + came.size()));
                for (gc_gid x : gone) { moved.insert(x, 1); seen.add(x); }
                for (gc_gid x : came) { moved.insert(x, 2); seen.add(x); }
                if (owned_host.empty() && nowned > 0) st = read_owned();
                if (st == GC_OK)
                    real_mask_follow(ctx, owned_host, moved, seen, carried);
            }
            if (st == GC_OK && mask_dev) defer_mask_follow(gone, came);
            if (st == GC_OK) st = owned_commit(d, mc, gone, came);
        }
        if (st != GC_OK) mc.owned_n = -1;
        const int owned_pending = st == GC_OK;
        int synced = 0;
        if (st == GC_OK && (!moving_hgroups.empty() || !fin_r.empty())) {
            std::sort(moving_hgroups.begin(), moving_hgroups.end());
            std::sort(fin_r.begin(), fin_r.end(), rigid_less);
            st = rigid_rows_commit(d, mc, moving_hgroups, fin_r);
            synced = st == GC_OK;
        }
        if (owned_pending) {
            const gc_status ov = st == GC_OK ? owned_verify(d, mc, synced) : st;
            if (ov != GC_OK) mc.owned_n = -1;
            if (st == GC_OK) st = ov;
        }
        if (st == GC_OK && owned_pending)
            st = native_terms_derive(ctx,
                                     (const gc_gid *)mc.owned_tab[mc.owned_cur].p,
                                     mc.owned_n);
        else if (st == GC_OK)
            st = GC_E_STATE;
        return st;
    }

    /* The selected rows, replaced by their follow: host work handed to the
     * rebuild's next device waits; the term stage drains it first. */
    void defer_mask_follow(const std::vector<gc_gid> &gone,
                           const std::vector<gc_gid> &came)
    {
        mf->gone = gone;
        mf->came = came;
        mf->carried = carried;
        const std::shared_ptr<MaskFollow> f = mf;
        gc_context *const c = ctx;
        gcn::host_defer([f]() {
            MaskFollow &m = *f;
            std::vector<std::pair<gc_gid, gc_u8> > fl;
            fl.reserve(2 * m.named.size() + m.cu.size());
            for (size_t i = 0; i < m.named.size(); ++i) {
                fl.push_back(std::make_pair(m.named[i].a, m.rf.empty() ? (gc_u8)0 : m.rf[2 * i]));
                fl.push_back(std::make_pair(m.named[i].b, m.rf.empty() ? (gc_u8)0 : m.rf[2 * i + 1]));
            }
            for (size_t j = 0; j < m.cu.size(); ++j)
                fl.push_back(std::make_pair(m.cu[j], m.cf.empty() ? (gc_u8)0 : m.cf[j]));
            /* by (GID, flag): the flag pass, then the stable GID passes */
            radix_sort(fl, [](const std::pair<gc_gid, gc_u8> &x) { return (gc_i64)x.second; });
            if (!radix_sort(fl, [](const std::pair<gc_gid, gc_u8> &x) { return x.first; }))
                std::sort(fl.begin(), fl.end());
            for (const auto &x : fl)
                if (m.u.empty() || m.u.back() != x.first) {
                    m.u.push_back(x.first);
                    m.uf.push_back(x.second);
                }
            return GC_OK;
        });
        gcn::host_defer([f, c]() {
            MaskFollow &m = *f;
            KeyMap moved;           /* 1 = left this rank, 2 = arrived */
            moved.reset((gc_i64)(m.gone.size() + m.came.size()));
            for (gc_gid x : m.gone) moved.insert(x, 1);
            for (gc_gid x : m.came) moved.insert(x, 2);
            const auto owned_before = [&](gc_gid x) {
                const auto it = std::lower_bound(m.u.begin(), m.u.end(), x);
                return it != m.u.end() && *it == x &&
                       m.uf[(size_t)(it - m.u.begin())];
            };
            std::vector<MaskRow> out;
            mask_follow_rows(m.named, moved, owned_before, m.carried, &out);
            return native_mask_replace(c, out.data(), (gc_i64)out.size());
        });
    }
};
#endif

gc_status migration_txn_native_run(gc_context *ctx, gc_i64 *moved_groups,
                                   gc_i64 *moved_atoms, gc_i32 *did_migrate,
                                   const struct gcn_pack_check *pack)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (moved_groups) *moved_groups = 0;
    if (moved_atoms) *moved_atoms = 0;
    if (did_migrate) *did_migrate = 0;
    if (!ctx || !d) return GC_E_ARG;
    if (ctx->nproc == 1) return GC_OK;       /* nothing can be foreign  */
    if (!d->dist) return GC_E_STATE;
#ifndef HAVE_MPI_GENESIS
    (void)pack;
    return GC_E_UNSUPPORTED;
#else
    MigTxn t(ctx);
    gc_status ret = GC_OK;
    const gc_status st = t.find_edges(t.classify(pack));
    if (!t.vote(st, &ret)) return ret;
    if (did_migrate) *did_migrate = 1;
    t.prepare(st);
    if (!t.make_plan(&ret)) return ret;
    t.exchange();
    t.validate();
    t.resolve();
    t.mask_select();
    /* the commit: the checked build votes; the product build returns */
    const gc_status commit = native_dist_step_vote(ctx, t.local,
                                                   "migration_commit");
    if (commit != GC_OK) return commit;
    const gc_status pub = t.publish();
    if (moved_groups) *moved_groups = t.n_send_g;
    if (moved_atoms) *moved_atoms = t.n_send_a;
    /* Past the commit a local failure is no refusal; the ranks leave together. */
    return native_dist_step_vote(ctx, pub, "migration_publish");
#endif
}

}  /* namespace gcn */
