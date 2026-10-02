/*
 * gpu_core.cpp : the device-native core's context, grouped SoA import and
 *                export, canonical term records, group descriptors and the
 *                owner-rule reference checker.
 *
 * Contract: doc/21_GPU_Native.rst; ABI: gpu_core_abi.h. Host-portable C++ with no
 * CUDA and no MPI; device residency is gpu_core_device.cu
 * (gpu_core_device_stub.cpp when CUDA is absent). Reads no environment
 * variable.
 */

#include "gpu_core_internal.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <utility>

/* ---- Tables from the contract, section 4.3 ---- */

extern "C" const gc_i32 gc_term_arity[GC_TERM_NKIND] = {
    2,  /* GC_TERM_BOND     */
    3,  /* GC_TERM_ANGLE    */
    4,  /* GC_TERM_DIHEDRAL */
    4,  /* GC_TERM_IMPROPER */
    8,  /* GC_TERM_CMAP     */
    2,  /* GC_TERM_NB14     */
    2   /* GC_TERM_EXCL     */
};

extern "C" const gc_i32 gc_term_pbc_codes[GC_TERM_NKIND] = {
    1, 3, 3, 3, 6, 0, 0
};

/* One-based endpoint numbers, in GENESIS's listed order. */
extern "C" const gc_i32 gc_term_owner_endpoint[GC_TERM_NKIND][2] = {
    {1, 2},  /* bond     */
    {1, 3},  /* angle    */
    {1, 4},  /* dihedral */
    {1, 4},  /* improper */
    {1, 8},  /* CMAP     */
    {1, 2},  /* 1-4 pair */
    {1, 2}   /* excluded */
};

namespace gcn {

/* ---- Checked integer arithmetic ---- */

static const gc_i64 kI64Max = (gc_i64)0x7fffffffffffffffLL;

bool mul_checked(gc_i64 a, gc_i64 b, gc_i64 *out)
{
    if (a < 0 || b < 0) return false;
    if (a != 0 && b > kI64Max / a) return false;
    *out = a * b;
    return true;
}

bool add_checked(gc_i64 a, gc_i64 b, gc_i64 *out)
{
    if (a < 0 || b < 0) return false;
    if (a > kI64Max - b) return false;
    *out = a + b;
    return true;
}

bool narrow(gc_i64 v, gc_i32 *out)
{
    if (v < 0 || v > (gc_i64)GC_MAX_LOCAL_INDEX) return false;
    *out = (gc_i32)v;
    return true;
}

/* ---- KeyMap ---- */

void KeyMap::reset(gc_i64 entries)
{
    gc_i64 cap = 16;
    while (cap < entries * 2 + 1) cap <<= 1;
    key_.assign((size_t)cap, empty());
    val_.assign((size_t)cap, -1);
    mask_ = cap - 1;
}

static gc_u64 mix64(gc_u64 x)
{
    x ^= x >> 33;
    x *= (gc_u64)0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= (gc_u64)0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

bool KeyMap::insert(gc_i64 key, gc_i32 value)
{
    if (mask_ == 0) return false;
    gc_i64 i = (gc_i64)(mix64((gc_u64)key) & (gc_u64)mask_);
    for (;;) {
        if (key_[(size_t)i] == empty()) {
            key_[(size_t)i] = key;
            val_[(size_t)i] = value;
            return true;
        }
        if (key_[(size_t)i] == key) return false;   /* already present */
        i = (i + 1) & mask_;
    }
}

gc_i32 KeyMap::find(gc_i64 key) const
{
    if (mask_ == 0) return -1;
    gc_i64 i = (gc_i64)(mix64((gc_u64)key) & (gc_u64)mask_);
    for (;;) {
        if (key_[(size_t)i] == empty()) return -1;
        if (key_[(size_t)i] == key) return val_[(size_t)i];
        i = (i + 1) & mask_;
    }
}

gc_i64 KeyMap::bytes() const
{
    return (gc_i64)key_.size() * (gc_i64)sizeof(gc_i64)
         + (gc_i64)val_.size() * (gc_i64)sizeof(gc_i32);
}

namespace {
struct IdentityLess {
    bool operator()(const AtomIdentity &a, const AtomIdentity &b) const {
        if (a.gid != b.gid) return a.gid < b.gid;
        return a.image < b.image;
    }
    bool operator()(const AtomIdentity &a,
                    const std::pair<gc_gid, gc_image> &b) const {
        if (a.gid != b.first) return a.gid < b.first;
        return a.image < b.second;
    }
};
struct IdentityGidLess {
    bool operator()(const AtomIdentity &a, gc_gid gid) const { return a.gid < gid; }
};
}

gc_status AtomIdentityMap::build(const std::vector<gc_gid> &owned,
                                 const std::vector<gc_gid> &ghost_gid,
                                 const std::vector<gc_image> &ghost_image)
{
    if (ghost_gid.size() != ghost_image.size()) return GC_E_ARG;
    gc_i64 total; gc_i32 last;
    if (!add_checked((gc_i64)owned.size(), (gc_i64)ghost_gid.size(), &total))
        return GC_E_OVERFLOW;
    if (total > 0 && !narrow(total - 1, &last)) return GC_E_CAPACITY;
    std::vector<AtomIdentity> next; next.reserve((size_t)total);
    for (size_t i = 0; i < owned.size(); ++i) {
        if (owned[i] <= 0) return GC_E_ARG;
        AtomIdentity a = { owned[i], 0, (gc_i32)i }; next.push_back(a);
    }
    for (size_t i = 0; i < ghost_gid.size(); ++i) {
        if (ghost_gid[i] <= 0 || ((gc_u64)ghost_image[i] >> 48) != 0)
            return GC_E_ARG;
        AtomIdentity a = { ghost_gid[i], ghost_image[i], (gc_i32)(-(gc_i64)i - 1) };
        next.push_back(a);
    }
    /* (gid, image) order: LSD radix sort on the GID (11 bits a pass), then each
     * run of one GID by image. */
    radix_sort(next, [](const AtomIdentity &a) { return a.gid; });  /* GIDs > 0 */
    for (size_t i = 0, j; i < next.size(); i = j) {
        for (j = i + 1; j < next.size() && next[j].gid == next[i].gid; ++j) {}
        if (j - i > 1) std::sort(next.begin() + i, next.begin() + j, IdentityLess());
    }
    for (size_t i = 1; i < next.size(); ++i)
        if (next[i].gid == next[i-1].gid && next[i].image == next[i-1].image)
            return GC_E_MISMATCH;
    entries_.swap(next); return GC_OK;
}

gc_status AtomIdentityMap::find_exact(gc_gid gid, gc_image image, gc_i32 *slot) const
{
    if (!slot) return GC_E_ARG;
    const std::pair<gc_gid,gc_image> key(gid,image);
    std::vector<AtomIdentity>::const_iterator it =
        std::lower_bound(entries_.begin(), entries_.end(), key, IdentityLess());
    if (it == entries_.end() || it->gid != gid || it->image != image)
        return GC_E_ENDPOINT;
    *slot = it->slot; return GC_OK;
}

gc_status AtomIdentityMap::find_unique(gc_gid gid, gc_i32 *slot) const
{
    if (!slot) return GC_E_ARG;
    std::vector<AtomIdentity>::const_iterator it =
        std::lower_bound(entries_.begin(), entries_.end(), gid, IdentityGidLess());
    if (it == entries_.end() || it->gid != gid) return GC_E_ENDPOINT;
    std::vector<AtomIdentity>::const_iterator nx = it + 1;
    if (nx != entries_.end() && nx->gid == gid) return GC_E_MISMATCH;
    *slot = it->slot; return GC_OK;
}

gc_i64 AtomIdentityMap::bytes() const
{
    return (gc_i64)entries_.capacity() * (gc_i64)sizeof(AtomIdentity);
}

/* ---- TermPool ---- */

void TermPool::clear()
{
    count = 0;
    endpoint.clear(); term_id.clear(); param_id.clear(); image.clear(); pbc.clear();
    genesis_cell.clear(); owner_cell.clear(); owned_here.clear();
    param_payload.clear();
    param_count = 0;
}

gc_i64 TermPool::bytes() const
{
    return (gc_i64)(endpoint.size() * sizeof(gc_gid)
                  + term_id.size()  * sizeof(gc_term_id)
                  + param_id.size() * sizeof(gc_i32)
                  + image.size()    * sizeof(gc_i32)
                  + pbc.size()      * sizeof(gc_i32)
                  + genesis_cell.size() * sizeof(gc_i32)
                  + owner_cell.size()   * sizeof(gc_i32)
                  + owned_here.size()   * sizeof(gc_u8));
}

gc_i64 TermPool::execution_bytes() const
{
    return (gc_i64)(endpoint_image.size() * sizeof(gc_image)
                  + endpoint_image_valid.size() * sizeof(gc_u8));
}

gc_status TermPool::resolve(const AtomIdentityMap &map, size_t at,
                            gc_i32 *slot) const
{
    if (at >= endpoint.size() || at >= endpoint_image_valid.size())
        return GC_E_STATE;
    if (endpoint_image_valid[at])
        return map.find_exact(endpoint[at], endpoint_image[at], slot);
    return map.find_unique(endpoint[at], slot);
}

}  /* namespace gcn */

/* ---- Context ---- */

gc_context_s::gc_context_s()
  : rank(0), nproc(1), replica(0), comm(0), epoch(0),
    ncell_local(0), ncell_boundary(0), ncell(0),
    num_owned(0), num_ghost(0),
    num_groups(0), num_group_members(0),
    real_mask_ready(0), native(0)
{
    std::memset(&geo,   0, sizeof(geo));
    std::memset(&guard, 0, sizeof(guard));
    std::memset(native_outer_send, 0, sizeof(native_outer_send));
    std::memset(native_outer_recv, 0, sizeof(native_outer_recv));
    for (int k = 0; k < GC_TERM_NKIND; ++k) {
        term[k].kind  = k;
        term[k].arity = gc_term_arity[k];
    }
}

/* ---- Descriptor validation ---- */

static gc_status validate_state(const gc_state_desc *s)
{
    if (s == 0 || s->geometry == 0 || s->cells == 0 || s->atoms == 0 ||
        s->groups == 0) return GC_E_ARG;
    if (s->num_terms < 0 || s->num_terms > GC_TERM_NKIND) return GC_E_ARG;
    if (s->num_terms > 0 && s->terms == 0) return GC_E_ARG;
    for (gc_i32 i = 0; i < s->num_terms; ++i) {
        const gc_term_desc *t = &s->terms[i];
        if (t->kind < 0 || t->kind >= GC_TERM_NKIND) return GC_E_ARG;
        if (t->arity != gc_term_arity[t->kind]) return GC_E_ARITY;
        if (t->num_term == 0 || t->list == 0) return GC_E_ARG;
        if (gc_term_pbc_codes[t->kind] > 0 &&
            (t->pbc == 0 || t->pbc_cell_stride <= 0)) return GC_E_ARG;
        const bool has_endpoint_image = t->endpoint_image != 0;
        if ((t->endpoint_cell != 0) != has_endpoint_image ||
            (t->endpoint_slot != 0) != has_endpoint_image ||
            t->endpoint_count < 0 ||
            (!has_endpoint_image && t->endpoint_count != 0)) return GC_E_ARG;
    }
    if (s->num_pairterms < 0 || s->num_pairterms > 2) return GC_E_ARG;
    if (s->num_pairterms > 0 && s->pairterms == 0) return GC_E_ARG;
    for (gc_i32 i = 0; i < s->num_pairterms; ++i) {
        const gc_pairterm_desc *pt = &s->pairterms[i];
        if (pt->kind != GC_TERM_NB14 && pt->kind != GC_TERM_EXCL)
            return GC_E_ARG;
        if (pt->num_term == 0 || pt->list == 0) return GC_E_ARG;
        if (pt->list_term_stride != 4) return GC_E_ARG;
        if (pt->ncell_local < 0) return GC_E_ARG;
    }
    const gc_geometry_desc *g = s->geometry;
    if (g->ncell_local < 0 || g->ncell_boundary < 0) return GC_E_ARG;
    for (int d = 0; d < 3; ++d) {
        if (g->cell[d] <= 0) return GC_E_ARG;
        if (g->num_domain[d] <= 0) return GC_E_ARG;
    }
    if (s->cells->ncell != (gc_i64)g->ncell_local + (gc_i64)g->ncell_boundary)
        return GC_E_ARG;
    if (s->atoms->ncell != s->cells->ncell) return GC_E_ARG;
    if (s->cells->cell_l2gx == 0 || s->cells->cell_l2gy == 0 ||
        s->cells->cell_l2gz == 0 || s->cells->cell_tie_key == 0)
        return GC_E_ARG;
    if (s->atoms->num_atom == 0 || s->atoms->gid == 0 || s->atoms->coord == 0)
        return GC_E_ARG;
    return GC_OK;
}

/* ---- Public scalars ---- */

extern "C" const char *gpu_core_status_string(gc_i32 s)
{
    switch (s) {
      case GC_OK:            return "ok";
      case GC_E_ARG:         return "bad argument";
      case GC_E_ABI:         return "abi mismatch";
      case GC_E_CAPACITY:    return "local index capacity";
      case GC_E_UNSUPPORTED: return "unsupported";
      case GC_E_DEVICE:      return "device error";
      case GC_E_NOMEM:       return "out of memory";
      case GC_E_OVERFLOW:    return "integer overflow";
      case GC_E_MISMATCH:    return "exact comparison mismatch";
      case GC_E_EPOCH:       return "epoch changed under a consumer";
      case GC_E_OWNER:       return "owner rule disagreement";
      case GC_E_ENDPOINT:    return "unresolved endpoint";
      case GC_E_ARITY:       return "arity";
      case GC_E_STATE:       return "called out of order";
      default:               return "unknown";
    }
}

/* ---- Eligibility ---- */

extern "C" gc_status gpu_core_try_setup(const gc_state_desc *state,
                                        const char **reason_out)
{
    const char *reason = "eligible";
    gc_status rc = validate_state(state);
    if (rc != GC_OK) {
        if (reason_out) *reason_out = "descriptor";
        return rc;
    }

    /* FP64 is the only native precision (doc/21_GPU_Native.rst); the bridge sets
     * geometry.host_fp64 to 1 for an --enable-double build. */
    if (state->geometry->host_fp64 != 1) {
        if (reason_out) *reason_out = "precision: native core is FP64 only";
        return GC_E_UNSUPPORTED;
    }

    /* Capacity admission (doc/21_GPU_Native.rst): only the cell count can be
     * judged before the atom scan. */
    gc_i64 ncell = state->cells->ncell;
    if (ncell > (gc_i64)GC_MAX_LOCAL_INDEX) {
        if (reason_out) *reason_out = "capacity: cells exceed 32-bit index";
        return GC_E_CAPACITY;
    }

    if (state->geometry->table_support_radius <= 0.0 &&
        state->geometry->prune_support_radius <= 0.0) {
        if (reason_out)
            *reason_out = "no initialized force-table support radius";
        return GC_E_UNSUPPORTED;
    }
    /* The support the list guard uses (the tables' when initialized, else the
     * prune radius): a pair beyond it contributes an exact zero, so the list
     * must exceed it. */
    const gc_f64 support = (state->geometry->table_support_radius > 0.0)
                         ? state->geometry->table_support_radius
                         : state->geometry->prune_support_radius;
    if (state->geometry->pairlistdist <= support) {
        if (reason_out)
            *reason_out = "pairlistdist does not exceed the support radius";
        return GC_E_UNSUPPORTED;
    }

    if (reason_out) *reason_out = reason;
    return GC_OK;
}

extern "C" gc_status gpu_core_create(const gc_state_desc *state,
                                     gc_context **out)
{
    if (out == 0) return GC_E_ARG;
    *out = 0;
    gc_status rc = validate_state(state);
    if (rc != GC_OK) return rc;
    gc_context *ctx = new (std::nothrow) gc_context_s();
    if (ctx == 0) return GC_E_NOMEM;
    ctx->rank    = state->rank;
    ctx->nproc   = state->nproc;
    ctx->replica = state->replica;
    ctx->comm    = state->comm;
    *out = ctx;
    return GC_OK;
}

extern "C" gc_status gpu_core_destroy(gc_context *ctx)
{
    if (ctx == 0) return GC_E_ARG;
    gcn::native_context_release(ctx);
    delete ctx;
    return GC_OK;
}

/* ---- The owner rule: a port of assign_cell_interaction (sp_domain.fpp),
 * doc/21_GPU_Native.rst ---- */

namespace {

inline gc_i32 iabs32(gc_i32 v) { return v < 0 ? -v : v; }

/* The key of a cell's EXTENDED global coordinate: unique, since a wrapped
 * boundary cell is stored at 0 or cell+1. */
inline gc_cell_key cell_key_of(const gc_context *ctx,
                               gc_i32 gx, gc_i32 gy, gc_i32 gz)
{
    const gc_cell_key ex = (gc_cell_key)ctx->geo.cell[0] + 2;
    const gc_cell_key ey = (gc_cell_key)ctx->geo.cell[1] + 2;
    return (((gc_cell_key)gz * ey) + (gc_cell_key)gy) * ex + (gc_cell_key)gx;
}

/* The midpoint regime of assign_cell_interaction, used when at least one cell
 * is a boundary cell. Arguments arrive in GENESIS's order: the periodic shift
 * is antisymmetric and the sums that use it are not. */
gc_i32 cell_pair_midpoint(const gc_context *ctx, gc_i32 a, gc_i32 b)
{
    const gc_i32 x1 = ctx->cell_gx[(size_t)a];
    const gc_i32 y1 = ctx->cell_gy[(size_t)a];
    const gc_i32 z1 = ctx->cell_gz[(size_t)a];
    const gc_i32 x2 = ctx->cell_gx[(size_t)b];
    const gc_i32 y2 = ctx->cell_gy[(size_t)b];
    const gc_i32 z2 = ctx->cell_gz[(size_t)b];

    const gc_i32 cx = ctx->geo.cell[0];
    const gc_i32 cy = ctx->geo.cell[1];
    const gc_i32 cz = ctx->geo.cell[2];

    /* Step 1: minimum image, and the shift (applied only when the axis is not
     * decomposed). */
    gc_i32 dx0 = iabs32(x1 - x2), dxm = iabs32(x1 - x2 - cx), dxp = iabs32(x1 - x2 + cx);
    gc_i32 dy0 = iabs32(y1 - y2), dym = iabs32(y1 - y2 - cy), dyp = iabs32(y1 - y2 + cy);
    gc_i32 dz0 = iabs32(z1 - z2), dzm = iabs32(z1 - z2 - cz), dzp = iabs32(z1 - z2 + cz);

    gc_i32 mx = dx0 < dxm ? dx0 : dxm; if (dxp < mx) mx = dxp;
    gc_i32 my = dy0 < dym ? dy0 : dym; if (dyp < my) my = dyp;
    gc_i32 mz = dz0 < dzm ? dz0 : dzm; if (dzp < mz) mz = dzp;

    gc_i32 movex = 0, movey = 0, movez = 0;
    if      (mx == dxm && ctx->geo.num_domain[0] == 1) movex = -cx;
    else if (mx == dxp && ctx->geo.num_domain[0] == 1) movex =  cx;
    if      (my == dym && ctx->geo.num_domain[1] == 1) movey = -cy;
    else if (my == dyp && ctx->geo.num_domain[1] == 1) movey =  cy;
    if      (mz == dzm && ctx->geo.num_domain[2] == 1) movez = -cz;
    else if (mz == dzp && ctx->geo.num_domain[2] == 1) movez =  cz;

    /* Step 2: the stencil test, on the shifted separation. */
    const gc_i32 sepx = x1 - x2 + movex;
    const gc_i32 sepy = y1 - y2 + movey;
    const gc_i32 sepz = z1 - z2 + movez;
    if (iabs32(sepx) > 2 || iabs32(sepy) > 2 || iabs32(sepz) > 2) return -1;

    /* Steps 3-5.  Note the asymmetry: the x midpoint uses the SHIFTED sum,
     * the y and z midpoints use the UNSHIFTED sum.  The seam test uses the
     * shifted sum on every axis.  This is GENESIS's, not a simplification. */
    const gc_i32 sumx = x1 + x2 + movex;
    const gc_i32 sumy = y1 + y2 + movey;
    const gc_i32 sumz = z1 + z2 + movez;

    const gc_f64 na = ctx->cell_tie[(size_t)a];
    const gc_f64 nb = ctx->cell_tie[(size_t)b];
    const bool   a_lighter = (na < nb);

    gc_i32 i1, i2, i3;

    if (sumx == 2 * ctx->geo.cell_start[0] - 1 ||
        sumx == 2 * ctx->geo.cell_end[0]   + 1) {
        i1 = a_lighter ? x1 : x2;
    } else {
        i1 = sumx / 2;
    }

    if (sumy == 2 * ctx->geo.cell_start[1] - 1 ||
        sumy == 2 * ctx->geo.cell_end[1]   + 1) {
        if (ctx->geo.num_domain[1] > 1) i2 = a_lighter ? y1 : y2;
        else                            i2 = (y1 + y2) / 2;
    } else {
        i2 = (y1 + y2) / 2;
    }

    if (sumz == 2 * ctx->geo.cell_start[2] - 1 ||
        sumz == 2 * ctx->geo.cell_end[2]   + 1) {
        if (ctx->geo.num_domain[2] > 1) i3 = a_lighter ? z1 : z2;
        else                            i3 = (z1 + z2) / 2;
    } else {
        i3 = (z1 + z2) / 2;
    }

    if (i1 < 0 || i1 > cx + 1 || i2 < 0 || i2 > cy + 1 ||
        i3 < 0 || i3 > cz + 1)
        return -1;

    return ctx->cell_map.find(cell_key_of(ctx, i1, i2, i3));
}

/* The whole rule; assign_cell_interaction has four regimes that do not agree:
 *   a == b                 the cell itself;
 *   both local             the LOWER local index, when the cells are within the
 *                          +-2 stencil under the minimum image (no midpoint,
 *                          no tie key);
 *   one local, one ghost   the midpoint regime, LOCAL cell first;
 *   both ghosts            the midpoint regime, LOWER index first.
 * Returns the local cell index of the pair cell, or -1 when the pair is
 * outside the stencil or its cell is not this rank's. */
gc_i32 cell_pair_rule(const gc_context *ctx, gc_i32 a, gc_i32 b)
{
    if (a < 0 || b < 0 || (gc_i64)a >= ctx->ncell || (gc_i64)b >= ctx->ncell)
        return -1;
    if (a == b) return a;

    const bool a_local = ((gc_i64)a < ctx->ncell_local);
    const bool b_local = ((gc_i64)b < ctx->ncell_local);

    if (a_local && b_local) {
        const gc_i32 cx = ctx->geo.cell[0];
        const gc_i32 cy = ctx->geo.cell[1];
        const gc_i32 cz = ctx->geo.cell[2];
        const gc_i32 dx = ctx->cell_gx[(size_t)a] - ctx->cell_gx[(size_t)b];
        const gc_i32 dy = ctx->cell_gy[(size_t)a] - ctx->cell_gy[(size_t)b];
        const gc_i32 dz = ctx->cell_gz[(size_t)a] - ctx->cell_gz[(size_t)b];
        gc_i32 mx = iabs32(dx); if (iabs32(dx - cx) < mx) mx = iabs32(dx - cx);
        if (iabs32(dx + cx) < mx) mx = iabs32(dx + cx);
        gc_i32 my = iabs32(dy); if (iabs32(dy - cy) < my) my = iabs32(dy - cy);
        if (iabs32(dy + cy) < my) my = iabs32(dy + cy);
        gc_i32 mz = iabs32(dz); if (iabs32(dz - cz) < mz) mz = iabs32(dz - cz);
        if (iabs32(dz + cz) < mz) mz = iabs32(dz + cz);
        if (mx > 2 || my > 2 || mz > 2) return -1;
        return (a < b) ? a : b;
    }

    if (a_local)  return cell_pair_midpoint(ctx, a, b);
    if (b_local)  return cell_pair_midpoint(ctx, b, a);
    return cell_pair_midpoint(ctx, (a < b) ? a : b, (a < b) ? b : a);
}

}  /* anonymous namespace */

/* ---- Import ---- */

namespace {

struct StagedAtom {
    gc_gid gid;
    gc_i32 cell;
    gc_i32 slot;      /* one-based slot inside the cell                  */
};

struct GroupRec {
    gc_cell_key key;
    gc_u8       kind;
    gc_gid      gid;      /* representative GID                          */
    gc_i32      cell;
    gc_i32      first;    /* index into the staged member list           */
    gc_i32      nmember;
};

bool group_less(const GroupRec &l, const GroupRec &r)
{
    if (l.key  != r.key)  return l.key  < r.key;
    if (l.kind != r.kind) return l.kind < r.kind;
    return l.gid < r.gid;
}

/* Orders parameter payloads by their bytes so equal payloads form one run. */
struct PayloadLess {
    const gc_f64 *pay;
    gc_i64 stride;
    PayloadLess(const gc_f64 *p, gc_i64 s) : pay(p), stride(s) {}
    bool operator()(gc_i64 x, gc_i64 y) const {
        return std::memcmp(&pay[x * stride], &pay[y * stride],
                           (size_t)(stride * (gc_i64)sizeof(gc_f64))) < 0;
    }
};

/* Canonical-key comparator over a term pool (doc/21_GPU_Native.rst): the whole
 * immutable record (kind, ordered endpoint GIDs, parameter payload, image
 * flag). Endpoints alone are not an identity: CHARMM gives one dihedral
 * quartet several terms differing in periodicity, phase and force constant.
 * Parameters are compared by payload bytes, not local id, so ranks that number
 * a parameter differently still agree. Returns -1, 0 or +1. */
int term_key_cmp(const gcn::TermPool &p, gc_i64 x, gc_i64 y)
{
    const gc_gid *a = &p.endpoint[(size_t)(x * p.arity)];
    const gc_gid *b = &p.endpoint[(size_t)(y * p.arity)];
    for (gc_i32 e = 0; e < p.arity; ++e) {
        if (a[e] != b[e]) return (a[e] < b[e]) ? -1 : 1;
    }
    if (p.param_stride > 0) {
        const gc_f64 *pa =
            &p.param_payload[(size_t)((gc_i64)p.param_id[(size_t)x] * p.param_stride)];
        const gc_f64 *pb =
            &p.param_payload[(size_t)((gc_i64)p.param_id[(size_t)y] * p.param_stride)];
        const int c = std::memcmp(pa, pb,
                                  (size_t)(p.param_stride * (gc_i64)sizeof(gc_f64)));
        if (c != 0) return (c < 0) ? -1 : 1;
    }
    if (p.image[(size_t)x] != p.image[(size_t)y])
        return (p.image[(size_t)x] < p.image[(size_t)y]) ? -1 : 1;
    return 0;
}

/* A total order over the pool, so that the stable ids assigned at import
 * and the order the export emits are the same order. */
struct KeyLess {
    const gcn::TermPool *p;
    explicit KeyLess(const gcn::TermPool *pp) : p(pp) {}
    bool operator()(gc_i64 x, gc_i64 y) const {
        const int c = term_key_cmp(*p, x, y);
        if (c != 0) return c < 0;
        return x < y;
    }
};

}  /* anonymous namespace */

static gc_status gpu_core_import_host(gc_context *ctx, const gc_state_desc *s)
{
    if (ctx == 0) return GC_E_ARG;
    gc_status rc = validate_state(s);
    if (rc != GC_OK) return rc;

    /* An import replaces the atom/cell epoch, so the old GID exclusion set is
     * invalid even if this import refuses; the bridge supplies a fresh stock
     * mask after every successful import. */
    ctx->real_mask_ready = 0;
    ctx->real_mask_gid.clear();

    const gc_geometry_desc *g  = s->geometry;
    const gc_cell_desc     *cd = s->cells;
    const gc_atom_desc     *ad = s->atoms;
    const gc_group_desc    *gd = s->groups;

    ctx->geo            = *g;
    ctx->ncell_local    = g->ncell_local;
    ctx->ncell_boundary = g->ncell_boundary;
    ctx->ncell          = cd->ncell;
    ctx->rank           = s->rank;
    ctx->nproc          = s->nproc;
    ctx->replica        = s->replica;
    ctx->comm           = s->comm;

    const gc_i64 ncell = ctx->ncell;

    /* ---------------- cells ---------------- */
    ctx->cell_gx.resize((size_t)ncell);
    ctx->cell_gy.resize((size_t)ncell);
    ctx->cell_gz.resize((size_t)ncell);
    ctx->cell_key.resize((size_t)ncell);
    ctx->cell_tie.resize((size_t)ncell);
    ctx->cell_map.reset(ncell);

    for (gc_i64 i = 0; i < ncell; ++i) {
        /* one-based EXTENDED coordinates (a wrapped boundary cell is at 0 or
         * cell+1), kept as they arrive */
        gc_i32 gx = cd->cell_l2gx[(size_t)i];
        gc_i32 gy = cd->cell_l2gy[(size_t)i];
        gc_i32 gz = cd->cell_l2gz[(size_t)i];
        if (gx < 0 || gx > g->cell[0] + 1 ||
            gy < 0 || gy > g->cell[1] + 1 ||
            gz < 0 || gz > g->cell[2] + 1) return GC_E_ARG;
        ctx->cell_gx[(size_t)i] = gx;
        ctx->cell_gy[(size_t)i] = gy;
        ctx->cell_gz[(size_t)i] = gz;
        const gc_cell_key key = cell_key_of(ctx, gx, gy, gz);
        ctx->cell_key[(size_t)i] = key;
        ctx->cell_tie[(size_t)i] = cd->cell_tie_key[(size_t)i];
        /* extended coordinates are unique: a shared key is a malformed geometry */
        if (!ctx->cell_map.insert(key, (gc_i32)i)) return GC_E_UNSUPPORTED;
    }

    /* ---------------- stage the padded atom slots ---------------- */
    gc_i64 owned_total = 0, ghost_total = 0;
    for (gc_i64 i = 0; i < ncell; ++i) {
        gc_i64 n = (gc_i64)ad->num_atom[(size_t)i];
        if (n < 0) return GC_E_ARG;
        if (i < ctx->ncell_local) {
            if (!gcn::add_checked(owned_total, n, &owned_total))
                return GC_E_OVERFLOW;
        } else {
            if (!gcn::add_checked(ghost_total, n, &ghost_total))
                return GC_E_OVERFLOW;
        }
    }
    {
        gc_i32 dummy;
        gc_i64 total = 0;
        if (!gcn::add_checked(owned_total, ghost_total, &total))
            return GC_E_OVERFLOW;
        if (!gcn::narrow(total, &dummy)) return GC_E_CAPACITY;
    }

    /* cell_base[i]: first staged index of cell i's atoms in (cell, slot) order,
     * without padding */
    std::vector<gc_i64> cell_base((size_t)ncell + 1, 0);
    for (gc_i64 i = 0; i < ncell; ++i)
        cell_base[(size_t)i + 1] = cell_base[(size_t)i] + ad->num_atom[(size_t)i];

    const gc_i64 staged_total = cell_base[(size_t)ncell];
    std::vector<StagedAtom> staged((size_t)staged_total);
    for (gc_i64 i = 0; i < ncell; ++i) {
        const gc_i64 n    = ad->num_atom[(size_t)i];
        const gc_i64 base = cell_base[(size_t)i];
        for (gc_i64 k = 0; k < n; ++k) {
            gc_i64 src;
            if (!gcn::mul_checked(i, ad->gid_cell_stride, &src))
                return GC_E_OVERFLOW;
            src += k;
            StagedAtom &a = staged[(size_t)(base + k)];
            a.gid  = (gc_gid)ad->gid[(size_t)src];
            a.cell = (gc_i32)i;
            a.slot = (gc_i32)(k + 1);
            if (a.gid <= 0) return GC_E_ARG;
        }
    }

    /* ---------------- ghosts ---------------- */
    ctx->num_ghost = ghost_total;
    ctx->ghost_gid.resize((size_t)ghost_total);
    ctx->ghost_image.resize((size_t)ghost_total);
    ctx->ghost_coord.resize((size_t)ghost_total * 3);
    ctx->ghost_charge.resize((size_t)ghost_total);
    ctx->ghost_cls.resize((size_t)ghost_total);
    ctx->ghost_cell.resize((size_t)ghost_total);
    {
        gc_i64 w = 0;
        for (gc_i64 i = ctx->ncell_local; i < ncell; ++i) {
            /* image of every ghost in this cell: (extended - in-range coordinate)
             * / cell count, per axis */
            gc_i32 ix = 0, iy = 0, iz = 0;
            if (cd->cell_l2gx_orig && cd->cell_l2gy_orig && cd->cell_l2gz_orig) {
                ix = (ctx->cell_gx[(size_t)i] - cd->cell_l2gx_orig[(size_t)i]) / g->cell[0];
                iy = (ctx->cell_gy[(size_t)i] - cd->cell_l2gy_orig[(size_t)i]) / g->cell[1];
                iz = (ctx->cell_gz[(size_t)i] - cd->cell_l2gz_orig[(size_t)i]) / g->cell[2];
            }
            const gc_image img = ((gc_image)(gc_i16)ix & 0xffffLL)
                               | (((gc_image)(gc_i16)iy & 0xffffLL) << 16)
                               | (((gc_image)(gc_i16)iz & 0xffffLL) << 32);
            const gc_i64 n    = ad->num_atom[(size_t)i];
            const gc_i64 base = cell_base[(size_t)i];
            for (gc_i64 k = 0; k < n; ++k, ++w) {
                const gc_i64 sa = base + k;
                gc_i64 xyz;
                if (!gcn::mul_checked(i, ad->xyz_cell_stride, &xyz))
                    return GC_E_OVERFLOW;
                xyz += k * 3;
                gc_i64 sc;
                if (!gcn::mul_checked(i, ad->gid_cell_stride, &sc))
                    return GC_E_OVERFLOW;
                sc += k;
                ctx->ghost_gid[(size_t)w]   = staged[(size_t)sa].gid;
                ctx->ghost_image[(size_t)w] = img;
                ctx->ghost_coord[(size_t)(w * 3 + 0)] = ad->coord[(size_t)(xyz + 0)];
                ctx->ghost_coord[(size_t)(w * 3 + 1)] = ad->coord[(size_t)(xyz + 1)];
                ctx->ghost_coord[(size_t)(w * 3 + 2)] = ad->coord[(size_t)(xyz + 2)];
                ctx->ghost_charge[(size_t)w] = ad->charge ? ad->charge[(size_t)sc] : 0.0;
                ctx->ghost_cls[(size_t)w]    = ad->atom_cls ? ad->atom_cls[(size_t)sc] : 0;
                ctx->ghost_cell[(size_t)w]   = (gc_i32)i;
            }
        }
    }

    /* ---------------- groups over owned atoms ---------------- */
    std::vector<gc_u8> claimed((size_t)cell_base[(size_t)ctx->ncell_local], 0);
    std::vector<GroupRec> groups;
    std::vector<gc_i32>   members;   /* staged indices, in group order    */
    groups.reserve((size_t)owned_total);
    members.reserve((size_t)owned_total);

    if (gd->num_water && gd->water_list && gd->water_atom_count > 0) {
        for (gc_i64 i = 0; i < ctx->ncell_local; ++i) {
            const gc_i64 nw = gd->num_water[(size_t)i];
            for (gc_i64 iw = 0; iw < nw; ++iw) {
                GroupRec r;
                r.key     = ctx->cell_key[(size_t)i];
                r.kind    = (gc_u8)GC_GROUP_WATER;
                r.cell    = (gc_i32)i;
                r.first   = (gc_i32)members.size();
                r.nmember = gd->water_atom_count;
                for (gc_i32 m = 0; m < gd->water_atom_count; ++m) {
                    gc_i64 idx;
                    if (!gcn::mul_checked(i, gd->water_cell_stride, &idx))
                        return GC_E_OVERFLOW;
                    idx += iw * gd->water_atom_count + m;
                    const gc_i32 slot = gd->water_list[(size_t)idx];
                    if (slot < 1 || (gc_i64)slot > ad->num_atom[(size_t)i])
                        return GC_E_ARITY;
                    const gc_i64 sa = cell_base[(size_t)i] + (slot - 1);
                    if (claimed[(size_t)sa]) return GC_E_ARITY;
                    claimed[(size_t)sa] = 1;
                    members.push_back((gc_i32)sa);
                    if (m == 0) r.gid = staged[(size_t)sa].gid;
                }
                groups.push_back(r);
            }
        }
    }

    if (gd->hgr_local && gd->hgr_bond_list && gd->hgr_max_h > 0) {
        if (gd->hgr_max_h > GC_MAX_HGROUP_H) return GC_E_ARITY;
        for (gc_i64 i = 0; i < ctx->ncell_local; ++i) {
            for (gc_i32 j = 1; j <= gd->hgr_max_h; ++j) {
                gc_i64 li;
                if (!gcn::mul_checked(i, gd->hgr_local_cell_stride, &li))
                    return GC_E_OVERFLOW;
                li += (gc_i64)(j - 1) * gd->hgr_local_h_stride;
                const gc_i64 ng = gd->hgr_local[(size_t)li];
                for (gc_i64 k = 0; k < ng; ++k) {
                    GroupRec r;
                    r.key     = ctx->cell_key[(size_t)i];
                    r.kind    = (gc_u8)GC_GROUP_HGROUP;
                    r.cell    = (gc_i32)i;
                    r.first   = (gc_i32)members.size();
                    r.nmember = j + 1;   /* heavy atom plus j hydrogens */
                    for (gc_i32 m = 0; m < j + 1; ++m) {
                        gc_i64 idx;
                        if (!gcn::mul_checked(i, gd->hgr_list_cell_stride, &idx))
                            return GC_E_OVERFLOW;
                        idx += (gc_i64)(j - 1) * gd->hgr_list_h_stride
                             + k * gd->hgr_list_group_stride
                             + (gc_i64)m * gd->hgr_list_member_stride;
                        const gc_i32 slot = gd->hgr_bond_list[(size_t)idx];
                        if (slot < 1 || (gc_i64)slot > ad->num_atom[(size_t)i])
                            return GC_E_ARITY;
                        const gc_i64 sa = cell_base[(size_t)i] + (slot - 1);
                        if (claimed[(size_t)sa]) return GC_E_ARITY;
                        claimed[(size_t)sa] = 1;
                        members.push_back((gc_i32)sa);
                        if (m == 0) r.gid = staged[(size_t)sa].gid;
                    }
                    groups.push_back(r);
                }
            }
        }
    }

    /* everything else is a singleton */
    for (gc_i64 i = 0; i < ctx->ncell_local; ++i) {
        const gc_i64 n    = ad->num_atom[(size_t)i];
        const gc_i64 base = cell_base[(size_t)i];
        for (gc_i64 k = 0; k < n; ++k) {
            const gc_i64 sa = base + k;
            if (claimed[(size_t)sa]) continue;
            claimed[(size_t)sa] = 1;
            GroupRec r;
            r.key     = ctx->cell_key[(size_t)i];
            r.kind    = (gc_u8)GC_GROUP_SINGLE;
            r.cell    = (gc_i32)i;
            r.gid     = staged[(size_t)sa].gid;
            r.first   = (gc_i32)members.size();
            r.nmember = 1;
            members.push_back((gc_i32)sa);
            groups.push_back(r);
        }
    }

    /* Every owned atom in exactly one group (doc/21_GPU_Native.rst). */
    if ((gc_i64)members.size() != owned_total) return GC_E_ARITY;

    /* ---------------- sort groups, scatter members ---------------- */
    std::stable_sort(groups.begin(), groups.end(), group_less);

    ctx->num_groups        = (gc_i64)groups.size();
    ctx->num_group_members = owned_total;
    ctx->num_owned         = owned_total;

    {
        gc_i32 dummy;
        if (!gcn::narrow(ctx->num_groups, &dummy)) return GC_E_CAPACITY;
        if (!gcn::narrow(ctx->num_owned,  &dummy)) return GC_E_CAPACITY;
    }

    ctx->group_offset.assign((size_t)ctx->num_groups + 1, 0);
    ctx->group_member.resize((size_t)owned_total);
    ctx->group_rep.resize((size_t)ctx->num_groups);
    ctx->group_kind.resize((size_t)ctx->num_groups);
    ctx->group_gid.resize((size_t)ctx->num_groups);
    ctx->group_cell.resize((size_t)ctx->num_groups);

    ctx->atom_gid.resize((size_t)owned_total);
    ctx->atom_coord.resize((size_t)owned_total * 3);
    ctx->atom_vel.resize((size_t)owned_total * 3);
    ctx->atom_charge.resize((size_t)owned_total);
    ctx->atom_mass.resize((size_t)owned_total);
    ctx->atom_inv_mass.resize((size_t)owned_total);
    ctx->atom_cls.resize((size_t)owned_total);
    ctx->atom_cell.resize((size_t)owned_total);
    ctx->atom_group.resize((size_t)owned_total);

    {
        gc_i64 w = 0;
        for (gc_i64 gi = 0; gi < ctx->num_groups; ++gi) {
            const GroupRec &r = groups[(size_t)gi];
            ctx->group_offset[(size_t)gi] = w;
            ctx->group_rep[(size_t)gi]    = (gc_i32)w;
            ctx->group_kind[(size_t)gi]   = r.kind;
            ctx->group_gid[(size_t)gi]    = r.gid;
            ctx->group_cell[(size_t)gi]   = r.cell;
            for (gc_i32 m = 0; m < r.nmember; ++m, ++w) {
                /* each member is scattered into its final compact owned slot */
                ctx->group_member[(size_t)w] = (gc_i32)w;
                const gc_i64 sa = members[(size_t)(r.first + m)];
                const StagedAtom &a = staged[(size_t)sa];
                const gc_i64 icel = a.cell;
                const gc_i64 k    = a.slot - 1;
                gc_i64 xyz, sc;
                if (!gcn::mul_checked(icel, ad->xyz_cell_stride, &xyz))
                    return GC_E_OVERFLOW;
                xyz += k * 3;
                if (!gcn::mul_checked(icel, ad->gid_cell_stride, &sc))
                    return GC_E_OVERFLOW;
                sc += k;
                ctx->atom_gid[(size_t)w] = a.gid;
                ctx->atom_coord[(size_t)(w * 3 + 0)] = ad->coord[(size_t)(xyz + 0)];
                ctx->atom_coord[(size_t)(w * 3 + 1)] = ad->coord[(size_t)(xyz + 1)];
                ctx->atom_coord[(size_t)(w * 3 + 2)] = ad->coord[(size_t)(xyz + 2)];
                if (ad->velocity) {
                    ctx->atom_vel[(size_t)(w * 3 + 0)] = ad->velocity[(size_t)(xyz + 0)];
                    ctx->atom_vel[(size_t)(w * 3 + 1)] = ad->velocity[(size_t)(xyz + 1)];
                    ctx->atom_vel[(size_t)(w * 3 + 2)] = ad->velocity[(size_t)(xyz + 2)];
                }
                ctx->atom_charge[(size_t)w]   = ad->charge   ? ad->charge[(size_t)sc]   : 0.0;
                ctx->atom_mass[(size_t)w]     = ad->mass     ? ad->mass[(size_t)sc]     : 0.0;
                ctx->atom_inv_mass[(size_t)w] = ad->inv_mass ? ad->inv_mass[(size_t)sc] : 0.0;
                ctx->atom_cls[(size_t)w]      = ad->atom_cls ? ad->atom_cls[(size_t)sc] : 0;
                ctx->atom_cell[(size_t)w]     = a.cell;
                ctx->atom_group[(size_t)w]    = (gc_i32)gi;
            }
        }
        ctx->group_offset[(size_t)ctx->num_groups] = w;
        if (w != owned_total) return GC_E_ARITY;
    }

    /* ---------------- GID map ---------------- */
    ctx->gid_map.reset(owned_total + ghost_total);
    for (gc_i64 i = 0; i < owned_total; ++i) {
        if (!ctx->gid_map.insert(ctx->atom_gid[(size_t)i], (gc_i32)i)) {
            /* One rank owning one GID twice is a malformed state. */
            return GC_E_ARITY;
        }
    }
    for (gc_i64 i = 0; i < ghost_total; ++i) {
        /* A ghost of a GID this rank also owns keeps the owned entry; a
         * ghost of a remote GID is encoded as -(index)-1. */
        ctx->gid_map.insert(ctx->ghost_gid[(size_t)i], (gc_i32)(-i - 1));
    }

    /* GID alone is not a local identity once periodic copies are admitted:
     * keep the legacy map for one-rank paths, the exact map for topology and
     * distributed execution. */
    {
        const gc_status idst = ctx->identity_map.build(
            ctx->atom_gid, ctx->ghost_gid, ctx->ghost_image);
        if (idst != GC_OK) return idst;
    }

    /* ---------------- terms ---------------- */
    for (int k = 0; k < GC_TERM_NKIND; ++k) ctx->term[k].clear();

    for (gc_i32 ti = 0; ti < s->num_terms; ++ti) {
        const gc_term_desc *td = &s->terms[ti];
        gcn::TermPool &p = ctx->term[td->kind];
        p.kind  = td->kind;
        p.arity = td->arity;

        gc_i64 total = 0;
        for (gc_i64 i = 0; i < td->ncell; ++i) {
            if (td->num_term[(size_t)i] < 0) return GC_E_ARG;
            if (!gcn::add_checked(total, td->num_term[(size_t)i], &total))
                return GC_E_OVERFLOW;
        }
        {
            gc_i32 dummy;
            if (!gcn::narrow(total, &dummy)) return GC_E_CAPACITY;
        }

        gc_i64 nendpoint;
        if (!gcn::mul_checked(total, (gc_i64)td->arity, &nendpoint))
            return GC_E_OVERFLOW;
        if (td->endpoint_image && td->endpoint_count != nendpoint)
            return GC_E_ARG;
        p.count = total;
        p.endpoint.resize((size_t)nendpoint);
        p.endpoint_image.assign((size_t)nendpoint, 0);
        p.endpoint_image_valid.assign((size_t)nendpoint, 0);
        p.term_id.resize((size_t)total);
        p.param_id.assign((size_t)total, 0);
        p.image.assign((size_t)total, 0);
        const gc_i32 ncodes = gc_term_pbc_codes[td->kind];
        gc_i64 pbc_total;
        if (!gcn::mul_checked(total, (gc_i64)ncodes, &pbc_total))
            return GC_E_OVERFLOW;
        p.pbc.resize((size_t)pbc_total);
        p.genesis_cell.resize((size_t)total);
        p.owner_cell.assign((size_t)total, -1);
        p.owned_here.assign((size_t)total, 0);

        if (td->param_nreal < 0 || td->param_nreal > GC_MAX_PARAM_FIELD ||
            td->param_nint  < 0 || td->param_nint  > GC_MAX_PARAM_FIELD)
            return GC_E_ARG;
        for (gc_i32 fr = 0; fr < td->param_nreal; ++fr)
            if (td->param_real[fr] == 0) return GC_E_ARG;
        for (gc_i32 fi = 0; fi < td->param_nint; ++fi)
            if (td->param_int[fi] == 0) return GC_E_ARG;

        const gc_i64 pstride = (gc_i64)td->param_nreal + (gc_i64)td->param_nint;
        p.param_stride = pstride;
        std::vector<gc_f64> pay;
        if (pstride > 0 && total > 0) {
            gc_i64 bytes;
            if (!gcn::mul_checked(total, pstride, &bytes)) return GC_E_OVERFLOW;
            pay.assign((size_t)bytes, 0.0);
        }

        gc_i64 w = 0;
        for (gc_i64 i = 0; i < td->ncell; ++i) {
            const gc_i64 n = td->num_term[(size_t)i];
            if (ncodes > 0 && n > td->pbc_cell_stride / ncodes)
                return GC_E_ARG;
            for (gc_i64 t = 0; t < n; ++t, ++w) {
                gc_i64 base;
                if (!gcn::mul_checked(i, td->list_cell_stride, &base))
                    return GC_E_OVERFLOW;
                base += t * td->arity;
                for (gc_i32 e = 0; e < td->arity; ++e) {
                    const gc_gid gid = (gc_gid)td->list[(size_t)(base + e)];
                    if (gid <= 0) return GC_E_ARG;
                    const size_t at = (size_t)(w * td->arity + e);
                    p.endpoint[at] = gid;
                    gc_i32 slot = 0;
                    if (td->endpoint_image) {
                        const gc_i32 source_cell = td->endpoint_cell[at];
                        const gc_i32 source_slot = td->endpoint_slot[at];
                        if (source_cell < 0 || (gc_i64)source_cell >= ad->ncell ||
                            source_slot < 0 || source_slot >= ad->num_atom[(size_t)source_cell])
                            return GC_E_ENDPOINT;
                        gc_i64 source_offset;
                        if (!gcn::mul_checked(source_cell, ad->gid_cell_stride,
                                              &source_offset) ||
                            !gcn::add_checked(source_offset, source_slot,
                                              &source_offset))
                            return GC_E_OVERFLOW;
                        if ((gc_gid)ad->gid[(size_t)source_offset] != gid)
                            return GC_E_MISMATCH;
                        const gc_image image = td->endpoint_image[at];
                        if (((gc_u64)image >> 48) != 0) return GC_E_ENDPOINT;
                        p.endpoint_image[at] = image;
                        p.endpoint_image_valid[at] = 1;
                    } else {
                        const gc_status found = ctx->identity_map.find_unique(gid, &slot);
                        if (found == GC_E_MISMATCH) return GC_E_UNSUPPORTED;
                        if (found == GC_OK) {
                            p.endpoint_image[at] = slot >= 0 ? 0
                                : ctx->ghost_image[(size_t)(-slot - 1)];
                            p.endpoint_image_valid[at] = 1;
                        }
                    }
                }
                p.genesis_cell[(size_t)w] = (gc_i32)i;
                if (ncodes > 0) {
                    gc_i64 cb;
                    if (!gcn::mul_checked(i, td->pbc_cell_stride, &cb))
                        return GC_E_OVERFLOW;
                    for (gc_i32 e = 0; e < ncodes; ++e) {
                        const gc_i32 code = td->pbc[(size_t)(cb + t * ncodes + e)];
                        if (code < 0 || code > 26) return GC_E_ARG;
                        p.pbc[(size_t)(w * ncodes + e)] = code;
                    }
                }
                if (pstride > 0) {
                    gc_i64 pb;
                    if (!gcn::mul_checked(i, td->param_cell_stride, &pb))
                        return GC_E_OVERFLOW;
                    gc_f64 *slot = &pay[(size_t)(w * pstride)];
                    gc_i64 d = 0;
                    for (gc_i32 fr = 0; fr < td->param_nreal; ++fr, ++d)
                        slot[d] = td->param_real[fr][(size_t)(pb + t)];
                    /* a 32-bit integer is exactly representable as a double */
                    for (gc_i32 fi = 0; fi < td->param_nint; ++fi, ++d)
                        slot[d] = (gc_f64)td->param_int[fi][(size_t)(pb + t)];
                }
            }
        }

        /* Deduplicated parameter records by exact bit pattern
         * (doc/21_GPU_Native.rst): sort the payloads and walk runs of equal ones,
         * O(n log n). */
        if (pstride > 0 && total > 0) {
            std::vector<gc_i64> order((size_t)total);
            for (gc_i64 t = 0; t < total; ++t) order[(size_t)t] = t;
            std::sort(order.begin(), order.end(), PayloadLess(&pay[0], pstride));

            const gc_i64 nbytes = pstride * (gc_i64)sizeof(gc_f64);
            gc_i64 uniq = -1;
            for (gc_i64 o = 0; o < total; ++o) {
                const gc_i64 t = order[(size_t)o];
                const gc_f64 *want = &pay[(size_t)(t * pstride)];
                if (uniq < 0 ||
                    std::memcmp(want,
                                &p.param_payload[(size_t)(uniq * pstride)],
                                (size_t)nbytes) != 0) {
                    ++uniq;
                    for (gc_i64 d = 0; d < pstride; ++d)
                        p.param_payload.push_back(want[(size_t)d]);
                }
                gc_i32 narrowed;
                if (!gcn::narrow(uniq, &narrowed)) return GC_E_CAPACITY;
                p.param_id[(size_t)t] = narrowed;
            }
            p.param_count = uniq + 1;
        }

        /* stable term ids in ascending canonical-key order */
        if (total > 0) {
            std::vector<gc_i64> order((size_t)total);
            for (gc_i64 t = 0; t < total; ++t) order[(size_t)t] = t;
            std::sort(order.begin(), order.end(), KeyLess(&p));
            for (gc_i64 o = 0; o < total; ++o) {
                const gc_i64 t = order[(size_t)o];
                p.term_id[(size_t)t] =
                    ((gc_term_id)td->kind << 56) | (gc_term_id)o;
            }
        }

        /* execution owner (doc/21_GPU_Native.rst); the context is unpublished,
         * so cell_pair_rule is called directly */
        const gc_i32 e1 = gc_term_owner_endpoint[td->kind][0] - 1;
        const gc_i32 e2 = gc_term_owner_endpoint[td->kind][1] - 1;
        for (gc_i64 t = 0; t < total; ++t) {
            const gc_gid ga = p.endpoint[(size_t)(t * td->arity + e1)];
            const gc_gid gb = p.endpoint[(size_t)(t * td->arity + e2)];
            gc_i32 ia = 0, ib = 0;
            const gc_status ra = p.resolve(ctx->identity_map,
                (size_t)(t * td->arity + e1), &ia);
            const gc_status rb = p.resolve(ctx->identity_map,
                (size_t)(t * td->arity + e2), &ib);
            if (ra == GC_E_MISMATCH || rb == GC_E_MISMATCH) return GC_E_UNSUPPORTED;
            if (ra != GC_OK || rb != GC_OK) continue;      /* unresolved */
            const gc_i32 ca = (ia >= 0) ? ctx->atom_cell[(size_t)ia]
                                        : ctx->ghost_cell[(size_t)(-ia - 1)];
            const gc_i32 cb = (ib >= 0) ? ctx->atom_cell[(size_t)ib]
                                        : ctx->ghost_cell[(size_t)(-ib - 1)];
            const gc_i32 owner = cell_pair_rule(ctx, ca, cb);
            p.owner_cell[(size_t)t] = owner;
            p.owned_here[(size_t)t] =
                (owner >= 0 && (gc_i64)owner < ctx->ncell_local) ? 1 : 0;
        }
    }

    /* ---------------- the cell-pair-indexed kinds ---------------- */
    /* The 1-4 and excluded pairs (doc/21_GPU_Native.rst): GENESIS stores
     * (4, term, owner cell), two cells and two CELL-LOCAL, ONE-BASED slots;
     * resolving them to global ids makes the record canonical like every other
     * term. CHARMM scales neither the 1-4 charge product nor its LJ pair, so its
     * records carry no parameter; a scaling force field supplies
     * qq_scale/lj_scale (parameter stride two). */
    for (gc_i32 pi = 0; pi < s->num_pairterms; ++pi) {
        const gc_pairterm_desc *pt = &s->pairterms[pi];
        gcn::TermPool &p = ctx->term[pt->kind];
        p.kind  = pt->kind;
        p.arity = 2;

        const gc_i64 nscale = (pt->qq_scale && pt->lj_scale) ? 2 : 0;

        gc_i64 total = 0;
        for (gc_i64 i = 0; i < pt->ncell_local; ++i) {
            if (pt->num_term[(size_t)i] < 0) return GC_E_ARG;
            if (!gcn::add_checked(total, pt->num_term[(size_t)i], &total))
                return GC_E_OVERFLOW;
        }
        {
            gc_i32 dummy;
            if (!gcn::narrow(total, &dummy)) return GC_E_CAPACITY;
        }

        p.count = total;
        p.endpoint.resize((size_t)(total * 2));
        p.endpoint_image.assign((size_t)(total * 2), 0);
        p.endpoint_image_valid.assign((size_t)(total * 2), 0);
        p.term_id.resize((size_t)total);
        p.param_id.assign((size_t)total, 0);
        p.image.assign((size_t)total, 0);
        p.genesis_cell.resize((size_t)total);
        p.owner_cell.assign((size_t)total, -1);
        p.owned_here.assign((size_t)total, 0);
        p.param_stride = nscale;

        std::vector<gc_f64> pay;
        if (nscale > 0 && total > 0) {
            gc_i64 bytes;
            if (!gcn::mul_checked(total, nscale, &bytes)) return GC_E_OVERFLOW;
            pay.assign((size_t)bytes, 0.0);
        }

        const gc_atom_desc *ad = s->atoms;
        gc_i64 w = 0;
        for (gc_i64 i = 0; i < pt->ncell_local; ++i) {
            const gc_i64 n = pt->num_term[(size_t)i];
            for (gc_i64 t = 0; t < n; ++t, ++w) {
                gc_i64 base;
                if (!gcn::mul_checked(i, pt->list_cell_stride, &base))
                    return GC_E_OVERFLOW;
                base += t * 4;
                const gc_i32 icel = pt->list[(size_t)(base + 0)];
                const gc_i32 jcel = pt->list[(size_t)(base + 1)];
                const gc_i32 ix   = pt->list[(size_t)(base + 2)];
                const gc_i32 iy   = pt->list[(size_t)(base + 3)];
                if (icel < 1 || (gc_i64)icel > ad->ncell) return GC_E_ARG;
                if (jcel < 1 || (gc_i64)jcel > ad->ncell) return GC_E_ARG;
                if (ix < 1 || ix > ad->num_atom[(size_t)(icel - 1)])
                    return GC_E_ARG;
                if (iy < 1 || iy > ad->num_atom[(size_t)(jcel - 1)])
                    return GC_E_ARG;

                gc_i64 oa, ob;
                if (!gcn::mul_checked((gc_i64)(icel - 1),
                                      ad->gid_cell_stride, &oa))
                    return GC_E_OVERFLOW;
                if (!gcn::mul_checked((gc_i64)(jcel - 1),
                                      ad->gid_cell_stride, &ob))
                    return GC_E_OVERFLOW;
                const gc_gid ga = (gc_gid)ad->gid[(size_t)(oa + ix - 1)];
                const gc_gid gb = (gc_gid)ad->gid[(size_t)(ob + iy - 1)];
                if (ga <= 0 || gb <= 0) return GC_E_ARG;

                p.endpoint[(size_t)(w * 2 + 0)] = ga;
                p.endpoint[(size_t)(w * 2 + 1)] = gb;
                p.genesis_cell[(size_t)w] = (gc_i32)i;
                const gc_i32 source_cell[2] = { icel - 1, jcel - 1 };
                const gc_gid source_gid[2] = { ga, gb };
                for (gc_i32 ee = 0; ee < 2; ++ee) {
                    gc_i32 found_slot = 0;
                    /* Select the admitted image in the exact source cell. */
                    gc_status found = GC_E_ENDPOINT;
                    if (source_cell[ee] < ctx->ncell_local) {
                        /* An owned cell's slot is the identity copy (image 0). */
                        found = GC_OK;
                    } else {
                        for (gc_i64 gi = 0; gi < ctx->num_ghost; ++gi) {
                            if (ctx->ghost_gid[(size_t)gi] == source_gid[ee] &&
                                ctx->ghost_cell[(size_t)gi] == source_cell[ee]) {
                                const gc_image im = ctx->ghost_image[(size_t)gi];
                                found = ctx->identity_map.find_exact(source_gid[ee], im, &found_slot);
                                if (found == GC_OK) {
                                    p.endpoint_image[(size_t)(w * 2 + ee)] = im;
                                    p.endpoint_image_valid[(size_t)(w * 2 + ee)] = 1;
                                }
                                break;
                            }
                        }
                    }
                    if (found == GC_OK && source_cell[ee] < ctx->ncell_local)
                        p.endpoint_image_valid[(size_t)(w * 2 + ee)] = 1;
                    if (found != GC_OK) return GC_E_ENDPOINT;
                }

                if (nscale > 0) {
                    gc_i64 sb;
                    if (!gcn::mul_checked(i, pt->scale_cell_stride, &sb))
                        return GC_E_OVERFLOW;
                    pay[(size_t)(w * 2 + 0)] = pt->qq_scale[(size_t)(sb + t)];
                    pay[(size_t)(w * 2 + 1)] = pt->lj_scale[(size_t)(sb + t)];
                }
            }
        }

        if (nscale > 0 && total > 0) {
            std::vector<gc_i64> order((size_t)total);
            for (gc_i64 t = 0; t < total; ++t) order[(size_t)t] = t;
            std::sort(order.begin(), order.end(), PayloadLess(&pay[0], nscale));
            const gc_i64 nbytes = nscale * (gc_i64)sizeof(gc_f64);
            gc_i64 uniq = -1;
            for (gc_i64 o = 0; o < total; ++o) {
                const gc_i64 t = order[(size_t)o];
                const gc_f64 *want = &pay[(size_t)(t * nscale)];
                if (uniq < 0 ||
                    std::memcmp(want,
                                &p.param_payload[(size_t)(uniq * nscale)],
                                (size_t)nbytes) != 0) {
                    ++uniq;
                    for (gc_i64 d = 0; d < nscale; ++d)
                        p.param_payload.push_back(want[(size_t)d]);
                }
                gc_i32 narrowed;
                if (!gcn::narrow(uniq, &narrowed)) return GC_E_CAPACITY;
                p.param_id[(size_t)t] = narrowed;
            }
            p.param_count = uniq + 1;
        }

        if (total > 0) {
            std::vector<gc_i64> order((size_t)total);
            for (gc_i64 t = 0; t < total; ++t) order[(size_t)t] = t;
            std::sort(order.begin(), order.end(), KeyLess(&p));
            for (gc_i64 o = 0; o < total; ++o) {
                const gc_i64 t = order[(size_t)o];
                p.term_id[(size_t)t] =
                    ((gc_term_id)pt->kind << 56) | (gc_term_id)o;
            }
        }

        /* the owner by the same rule as GENESIS's genesis_cell */
        for (gc_i64 t = 0; t < total; ++t) {
            const gc_gid ga = p.endpoint[(size_t)(t * 2 + 0)];
            const gc_gid gb = p.endpoint[(size_t)(t * 2 + 1)];
            const gc_i32 ia = ctx->gid_map.find(ga);
            const gc_i32 ib = ctx->gid_map.find(gb);
            if (ia == -1 || ib == -1) continue;
            const gc_i32 ca = (ia >= 0) ? ctx->atom_cell[(size_t)ia]
                                        : ctx->ghost_cell[(size_t)(-ia - 1)];
            const gc_i32 cb = (ib >= 0) ? ctx->atom_cell[(size_t)ib]
                                        : ctx->ghost_cell[(size_t)(-ib - 1)];
            const gc_i32 owner = cell_pair_rule(ctx, ca, cb);
            p.owner_cell[(size_t)t] = owner;
            p.owned_here[(size_t)t] =
                (owner >= 0 && (gc_i64)owner < ctx->ncell_local) ? 1 : 0;
        }
    }

    /* ---------------- list-validity guard plan ---------------- */
    {
        gcn::GuardPlan &lg = ctx->guard;
        lg.pairlistdist = g->pairlistdist;
        /* The guard protects exactly the pairs with a non-zero interaction, which
         * ends at the TABLES' support, not the kernel's prune radius: the kernel
         * reads rows L and L+1 of table_grad and table_ene, L = int(rc^2 rho /
         * r^2), and every PME linear table is zero below its first filled row,
         * whose electrostatic gradient is never zero. gpu_core_support_radius
         * (first non-zero grad row i, radius sqrt(rc^2 rho / (i-1))) is therefore
         * also the ene support; a pair beyond it contributes an exact zero, so
         * leaving it out of the list is exact. The prune radius only decides
         * which zero pairs the kernel skips. gpu_core_try_setup requires
         * pairlistdist to exceed this support. */
        lg.support_radius = (g->table_support_radius > 0.0)
                          ? g->table_support_radius : g->prune_support_radius;
        /* a conservative roundoff allowance: one ULP of the box scale times the
         * coordinate updates a list epoch can accumulate */
        const gc_f64 roundoff_margin = 1.0e-9 * (g->system_size[0] > 1.0
                                                 ? g->system_size[0] : 1.0);
        lg.half_skin = 0.5 * (lg.pairlistdist - lg.support_radius
                              - roundoff_margin);
    }

    return GC_OK;
}

extern "C" gc_status gpu_core_import(gc_context *ctx, const gc_state_desc *s)
{
    if (!ctx) return GC_E_ARG;
    /* All ranks complete the host descriptor/identity admission before
     * any rank enters the device attach's collective route probe. */
    gc_status st = gpu_core_import_host(ctx, s);
    st = gcn::native_dist_vote(ctx, st, "import_host");
    if (st != GC_OK) return st;
    st = gcn::native_context_attach(ctx);
    st = gcn::native_dist_vote(ctx, st, "import_device");
    if (st != GC_OK) return st;
    ctx->epoch = ctx->epoch + 1;  /* publish only after rank-wide admission */
    return GC_OK;
}

extern "C" gc_status gpu_core_posres_set(gc_context *ctx, gc_i64 n,
                                         const gc_i32 *gid, const gc_f64 *ref,
                                         const gc_f64 *par)
{
    if (!ctx || n < 0 || (n > 0 && (!gid || !ref || !par))) return GC_E_ARG;
    if (!ctx->native || ctx->epoch == 0) return GC_E_STATE;
    /* one record of eight words per restraint: the GID, then k, w_xyz and
     * ref_xyz bit for bit, so the gather moves them exactly */
    std::vector<gc_i64> mine((size_t)(8 * n)), all;
    for (gc_i64 i = 0; i < n; ++i) {
        const gc_f64 v[7] = { par[4*i], par[4*i+1], par[4*i+2], par[4*i+3],
                              ref[3*i], ref[3*i+1], ref[3*i+2] };
        mine[(size_t)(8*i)] = (gc_i64)gid[i];
        std::memcpy(&mine[(size_t)(8*i + 1)], v, sizeof(v));
    }
    gc_status st = gcn::native_dist_gather(ctx, mine, &all);
    if (st != GC_OK) return st;
    const gc_i64 total = (gc_i64)all.size() / 8;
    std::vector<gc_gid> g((size_t)total);
    std::vector<gc_f64> p((size_t)(7 * total));
    for (gc_i64 i = 0; i < total; ++i) {
        g[(size_t)i] = (gc_gid)all[(size_t)(8*i)];
        if (g[(size_t)i] < 1) return GC_E_ARG;
        std::memcpy(&p[(size_t)(7*i)], &all[(size_t)(8*i + 1)], 7 * sizeof(gc_f64));
    }
    st = gcn::native_posres_upload(ctx, total, total ? &g[0] : 0,
                                   total ? &p[0] : 0);
    ctx->posres_count = (st == GC_OK) ? total : 0;
    return st;
}

extern "C" gc_status gpu_core_import_real_mask(gc_context *ctx,
    gc_i64 ncell_local, gc_i64 max_atom,
    const gc_i32 *num_atom, const gc_i32 *gid_padded,
    gc_i64 near_pairs, const gc_i16 *cell_pairlist1,
    const gc_u8 *mask_self, const gc_u8 *mask_near)
{
    if (!ctx || ctx->epoch == 0 || !ctx->native || ctx->real_mask_ready)
        return GC_E_STATE;
    if (ncell_local != ctx->ncell_local || max_atom <= 0 ||
        near_pairs < 0 || near_pairs > GC_MAX_LOCAL_INDEX ||
        !num_atom || !gid_padded || !mask_self ||
        (near_pairs > 0 && (!cell_pairlist1 || !mask_near)))
        return GC_E_ARG;
    gc_i64 cell_stride, total_bytes, self_bytes;
    if (!gcn::mul_checked(max_atom, max_atom, &cell_stride) ||
        !gcn::mul_checked(ncell_local, cell_stride, &self_bytes) ||
        !gcn::mul_checked(near_pairs, cell_stride, &total_bytes))
        return GC_E_OVERFLOW;

    /* the padded slots must be a bijection with the imported owned GIDs, so a
     * zero stock bit cannot name another atom after the native sort */
    std::vector<gc_u8> seen((size_t)ctx->num_owned, 0);
    gc_i64 owned = 0;
    for (gc_i64 c = 0; c < ncell_local; ++c) {
        const gc_i32 n = num_atom[(size_t)c];
        if (n < 0 || (gc_i64)n > max_atom) return GC_E_ARITY;
        for (gc_i32 a = 0; a < n; ++a) {
            const gc_gid gid = gid_padded[(size_t)(c * max_atom + a)];
            const gc_i32 index = ctx->gid_map.find(gid);
            if (gid <= 0 || index < 0 ||
                ctx->atom_cell[(size_t)index] != c || seen[(size_t)index])
                return GC_E_MISMATCH;
            seen[(size_t)index] = 1;
            ++owned;
        }
    }
    if (owned != ctx->num_owned) return GC_E_MISMATCH;

    struct MaskAtom { gc_gid gid; gc_image image; };
    struct MaskPair { MaskAtom a, b; };
    const auto atom_less = [](const MaskAtom &a, const MaskAtom &b) {
        return a.gid < b.gid || (a.gid == b.gid && a.image < b.image);
    };
    const auto atom_equal = [](const MaskAtom &a, const MaskAtom &b) {
        return a.gid == b.gid && a.image == b.image;
    };
    const auto cell_image = [ctx](gc_i32 c) -> gc_image {
        const gc_i32 n[3] = { ctx->geo.cell[0], ctx->geo.cell[1], ctx->geo.cell[2] };
        const gc_i32 e[3] = { ctx->cell_gx[(size_t)c],
                              ctx->cell_gy[(size_t)c],
                              ctx->cell_gz[(size_t)c] };
        gc_i32 im[3] = {0,0,0};
        for (int k=0; k<3; ++k) {
            if (e[k] <= 0) im[k] = -1;
            else if (e[k] > n[k]) im[k] = +1;
        }
        return ((gc_image)(gc_i16)im[0] & 0xffffLL)
             | (((gc_image)(gc_i16)im[1] & 0xffffLL) << 16)
             | (((gc_image)(gc_i16)im[2] & 0xffffLL) << 32);
    };
    const auto make_pair = [&](gc_i32 ca, gc_i32 aa, gc_i32 cb, gc_i32 bb) {
        MaskAtom x = { gid_padded[(size_t)((gc_i64)ca * max_atom + aa)], cell_image(ca) };
        MaskAtom y = { gid_padded[(size_t)((gc_i64)cb * max_atom + bb)], cell_image(cb) };
        MaskPair q;
        if (atom_less(y,x)) { q.a=y; q.b=x; } else { q.a=x; q.b=y; }
        return q;
    };

    /* Stock's bits are 0 or 1. Zero
     * bits are rare, so each run of bytes is searched with memchr; the found
     * order does not matter, the list is sorted below. */
    /* Appends the pairs of every zero byte in row[lo, hi). */
    std::vector<MaskPair> zero;
    const auto scan = [&](const gc_u8 *row, gc_i32 lo, gc_i32 hi,
                          gc_i32 cc, gc_i32 col, gc_i32 cr) {
        const gc_u8 *q = row + lo, *end = row + hi;
        while (q < end && (q = (const gc_u8 *)std::memchr(q, 0, (size_t)(end - q)))) {
            zero.push_back(make_pair(cc, col, cr, (gc_i32)(q - row)));
            ++q;
        }
    };
    for (gc_i64 c = 0; c < ncell_local; ++c) {
        const gc_i32 n = num_atom[(size_t)c];
        for (gc_i32 a = 0; a < n; ++a)
            scan(&mask_self[(size_t)(c * cell_stride + (gc_i64)a * max_atom)],
                 a + 1, n, (gc_i32)c, a, (gc_i32)c);
    }
    /* Stock's near mask is indexed (lower cell's atom, higher cell's atom):
     * the column is the lower cell's atom, the contiguous row the higher's. */
    for (gc_i64 p = 0; p < near_pairs; ++p) {
        const gc_i32 ca = cell_pairlist1[(size_t)(2 * p)] - 1;
        const gc_i32 cb = cell_pairlist1[(size_t)(2 * p + 1)] - 1;
        if (ca < 0 || cb < 0 || ca >= ctx->ncell || cb >= ctx->ncell || ca == cb)
            return GC_E_ARG;
        const gc_i32 lo_cell = ca < cb ? ca : cb, hi_cell = ca < cb ? cb : ca;
        const gc_i32 nc = num_atom[(size_t)lo_cell], nr = num_atom[(size_t)hi_cell];
        if (nc < 0 || nr < 0 || (gc_i64)nc > max_atom || (gc_i64)nr > max_atom)
            return GC_E_ARITY;
        for (gc_i32 col = 0; col < nc; ++col)
            scan(&mask_near[(size_t)(p * cell_stride + (gc_i64)col * max_atom)],
                 0, nr, lo_cell, col, hi_cell);
    }
    std::sort(zero.begin(), zero.end(), [&](const MaskPair &x, const MaskPair &y) {
        if (atom_less(x.a,y.a)) return true;
        if (atom_less(y.a,x.a)) return false;
        return atom_less(x.b,y.b);
    });
    zero.erase(std::unique(zero.begin(), zero.end(), [&](const MaskPair &x, const MaskPair &y) {
        return atom_equal(x.a,y.a) && atom_equal(x.b,y.b);
    }), zero.end());

    /* Stock gives a cross-rank cell pair's zero bits to the rank its midpoint
     * rule picks (assign_cell_interaction); the native list evaluates the pair
     * on the owner of the cell its half-list rule picks (gcn_mask_home_other).
     * So each rank also takes the other ranks' zero bits naming one of its
     * owned atoms, re-imaged so that atom is the identity copy, and clears the
     * bits of the pairs it evaluates. */
    if (ctx->nproc > 1) {
        std::vector<gc_i64> mine, all;
        for (size_t i = 0; i < zero.size(); ++i) {
            if (ctx->gid_map.find(zero[i].a.gid) >= 0 &&
                ctx->gid_map.find(zero[i].b.gid) >= 0) continue;
            const gc_i64 w[4] = { zero[i].a.gid, zero[i].a.image,
                                  zero[i].b.gid, zero[i].b.image };
            mine.insert(mine.end(), w, w + 4);
        }
        gc_status st = gcn::native_dist_gather(ctx, mine, &all);
        st = gcn::native_dist_vote(ctx, st, "import_real_mask_gather");
        if (st != GC_OK) return st;
        for (size_t i = 0; i + 3 < all.size(); i += 4) {
            MaskAtom x = { all[i], all[i + 1] }, y = { all[i + 2], all[i + 3] };
            if (ctx->gid_map.find(x.gid) < 0) std::swap(x, y);
            if (ctx->gid_map.find(x.gid) < 0 || ctx->gid_map.find(y.gid) >= 0)
                continue;
            y.image = gcn::image_sub(y.image, x.image);
            x.image = 0;
            MaskPair q;
            if (atom_less(y, x)) { q.a = y; q.b = x; } else { q.a = x; q.b = y; }
            zero.push_back(q);
        }
        std::sort(zero.begin(), zero.end(), [&](const MaskPair &u, const MaskPair &v) {
            if (atom_less(u.a,v.a)) return true;
            if (atom_less(v.a,u.a)) return false;
            return atom_less(u.b,v.b);
        });
        zero.erase(std::unique(zero.begin(), zero.end(), [&](const MaskPair &u, const MaskPair &v) {
            return atom_equal(u.a,v.a) && atom_equal(u.b,v.b);
        }), zero.end());
    }
    if (zero.size() > (size_t)GC_MAX_LOCAL_INDEX) return GC_E_CAPACITY;
    ctx->real_mask_gid.clear();
    ctx->real_mask_image.clear();
    ctx->real_mask_gid.reserve(2 * zero.size());
    ctx->real_mask_image.reserve(2 * zero.size());
    for (size_t i = 0; i < zero.size(); ++i) {
        ctx->real_mask_gid.push_back(zero[i].a.gid);
        ctx->real_mask_image.push_back(zero[i].a.image);
        ctx->real_mask_gid.push_back(zero[i].b.gid);
        ctx->real_mask_image.push_back(zero[i].b.image);
    }
    ctx->real_mask_ready = 1;
    return GC_OK;
}

namespace {
struct HostSlot { gc_gid gid; gc_i32 cell, slot; };

/* Every occupied stock owned slot, including water, sorted by gid (id_g2l
 * covers solute atoms only). */
gc_status host_slots_by_gid(gc_i64 ncell, gc_i64 max_atom,
                            const gc_i32 *num_atom, const gc_i32 *id_l2g,
                            std::vector<HostSlot> *host)
{
    gc_i32 admitted;
    if (ncell > 0 && !gcn::narrow(ncell - 1, &admitted)) return GC_E_CAPACITY;
    for (gc_i64 c = 0; c < ncell; ++c) {
        const gc_i64 n = num_atom[(size_t)c];
        if (n < 0 || n > max_atom) return GC_E_ARG;
        gc_i64 base;
        if (!gcn::mul_checked(c, max_atom, &base)) return GC_E_OVERFLOW;
        for (gc_i64 s = 0; s < n; ++s) {
            gc_i64 at;
            if (!gcn::add_checked(base, s, &at)) return GC_E_OVERFLOW;
            if (!gcn::narrow(s, &admitted)) return GC_E_CAPACITY;
            const gc_gid g = id_l2g[(size_t)at];
            if (g <= 0) return GC_E_MISMATCH;
            host->push_back({g, (gc_i32)c, (gc_i32)s});
        }
    }
    std::sort(host->begin(), host->end(),
              [](const HostSlot &a, const HostSlot &b) { return a.gid < b.gid; });
    for (size_t i = 1; i < host->size(); ++i)
        if ((*host)[i].gid == (*host)[i - 1].gid) return GC_E_MISMATCH;
    return GC_OK;
}

const HostSlot *find_host_slot(const std::vector<HostSlot> &host, gc_gid g)
{
    const auto found = std::lower_bound(
        host.begin(), host.end(), g,
        [](const HostSlot &a, gc_gid key) { return a.gid < key; });
    return (found == host.end() || found->gid != g) ? 0 : &*found;
}
} // namespace

/* Match the current compact device order against every occupied stock
 * owned slot: a bijection. */
extern "C" gc_status gpu_core_map_host_slots(
    gc_i64 count, gc_i64 ncell, gc_i64 max_atom, const gc_i32 *num_atom,
    const gc_i32 *id_l2g, const gc_gid *gid,
    gc_i32 *cell_out, gc_i32 *slot_out)
{
    if (count < 0 || ncell < 0 || max_atom < 0) return GC_E_ARG;
    if (ncell > 0 && !num_atom) return GC_E_ARG;
    if (count > 0 && (!num_atom || !id_l2g || !gid ||
                      !cell_out || !slot_out || max_atom == 0)) return GC_E_ARG;
    gc_i32 admitted;
    if (count > 0 && !gcn::narrow(count - 1, &admitted)) return GC_E_CAPACITY;

    std::vector<HostSlot> host;
    host.reserve((size_t)count);
    const gc_status st = host_slots_by_gid(ncell, max_atom, num_atom, id_l2g,
                                           &host);
    if (st != GC_OK) return st;
    if ((gc_i64)host.size() != count) return GC_E_MISMATCH;
    std::vector<gc_gid> device(gid, gid + (size_t)count);
    std::sort(device.begin(), device.end());
    for (gc_i64 i = 0; i < count; ++i)
        if (host[(size_t)i].gid != device[(size_t)i]) return GC_E_MISMATCH;
    for (gc_i64 i = 0; i < count; ++i) {
        const HostSlot *found = find_host_slot(host, gid[(size_t)i]);
        if (!found) return GC_E_MISMATCH;
        cell_out[(size_t)i] = found->cell;
        slot_out[(size_t)i] = found->slot;
    }
    return GC_OK;
}

/* The same match at more than one rank, where the device may own atoms this
 * rank's host domain does not hold: those get cell and slot -1. */
extern "C" gc_status gpu_core_find_host_slots(
    gc_i64 count, gc_i64 ncell, gc_i64 max_atom, const gc_i32 *num_atom,
    const gc_i32 *id_l2g, const gc_gid *gid,
    gc_i32 *cell_out, gc_i32 *slot_out)
{
    if (count < 0 || ncell < 0 || max_atom < 0) return GC_E_ARG;
    if (ncell > 0 && (!num_atom || !id_l2g || max_atom == 0)) return GC_E_ARG;
    if (count > 0 && (!gid || !cell_out || !slot_out)) return GC_E_ARG;

    std::vector<HostSlot> host;
    const gc_status st = host_slots_by_gid(ncell, max_atom, num_atom, id_l2g,
                                           &host);
    if (st != GC_OK) return st;
    for (gc_i64 i = 0; i < count; ++i) {
        const HostSlot *found = find_host_slot(host, gid[(size_t)i]);
        cell_out[(size_t)i] = found ? found->cell : -1;
        slot_out[(size_t)i] = found ? found->slot : -1;
    }
    return GC_OK;
}

/* ---- Reports and counts ---- */

extern "C" gc_status gpu_core_counts(const gc_context *ctx,
                                     gc_i64 *owned_atoms, gc_i64 *halo_atoms)
{
    if (ctx == 0) return GC_E_ARG;
    if (ctx->epoch == 0) return GC_E_STATE;
    if (owned_atoms) *owned_atoms = ctx->num_owned;
    if (halo_atoms)  *halo_atoms  = ctx->num_ghost;
    return GC_OK;
}

extern "C" gc_status gpu_core_list_guard_plan(const gc_context *ctx,
                                              gc_f64 *pairlistdist,
                                              gc_f64 *support_radius,
                                              gc_f64 *half_skin)
{
    if (ctx == 0 || !pairlistdist || !support_radius || !half_skin)
        return GC_E_ARG;
    if (ctx->epoch == 0) return GC_E_STATE;
    *pairlistdist   = ctx->guard.pairlistdist;
    *support_radius = ctx->guard.support_radius;
    *half_skin      = ctx->guard.half_skin;
    return GC_OK;
}

/* ---- Export, in ascending GID order (doc/21_GPU_Native.rst) ---- */

/* ---- The owner-rule reference checker (doc/21_GPU_Native.rst) ---- */

/* ---- Device residency ---- */
