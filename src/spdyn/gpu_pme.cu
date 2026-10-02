/*
 * gpu_pme.cu : the reciprocal sum of the device-native core.
 *
 * Plans and runs the pencil pipeline once per force evaluation: pme_recip
 * (spread, solve, gather on one brick and one Z pencil), pencil_fft, and the
 * mesh exchange sessions that move the mesh between them.  On one rank the
 * brick is the whole mesh and one three-dimensional transform runs on it with
 * no move (`whole`).
 *
 * Each rank spreads and gathers its OWNED atoms only, so the sum needs no
 * ghost coordinate and leaves no ghost force.  A rank's brick is every mesh
 * cell an owned atom's stencil can reach (owner_brick), so neighbouring
 * bricks overlap.  The spread leaves integer fixed-point sums in the brick,
 * the brick -> X-real move ADDS them on arrival and the X-real owner folds
 * them into its pencil: integer sums are exact in any order, so the mesh is
 * the bits of one whole-mesh spread at every rank count.  The reverse move
 * copies each convolved cell back to every brick that holds it.
 *
 * Each rank's solve sums its own Z pencil; compute_dynvars sums the ranks.
 * No factor 1/2 and no self energy in the solve.
 */

#include "gpu_core_native.h"
#include "gpu_mesh_exchange_session.hpp"
#include "gpu_pencil_fft.hpp"
#include "gpu_pme_recip.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <utility>
#include <vector>

namespace pn = genesis_native_pencil;
namespace s4 = genesis_native_s4;
namespace ft = genesis_native_s4fft;
namespace pm = genesis_native_s4pme;

namespace {

/* The mesh's buffers and the six moves between them, in the order of one
 * evaluation.  kBrickWords and kXWords hold the spread's fixed-point sums (a
 * word per cell, as wide as a real). */
enum { kBrickWords, kBrick, kXWords, kXReal, kXCplx, kYCplx, kZCplx, kBufs };
enum { kBrickToX, kXToBrick, kXToY, kYToX, kYToZ, kZToY, kMoves };

const struct {
    bool brick;
    pn::Transpose transpose;
    pn::Direction dir;
    int src, dst;
} kMove[kMoves] = {
    {true,  pn::Transpose::x_to_y, pn::Direction::forward, kBrickWords, kXWords},
    {true,  pn::Transpose::x_to_y, pn::Direction::reverse, kXReal, kBrick},
    {false, pn::Transpose::x_to_y, pn::Direction::forward, kXCplx, kYCplx},
    {false, pn::Transpose::x_to_y, pn::Direction::reverse, kYCplx, kXCplx},
    {false, pn::Transpose::y_to_z, pn::Direction::forward, kYCplx, kZCplx},
    {false, pn::Transpose::y_to_z, pn::Direction::reverse, kZCplx, kYCplx},
};

/* Past the halo's plans (tag_base 1024) and the migration's (4096). */
const gc_i32 kTagBase = 8192;
const gc_i64 kDeadlineMs = GCX_DEADLINE_MS;

/* Across nodes, when every node holds the same number of ranks, numbered
 * contiguously, and whole z layers of domains (ranks run x fastest and z
 * slowest): the grid (ranks per node) x (nodes), numbered node-major
 * (PencilLayout::set_node_major).  An X pencil then lies in its node's z
 * range, so the brick moves and the x -> y transpose stay on the node and
 * only y -> z crosses the network.  Returns the ranks per node, or 0. */
int node_major(MPI_Comm comm, int rank, int nproc, int xy_domains)
{
    MPI_Comm node = MPI_COMM_NULL;
    if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                            &node) != MPI_SUCCESS)
        return 0;
    int local = 1, lr = 0, lo = rank;
    MPI_Comm_size(node, &local);
    MPI_Comm_rank(node, &lr);
    MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_INT, MPI_MIN, node);
    MPI_Comm_free(&node);
    int v[3] = {local > 1 && local < nproc && nproc % local == 0 &&
                lo == rank - lr && lo % local == 0 && xy_domains > 0 &&
                local % xy_domains == 0, local, -local};
    MPI_Allreduce(MPI_IN_PLACE, v, 3, MPI_INT, MPI_MIN, comm);
    return v[0] && v[1] == -v[2] ? local : 0;
}

/* pa x pb = n with pa the largest divisor not above sqrt(n).  slab: pa = 1,
 * so the x -> y transpose is a rank-local reorder and one all-to-all (y -> z)
 * is left per direction. */
pn::ProcessGrid process_grid(pn::count_t n, bool slab)
{
    if (slab) return pn::ProcessGrid{1, n};
    pn::count_t a = 1;
    for (pn::count_t f = 1; f * f <= n; ++f)
        if (n % f == 0) a = f;
    return pn::ProcessGrid{a, n / a};
}

/* How far an owned atom may be from its domain, in list buffers
 * (pairlistdist - cutoffdist), when the brick is planned and when a new box
 * is checked against it.  An atom beyond the brick poisons the sum
 * (gpu_pme_recip.hpp), never drops charge. */
const double kPlanDrift  = 2.0;
const double kReboxDrift = 1.0;

/* The brick of this rank's owned atoms: every mesh cell their stencils can
 * reach.  On an axis with one domain, the whole axis; otherwise the domain's
 * extent in grid units widened by `drift` list buffers on both sides and by
 * order - 1 cells below (a stencil reaches down from its atom's grid point).
 * A domain starts at (origin + start * cell_size) * N / box, the cell
 * assignment's own shift (gpu_rebuild.cu, gcn_cell_of_coord). */
pm::BrickBox owner_brick(const gc_context *ctx, const struct gcn_layout &L,
                         const pm::RecipParams &p, double drift)
{
    const gc_geometry_desc &g = ctx->geo;
    long lo[3], len[3];
    for (int k = 0; k < 3; ++k) {
        const long N = (long)p.N[k];
        lo[k] = 0;
        len[k] = N;
        if (L.nd[k] == 1) continue;
        const double dom_lo = (g.origin[k] + L.start[k] * g.cell_size[k])
                            * p.r_scale[k];
        const double dom_hi = dom_lo + L.len[k] * g.cell_size[k] * p.r_scale[k];
        const double d = drift * (g.pairlistdist - g.cutoffdist) * p.r_scale[k];
        const long a = (long)std::floor(dom_lo - d) - (p.nbs - 1);
        const long b = (long)std::floor(dom_hi + d) + 1;
        if (b - a >= N) continue;
        lo[k] = ((a % N) + N) % N;
        len[k] = b - a;
    }
    pm::BrickBox r;
    r.x0 = lo[0]; r.y0 = lo[1]; r.z0 = lo[2];
    r.nx = len[0]; r.ny = len[1]; r.nz = len[2];
    return r;
}

/* Whether every cell of `in` lies in `out` (both periodic, as BrickBox). */
bool brick_holds(const pm::BrickBox &out, const pm::BrickBox &in, const int *N)
{
    const long o0[3] = {out.x0, out.y0, out.z0}, on[3] = {out.nx, out.ny, out.nz};
    const long i0[3] = {in.x0, in.y0, in.z0}, in_n[3] = {in.nx, in.ny, in.nz};
    for (int k = 0; k < 3; ++k) {
        if (on[k] == N[k]) continue;
        const long d = ((i0[k] - o0[k]) % N[k] + N[k]) % N[k];
        if (d + in_n[k] > on[k]) return false;
    }
    return true;
}

/* The mesh layout, decided once on rank 0 and broadcast so every rank builds
 * the same plans.  One node: slab, whose all-to-all stays on the node's GPU
 * links.  Across nodes: pencils, whose all-to-alls run within rows and
 * columns of the process grid. */
bool pme_slab(MPI_Comm comm)
{
    int nproc = 1, local = 1, slab = 0;
    MPI_Comm_size(comm, &nproc);
    if (nproc > 1) {
        MPI_Comm node;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
        MPI_Comm_size(node, &local);
        MPI_Comm_free(&node);
        slab = (local == nproc);
    }
    MPI_Bcast(&slab, 1, MPI_INT, 0, comm);
    return slab != 0;
}

gc_status pme_fail(int rank, const char *phase, const char *why)
{
    std::fprintf(stderr, "GPU_Core_Error> phase=%s rank=%d reason=%s\n",
                 phase, rank, why ? why : "");
    return GC_E_DEVICE;
}
gc_status pme_fail(const gc_context *ctx, const char *phase, const char *why)
{
    return pme_fail((int)ctx->rank, phase, why);
}

/* One status for every rank of `comm` (native_dist_vote's rule). */
gc_status mesh_vote(MPI_Comm comm, int rank, int nproc, gc_status local,
                    const char *phase)
{
    if (nproc <= 1) return local;
    int in = (int)local, out = 0;
    if (MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS)
        return GC_E_STATE;
    if (out != 0 && rank == 0)
        std::fprintf(stderr, "GPU_Core_Error> phase=%s collective_status=%d\n",
                     phase, out);
    return (gc_status)out;
}

}  /* anonymous namespace */

struct gcn_pme_mesh {
    pn::PencilLayout layout;
    s4::ExchangePlan plan[kMoves];
    s4::MeshExchangeSession session[kMoves];
    bool created[kMoves];
    void *buf[kBufs];
    pn::count_t bytes[kBufs];
    ft::PencilFftPlans fft;
    pm::RecipDevice recip;
    pm::RecipInput input;       /* the plan's inputs; rebox swaps the box */
    pm::BrickBox brick;         /* owner_brick at kPlanDrift */
    pm::ZPencilBox zpencil;
    pm::FoldSelf fold_self;     /* kBrickToX's own part, added by the fold */
    gc_i64 epoch;
    cudaStream_t stream;        /* the stream fft, recip and moves run on */
    MPI_Comm comm;              /* the rank's communicator */
    int rank, nproc;            /* on comm */
    bool whole;                 /* one rank: brick <-> spectrum, no moves */
    bool spans_nodes;           /* comm spans nodes: some move holds the host */
    bool swap_xz;               /* the mesh's x is the box's z (MeshSetup)    */
    explicit gcn_pme_mesh(const pn::PencilLayout &l)
        : layout(l), created(), buf(), bytes(), input(), brick(), zpencil(),
          fold_self(), epoch(0), stream(0), comm(MPI_COMM_NULL), rank(0), nproc(1),
          whole(false), spans_nodes(false), swap_xz(false) {}
    gc_status move(int i, cudaStream_t s)
    {
        gc_status st = session[i].move(epoch, (void *)s);
        if (st != GC_OK) pme_fail(rank, "pme_exchange", session[i].last_error());
        return st;
    }
};

/* Publish this rank's solve partials [E, xx, xy, xz, yy, yz, zz] in the
 * reciprocal accumulator; the join reads the diagonal only. */
__global__ void gcn_kern_pme_publish(const gc_f64 *__restrict__ s,
                                     gc_f64 *__restrict__ acc, int want_energy,
                                     int swap_xz = 0)
{
    acc[GCN_PE_ENE]  = want_energy ? s[0] : 0.0;
    acc[GCN_PE_VIRX] = swap_xz ? s[6] : s[1];
    acc[GCN_PE_VIRY] = s[4];
    acc[GCN_PE_VIRZ] = swap_xz ? s[1] : s[6];
}

/* The Ewald self energy, -sum q^2 * el_fact * alpha / sqrt(pi), reduced over
 * owned slots; recomputed at every rebuild, so it cannot go stale. */
__global__ void gcn_kern_pme_self(const gc_f64 *__restrict__ charge,
                                  gc_i64 n, gc_f64 *__restrict__ part)
{
    __shared__ double sh[GCN_RED_BLOCK];
    double acc = 0.0;
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x)
        acc += charge[s] * charge[s];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int half = blockDim.x >> 1; half > 0; half >>= 1) {
        if ((int)threadIdx.x < half) sh[threadIdx.x] += sh[threadIdx.x + half];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}

namespace {

/* swap_xz: the mesh is laid out with the box's x and z exchanged, so the
 * transforms' first axis (whose lines every rank holds whole) is the box's z,
 * which the domain decomposition leaves unsplit when it splits x and y only.
 * A rank's brick then holds most of its own X pencil and the brick moves
 * carry little more than the stencil overlap between neighbouring domains.
 * Spread and gather read coordinates and write forces through the same
 * exchange (swapped_soa), and the diagonal virial is swapped back when it is
 * published; the mesh sums are the same integers either way. */
struct MeshSetup {
    pm::RecipInput in;
    bool single;                /* nonbond_precision = MIXED: an FP32 mesh */
    bool probe;                 /* the transport probe has not run yet */
    bool swap_xz = false;
    /* With swap_xz, the process grid of the domains: pa = domains along y
     * (the mesh's y), pb = domains along x (the mesh's z).  PencilLayout's
     * rank r holds (a, b) = (r / pb, r % pb), the domain order, so each
     * rank's X pencil is its own domain's block. */
    int pgrid[2] = {0, 0};
    /* Without swap_xz: the domains along x times along y (one z layer), for
     * node_major. */
    int xy_domains = 0;
};

pm::RecipInput swapped(pm::RecipInput in)
{
    std::swap(in.box[0], in.box[2]);
    std::swap(in.ngrid[0], in.ngrid[2]);
    return in;
}

pm::BrickBox swapped(pm::BrickBox b)
{
    std::swap(b.x0, b.z0);
    std::swap(b.nx, b.nz);
    return b;
}

/* An SoA array (x at [0,pitch), y, z after it) as the swapped mesh reads
 * it: x' is z and z' is x, so x' starts at 2 pitch and steps by -pitch. */
template <class T> T *swapped_soa(T *base, long pitch, long *spitch)
{
    *spitch = -pitch;
    return base + 2 * pitch;
}

/* Collective over `comm` (nproc ranks): the layout and this rank's boxes,
 * every rank's brick, the plans, the sessions, the buffers, the transforms
 * and the spectral solve, with votes between.  `why` is the caller's local
 * verdict so far and `brick` its brick.  *out holds whatever was built, for
 * native_pme_release, whatever the status. */
gc_status defer_brick_self(gcn_pme_mesh *m);

gc_status mesh_build(MPI_Comm comm, int rank, int nproc, const MeshSetup &ms,
                     const char *why, const pm::BrickBox &brick,
                     cudaStream_t stream, gcn_pme_mesh **out)
{
    const bool slab = pme_slab(comm);
    pn::ProcessGrid grid = ms.swap_xz
        ? pn::ProcessGrid{(pn::count_t)ms.pgrid[0], (pn::count_t)ms.pgrid[1]}
        : process_grid((pn::count_t)nproc, slab);
    const int local = !slab && !ms.swap_xz
                    ? node_major(comm, rank, nproc, ms.xy_domains) : 0;
    if (local > 0) {
        grid = pn::ProcessGrid{(pn::count_t)local, (pn::count_t)(nproc / local)};
        if (rank == 0) {
            std::printf("Native_Pme> pencil grid %ld x %ld, node-major: brick "
                        "moves and x -> y on the node\n", (long)grid.pa,
                        (long)grid.pb);
            std::fflush(stdout);
        }
    }
    const pn::count_t prank = (pn::count_t)rank;
    const pn::count_t real_w = ms.single ? 4 : 8, cplx_w = 2 * real_w;
    const bool whole = nproc == 1;
    pm::RecipParams prm = pm::RecipParams();
    gcn_pme_mesh *m = 0;
    try {
        prm = pm::make_recip_params(ms.in);
        if (!why) why = pm::spread_gather_admissible(prm);
        const pn::MeshShape mesh{(pn::count_t)prm.N[0], (pn::count_t)prm.N[1],
                                 (pn::count_t)prm.N[2]};
        m = *out = new gcn_pme_mesh(
            pn::PencilLayout(mesh, grid, (pn::count_t)nproc));
        if (local > 0) m->layout.set_node_major((pn::count_t)local);
        m->input = ms.in;
        m->swap_xz = ms.swap_xz;
        m->whole = whole;
        m->stream = stream;
        m->comm = comm;
        m->rank = rank;
        m->nproc = nproc;
        m->brick = brick;
        if (whole) {
            m->zpencil.kx0 = 0; m->zpencil.nkx = (long)(mesh.nx / 2 + 1);
            m->zpencil.ky0 = 0; m->zpencil.nky = (long)mesh.ny;
            m->zpencil.nz  = (long)mesh.nz;
            m->zpencil.kx_fastest = true;
        } else {
            const pn::Region z = m->layout.z_complex(prank);
            m->zpencil.kx0 = (long)z.x0; m->zpencil.nkx = (long)z.nx;
            m->zpencil.ky0 = (long)z.y0; m->zpencil.nky = (long)z.ny;
            m->zpencil.nz  = (long)z.nz;
        }
    } catch (const std::exception &e) {
        if (!why) why = e.what();
    }
    /* Every rank's brick: collective, so reached whatever the local verdict
     * (a refusing rank sends an empty brick). */
    std::vector<long> all(6 * (std::size_t)nproc, 0);
    {
        long mine[6] = {0, 0, 0, 0, 0, 0};
        if (!why) {
            const long v[6] = {brick.x0, brick.y0, brick.z0,
                               brick.nx, brick.ny, brick.nz};
            std::memcpy(mine, v, sizeof(mine));
        }
        if (MPI_Allgather(mine, 6, MPI_LONG, &all[0], 6, MPI_LONG, comm) !=
                MPI_SUCCESS && !why)
            why = "the brick exchange failed";
    }
    if (!why && !whole) try {
        std::vector<pn::Region> bricks((std::size_t)nproc);
        for (std::size_t r = 0; r < bricks.size(); ++r) {
            const long *v = &all[6 * r];
            bricks[r] = pn::Region{(pn::count_t)v[0], (pn::count_t)v[1],
                                   (pn::count_t)v[2], (pn::count_t)v[3],
                                   (pn::count_t)v[4], (pn::count_t)v[5]};
        }
        for (int i = 0; i < kMoves; ++i) {
            m->plan[i] = kMove[i].brick
                ? s4::build_brick_plan(m->layout, bricks, prank, kMove[i].dir,
                                       real_w, s4::Memory::device)
                : s4::build_transpose_plan(m->layout, prank, kMove[i].transpose,
                                           kMove[i].dir, cplx_w,
                                           s4::Memory::device);
        }
        if ((pn::count_t)m->brick.cells() != m->plan[kBrickToX].source_elements ||
            (pn::count_t)m->zpencil.points() != m->plan[kYToZ].target_elements)
            why = "the brick or Z pencil box is not its plan's";
    } catch (const std::exception &e) {
        why = e.what();
    }
    if (why)
        std::fprintf(stderr, "GPU_Core_Decline> phase=pme_plan rank=%d "
                     "reason=%s\n", rank, why);
    gc_status st = mesh_vote(comm, rank, nproc, why ? GC_E_UNSUPPORTED : GC_OK,
                             "pme_plan");
    if (st != GC_OK) return st;
    /* The transport needs its start-up probe before any plan; the dist state
     * runs it when there is one, and one rank needs none. */
    if (ms.probe && !whole) {
        gcx_probe_report probe;
        std::memset(&probe, 0, sizeof(probe));
        st = gcx_probe_run((gcx_comm)MPI_Comm_c2f(comm), &probe);
        if (st != GC_OK) return pme_fail(rank, "pme_probe", "gcx_probe_run");
    }
    /* Whether the communicator spans nodes: a move over the network holds the
     * host until it lands (native_pme_crosses_nodes). */
    if (!whole) {
        MPI_Comm node = MPI_COMM_NULL;
        int nsize = nproc;
        if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                &node) == MPI_SUCCESS) {
            MPI_Comm_size(node, &nsize);
            MPI_Comm_free(&node);
        }
        int spans = nsize < nproc, any = 0;
        MPI_Allreduce(&spans, &any, 1, MPI_INT, MPI_MAX, comm);
        if (m) m->spans_nodes = any != 0;
    }
    /* Collective: create votes internally, so every rank gets one answer. */
    void *ub = 0;
    int have_ub = 0;
    MPI_Comm_get_attr(comm, MPI_TAG_UB, &ub, &have_ub);
    const gc_i64 tag_ub = (have_ub && ub) ? *(int *)ub : 32767;
    const gcx_comm fcomm = (gcx_comm)MPI_Comm_c2f(comm);
    for (int i = 0; i < kMoves && !whole; ++i) {
        /* each plan takes its own tag span, refused past MPI_TAG_UB */
        const gc_i64 base = (gc_i64)kTagBase + (gc_i64)i * GCX_TAG_SPAN;
        st = base + GCX_TAG_SPAN - 1 > tag_ub ? GC_E_CAPACITY : GC_OK;
        if (st == GC_OK)
            st = m->session[i].create(m->plan[i], s4::Memory::device, fcomm,
                                      (gc_i32)base, kDeadlineMs);
        if (st != GC_OK) {
            const char *e = m->session[i].last_error();
            pme_fail(rank, "pme_session",
                     e && *e ? e : s4::mesh_session_state_name(st));
            return st;
        }
        m->created[i] = true;
    }
    /* Local again: buffers, transforms and the spectral solve, then one vote. */
    const pn::count_t cells = (pn::count_t)m->brick.cells();
    const pn::count_t xcells = whole ? 0 : m->plan[kBrickToX].target_elements;
    const pn::count_t elems[kBufs] = {
        cells, cells, xcells, xcells,
        whole ? 0 : m->plan[kXToY].source_elements,
        whole ? 0 : m->plan[kXToY].target_elements,
        whole ? (pn::count_t)m->zpencil.points()
              : m->plan[kYToZ].target_elements};
    const pn::count_t width[kBufs] = {real_w, real_w, real_w, real_w,
                                      cplx_w, cplx_w, cplx_w};
    for (int b = 0; b < kBufs && st == GC_OK; ++b) {
        m->bytes[b] = elems[b] ? elems[b] * width[b] : 1;
        st = gcn::dev_calloc(&m->buf[b], (gc_i64)m->bytes[b]);
    }
    for (int i = 0; i < kMoves && st == GC_OK; ++i)
        if (m->created[i])
            st = m->session[i].attach(m->buf[kMove[i].src],
                                      m->bytes[kMove[i].src],
                                      m->buf[kMove[i].dst],
                                      m->bytes[kMove[i].dst]);
    /* The brick's own cells go straight from its sums into the fold, which
     * reads them where it reads the pencil's (pm::FoldSelf). */
    if (st == GC_OK && m->created[kBrickToX])
        st = defer_brick_self(m);
    if (st == GC_OK) {
        if (whole) {
            if (!m->fft.build_3d(prm.N[0], prm.N[1], prm.N[2],
                                 (void *)m->stream, ms.single))
                st = pme_fail(rank, "pme_fft", m->fft.last_error());
        } else try {
            const ft::FftAxes axes = ft::fft_axes(
                prm.N[0], prm.N[1], prm.N[2], m->plan[kXToY].source_elements,
                m->plan[kYToZ].source_elements, m->plan[kYToZ].target_elements);
            if (!m->fft.build(axes, (void *)m->stream, ms.single))
                st = pme_fail(rank, "pme_fft", m->fft.last_error());
        } catch (const std::exception &e) {
            st = pme_fail(rank, "pme_fft", e.what());
        }
    }
    if (st == GC_OK && !m->recip.build(prm, (void *)m->stream, ms.single))
        st = pme_fail(rank, "pme_recip", m->recip.last_error());
    return mesh_vote(comm, rank, nproc, st, "pme_build");
}

/* Collective when a plan exists: the sessions are destroyed together. */
void mesh_release(gcn_pme_mesh *m)
{
    if (m == 0) return;
    for (int i = 0; i < kMoves; ++i)
        if (m->created[i]) m->session[i].destroy();
    for (int b = 0; b < kBufs; ++b) gcn::dev_release(&m->buf[b]);
    delete m;
}

/* The forward brick move's self copies as fold boxes.  Each copy lands on
 * the canonical X-real pencil (rows of the mesh's nx), so its target offset
 * names the box's corner. */
gc_status defer_brick_self(gcn_pme_mesh *m)
{
    std::vector<s4::Copy3D> c;
    std::memset(&m->fold_self, 0, sizeof m->fold_self);
    if (!m->session[kBrickToX].defer_self(&c, pm::kMaxFoldSelf)) return GC_OK;
    pm::FoldSelf &f = m->fold_self;
    const pn::count_t cells = m->plan[kBrickToX].target_elements;
    f.src = m->buf[kBrickWords];
    for (std::size_t k = 0; k < c.size(); ++k) {
        const s4::Copy3D &q = c[k];
        if (k == 0) { f.d0 = (long)q.dst_stride[0]; f.d1 = (long)q.dst_stride[1]; }
        if (q.empty()) continue;
        if ((long)q.dst_stride[0] != f.d0 || (long)q.dst_stride[1] != f.d1 ||
            q.dst_stride[2] != 1 || f.d1 <= 0 || f.d0 < f.d1 ||
            q.dst_span() > cells ||
            (long)(q.dst_offset % f.d1) + (long)q.n[2] > f.d1 ||
            (long)((q.dst_offset % f.d0) / f.d1) + (long)q.n[1] > f.d0 / f.d1)
            return pme_fail(m->rank, "pme_fold", "brick self copy is not a pencil box");
        pm::FoldSelf::Box &b = f.box[f.n++];
        const long o = (long)q.dst_offset;
        b.z0 = (int)(o / f.d0);
        b.y0 = (int)((o % f.d0) / f.d1);
        b.x0 = (int)(o % f.d1);
        b.nz = (int)q.n[0];
        b.ny = (int)q.n[1];
        b.nx = (int)q.n[2];
        b.off = (long)q.src_offset;
        b.s0 = (long)q.src_stride[0];
        b.s1 = (long)q.src_stride[1];
        b.s2 = (long)q.src_stride[2];
    }
    return GC_OK;
}

}  /* anonymous namespace */

namespace gcn {

gc_status native_pme_plan(gc_context *ctx, const gc_pme_desc *desc)
{
    struct gcn_device *d = ctx->native;
    if (d->pme_mesh) return GC_E_STATE;
    /* Local and deterministic: the parameters and this rank's brick; then the collective build. */
    MeshSetup ms;
    for (int k = 0; k < 3; ++k) {
        ms.in.box[k]   = d->box[k];
        ms.in.ngrid[k] = desc->ngrid[k];
    }
    ms.in.order   = desc->n_bspline;
    ms.in.alpha   = desc->alpha;
    ms.in.elecoef = desc->elecoef;
    ms.in.dielec  = desc->dielec_const;
    /* nonbond_precision = MIXED: an FP32 mesh, spectrum and transforms. */
    ms.single = d->tab.nonbond_precision == GC_NONBOND_MIXED;
    ms.probe  = d->dist == 0;
    /* The box's z unsplit and x split: the swapped layout (MeshSetup). */
    ms.swap_xz = ctx->nproc > 1 && d->layout.nd[2] == 1 &&
                 d->layout.nd[0] > 1;
    if (ms.swap_xz) {
        ms.pgrid[0] = d->layout.nd[1];
        ms.pgrid[1] = d->layout.nd[0];
    } else {
        ms.xy_domains = d->layout.nd[0] * d->layout.nd[1];
    }
    const char *why = 0;
    pm::BrickBox brick = pm::BrickBox();
    try {
        brick = owner_brick(ctx, d->layout, pm::make_recip_params(ms.in),
                            kPlanDrift);
    } catch (const std::exception &e) {
        why = e.what();
    }
    if (ms.swap_xz) {
        ms.in = swapped(ms.in);
        brick = swapped(brick);
    }
    gc_status st = mesh_build(MPI_Comm_f2c((MPI_Fint)ctx->comm), (int)ctx->rank,
                              (int)ctx->nproc, ms, why, brick, d->pme_stream,
                              &d->pme_mesh);
    if (st != GC_OK) return st;
    /* Ewald self-energy prefactor: u_self = -sum q^2 * el_fact * alpha /
     * sqrt(pi) (sp_energy_pme_opt_1dalltoall.fpp). */
    const double pi = 3.14159265358979323846;
    d->pme_self_fact = -(desc->elecoef / desc->dielec_const) * desc->alpha
                     / sqrt(pi);
    d->reciprocal_ready = 1;
    return GC_OK;
}

/* One reduction over the owned charges, times the prefactor the plan fixed;
 * recomputed per evaluation so a charge or box change cannot leave it stale. */
gc_f64 native_pme_self_energy(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;
    if (d->num_owned <= 0) return 0.0;

    gc_i64 nb = (d->num_owned + GCN_RED_BLOCK - 1) / GCN_RED_BLOCK;
    if (nb > GCN_MAX_BLOCKS) nb = GCN_MAX_BLOCKS;
    if (nb < 1) nb = 1;

    gcn_kern_pme_self<<<(unsigned)nb, GCN_RED_BLOCK, 0, d->stream>>>(
        d->charge, d->num_owned, d->reduce_partial);

    double q2 = 0.0;
    if (native_reduce(ctx, &q2, 1) != GC_OK) return 0.0;
    return d->pme_self_fact * q2;
}

/* The reciprocal sum for the current box, on the mesh and order the plan
 * fixed (pme_pre on a barostat step).  The brick must still hold the owned
 * atoms' stencils; checked, because the domains move in grid units with the
 * box. */
gc_status native_pme_rebox(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;
    gcn_pme_mesh *m = d->pme_mesh;
    if (m == 0) return GC_E_STATE;
    const char *why = 0;
    try {
        pm::RecipInput in = m->swap_xz ? swapped(m->input) : m->input;
        for (int k = 0; k < 3; ++k) in.box[k] = d->box[k];
        const pm::RecipParams box_prm = pm::make_recip_params(in);
        const pm::RecipParams prm = m->swap_xz
            ? pm::make_recip_params(swapped(in)) : box_prm;
        why = pm::spread_gather_admissible(prm);
        pm::BrickBox need = owner_brick(ctx, d->layout, box_prm, kReboxDrift);
        if (m->swap_xz) need = swapped(need);
        if (!why && !brick_holds(m->brick, need, prm.N))
            why = "the new box moves owned atoms' stencils past the PME brick";
        if (!why && !m->recip.rebox(prm)) why = m->recip.last_error();
    } catch (const std::exception &e) {
        why = e.what();
    }
    if (why) {
        pme_fail(ctx, "pme_rebox", why);
        return GC_E_UNSUPPORTED;
    }
    return GC_OK;
}

/* Collective when a plan exists: the sessions are destroyed together. */
gc_status native_pme_release(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;
    if (d == 0) return GC_OK;
    d->reciprocal_ready = 0;
    gcn_pme_mesh *m = d->pme_mesh;
    if (m == 0) return GC_OK;
    mesh_release(m);
    d->pme_mesh = 0;
    return GC_OK;
}

gc_status native_pme_run(gc_context *ctx, gc_i32 want_energy,
                         gc_i32 want_scalars, int forked)
{
    struct gcn_device *d = ctx->native;
    gcn_pme_mesh *m = d->pme_mesh;
    if (m == 0) return GC_E_STATE;
    const long n = (long)d->num_owned;
    const long pitch = (long)d->pitch;
    void *brick = m->buf[kBrick];
    gc_status st;
    ++m->epoch;
    /* On its own stream the sum forks after this point of `stream` (the
     * step's coordinates are final) and native_pme_join orders `stream`
     * after it. */
    cudaStream_t ps = m->stream;
    if (ps != d->stream &&
        ((!forked && cudaEventRecord(d->pme_fork, d->stream) != cudaSuccess) ||
         cudaStreamWaitEvent(ps, d->pme_fork, 0) != cudaSuccess))
        return pme_fail(ctx, "pme_fork", "event");
    /* The brick's sums start from zero (one rank's fold does it in place). */
    if (!m->whole && cudaMemsetAsync(m->buf[kBrickWords], 0,
                                     (size_t)m->bytes[kBrickWords], ps) !=
                         cudaSuccess)
        return pme_fail(ctx, "pme_spread", "clear");
    long mpitch = pitch;
    const double *mcoord = m->swap_xz ? swapped_soa(d->coord, pitch, &mpitch)
                                      : d->coord;
    if (!m->recip.spread(mcoord, d->charge, n, mpitch, m->brick,
                         m->buf[kBrickWords]))
        return pme_fail(ctx, "pme_spread", m->recip.last_error());
    if (m->whole) {
        if (!m->recip.fold(m->buf[kBrickWords], m->brick.cells(), brick))
            return pme_fail(ctx, "pme_fold", m->recip.last_error());
        if (!m->fft.exec_forward_x(brick, m->buf[kZCplx]))
            return pme_fail(ctx, "pme_fft", m->fft.last_error());
        if (!m->recip.solve(m->buf[kZCplx], m->zpencil, want_scalars != 0))
            return pme_fail(ctx, "pme_solve", m->recip.last_error());
        if (want_scalars)
            gcn_kern_pme_publish<<<1, 1, 0, ps>>>(m->recip.scalars_device(),
                                                  d->acc_recip, (int)want_energy);
        if (!m->fft.exec_inverse_x(m->buf[kZCplx], brick))
            return pme_fail(ctx, "pme_fft", m->fft.last_error());
    } else {
        if ((st = m->move(kBrickToX, ps)) != GC_OK) return st;
        if (!m->recip.fold(m->buf[kXWords],
                           (long)m->plan[kBrickToX].target_elements,
                           m->buf[kXReal], &m->fold_self))
            return pme_fail(ctx, "pme_fold", m->recip.last_error());
        if (!m->fft.exec_forward_x(m->buf[kXReal], m->buf[kXCplx]))
            return pme_fail(ctx, "pme_fft", m->fft.last_error());
        if ((st = m->move(kXToY, ps)) != GC_OK) return st;
        if (!m->fft.exec_forward_y(m->buf[kYCplx]))
            return pme_fail(ctx, "pme_fft", m->fft.last_error());
        if ((st = m->move(kYToZ, ps)) != GC_OK) return st;
        if (!m->fft.exec_forward_z(m->buf[kZCplx]))
            return pme_fail(ctx, "pme_fft", m->fft.last_error());
        if (!m->recip.solve(m->buf[kZCplx], m->zpencil, want_scalars != 0))
            return pme_fail(ctx, "pme_solve", m->recip.last_error());
        if (want_scalars)
            gcn_kern_pme_publish<<<1, 1, 0, ps>>>(m->recip.scalars_device(),
                                                  d->acc_recip, (int)want_energy,
                                                  (int)m->swap_xz);
        /* Back to the brick, unnormalised, as vol_fact4 in the gather expects. */
        if (!m->fft.exec_inverse_z(m->buf[kZCplx]))
            return pme_fail(ctx, "pme_fft", m->fft.last_error());
        if ((st = m->move(kZToY, ps)) != GC_OK) return st;
        if (!m->fft.exec_inverse_y(m->buf[kYCplx]))
            return pme_fail(ctx, "pme_fft", m->fft.last_error());
        if ((st = m->move(kYToX, ps)) != GC_OK) return st;
        if (!m->fft.exec_inverse_x(m->buf[kXCplx], m->buf[kXReal]))
            return pme_fail(ctx, "pme_fft", m->fft.last_error());
        if ((st = m->move(kXToBrick, ps)) != GC_OK) return st;
    }
    double *mforce = m->swap_xz ? swapped_soa(d->force_recip, pitch, &mpitch)
                                : d->force_recip;
    if (!m->recip.gather(mcoord, d->charge, n, mpitch, m->brick, brick,
                         mforce))
        return pme_fail(ctx, "pme_gather", m->recip.last_error());
    if (ps != d->stream && cudaEventRecord(d->pme_done, ps) != cudaSuccess)
        return pme_fail(ctx, "pme_done", "event");
    return dev_launched("pme");
}

int native_pme_crosses_nodes(const gc_context *ctx)
{
    const gcn_pme_mesh *m = ctx->native ? ctx->native->pme_mesh : 0;
    return m != 0 && m->spans_nodes;
}

/* A step captured in a CUDA graph: every move may be captured on its next
 * epoch (MeshExchangeSession::steady); a replay runs each move's host side. */
int native_pme_graph_ready(const gc_context *ctx)
{
    const gcn_pme_mesh *m = ctx->native ? ctx->native->pme_mesh : 0;
    if (m == 0) return 0;
    for (int i = 0; i < kMoves; ++i)
        if (!m->session[i].steady(m->epoch + 1)) return 0;
    return 1;
}

int native_pme_ce_candidate(const gc_context *ctx)
{
    const gcn_pme_mesh *m = ctx->native ? ctx->native->pme_mesh : 0;
    if (m == 0) return 0;
    for (int i = 0; i < kMoves; ++i)
        if (m->session[i].copy_engine_candidate())
            return 1;
    return 0;
}

void native_pme_allow_ce(gc_context *ctx, int on)
{
    gcn_pme_mesh *m = ctx->native ? ctx->native->pme_mesh : 0;
    if (m == 0) return;
    for (int i = 0; i < kMoves; ++i) m->session[i].allow_copy_engine(on != 0);
}

void native_pme_graph_capability(const gc_context *ctx, int *fused,
                                 int *stores)
{
    const gcn_pme_mesh *m = ctx->native ? ctx->native->pme_mesh : 0;
    *fused = *stores = m != 0;
    if (m == 0) return;
    for (int i = 0; i < kMoves; ++i) {
        bool f = false, s = false;
        m->session[i].graph_capability(&f, &s);
        if (!f) *fused = 0;
        if (!s) *stores = 0;
    }
}

int native_pme_graph_slot(const gc_context *ctx)
{
    const gcn_pme_mesh *m = ctx->native ? ctx->native->pme_mesh : 0;
    if (m == 0) return 0;
    for (int i = 0; i < kMoves; ++i)
        if (m->session[i].slot_bound())
            return (int)((m->epoch + 1) % GCX_SLOTS);
    return 0;
}

gc_status native_pme_step_replay(gc_context *ctx)
{
    gcn_pme_mesh *m = ctx->native ? ctx->native->pme_mesh : 0;
    if (m == 0) return GC_E_STATE;
    ++m->epoch;
    for (int i = 0; i < kMoves; ++i) {
        const gc_status st = m->session[i].replay_epoch(m->epoch);
        if (st != GC_OK) return pme_fail(ctx, "pme_replay", "unsteady epoch");
    }
    return GC_OK;
}

gc_status native_pme_join(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;
    gcn_pme_mesh *m = d->pme_mesh;
    if (m == 0 || m->stream == d->stream) return GC_OK;
    return cudaStreamWaitEvent(d->stream, d->pme_done, 0) == cudaSuccess
           ? GC_OK : GC_E_DEVICE;
}

}  /* namespace gcn */
