/*
 * gpu_core_xchg.h : the device-native core's exchange ABI.
 *
 * One transport, several schedules: coordinates, forces, migrating groups,
 * owner-directory requests and mesh transposes travel through the plan/token
 * interface; the caller supplies its own edge set and pack/unpack.  POD and
 * C-interoperable: no MPI or CUDA type crosses it (a communicator is the
 * Fortran handle, streams and events are void pointers).  Routes come from a
 * start-up probe and a two-round handshake, never from the environment.
 */

#ifndef GPU_CORE_XCHG_H
#define GPU_CORE_XCHG_H

#include "gpu_core_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Wire schema of gcx_wire_header; a receiver seeing another value refuses. */
#define GCX_WIRE_SCHEMA 1

/* Largest payload per MPI call (int count); larger transfers are split into
 * chunks in ascending byte offset on both endpoints. */
#define GCX_MPI_CHUNK_BYTES ((gc_i64)1 << 30)

/* Slots per edge: a slot is rewritten only after its previous use is acknowledged. */
#define GCX_SLOTS 2

/* Edge keys tell apart two edges of a plan to one peer and are part of the
 * message tag; six are reserved for (axis, direction) pairs. */
#define GCX_MAX_EDGE_KEY 32

/* Tags owned by a plan from its tag_base; a span past MPI_TAG_UB is refused. */
#define GCX_TAG_OPS   (GCX_MAX_EDGE_KEY * GCX_SLOTS * 3)
#define GCX_TAG_ROUNDS 4
#define GCX_TAG_SPAN  (GCX_TAG_OPS + GCX_TAG_ROUNDS * GCX_MAX_EDGE_KEY)

/* Bytes of a PCI bus id, the only device identity compared (ordinals are
 * process-local). */
#define GCX_DEVICE_ID_LEN 24

/* An edge keeps the peer route unless, beside compute, it is slower than
 * the staged reference by more than this factor at the edge's size. */
#define GCX_PEER_MARGIN 1.05

/* The caller's Fortran MPI communicator handle (MPI_Comm_f2c recovers it). */
typedef gc_i32 gcx_comm;

/* Ordered from most to least capable; a disagreeing pair is demoted to the
 * larger value, so both endpoints reach the same answer. */
typedef enum gcx_route {
    GCX_ROUTE_SELF       = 0,  /* the edge's peer is this rank              */
    GCX_ROUTE_LOCAL      = 1,  /* two ranks, one GPU: device-local copy     */
    GCX_ROUTE_PEER       = 2,  /* two GPUs of one node: peer copy over IPC  */
    GCX_ROUTE_MPI_DEVICE = 3,  /* MPI handed device pointers                */
    GCX_ROUTE_STAGED     = 4,  /* pinned host bounce, then MPI              */
    GCX_ROUTE_NKIND      = 5
} gcx_route;

/* What is known about a device-buffer MPI route; VERIFIED requires
 * affirmative evidence from the transport, never a bandwidth reading. */
typedef enum gcx_rdma {
    GCX_RDMA_STAGED   = 0,     /* the fabric refused device registration    */
    GCX_RDMA_UNKNOWN  = 1,     /* device pointers are accepted; how is not  */
    GCX_RDMA_VERIFIED = 2      /* the transport reported RDMA affirmatively */
} gcx_rdma;

/* Operation classes, each with its own tag block and request set. */
typedef enum gcx_op_kind {
    GCX_OP_HALO_COORD      = 0,
    GCX_OP_HALO_FORCE      = 1,
    GCX_OP_MIGRATE_PAYLOAD = 2,
    GCX_OP_MESH            = 3,
    GCX_OP_NKIND           = 4
} gcx_op_kind;

const char *gcx_route_name(gc_i32 route);
const char *gcx_rdma_name(gc_i32 rdma);
const char *gcx_op_name(gc_i32 op);

/* Header of every message of every route; it travels by MPI even for peer
 * routes, where it publishes completion of the device copy. */
typedef struct gcx_wire_header {
    gc_i32 schema;      /* GCX_WIRE_SCHEMA                                 */
    gc_i32 op;          /* gcx_op_kind                                     */
    gc_i32 src;         /* communicator-local source rank                  */
    gc_i32 dst;         /* communicator-local destination rank             */
    gc_i64 epoch;       /* the caller's epoch, monotone per plan           */
    gc_i64 sequence;    /* message ordinal within the plan, monotone       */
    gc_i64 bytes;       /* payload bytes that follow, may be zero          */
    gc_i64 records;     /* typed record count, caller defined              */
} gcx_wire_header;

/* One directed edge; capacities are the bytes admitted in each direction. */
typedef struct gcx_edge_desc {
    gc_i32 peer;             /* communicator-local rank                    */
    gc_i32 axis;             /* 0..2 for a dimensional edge, -1 otherwise  */
    gc_i32 dir;              /* -1 lower, +1 upper, 0 not dimensional      */
    /* key is this rank's name for the edge and partner the peer's, both below
     * GCX_MAX_EDGE_KEY.  -1 derives them: dimensional edges key on (axis,
     * direction) and pair with the opposite direction, plain edges on their
     * ordinal among edges to the same peer.  Asymmetric edge sets state them. */
    gc_i32 key;
    gc_i32 partner;
    gc_i32 pad0;
    gc_i64 send_capacity;    /* bytes this rank may send on this edge      */
    gc_i64 recv_capacity;    /* bytes this rank may receive on this edge   */
} gcx_edge_desc;

typedef struct gcx_plan_desc {
    const gcx_edge_desc *edge;
    gc_i64 num_edges;
    gc_i32 op;               /* gcx_op_kind: the class this plan serves    */
    gc_i32 tag_base;         /* first MPI tag; the plan uses a bounded run */
    gc_i32 wire_schema;      /* GCX_WIRE_SCHEMA                            */
    gc_i32 pad0;
    /* Wait bound before GC_E_STATE; zero takes the default. */
    gc_i64 deadline_ms;
} gcx_plan_desc;

typedef struct gcx_plan_s  gcx_plan;
typedef struct gcx_token_s gcx_token;

/* What the plan decided. */
typedef struct gcx_route_report {
    gc_i64 edges[GCX_ROUTE_NKIND];
    gc_i64 bytes_sent;
    gc_i64 bytes_received;
    gc_i64 messages;
    gc_i64 chunks;           /* payload chunks above GCX_MPI_CHUNK_BYTES   */
    gc_i32 rdma;             /* gcx_rdma, for the mpi_device edges         */
    gc_i32 pad0;
} gcx_route_report;

/* Start-up probe, run once per communicator before any plan. */
typedef struct gcx_probe_report {
    gc_i32 rank;
    gc_i32 nproc;
    gc_i32 node_key;            /* the node's lowest communicator rank     */
    gc_i32 node_ranks;
    gc_i32 nodes;
    gc_i32 device_ordinal;      /* this process's numbering, never shared  */
    gc_i32 peer_capable;        /* node-local peers with CUDA peer access  */
    gc_i32 peer_admitted;       /* those whose link passed the read check  */
    gc_i32 mpi_device_capable;  /* the fabric registered device memory     */
    gc_i32 rdma;                /* gcx_rdma                                */
    gc_f64 bw_staged;           /* GB/s, the conservative reference        */
    gc_f64 bw_peer_worst;       /* GB/s per link, all links at once, large */
    gc_f64 margin;              /* GCX_PEER_MARGIN, the rule applied       */
    gc_f64 seconds;            /* probe wall clock, reported separately    */
    char   device_id[GCX_DEVICE_ID_LEN];
} gcx_probe_report;

/* Collective.  Forms the node communicator, resolves device identity, admits
 * peer links against the staged reference and decides whether device-pointer
 * MPI is usable (unknown support selects staging without a trial). */
gc_status gcx_probe_run(gcx_comm comm, gcx_probe_report *out);
gc_status gcx_probe_release(void);
/* The node-local GPUs this rank's plans route PEER, and whether any of those
 * links has native atomics (NVLink-class; PCIe peer links have none). */
void gcx_peer_fanout(gc_i32 *peers, gc_i32 *native_atomic);

/* Inter-node transport per exchange class ([DYNAMICS] gpu_route_mesh,
 * gpu_route_coord, gpu_route_force): MPI keeps the elected route, THREAD
 * hands inter-node edges to a transport thread.  Results are bitwise
 * identical for every choice.  Set on every rank before gcx_probe_run. */
typedef enum gcx_route_class {
    GCX_ROUTE_CLASS_MESH  = 0,
    GCX_ROUTE_CLASS_COORD = 1,
    GCX_ROUTE_CLASS_FORCE = 2,
    GCX_ROUTE_NCLASS      = 3
} gcx_route_class;
typedef enum gcx_route_mode {
    GCX_ROUTE_MODE_MPI      = 0,
    GCX_ROUTE_MODE_THREAD   = 1
} gcx_route_mode;
gc_status gcx_route_select(const gc_i32 mode[GCX_ROUTE_NCLASS]);

/* Device sum of nwords 64-bit integer words over every rank of comm, in
 * place on the caller's stream, through peers' mapped boards.  Integer
 * words give identical bits on every rank.  The device sum needs all ranks
 * on one node, peer capable, on distinct GPUs; otherwise, with host_thread,
 * a host thread sums over MPI (graph-capturable, no gcx_wsum_view).  *out
 * is null when neither applies.  Collective; needs the probe. */
typedef struct gcx_wsum gcx_wsum;
gc_status gcx_wsum_create(gcx_comm comm, gc_i32 nwords, gc_i32 host_thread,
                          gcx_wsum **out);
gc_status gcx_wsum_launch(gcx_wsum *w, gc_u64 *words, void *stream);
void      gcx_wsum_destroy(gcx_wsum *w);

/* The sum's operands, for a kernel that runs the sum itself. */
struct gcx_wsum_view {
    gc_u64 *const *boards;
    unsigned int *epoch;
    int nwords, nproc, rank;
    gc_u64 *post, *sum;                /* the host thread's words, or null */
};
gc_status gcx_wsum_view_of(const gcx_wsum *w, struct gcx_wsum_view *v);

#ifdef __CUDACC__
/* The sum by one thread block, as gcx_wsum_launch runs it. */
static __device__ __forceinline__ void gcx_wsum_sum(
    gc_u64 *__restrict__ words, const struct gcx_wsum_view v)
{
    __shared__ unsigned int ep;
    if (threadIdx.x == 0) { ep = *v.epoch + 1; *v.epoch = ep; }
    __syncthreads();
    const int m = 2 * v.nwords;
    if (v.post) {
        for (int k = threadIdx.x; k < m; k += blockDim.x) {
            const gc_u64 w = words[k >> 1];
            const gc_u64 half = (k & 1) ? (w >> 32) : (w & 0xffffffffull);
            ((volatile gc_u64 *)v.post)[k] = ((gc_u64)ep << 32) | half;
        }
        __threadfence_system();
        __syncthreads();               /* every read of words is done */
        for (int j = threadIdx.x; j < v.nwords; j += blockDim.x) {
            const volatile gc_u64 *e = v.sum + 2 * j;
            gc_u64 lo, hi;
            do { lo = e[0]; } while ((unsigned int)(lo >> 32) != ep);
            do { hi = e[1]; } while ((unsigned int)(hi >> 32) != ep);
            words[j] = (lo & 0xffffffffull) | (hi << 32);
        }
        return;
    }
    const size_t slot = (size_t)(ep & 1) * v.nproc * m;
    for (int i = threadIdx.x; i < v.nproc * m; i += blockDim.x) {
        const int r = i / m, k = i - r * m;
        const gc_u64 w = words[k >> 1];
        const gc_u64 half = (k & 1) ? (w >> 32) : (w & 0xffffffffull);
        volatile gc_u64 *dst = v.boards[r] + slot + (size_t)v.rank * m + k;
        *dst = ((gc_u64)ep << 32) | half;
    }
    __syncthreads();                   /* every read of words is done */
    const volatile gc_u64 *own = v.boards[v.rank] + slot;
    for (int j = threadIdx.x; j < v.nwords; j += blockDim.x) {
        gc_u64 sum = 0;
        for (int r = 0; r < v.nproc; ++r) {
            const volatile gc_u64 *e = own + (size_t)r * m + 2 * j;
            gc_u64 lo, hi;
            do { lo = e[0]; } while ((unsigned int)(lo >> 32) != ep);
            do { hi = e[1]; } while ((unsigned int)(hi >> 32) != ep);
            sum += (lo & 0xffffffffull) | (hi << 32);
        }
        words[j] = sum;
    }
}
#endif

/* A device-sequenced edge of a mesh plan, driven from the caller's stream.
 * Slot s of the peer's landing area is at peer_slots + s * send_capacity, of
 * this rank's at my_slots + s * recv_capacity; sig words: [0] ready, [1]
 * ack.  `stores`: the caller's kernel stores into the peer's slots (chosen
 * by the probe, one choice per node); otherwise it copies with copy engines. */
typedef struct gcx_device_edge {
    char   *peer_slots;
    char   *my_slots;
    unsigned long long *peer_sig;
    unsigned long long *my_sig;
    gc_i64  send_capacity;
    gc_i64  recv_capacity;
    gc_i32  devsig;
    gc_i32  stores;
} gcx_device_edge;
gc_status gcx_plan_device_edge(const gcx_plan *plan, gc_i64 edge,
                               gcx_device_edge *out);

/* The device-sequenced move, shared by the halo passes and the mesh moves.
 * The caller packs, signals and unpacks on its own stream; the plan keeps
 * the epochs: the last one taken, each slot's last user, and the one the
 * caller's device sequence word (gcx_seq) holds.
 *
 * gcx_move_edges: every edge's device view (edges[num_edges], may be null);
 * fused when every remote edge is device-sequenced, stores when every remote
 * edge also stores into the peer's slots. */
gc_status gcx_move_edges(const gcx_plan *plan, gcx_device_edge *edges,
                         gc_i32 *fused, gc_i32 *stores);
/* Claims `epoch` (must increase) and its slot on a plan whose remote edges
 * are all device-sequenced.  `prev` is the epoch that last used the slot (0:
 * none), whose acknowledgement the caller awaits before writing the peers.
 * Captured on `stream`, the move must be steady (gcx_move_steady) and
 * `stride` is what its launches advance the sequence word by; else 0. */
typedef struct gcx_move_open {
    gc_i32 slot;
    gc_i32 pad0;
    gc_i64 prev;
    unsigned long long stride;
} gcx_move_open;
gc_status gcx_move_claim(gcx_plan *plan, gc_i64 epoch, void *stream,
                         gcx_move_open *out);
/* Whether the move claiming `epoch` may be captured: the sequence word holds
 * the last epoch and the slot's previous user is GCX_SEQ_BACK(stride) back.
 * `same_slot`: the capture also fixes the slot (stride a multiple of
 * GCX_SLOTS). */
int gcx_move_steady(const gcx_plan *plan, gc_i64 epoch, int same_slot);
/* A replayed capture's host side: the steady claim of `epoch`, no launch. */
gc_status gcx_move_replay(gcx_plan *plan, gc_i64 epoch);

/* Epoch and awaited epoch of such an operation, in device memory so a
 * captured graph takes no per-step argument.  stride 0: the first kernel
 * stores the host's values; stride > 0: the device advances the word and the
 * slot's previous user is GCX_SEQ_BACK(stride) epochs earlier. */
typedef struct gcx_seq {
    unsigned long long epoch;
    unsigned long long prev;
} gcx_seq;
#define GCX_SEQ_BACK(stride) \
    ((stride) % GCX_SLOTS ? (stride) * GCX_SLOTS : (stride))

/* Milliseconds an exchange waits for a peer on host and device before
 * reporting it lost; override with -DGCX_DEADLINE_MS. */
#ifndef GCX_DEADLINE_MS
#define GCX_DEADLINE_MS 30000
#endif

#ifdef __CUDACC__
/* Bounded device wait on a signal word; past the deadline the kernel traps. */
#define GCX_SIG_DEADLINE_NS ((unsigned long long)(GCX_DEADLINE_MS) * 1000000ull)
__device__ inline void gcx_sig_wait(const volatile unsigned long long *f,
                                    unsigned long long v)
{
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    while (*f < v) {
        __nanosleep(64);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        /* A reading behind the first restarts the count (unsigned wrap). */
        if (t < t0) t0 = t;
        else if (t - t0 > GCX_SIG_DEADLINE_NS) __trap();
    }
}

__device__ inline gcx_seq gcx_seq_open(gcx_seq *q, unsigned long long epoch,
                                       unsigned long long prev,
                                       unsigned long long stride)
{
    gcx_seq s;
    if (stride == 0) {
        s.epoch = epoch;
        s.prev = prev;
    } else {
        s.epoch = q->epoch + stride;
        s.prev = s.epoch - GCX_SEQ_BACK(stride);
    }
    *q = s;
    return s;
}
#endif
/* Collective, before any device allocation: the native core's rank -> GPU
 * permutation.  nd = domain grid, box = box lengths.  Leaves the default
 * device when it does not apply.  Returns on every rank the slowest measured
 * one-way neighbour link (link_gbs, GB/s; 0 if unmeasured) and the device
 * throughput (units: SMs x GHz, smallest over comm). */
void gcx_place_device(gcx_comm comm, const gc_i32 *nd, const double *box,
                      double *link_gbs, double *units);

/* Collective.  Elects a route per edge, runs the two-round handshake
 * (proposal with deterministic demotion, then confirmation; a mismatch is
 * GC_E_MISMATCH) and logs one line per route class. */
gc_status gcx_plan_create(const gcx_plan_desc *desc, gcx_comm comm,
                          gcx_plan **out);
gc_status gcx_plan_destroy(gcx_plan *plan);
/* Edge identities from the frozen plan, including the peer's partner key. */
gc_status gcx_plan_edge_identity(const gcx_plan *plan, gc_i64 edge,
                                 gc_i32 *key, gc_i32 *partner, gc_i32 *peer);

typedef struct gcx_buffer {
    void  *ptr;
    gc_i64 bytes;
} gcx_buffer;

typedef struct gcx_op_desc {
    gc_i64 epoch;                 /* refused when it does not increase     */
    gc_i32 op;                    /* must equal the plan's op              */
    /* Non-zero: sizes differ from the capacities, so headers complete first and
     * capacity is checked against the arriving counts.  Zero: fixed-size; each
     * edge receives its capacity or its receive buffer's bytes if fewer. */
    gc_i32 variable;
    const gcx_buffer *send;       /* [num_edges]                           */
    const gcx_buffer *recv;       /* [num_edges]                           */
    const gc_i64 *send_bytes;     /* [num_edges], null = the buffer's bytes */
    const gc_i64 *send_records;   /* [num_edges], may be null              */
    gc_i64 *recv_bytes;           /* [num_edges] out, may be null          */
    gc_i64 *recv_records;         /* [num_edges] out, may be null          */
    /* Producer event, recorded after the work that fills the send buffers and
     * clears the receive buffers; the transport stream waits for it.  The
     * transport stream is non-blocking and unordered against the legacy
     * default stream, so a null event asserts that every buffer is already
     * visible to the device (cudaMemcpy/cudaMemset returning is not enough). */
    void *producer_event;
} gcx_op_desc;

/* Post receives, wait for the producer, dispatch sends; returns a token.
 * No device synchronization.  One outstanding epoch per plan (a second begin
 * is GC_E_STATE).  The caller must not write send buffers or read receive
 * buffers until gcx_consume, nor free either until gcx_release. */
gc_status gcx_begin(gcx_plan *plan, const gcx_op_desc *op, gcx_token **out);

/* Drive MPI on the calling host thread; never blocks. */
gc_status gcx_progress(gcx_plan *plan);

/* Make the received payload visible to consumer_stream: completes the
 * transport, validates every header against the plan and epoch, issues the
 * landing copies a peer route needs. */
gc_status gcx_consume(gcx_token *token, void *consumer_stream);

/* Permission to reuse the slot; the acknowledgement waits for consumer_done,
 * so the peer cannot overwrite a buffer still being read. */
gc_status gcx_release(gcx_token *token, void *consumer_done);

/* Drop a refused operation after a gcx_consume error: cancels what is in
 * flight and faults the plan.  The run stops; there is no recovery. */
gc_status gcx_abort(gcx_token *token);

/* Three-pass dimensional halo: coordinates forward x -> y -> z, ghost forces
 * back z -> y -> x, additively, along the recorded reverse route.  Six
 * plans; the same rank on both sides of a short periodic dimension is two
 * distinct edges. */
typedef struct gcx_halo_desc {
    gc_i32 num_domain[3];
    gc_i32 neighbour_lower[3];    /* communicator-local ranks              */
    gc_i32 neighbour_upper[3];
    gc_i32 tag_base;
    gc_i64 coord_capacity[3];     /* bytes per side per axis               */
    gc_i64 force_capacity[3];
    gc_i64 deadline_ms;
} gcx_halo_desc;

typedef struct gcx_halo_s gcx_halo;

gc_status gcx_halo_create(const gcx_halo_desc *desc, gcx_comm comm,
                          gcx_halo **out);
gc_status gcx_halo_destroy(gcx_halo *halo);

/* The plan of one pass; forward is x,y,z, reverse is z,y,x (pass p uses
 * axis 2-p). */
gc_status gcx_halo_plan(gcx_halo *halo, gc_i32 pass, gc_i32 reverse,
                        gcx_plan **out);

/* One forward coordinate pass and one reverse force pass.  The reverse pass
 * is summed by the caller's unpack kernel, not by the transport. */
gc_status gcx_halo_forward(gcx_halo *halo, gc_i32 pass,
                           const gcx_op_desc *op, gcx_token **out);
gc_status gcx_halo_reverse(gcx_halo *halo, gc_i32 pass,
                           const gcx_op_desc *op, gcx_token **out);

/* Migration: the payload plan.  It takes the second tag span after
 * tag_base, so its tags are those it had beside a counts plan. */
typedef struct gcx_migration_desc {
    const gcx_edge_desc *edge;
    gc_i64 num_edges;
    gc_i32 tag_base;
    gc_i32 pad0;
    gc_i64 deadline_ms;
} gcx_migration_desc;

typedef struct gcx_migration_s gcx_migration;

gc_status gcx_migration_create(const gcx_migration_desc *desc, gcx_comm comm,
                               gcx_migration **out);
gc_status gcx_migration_destroy(gcx_migration *mig);

/* Exchange the payload. */
gc_status gcx_migration_payload(gcx_migration *mig, const gcx_op_desc *op,
                                gcx_token **out);

#ifdef __cplusplus
}
#endif

#endif /* GPU_CORE_XCHG_H */
