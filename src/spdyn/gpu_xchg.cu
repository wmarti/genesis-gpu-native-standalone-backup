/*
 * gpu_xchg.cu : the device-native core's one exchange primitive.
 *
 * A plan is a set of directed edges whose routes are elected once from the
 * start-up probe and agreed by both endpoints; an operation returns a token,
 * not a global synchronization.  The three-pass halo and the migration
 * exchange are schedules built on it.
 */

#include "gpu_core_xchg.h"
#include "gpu_core_internal.h"

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <dirent.h>
#include <sched.h>
#include <cstdlib>
#endif

#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif

/* Private types of the exchange primitive; only the gcx_ entry points of
 * gpu_core_xchg.h are public. */
namespace gcx {

/* Collective start-up probe result, cached for the process; routes are
 * frozen for the run. */
struct Probe {
    bool        done;
    MPI_Comm    comm;          /* the communicator it was run on           */
    MPI_Comm    node;          /* MPI_COMM_TYPE_SHARED split of it         */
    int         rank, nproc;
    int         node_rank, node_size;
    int         node_key;      /* the node's lowest comm rank              */
    int         nodes;
    int         device;        /* this process's ordinal                   */
    std::string device_id;     /* PCI bus id: the only identity exchanged  */
    std::vector<int>  peer_node;      /* node key of that rank             */
    std::vector<char> same_node;
    std::vector<char> same_device;    /* node-local and one GPU            */
    std::vector<char> peer_capable;   /* node-local, two GPUs, IPC usable  */
    std::vector<char> peer_admitted;  /* those whose link passed the check */
    std::vector<int>  peer_ordinal;   /* that GPU here, -1 = not visible   */
    std::vector<char> peer_atomic;    /* the link has native atomics       */
    std::vector<char> peer_used;      /* some plan routes an edge PEER     */
    /* Staged reference bandwidth at the small and large probed sizes. */
    double      bw_staged_small;
    double      bw_staged;            /* the large size                    */
    /* Measured node-local peer links, recorded whether admitted or not. */
    std::vector<double> peer_bw_small;
    std::vector<double> peer_bw_large;
    double      bw_peer_worst;        /* worst measured, large size        */
    /* Seconds per move at the two probed sizes: SM stores (peer) and
     * pinned-memory copies (staged), idle and beside a compute kernel. */
    double      t_peer[2];
    double      t_staged_fan[2];
    double      t_peer_load[2];
    double      t_staged_load[2];
    /* Copy-engine peer copies, idle and beside compute. */
    double      t_ce[2];
    double      t_ce_load[2];
    bool        mpi_device_capable;
    /* Some GPU hosts several ranks: device spin-waits (devsig) are off. */
    bool        shared_gpu;
    gc_i32      rdma;
    /* Transport thread usable: MPI_THREAD_MULTIPLE, device buffers, GPUDirect RDMA. */
    bool        relay_ok;
    gc_i32      route_mode[GCX_ROUTE_NCLASS];
    double      seconds;

    Probe();
};

Probe &probe();

struct Edge {
    int    peer;
    int    axis, dir;
    int    key, partner;      /* the two endpoints' names for this edge    */
    gc_i64 send_capacity;
    gc_i64 recv_capacity;
    gc_i32 route;             /* gcx_route, frozen by the handshake        */
    /* peer_base: mapping of the peer's exported memory (for
     * cudaIpcCloseMemHandle); peer_inbox: this edge's slice of it. */
    void  *peer_base;
    void  *peer_inbox;
    gc_i64 inbox_offset;      /* where in MY inbox this edge's slots are   */
    bool   ipc_open;
    /* Node-local staged edge writing into the peer's mapped host slot. */
    bool   shared;
    char  *shared_peer;       /* the peer's slots for this edge            */
    char  *shared_peer_dev;   /* the same, as the device addresses it      */
    /* Device-sequenced edge: signal words replace the MPI acknowledgement. */
    bool   devsig;
    /* Shared staged edge sequenced by signal words in the host landing areas. */
    bool   shmsig;
    /* Inter-node edge carried by the transport thread (relay). */
    bool   relay;
};

struct EdgeFlight {
    gcx_wire_header send_head;
    gcx_wire_header recv_head;
    gc_i64          recv_bytes;
    gc_i64          recv_records;
    bool            sends_done;
};

/* Transport phases, driven by gcx_progress without blocking. */
enum Phase { PH_IDLE = 0, PH_PRODUCER, PH_HEADERS, PH_PAYLOAD, PH_DONE };

struct Plan {
    MPI_Comm comm;
    int      rank, nproc;
    gc_i32   op;
    gc_i32   tag_base;
    gc_i32   wire_schema;
    gc_i64   deadline_ms;

    std::vector<Edge> edge;

    /* Device inbox written by peer senders and published by gcx_consume. */
    void   *inbox;
    gc_i64  inbox_bytes;
    cudaIpcMemHandle_t inbox_handle;
    bool    inbox_exported;

    /* Pinned host bounce: one send and one receive region of host_span bytes
     * per edge per slot. */
    void   *host_send;
    void   *host_recv;
    gc_i64  host_span;
    /* Node-shared landing area and every mapping held. */
    char   *shm_base;
    char   *shm_dev;          /* shm_base as the device addresses it       */
    std::vector<std::pair<void *, gc_i64> > shm_maps;

    cudaStream_t transport;
    cudaEvent_t  send_ready;   /* this epoch's device dispatch is done     */
    cudaEvent_t  recv_ready;   /* the landing copies are done              */
    /* Landing stream so H2D of arrivals overlaps D2H of departures. */
    cudaStream_t landing;
    cudaEvent_t  land_ready;
    bool         stream_owned;
    /* The streams belong to the plan chain (shared_chain_streams). */
    bool         stream_shared;

    /* One outstanding epoch per plan. */
    gc_i64  epoch;
    gc_i64  sequence;
    int     slot;
    bool    outstanding;
    /* A refused operation faults the plan; it refuses further work. */
    bool    faulted;
    Phase   phase;
    double  started;
    bool    variable;
    bool    host_edges;       /* a remote edge that is neither devsig, shmsig nor relay */
    bool    shm_edges;        /* a shmsig edge: no caller-driven device path */
    gc_i64  slot_epoch[GCX_SLOTS];   /* the epoch last sent from each slot  */
    gc_i64  seq_epoch;        /* the epoch a device-sequenced move last opened */

    /* This epoch's buffers, copied from the caller's descriptor. */
    std::vector<gcx_buffer> cur_send, cur_recv;
    std::vector<gc_i64>     cur_send_bytes, cur_send_records;
    gc_i64                 *out_recv_bytes;
    gc_i64                 *out_recv_records;

    /* Per-slot acknowledgements; this rank's own is posted from gcx_progress
     * once its consumer event has fired. */
    std::vector<MPI_Request> ack_send[GCX_SLOTS];
    std::vector<MPI_Request> ack_recv[GCX_SLOTS];
    std::vector<gc_i64>      ack_word[GCX_SLOTS];
    std::vector<gc_i64>      ack_in[GCX_SLOTS];
    bool                     ack_pending[GCX_SLOTS];
    cudaEvent_t              ack_event;
    int                      ack_slot;
    bool                     ack_owed;
    bool                     ack_variable;   /* the owed release's operation */

    std::vector<EdgeFlight>  flight;
    std::vector<MPI_Request> req;    /* headers                            */
    std::vector<MPI_Request> preq;   /* payload chunks                     */

    gcx_route_report report;

    /* Transport-thread side: completed epoch word (pinned, device-readable),
     * last handed epoch, and whether the operation in flight is relayed. */
    volatile unsigned long long *relay_word;
    unsigned long long          *relay_word_dev;
    gc_i64                       relay_last;
    bool                         relay_op;

    Plan();
};

/* Conservative reconciliation of two proposed routes. */
gc_i32 reconcile(gc_i32 mine, gc_i32 theirs);

/* One log line per route class that has edges. */
void log_plan(const Plan &p);

}  /* namespace gcx */

struct gcx_plan_s  { gcx::Plan p; };

struct gcx_token_s {
    gcx_plan *owner;
    gc_i64    epoch;
    int       slot;
    bool      consumed;
};

struct gcx_halo_s { gcx_plan *forward[3]; gcx_plan *reverse[3]; };

struct gcx_migration_s { gcx_plan *payload; };

namespace gcx {

gc_i32 reconcile(gc_i32 mine, gc_i32 theirs)
{
    /* The larger route is the conservative common answer. */
    return mine > theirs ? mine : theirs;
}

Probe::Probe()
    : done(false), comm(MPI_COMM_NULL), node(MPI_COMM_NULL),
      rank(0), nproc(1), node_rank(0), node_size(1), node_key(0), nodes(1),
      device(-1), bw_staged(0.0), bw_peer_worst(0.0),
      mpi_device_capable(false), shared_gpu(false), rdma(GCX_RDMA_STAGED),
      relay_ok(false),
      seconds(0.0)
{
    for (int k = 0; k < GCX_ROUTE_NCLASS; ++k) route_mode[k] = GCX_ROUTE_MODE_MPI;
    t_peer[0] = t_peer[1] = t_staged_fan[0] = t_staged_fan[1] = 0.0;
    t_peer_load[0] = t_peer_load[1] = t_staged_load[0] = t_staged_load[1] = 0.0;
    t_ce[0] = t_ce[1] = t_ce_load[0] = t_ce_load[1] = 0.0;
}

Probe &probe()
{
    static Probe p;
    return p;
}

/* The gcx_route_class an operation belongs to, or -1 (none). */
static int route_class(gc_i32 op)
{
    return op == GCX_OP_MESH ? GCX_ROUTE_CLASS_MESH :
           op == GCX_OP_HALO_COORD ? GCX_ROUTE_CLASS_COORD :
           op == GCX_OP_HALO_FORCE ? GCX_ROUTE_CLASS_FORCE : -1;
}

static const char *route_mode_name(gc_i32 m)
{
    switch (m) {
    case GCX_ROUTE_MODE_MPI:      return "MPI";
    case GCX_ROUTE_MODE_THREAD:   return "THREAD";
    default:                      return "invalid";
    }
}

Plan::Plan()
    : comm(MPI_COMM_NULL), rank(0), nproc(1), op(0), tag_base(0),
      wire_schema(GCX_WIRE_SCHEMA), deadline_ms(0),
      inbox(0), inbox_bytes(0), inbox_exported(false),
      host_send(0), host_recv(0), host_span(0), shm_base(0), shm_dev(0),
      transport(0), send_ready(0), recv_ready(0), landing(0), land_ready(0),
      stream_owned(false), stream_shared(false),
      epoch(0), sequence(0), slot(0), outstanding(false), faulted(false),
      phase(PH_IDLE), started(0.0), variable(false), host_edges(true),
      shm_edges(false), seq_epoch(0),
      out_recv_bytes(0), out_recv_records(0),
      ack_event(0), ack_slot(0), ack_owed(false), ack_variable(false),
      relay_word(0), relay_word_dev(0), relay_last(0), relay_op(false)
{
    std::memset(&inbox_handle, 0, sizeof(inbox_handle));
    for (int s = 0; s < GCX_SLOTS; ++s) { ack_pending[s] = false; slot_epoch[s] = 0; }
    std::memset(&report, 0, sizeof(report));
}

void log_plan(const Plan &p)
{
    if (p.rank != 0) return;
    for (int r = 0; r < GCX_ROUTE_NKIND; ++r) {
        if (p.report.edges[r] == 0) continue;
        gc_i64 cap = 0;
        for (size_t e = 0; e < p.edge.size(); ++e)
            if (p.edge[e].route == r) cap += p.edge[e].recv_capacity;
        std::fprintf(stdout,
                     "Native_Xchg> plan=%-15s route=%-10s edges=%lld "
                     "recv_capacity=%lld rdma=%s\n",
                     gcx_op_name(p.op), gcx_route_name(r),
                     (long long)p.report.edges[r], (long long)cap,
                     (r == GCX_ROUTE_MPI_DEVICE) ? gcx_rdma_name(probe().rdma)
                                                 : "n/a");
    }
    std::fflush(stdout);
}

}  /* namespace gcx */

using namespace gcx;
using gcn::add_checked;
using gcn::mul_checked;


namespace {

/* Message tag, disambiguated by the receiving side's edge key. */
int tag_of(const Plan &p, int key, int slot, int kind)
{
    return p.tag_base + (key * GCX_SLOTS + slot) * 3 + kind;
}

char *host_slot(const Plan &p, void *base, size_t e, int slot)
{
    return (char *)base + ((gc_i64)e * GCX_SLOTS + slot) * p.host_span;
}

/* An edge's landing region: its slots rounded to 16 bytes, then two signal
 * words, [0] ready and [1] ack, written by the peer and polled by this
 * rank's device. */
gc_i64 sig_offset(gc_i64 capacity)
{
    return (capacity * GCX_SLOTS + 15) / 16 * 16;
}

unsigned long long *my_sig(const Plan &p, size_t e)
{
    return (unsigned long long *)((char *)p.inbox + p.edge[e].inbox_offset
                                  + sig_offset(p.edge[e].recv_capacity));
}

unsigned long long *peer_sig(const Plan &p, size_t e)
{
    return (unsigned long long *)((char *)p.edge[e].peer_inbox
                                  + sig_offset(p.edge[e].send_capacity));
}

unsigned long long *my_shm_sig(const Plan &p, size_t e)
{
    return (unsigned long long *)(p.shm_dev + p.edge[e].inbox_offset
                                  + sig_offset(p.edge[e].recv_capacity));
}

unsigned long long *peer_shm_sig(const Plan &p, size_t e)
{
    return (unsigned long long *)(p.edge[e].shared_peer_dev
                                  + sig_offset(p.edge[e].send_capacity));
}

__global__ void k_sig_wait(const volatile unsigned long long *f,
                           unsigned long long v)
{
    gcx_sig_wait(f, v);
}

__global__ void k_sig_set(volatile unsigned long long *f, unsigned long long v)
{
    __threadfence_system();
    *f = v;
}

char *inbox_slot(const Plan &p, size_t e, int slot)
{
    return (char *)p.inbox + p.edge[e].inbox_offset
           + (gc_i64)slot * p.edge[e].recv_capacity;
}

/* Slot `slot` of the PEER's landing area.  The stride is this rank's send
 * capacity (plan creation checks it equals the peer's receive capacity),
 * not this rank's receive capacity, which differs on asymmetric edges. */
char *peer_slot(const Plan &p, size_t e, int slot)
{
    return (char *)p.edge[e].peer_inbox
           + (gc_i64)slot * p.edge[e].send_capacity;
}

/* An edge whose fixed-size operation needs no header from the host. */
bool quiet(const Plan &p, size_t e)
{
    return (p.edge[e].devsig || p.edge[e].shmsig || p.edge[e].relay) && !p.variable;
}

/* What a fixed-size edge receives: its capacity, or the receive buffer's
 * bytes when that is smaller. */
gc_i64 fixed_recv_bytes(const Plan &p, size_t e)
{
    gc_i64 n = p.edge[e].recv_capacity;
    if (p.cur_recv[e].ptr != 0 && p.cur_recv[e].bytes < n)
        n = p.cur_recv[e].bytes;
    return n;
}

#ifdef __linux__
bool widen_to_rank_cpus(cpu_set_t *saved);
#endif

/* Widens the calling thread's CPU affinity to the union of the process's
 * threads, saving the old mask in `saved`; false, nothing changed, if it
 * cannot be read or adds no CPU. */
#ifdef __linux__
bool widen_to_rank_cpus(cpu_set_t *saved)
{
    if (sched_getaffinity(0, sizeof(*saved), saved) != 0) return false;
    cpu_set_t all;
    CPU_ZERO(&all);
    DIR *d = opendir("/proc/self/task");
    if (d == 0) return false;
    for (struct dirent *de; (de = readdir(d)) != 0; ) {
        const int tid = std::atoi(de->d_name);
        cpu_set_t m;
        if (tid > 0 && sched_getaffinity(tid, sizeof(m), &m) == 0)
            CPU_OR(&all, &all, &m);
    }
    closedir(d);
    CPU_OR(&all, &all, saved);
    if (CPU_EQUAL(&all, saved)) return false;
    return sched_setaffinity(0, sizeof(all), &all) == 0;
}
#endif

/* ---- The transport thread (THREAD route mode) ----------------------
 * A second thread carries the inter-node edges of the fixed-size mesh and
 * halo operations and makes no CUDA call: the transport stream copies the
 * payload into the edge's pinned host slot and sets a ready word; the thread
 * moves it with MPI (MPI_THREAD_MULTIPLE) and sets a done word; the stream
 * waits for it on the device and copies into the caller's buffer.  Bytes
 * and arithmetic order are unchanged.  Host progress beside a device that
 * waits on signal words follows Pall et al., J. Chem. Phys. 153, 134110
 * (2020). */
struct RelayMsg {
    char  *buf;
    gc_i64 bytes;
    int    peer, tag;
    bool   send;
};

struct RelayJob {
    Plan                 *plan;
    gc_i64                epoch;
    MPI_Comm              comm;
    double                deadline_s;
    std::vector<RelayMsg> msg;
};

struct Relay {
    std::mutex              m;
    std::condition_variable cv;
    std::deque<RelayJob>    q;
    std::thread             th;
    bool                    running, stop;
    volatile int            fault;
    Relay() : running(false), stop(false), fault(0) {}
};

/* Never destroyed: a static destructor at exit would destroy the condition
 * variable under a waiting thread. */
Relay &relay()
{
    static Relay *r = new Relay;
    return *r;
}

/* Every handed-over operation is in flight at once: handling them one at a
 * time could deadlock when ranks hand them over in different orders. */
struct RelayFlight {
    RelayJob                 job;
    std::vector<MPI_Request> rq;
    bool                     posted;
    double                   t0;
};

void relay_post(RelayFlight &f)
{
    for (int pass = 0; pass < 2; ++pass)               /* receives first */
        for (size_t k = 0; k < f.job.msg.size(); ++k) {
            const RelayMsg &g = f.job.msg[k];
            if (g.send != (pass == 1)) continue;
            for (gc_i64 o = 0; o < g.bytes; o += GCX_MPI_CHUNK_BYTES) {
                const gc_i64 n = std::min(g.bytes - o, GCX_MPI_CHUNK_BYTES);
                MPI_Request x;
                if (g.send) MPI_Isend(g.buf + o, (int)n, MPI_BYTE, g.peer, g.tag, f.job.comm, &x);
                else        MPI_Irecv(g.buf + o, (int)n, MPI_BYTE, g.peer, g.tag, f.job.comm, &x);
                f.rq.push_back(x);
            }
        }
    f.posted = true;
}

void relay_main()
{
    Relay &r = relay();
    std::vector<RelayFlight> fl;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(r.m);
            if (fl.empty())
                r.cv.wait(lk, [&] { return r.stop || !r.q.empty(); });
            while (!r.q.empty()) {
                RelayFlight f;
                f.job = r.q.front();
                f.posted = false;
                f.t0 = MPI_Wtime();
                fl.push_back(f);
                r.q.pop_front();
            }
            if (fl.empty() && r.stop) return;
        }
        for (size_t k = 0; k < fl.size(); ) {
            RelayFlight &f = fl[k];
            bool ok = true, done = false;
            if (!f.posted &&
                __atomic_load_n(f.job.plan->relay_word + 1, __ATOMIC_ACQUIRE) >=
                    (unsigned long long)f.job.epoch)
                relay_post(f);
            if (ok && f.posted) {
                int all = 1;
                if (!f.rq.empty())
                    MPI_Testall((int)f.rq.size(), &f.rq[0], &all, MPI_STATUSES_IGNORE);
                done = all != 0;
            }
            if (ok && !done && MPI_Wtime() - f.t0 > f.job.deadline_s) ok = false;
            if (!ok) {
                /* never written: the device wait traps at its deadline */
                r.fault = 1;
                std::fprintf(stderr, "Native_Xchg> transport thread: plan=%s epoch=%lld "
                             "did not complete\n", gcx_op_name(f.job.plan->op),
                             (long long)f.job.epoch);
            } else if (done) {
                __atomic_store_n(f.job.plan->relay_word, (unsigned long long)f.job.epoch,
                                 __ATOMIC_RELEASE);
            }
            if (!ok || done) fl.erase(fl.begin() + k);
            else ++k;
        }
        if (!fl.empty()) std::this_thread::yield();
    }
}

/* Stops and joins the thread once its operations complete or expire. */
void relay_stop()
{
    Relay &r = relay();
    if (!r.running) return;
    {
        std::lock_guard<std::mutex> lk(r.m);
        r.stop = true;
    }
    r.cv.notify_all();
    r.th.join();
    r.running = false;
}

/* Runs at MPI_Finalize (MPI_COMM_SELF attribute deletion), while MPI is
 * still usable. */
int relay_at_finalize(MPI_Comm, int, void *, void *)
{
    relay_stop();
    return MPI_SUCCESS;
}

/* Starts the thread once per process, on all of this rank's CPUs. */
void relay_start()
{
    Relay &r = relay();
    if (r.running) return;
#ifdef __linux__
    cpu_set_t bound;
    const bool widened = widen_to_rank_cpus(&bound);
#endif
    r.th = std::thread(relay_main);
#ifdef __linux__
    if (widened) sched_setaffinity(0, sizeof(bound), &bound);
#endif
    r.running = true;
    int key = MPI_KEYVAL_INVALID;
    MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, relay_at_finalize, &key, 0);
    MPI_Comm_set_attr(MPI_COMM_SELF, key, 0);
}

/* Waits (bounded) until the thread has completed every operation of `p`. */
bool relay_wait(const Plan &p)
{
    if (p.relay_word == 0) return true;
    const double t0 = MPI_Wtime();
    while (__atomic_load_n(p.relay_word, __ATOMIC_ACQUIRE) < (unsigned long long)p.relay_last) {
        if (relay().fault || (MPI_Wtime() - t0) * 1000.0 > (double)p.deadline_ms) return false;
        std::this_thread::yield();
    }
    return true;
}

/* Post this rank's acknowledgement of a slot, once its consumer is done. */
void flush_ack(Plan &p)
{
    if (!p.ack_owed) return;
    if (p.ack_event && cudaEventQuery(p.ack_event) == cudaErrorNotReady)
        return;
    int slot = p.ack_slot;
    p.ack_word[slot].assign(p.edge.size(), 0);
    p.ack_send[slot].clear();
    for (size_t i = 0; i < p.edge.size(); ++i) {
        if (p.edge[i].peer == p.rank || p.edge[i].devsig || p.edge[i].shmsig ||
            (p.edge[i].relay && !p.ack_variable)) continue;
        p.ack_word[slot][i] = p.epoch;
        MPI_Request q;
        MPI_Isend(&p.ack_word[slot][i], 1, MPI_LONG_LONG, p.edge[i].peer,
                  tag_of(p, p.edge[i].partner, slot, 2), p.comm, &q);
        p.ack_send[slot].push_back(q);
    }
    p.ack_pending[slot] = true;
    p.ack_owed = false;
}

/* Let one request set finish, cancelling what is unmatched at the deadline
 * (a refused operation leaves messages no peer will match). */
void drain(std::vector<MPI_Request> &r, double deadline_s)
{
    double t0 = MPI_Wtime();
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i] == MPI_REQUEST_NULL) continue;
        int done = 0;
        for (;;) {
            MPI_Test(&r[i], &done, MPI_STATUS_IGNORE);
            if (done) break;
            if (MPI_Wtime() - t0 > deadline_s) break;
        }
        if (!done) {
            MPI_Cancel(&r[i]);
            MPI_Wait(&r[i], MPI_STATUS_IGNORE);
        }
    }
    r.clear();
}

/* Bring a plan to rest so it can be destroyed: publish any owed
 * acknowledgement, then finish or cancel every request it still owns. */
void quiesce(Plan &p)
{
    if (p.ack_owed) {
        if (p.ack_event) cudaEventSynchronize(p.ack_event);
        p.ack_event = 0;
        flush_ack(p);
    }
    drain(p.req, 0.5);
    drain(p.preq, 0.5);
    for (int s = 0; s < GCX_SLOTS; ++s) {
        drain(p.ack_send[s], 0.5);
        drain(p.ack_recv[s], 0.5);
        p.ack_pending[s] = false;
    }
    p.outstanding = false;
    p.phase = PH_IDLE;
}

}  /* anonymous namespace */

extern "C" const char *gcx_route_name(gc_i32 route)
{
    switch (route) {
    case GCX_ROUTE_SELF:       return "self";
    case GCX_ROUTE_LOCAL:      return "local";
    case GCX_ROUTE_PEER:       return "peer";
    case GCX_ROUTE_MPI_DEVICE: return "mpi_device";
    case GCX_ROUTE_STAGED:     return "staged";
    default:                   return "invalid";
    }
}

extern "C" const char *gcx_rdma_name(gc_i32 rdma)
{
    switch (rdma) {
    case GCX_RDMA_STAGED:   return "staged";
    case GCX_RDMA_UNKNOWN:  return "unknown";
    case GCX_RDMA_VERIFIED: return "verified";
    default:                return "invalid";
    }
}

extern "C" const char *gcx_op_name(gc_i32 op)
{
    switch (op) {
    case GCX_OP_HALO_COORD:      return "halo_coord";
    case GCX_OP_HALO_FORCE:      return "halo_force";
    case GCX_OP_MIGRATE_PAYLOAD: return "migrate_payload";
    case GCX_OP_MESH:            return "mesh";
    default:                     return "invalid";
    }
}

#define GCX_PROBE_SMALL ((gc_i64)16 * 1024)
#define GCX_PROBE_LARGE ((gc_i64)4 * 1024 * 1024)
#define GCX_PROBE_ITERS 7

namespace {

struct ProbeArena {
    void *inbox;          /* exported: every node peer pushes in here      */
    void *src;            /* this rank's source payload                    */
    void *host;           /* pinned, for the staged reference              */
    std::vector<void *>             mapped;   /* peer inbox, per node rank */
    std::vector<cudaIpcMemHandle_t> handle;
    ProbeArena() : inbox(0), src(0), host(0) {}
};

ProbeArena &arena()
{
    static ProbeArena a;
    return a;
}

/* Probe peer reading: this rank's payload stored into every linked peer's
 * inbox (blockIdx.y selects the peer). */
__global__ void k_probe_push(void *const *dst, const uint4 *__restrict__ src, long long n)
{
    uint4 *d = (uint4 *)dst[blockIdx.y];
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        d[i] = src[i];
}

/* Probe stand-in for step compute: many short blocks over every SM. */
__global__ void k_probe_load(long long cycles)
{
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

double median(std::vector<double> &v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double gbs(gc_i64 bytes, double seconds)
{
    return seconds > 0.0 ? 1.0e-9 * (double)bytes / seconds : 0.0;
}

/* Does this MPI accept device pointers?  Asked before any trial, since an
 * unsupported trial may abort rather than return an error. */
bool mpi_claims_cuda(void)
{
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() == 1;
#else
    return false;
#endif
}

}  /* anonymous namespace */

extern "C" void gcx_peer_fanout(gc_i32 *peers, gc_i32 *native_atomic)
{
    const Probe &pr = probe();
    int n = 0, at = 0;
    for (size_t r = 0; r < pr.peer_used.size(); ++r)
        if (pr.peer_used[r]) { ++n; at |= pr.peer_atomic[r]; }
    *peers = n;
    *native_atomic = at;
}

extern "C" gc_status gcx_probe_run(gcx_comm comm_f, gcx_probe_report *out)
{
    if (out == 0) return GC_E_ARG;

    Probe &pr = probe();
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)comm_f);
    if (comm == MPI_COMM_NULL) return GC_E_ARG;

    double t0 = MPI_Wtime();

    MPI_Comm_rank(comm, &pr.rank);
    MPI_Comm_size(comm, &pr.nproc);
    pr.comm = comm;

    /* Node communicator (MPI_COMM_TYPE_SHARED), keyed by comm rank so that shm
     * rank 0 is the node's lowest rank. */
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, pr.rank, MPI_INFO_NULL,
                        &pr.node);
    MPI_Comm_rank(pr.node, &pr.node_rank);
    MPI_Comm_size(pr.node, &pr.node_size);

    int key = pr.rank;
    MPI_Bcast(&key, 1, MPI_INT, 0, pr.node);
    pr.node_key = key;
    pr.peer_node.assign(pr.nproc, 0);
    MPI_Allgather(&pr.node_key, 1, MPI_INT, &pr.peer_node[0], 1, MPI_INT, comm);
    {
        std::vector<int> k(pr.peer_node);
        std::sort(k.begin(), k.end());
        k.erase(std::unique(k.begin(), k.end()), k.end());
        pr.nodes = (int)k.size();
    }
    pr.same_node.assign(pr.nproc, 0);
    for (int r = 0; r < pr.nproc; ++r)
        pr.same_node[r] = (pr.peer_node[r] == pr.node_key) ? 1 : 0;

    /* Device identity is the PCI bus id, never a CUDA ordinal; bus ids are
     * gathered on the node communicator only. */
    if (cudaGetDevice(&pr.device) != cudaSuccess) return GC_E_DEVICE;
    char id[GCX_DEVICE_ID_LEN];
    std::memset(id, 0, sizeof(id));
    if (cudaDeviceGetPCIBusId(id, GCX_DEVICE_ID_LEN - 1, pr.device)
        != cudaSuccess)
        return GC_E_DEVICE;
    pr.device_id = id;

    std::vector<char> node_id((size_t)pr.node_size * GCX_DEVICE_ID_LEN, 0);
    MPI_Allgather(id, GCX_DEVICE_ID_LEN, MPI_CHAR,
                  &node_id[0], GCX_DEVICE_ID_LEN, MPI_CHAR, pr.node);
    std::vector<int> node_of_rank(pr.node_size, 0);
    MPI_Allgather(&pr.rank, 1, MPI_INT, &node_of_rank[0], 1, MPI_INT, pr.node);

    pr.same_device.assign(pr.nproc, 0);
    pr.peer_capable.assign(pr.nproc, 0);
    pr.peer_admitted.assign(pr.nproc, 0);
    pr.peer_ordinal.assign(pr.nproc, -1);
    pr.peer_atomic.assign(pr.nproc, 0);
    pr.peer_used.assign(pr.nproc, 0);

    int visible = 0;
    cudaGetDeviceCount(&visible);
    for (int nr = 0; nr < pr.node_size; ++nr) {
        int r = node_of_rank[nr];
        const char *pid = &node_id[(size_t)nr * GCX_DEVICE_ID_LEN];
        if (r == pr.rank) { pr.peer_ordinal[r] = pr.device; continue; }
        for (int d = 0; d < visible; ++d) {
            char cand[GCX_DEVICE_ID_LEN];
            std::memset(cand, 0, sizeof(cand));
            if (cudaDeviceGetPCIBusId(cand, GCX_DEVICE_ID_LEN - 1, d)
                == cudaSuccess && std::strcmp(cand, pid) == 0) {
                pr.peer_ordinal[r] = d;
                break;
            }
        }
        if (std::strcmp(pid, id) == 0) {
            pr.same_device[r] = 1;          /* one GPU, two ranks          */
        } else if (pr.peer_ordinal[r] >= 0) {
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, pr.device, pr.peer_ordinal[r])
                == cudaSuccess && can) {
                pr.peer_capable[r] = 1;
                int at = 0;
                if (cudaDeviceGetP2PAttribute(&at, cudaDevP2PAttrNativeAtomicSupported,
                                              pr.device, pr.peer_ordinal[r])
                    == cudaSuccess && at)
                    pr.peer_atomic[r] = 1;
                cudaGetLastError();
            }
        }
    }

    /* Staged reference: device -> pinned host -> device; node-local routes must
     * beat it by the margin. */
    ProbeArena &a = arena();
    if (cudaMalloc(&a.src, (size_t)GCX_PROBE_LARGE) != cudaSuccess)
        return GC_E_NOMEM;
    if (cudaMalloc(&a.inbox, (size_t)GCX_PROBE_LARGE) != cudaSuccess)
        return GC_E_NOMEM;
    if (cudaHostAlloc(&a.host, (size_t)GCX_PROBE_LARGE, cudaHostAllocDefault)
        != cudaSuccess)
        return GC_E_NOMEM;
    cudaMemset(a.src, 0x5a, (size_t)GCX_PROBE_LARGE);

    /* Measured at both probed sizes; a peer reading is compared with the staged
     * reading of the same size (latency versus bandwidth bound). */
    const gc_i64 sizes[2] = { GCX_PROBE_SMALL, GCX_PROBE_LARGE };
    double staged_bw[2] = { 0.0, 0.0 };
    for (int si = 0; si < 2; ++si) {
        cudaMemcpy(a.host, a.src, (size_t)sizes[si], cudaMemcpyDeviceToHost);
        cudaMemcpy(a.inbox, a.host, (size_t)sizes[si], cudaMemcpyHostToDevice);
        std::vector<double> t;
        for (int it = 0; it < GCX_PROBE_ITERS; ++it) {
            double s = MPI_Wtime();
            cudaMemcpy(a.host, a.src, (size_t)sizes[si],
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(a.inbox, a.host, (size_t)sizes[si],
                       cudaMemcpyHostToDevice);
            t.push_back(MPI_Wtime() - s);
        }
        staged_bw[si] = gbs(sizes[si], median(t));
    }
    pr.bw_staged_small = staged_bw[0];
    pr.bw_staged       = staged_bw[1];

    /* Peer links: the peer's memory is mapped over the node communicator and
     * released by gcx_probe_release. */
    a.handle.assign((size_t)pr.node_size, cudaIpcMemHandle_t());
    a.mapped.assign((size_t)pr.node_size, (void *)0);
    cudaIpcMemHandle_t mine;
    std::memset(&mine, 0, sizeof(mine));
    int export_ok = (cudaIpcGetMemHandle(&mine, a.inbox) == cudaSuccess);
    MPI_Allgather(&mine, sizeof(mine), MPI_BYTE,
                  &a.handle[0], sizeof(mine), MPI_BYTE, pr.node);
    std::vector<int> exported(pr.node_size, 0);
    MPI_Allgather(&export_ok, 1, MPI_INT, &exported[0], 1, MPI_INT, pr.node);

    for (int nr = 0; nr < pr.node_size; ++nr) {
        int r = node_of_rank[nr];
        if (r == pr.rank || !exported[nr]) continue;
        if (!pr.peer_capable[r] && !pr.same_device[r]) continue;
        void *p = 0;
        if (cudaIpcOpenMemHandle(&p, a.handle[nr],
                                 cudaIpcMemLazyEnablePeerAccess)
            == cudaSuccess)
            a.mapped[nr] = p;
    }

    /* Reported peer access is not proof that it works (e.g. IOMMU translation):
     * a mapping is used only after a read through it returns the owner's
     * per-rank byte, required of both ends of the pair; otherwise it is staged. */
    {
        cudaMemset(a.inbox, 0xa5 ^ pr.node_rank, (size_t)GCX_PROBE_SMALL);
        cudaDeviceSynchronize();
        MPI_Barrier(pr.node);
        std::vector<int> good((size_t)pr.node_size, 0);
        const unsigned char *h = (const unsigned char *)a.host;
        for (int nr = 0; nr < pr.node_size; ++nr) {
            if (a.mapped[nr] == 0 ||
                cudaMemcpy(a.host, a.mapped[nr], (size_t)GCX_PROBE_SMALL,
                           cudaMemcpyDefault) != cudaSuccess)
                continue;
            const unsigned char want = (unsigned char)(0xa5 ^ nr);
            gc_i64 k = 0;
            while (k < GCX_PROBE_SMALL && h[k] == want) ++k;
            good[(size_t)nr] = k == GCX_PROBE_SMALL;
        }
        cudaGetLastError();
        std::vector<int> all((size_t)pr.node_size * pr.node_size, 0);
        MPI_Allgather(&good[0], pr.node_size, MPI_INT, &all[0], pr.node_size,
                      MPI_INT, pr.node);
        for (int nr = 0; nr < pr.node_size; ++nr) {
            if (a.mapped[nr] == 0 ||
                (all[(size_t)pr.node_rank * pr.node_size + nr] &&
                 all[(size_t)nr * pr.node_size + pr.node_rank]))
                continue;
            cudaIpcCloseMemHandle(a.mapped[nr]);
            a.mapped[nr] = 0;
            pr.peer_capable[node_of_rank[nr]] = 0;
        }
    }

    /* Both routes are measured as a step drives them (all ranks moving to all
     * node peers at once: SM-store copies against pinned-memory copies) so plans
     * decide per edge (peer_path). */
    std::vector<void *> links;
    int capable = 0;
    for (int nr = 0; nr < pr.node_size; ++nr) {
        const int r = node_of_rank[nr];
        if (a.mapped[nr] == 0) continue;
        pr.peer_admitted[r] = 1;
        if (pr.same_device[r]) continue;   /* one GPU: nothing to time */
        ++capable;
        links.push_back(a.mapped[nr]);
    }
    int any_links = capable > 0, node_links = 0;
    MPI_Allreduce(&any_links, &node_links, 1, MPI_INT, MPI_MAX, pr.node);
    pr.t_peer[0] = pr.t_peer[1] = 0.0;
    pr.t_staged_fan[0] = pr.t_staged_fan[1] = 0.0;
    pr.t_peer_load[0] = pr.t_peer_load[1] = 0.0;
    pr.t_staged_load[0] = pr.t_staged_load[1] = 0.0;
    pr.t_ce[0] = pr.t_ce[1] = pr.t_ce_load[0] = pr.t_ce_load[1] = 0.0;
    pr.bw_peer_worst = 0.0;
    pr.peer_bw_small.assign(pr.nproc, 0.0);
    pr.peer_bw_large.assign(pr.nproc, 0.0);
    if (node_links) {
        void **d_links = 0;
        bool ok = links.empty() ||
            (cudaMalloc((void **)&d_links, links.size() * sizeof(void *)) == cudaSuccess &&
             cudaMemcpy(d_links, &links[0], links.size() * sizeof(void *),
                        cudaMemcpyHostToDevice) == cudaSuccess);
        const int k = (int)links.size();
        cudaStream_t ls = 0, cs = 0;
        ok = ok && cudaStreamCreateWithFlags(&ls, cudaStreamNonBlocking) == cudaSuccess &&
             cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) == cudaSuccess;
        int nsm = 1, tpm = 2048, khz = 1000000;
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, pr.device);
        cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, pr.device);
        cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, pr.device);
        /* one move of each route, beside `load_us` of stand-in compute (0 = none) */
        auto move = [&](int route, int si, double load_us) {
            const long long blk_cycles = 2LL * khz / 1000;         /* ~2 us */
            const long long nblk = load_us > 0.0
                ? (long long)(load_us / 2.0) * nsm * (tpm / 256) : 0;
            if (nblk > 0)
                k_probe_load<<<(unsigned)std::min(nblk, 1LL << 30), 256, 0, ls>>>(blk_cycles);
            if (route == 0) {
                k_probe_push<<<dim3(64, (unsigned)k), 256, 0, cs>>>(
                    d_links, (const uint4 *)a.src, (long long)(sizes[si] / 16));
            } else if (route == 2) {
                for (int j = 0; j < k && ok; ++j)
                    ok = cudaMemcpyAsync(links[(size_t)j], a.src, (size_t)sizes[si],
                                         cudaMemcpyDefault, cs) == cudaSuccess;
            } else {
                for (int j = 0; j < k && ok; ++j)
                    ok = cudaMemcpyAsync(a.host, a.src, (size_t)sizes[si],
                                         cudaMemcpyDeviceToHost, cs) == cudaSuccess;
                ok = ok && cudaStreamSynchronize(cs) == cudaSuccess;
                for (int j = 0; j < k && ok; ++j)
                    ok = cudaMemcpyAsync(a.inbox, a.host, (size_t)sizes[si],
                                         cudaMemcpyHostToDevice, cs) == cudaSuccess;
            }
            ok = ok && cudaGetLastError() == cudaSuccess &&
                 cudaDeviceSynchronize() == cudaSuccess;
        };
        /* idle readings first (they size the load), then the loaded ones */
        for (int pass = 0; pass < 2; ++pass)
            for (int si = 0; si < 2; ++si)
                for (int route = 0; route < 3; ++route) {
                    const double load_us = pass ? 1e6 * pr.t_staged_fan[si] : 0.0;
                    std::vector<double> t;
                    /* one untimed pass: first touch of a lazily enabled mapping */
                    for (int it = 0; it <= GCX_PROBE_ITERS; ++it) {
                        MPI_Barrier(pr.node);
                        const double s = MPI_Wtime();
                        if (ok && k > 0) move(route, si, load_us);
                        if (it > 0) t.push_back(MPI_Wtime() - s);
                    }
                    double *out = pass ? (route == 0 ? pr.t_peer_load : route == 1 ? pr.t_staged_load : pr.t_ce_load)
                                       : (route == 0 ? pr.t_peer : route == 1 ? pr.t_staged_fan : pr.t_ce);
                    out[si] = median(t);
                }
        /* One reading per node, the slowest rank's, so both ends of an edge agree
         * on fusing (gcx_plan_device_edge). */
        double *reads[] = {pr.t_peer, pr.t_staged_fan, pr.t_peer_load,
                           pr.t_staged_load, pr.t_ce, pr.t_ce_load};
        for (double *r : reads)
            MPI_Allreduce(MPI_IN_PLACE, r, 2, MPI_DOUBLE, MPI_MAX, pr.node);
        if (ls) cudaStreamDestroy(ls);
        if (cs) cudaStreamDestroy(cs);
        if (d_links) cudaFree(d_links);
        if (pr.rank == 0 && k > 0)
            std::fprintf(stdout, "Native_Xchg> peer probe (%d links at once): "
                         "peer %.1f / %.1f us, staged %.1f / %.1f us at %lld / %lld bytes; "
                         "beside compute peer %.1f / %.1f us, staged %.1f / %.1f us; "
                         "peer copy-engine %.1f / %.1f us, beside compute %.1f / %.1f us%s\n",
                         k, 1e6 * pr.t_peer[0], 1e6 * pr.t_peer[1],
                         1e6 * pr.t_staged_fan[0], 1e6 * pr.t_staged_fan[1],
                         (long long)sizes[0], (long long)sizes[1],
                         1e6 * pr.t_peer_load[0], 1e6 * pr.t_peer_load[1],
                         1e6 * pr.t_staged_load[0], 1e6 * pr.t_staged_load[1],
                         1e6 * pr.t_ce[0], 1e6 * pr.t_ce[1],
                         1e6 * pr.t_ce_load[0], 1e6 * pr.t_ce_load[1],
                         ok ? "" : " (a copy failed)");
        /* a failed copy is not a fast copy: its links are staged */
        if (!ok) {
            cudaGetLastError();
            for (int r = 0; r < pr.nproc; ++r)
                if (!pr.same_device[r]) pr.peer_admitted[r] = 0;
            capable = 0;
        }
        for (int nr = 0; nr < pr.node_size; ++nr) {
            const int r = node_of_rank[nr];
            if (!pr.peer_admitted[r] || pr.same_device[r]) continue;
            pr.peer_bw_small[r] = gbs(sizes[0], pr.t_peer[0]);
            pr.peer_bw_large[r] = gbs(sizes[1], pr.t_peer[1]);
            pr.bw_peer_worst = pr.peer_bw_large[r];
        }
    }
    int admitted = 0;
    for (int r = 0; r < pr.nproc; ++r)
        if (pr.peer_admitted[r] && !pr.same_device[r] && r != pr.rank) ++admitted;
    MPI_Barrier(pr.node);

    /* Device-buffer MPI: capability first, then a one-element ring round trip
     * on a duplicate communicator whose error handler records instead of
     * aborting.  Skipped on one node, where no edge can take this route. */
    int dev_ok = 0;
    if (pr.nodes > 1 && mpi_claims_cuda()) {
        MPI_Comm trial;
        MPI_Comm_dup(comm, &trial);
        MPI_Comm_set_errhandler(trial, MPI_ERRORS_RETURN);
        void *d = 0, *rbuf = 0;
        if (cudaMalloc(&d, sizeof(gc_i64)) == cudaSuccess &&
            cudaMalloc(&rbuf, sizeof(gc_i64)) == cudaSuccess) {
            gc_i64 token = (gc_i64)pr.rank + 1, got = 0;
            cudaMemcpy(d, &token, sizeof(token), cudaMemcpyHostToDevice);
            /* The receive buffer is read back only after a successful trial (a refused
             * one never writes it); zero is never a valid token. */
            cudaMemset(rbuf, 0, sizeof(gc_i64));
            cudaDeviceSynchronize();
            int up = (pr.rank + 1) % pr.nproc;
            int dn = (pr.rank + pr.nproc - 1) % pr.nproc;
            int e = MPI_Sendrecv(d, sizeof(gc_i64), MPI_BYTE, up, 0,
                                 rbuf, sizeof(gc_i64), MPI_BYTE, dn, 0,
                                 trial, MPI_STATUS_IGNORE);
            if (e == MPI_SUCCESS &&
                cudaMemcpy(&got, rbuf, sizeof(got), cudaMemcpyDeviceToHost)
                    == cudaSuccess)
                dev_ok = (got == (gc_i64)dn + 1) ? 1 : 0;
        }
        if (d)    cudaFree(d);
        if (rbuf) cudaFree(rbuf);
        MPI_Comm_free(&trial);
    }
    int all_ok = 0;
    MPI_Allreduce(&dev_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    pr.mpi_device_capable = (all_ok != 0);
    {
        int mine = 0, any = 0;
        for (int r = 0; r < pr.nproc; ++r)
            if (r != pr.rank && pr.same_device[r]) mine = 1;
        MPI_Allreduce(&mine, &any, 1, MPI_INT, MPI_MAX, comm);
        pr.shared_gpu = any != 0;
    }
    if (pr.rank == 0) {
        static const char *const cls[GCX_ROUTE_NCLASS] = { "mesh", "coord", "force" };
        std::fprintf(stdout, "Native_Xchg> route request:");
        for (int k = 0; k < GCX_ROUTE_NCLASS; ++k)
            std::fprintf(stdout, " %s=%s", cls[k], route_mode_name(pr.route_mode[k]));
        std::fprintf(stdout, "\n");
        std::fflush(stdout);
    }
    /* Transport thread: THREAD on some class, more than one node, every rank
     * MPI_THREAD_MULTIPLE-capable and no shared GPU (as devsig).  One vote. */
    bool want_relay = false;
    for (int k = 0; k < GCX_ROUTE_NCLASS; ++k)
        want_relay = want_relay || pr.route_mode[k] == GCX_ROUTE_MODE_THREAD;
    want_relay = want_relay && pr.nodes > 1;
    if (want_relay && !pr.relay_ok) {
        int level = MPI_THREAD_SINGLE;
        MPI_Query_thread(&level);
        int mine = (level >= MPI_THREAD_MULTIPLE && !pr.shared_gpu) ? 1 : 0, all = 0;
        MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MIN, comm);
        pr.relay_ok = all != 0;
        if (pr.relay_ok) relay_start();
        if (pr.rank == 0) {
            std::fprintf(stdout, "Native_Xchg> transport thread: %s\n",
                         pr.relay_ok ? "yes (inter-node edges of the mesh and the halo)"
                         : level < MPI_THREAD_MULTIPLE ? "no (MPI_THREAD_MULTIPLE not provided), MPI routes kept"
                         : "no (ranks share a GPU), MPI routes kept");
            std::fflush(stdout);
        }
    }
    /* CUDA-aware is not proof of GPUDirect RDMA: never written as verified. */
    pr.rdma = pr.mpi_device_capable ? GCX_RDMA_UNKNOWN : GCX_RDMA_STAGED;
    /* A node-local peer whose GPU is not visible has no device route (host
     * staged); said once. */
    int hidden = 0, any_hidden = 0;
    for (int r = 0; r < pr.nproc; ++r)
        if (r != pr.rank && pr.same_node[r] && !pr.same_device[r] &&
            pr.peer_ordinal[r] < 0) hidden = 1;
    MPI_Allreduce(&hidden, &any_hidden, 1, MPI_INT, MPI_MAX, comm);
    if (pr.rank == 0 && any_hidden)
        std::fprintf(stdout, "Native_Xchg> node-local GPUs are hidden from "
                     "each other (per-rank CUDA_VISIBLE_DEVICES): those moves "
                     "are host-staged; launch with every node GPU visible\n");
    if (pr.rank == 0) {
        std::fprintf(stdout, "Native_Xchg> MPI device buffers: %s\n",
                     pr.nodes == 1 ? "not used (one node)"
                     : pr.mpi_device_capable ? "yes (inter-node moves use CUDA-aware MPI)"
                                             : "no (inter-node moves are host-staged)");
        std::fflush(stdout);
    }

    pr.seconds = MPI_Wtime() - t0;
    pr.done = true;

    out->rank               = pr.rank;
    out->nproc              = pr.nproc;
    out->node_key           = pr.node_key;
    out->node_ranks         = pr.node_size;
    out->nodes              = pr.nodes;
    out->device_ordinal     = pr.device;
    out->peer_capable       = capable;
    out->peer_admitted      = admitted;
    out->mpi_device_capable = pr.mpi_device_capable ? 1 : 0;
    out->rdma               = pr.rdma;
    out->bw_staged          = pr.bw_staged;
    out->bw_peer_worst      = pr.bw_peer_worst;
    out->margin             = GCX_PEER_MARGIN;
    out->seconds            = pr.seconds;
    std::memset(out->device_id, 0, GCX_DEVICE_ID_LEN);
    std::strncpy(out->device_id, pr.device_id.c_str(), GCX_DEVICE_ID_LEN - 1);
    return GC_OK;
}

/* Called once, before anything is allocated on the device, when the run
 * asks for the native core.  When every node-local rank owns one of the
 * node's GPUs (ranks per node == visible GPUs <= 8), the node's first rank
 * times a copy between every ordered GPU pair and the node picks the rank ->
 * GPU permutation minimising sum(face weight / bandwidth) over neighbouring
 * domains (x fastest, periodic; weight = face area), taken only if it beats
 * the default by 5%.  Otherwise the default device stays. */
namespace {

double pair_cost(const std::vector<int> &perm, const std::vector<int> &grank,
                 const int nd[3], const double w[3], const std::vector<double> &bw,
                 int ndev)
{
    const int n = (int)perm.size();
    double c = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            int a[3], b[3];
            int gi = grank[i], gj = grank[j];
            a[0] = gi % nd[0]; a[1] = (gi / nd[0]) % nd[1]; a[2] = gi / (nd[0] * nd[1]);
            b[0] = gj % nd[0]; b[1] = (gj / nd[0]) % nd[1]; b[2] = gj / (nd[0] * nd[1]);
            double wt = 0.0;
            int diff = 0, ax = -1;
            for (int k = 0; k < 3; ++k) if (a[k] != b[k]) { ++diff; ax = k; }
            if (diff != 1 || nd[ax] < 2) continue;
            const int dd = (a[ax] - b[ax] + nd[ax]) % nd[ax];
            if (dd == 1 || dd == nd[ax] - 1) wt = nd[ax] == 2 ? 2.0 * w[ax] : w[ax];
            if (wt == 0.0) continue;
            const double f = bw[(size_t)perm[i] * ndev + perm[j]];
            const double r = bw[(size_t)perm[j] * ndev + perm[i]];
            c += wt * (1.0 / (f > 0 ? f : 1e-9) + 1.0 / (r > 0 ? r : 1e-9));
        }
    return c;
}

/* The slowest one-way rate (GB/s) between neighbouring domains' GPUs under
 * perm; 0 when none was measured. */
double link_floor(const std::vector<int> &perm, const std::vector<int> &grank,
                  const int nd[3], const std::vector<double> &bw, int ndev)
{
    const int n = (int)perm.size();
    double lo = HUGE_VAL;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            int a[3], b[3], diff = 0;
            const int gi = grank[i], gj = grank[j];
            a[0] = gi % nd[0]; a[1] = (gi / nd[0]) % nd[1]; a[2] = gi / (nd[0] * nd[1]);
            b[0] = gj % nd[0]; b[1] = (gj / nd[0]) % nd[1]; b[2] = gj / (nd[0] * nd[1]);
            for (int k = 0; k < 3; ++k) if (a[k] != b[k]) ++diff;
            if (diff != 1) continue;
            const double f = bw[(size_t)perm[i] * ndev + perm[j]];
            if (f > 0 && f < lo) lo = f;
        }
    return lo < HUGE_VAL ? lo : 0.0;
}

/* The device's throughput, SMs x clock (GHz), the smallest over comm. */
double device_units(MPI_Comm comm)
{
    int dev = 0, sm = 0, khz = 0;
    double u = 0.0, all = 0.0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) == cudaSuccess)
        u = (double)sm * (double)khz * 1e-6;
    cudaGetLastError();
    MPI_Allreduce(&u, &all, 1, MPI_DOUBLE, MPI_MIN, comm);
    return all;
}

/* gcx_place_device's sweeps over the node's links, and inter_node_rate's
 * timed moves */
#define GCX_PLACE_SWEEPS 3
#define GCX_NODE_MOVES   5

/* The one-way rate (GB/s) a rank sees to its counterpart on the next node,
 * with every rank moving at once.  Device buffers when MPI claims CUDA
 * support and a trial succeeds, else host-staged moves.  The fastest of the
 * moves' slowest-rank times sets the rate, the same on every rank.  HUGE_VAL
 * on one node or with unequal ranks per node. */
double inter_node_rate(MPI_Comm comm, MPI_Comm node, int *device_buffers)
{
    int rank = 0, nr = 0, ns = 1, lead = 0, nodes = 0, nmin = 0, nmax = 0;
    *device_buffers = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_rank(node, &nr);
    MPI_Comm_size(node, &ns);
    lead = nr == 0 ? 1 : 0;
    MPI_Allreduce(&lead, &nodes, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&ns, &nmin, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&ns, &nmax, 1, MPI_INT, MPI_MAX, comm);
    if (nodes < 2 || nmin != nmax) return HUGE_VAL;

    MPI_Comm col;
    MPI_Comm_split(comm, nr, rank, &col);
    MPI_Comm_set_errhandler(col, MPI_ERRORS_RETURN);
    int ci = 0;
    MPI_Comm_rank(col, &ci);
    const int up = (ci + 1) % nodes, dn = (ci + nodes - 1) % nodes;
    const size_t bytes = (size_t)4 << 20;
    void *ds = 0, *dr = 0, *hs = 0, *hr = 0;
    int dev = mpi_claims_cuda() && cudaMalloc(&ds, bytes) == cudaSuccess &&
              cudaMalloc(&dr, bytes) == cudaSuccess ? 1 : 0;
    if (dev) {
        cudaMemset(ds, 0, bytes);
        cudaDeviceSynchronize();
        dev = MPI_Sendrecv(ds, (int)bytes, MPI_BYTE, up, 0, dr, (int)bytes,
                           MPI_BYTE, dn, 0, col, MPI_STATUS_IGNORE)
              == MPI_SUCCESS ? 1 : 0;
    }
    int alldev = 0;
    MPI_Allreduce(&dev, &alldev, 1, MPI_INT, MPI_MIN, comm);
    if (!alldev) {
        if (!ds && cudaMalloc(&ds, bytes) != cudaSuccess) ds = 0;
        if (!dr && cudaMalloc(&dr, bytes) != cudaSuccess) dr = 0;
        if (cudaMallocHost(&hs, bytes) != cudaSuccess) hs = 0;
        if (cudaMallocHost(&hr, bytes) != cudaSuccess) hr = 0;
    }
    int ok = alldev || (ds && dr && hs && hr) ? 1 : 0, allok = 0;
    MPI_Allreduce(&ok, &allok, 1, MPI_INT, MPI_MIN, comm);
    /* Each move is timed between barriers, slowest rank kept; the fastest
     * move sets the rate. */
    double tit[GCX_NODE_MOVES], tmax[GCX_NODE_MOVES];
    for (int it = 0; it < GCX_NODE_MOVES; ++it) tit[it] = tmax[it] = HUGE_VAL;
    if (allok) {
        int e = MPI_SUCCESS;
        for (int it = -2; it < GCX_NODE_MOVES; ++it) {
            MPI_Barrier(comm);
            const double s = MPI_Wtime();
            if (alldev) {
                e |= MPI_Sendrecv(ds, (int)bytes, MPI_BYTE, up, 1, dr, (int)bytes,
                                  MPI_BYTE, dn, 1, col, MPI_STATUS_IGNORE);
            } else {
                cudaMemcpy(hs, ds, bytes, cudaMemcpyDeviceToHost);
                e |= MPI_Sendrecv(hs, (int)bytes, MPI_BYTE, up, 1, hr, (int)bytes,
                                  MPI_BYTE, dn, 1, col, MPI_STATUS_IGNORE);
                cudaMemcpy(dr, hr, bytes, cudaMemcpyHostToDevice);
            }
            if (it >= 0) tit[it] = MPI_Wtime() - s;
        }
        if (e != MPI_SUCCESS)
            for (int it = 0; it < GCX_NODE_MOVES; ++it) tit[it] = HUGE_VAL;
    }
    MPI_Allreduce(tit, tmax, GCX_NODE_MOVES, MPI_DOUBLE, MPI_MAX, comm);
    double t = HUGE_VAL;
    for (int it = 0; it < GCX_NODE_MOVES; ++it) if (tmax[it] < t) t = tmax[it];
    if (ds) cudaFree(ds);
    if (dr) cudaFree(dr);
    if (hs) cudaFreeHost(hs);
    if (hr) cudaFreeHost(hr);
    cudaGetLastError();
    MPI_Comm_free(&col);
    *device_buffers = alldev;
    return t > 0.0 && t < HUGE_VAL ? (double)bytes / t / 1e9 : HUGE_VAL;
}

/* link_gbs (the slowest node-local neighbour link, HUGE_VAL when none was
 * measured) becomes the slower of it and the inter-node rate, or 0 when
 * neither was measured.  One node: unchanged. */
void inter_node_floor(MPI_Comm comm, MPI_Comm node, int rank, double *link_gbs)
{
    int devbuf = 0;
    const double x = inter_node_rate(comm, node, &devbuf);
    const double in = *link_gbs;
    if (x < *link_gbs) *link_gbs = x;
    if (!(*link_gbs < HUGE_VAL)) *link_gbs = 0.0;
    if (rank == 0 && x < HUGE_VAL) {
        std::fprintf(stdout, "Native_Xchg> inter-node probe (4 MB to the next "
                     "node, every rank at once, %s): %.2f GB/s; neighbour link "
                     "%.2f -> %.2f GB/s\n", devbuf ? "device buffers" : "host-staged",
                     x, in < HUGE_VAL ? in : 0.0, *link_gbs);
        std::fflush(stdout);
    }
}

}  /* anonymous namespace */

extern "C" void gcx_place_device(gcx_comm comm_f, const gc_i32 *nd_in,
                                 const double *box, double *link_gbs,
                                 double *units)
{
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)comm_f);
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    *link_gbs = 0.0;
    *units = 0.0;
    const int nd[3] = { nd_in[0], nd_in[1], nd_in[2] };
    if (size < 2 || nd[0] * nd[1] * nd[2] != size) { *units = device_units(comm); return; }

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nr = 0, ns = 1, ndev = 0, cur = 0;
    MPI_Comm_rank(node, &nr);
    MPI_Comm_size(node, &ns);
    cudaGetDeviceCount(&ndev);
    cudaGetDevice(&cur);
    /* Every node must qualify, or none moves: one rule for the job. */
    int ok = (ns >= 2 && ns <= 8 && ndev == ns && cur == nr) ? 1 : 0, all = 0;
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, comm);

    std::vector<int> grank(ns, 0), devs(ns, 0);
    MPI_Allgather(&rank, 1, MPI_INT, &grank[0], 1, MPI_INT, node);
    MPI_Allgather(&cur, 1, MPI_INT, &devs[0], 1, MPI_INT, node);
    /* Links are still timed when every node-local rank has its own GPU, so
     * pme_grid sees the rate the run will use. */
    int meas = ns >= 2 && ns <= 8 && ndev >= ns && ndev <= 8 ? 1 : 0, mall = 0;
    {
        std::vector<int> seen(ndev > 0 ? ndev : 1, 0);
        for (int i = 0; i < ns && meas; ++i) {
            if (devs[i] < 0 || devs[i] >= ndev || seen[devs[i]]) meas = 0;
            else seen[devs[i]] = 1;
        }
    }
    MPI_Allreduce(&meas, &mall, 1, MPI_INT, MPI_MIN, comm);
    if (!all && !mall) {
        *link_gbs = HUGE_VAL;
        inter_node_floor(comm, node, rank, link_gbs);
        MPI_Comm_free(&node);
        *units = device_units(comm);
        return;
    }

    std::vector<int> perm(ns, 0);
    double c0 = 0.0, c1 = 0.0, slow = HUGE_VAL;
    std::vector<double> bw((size_t)ndev * ndev, 0.0);
    if (nr == 0) {
        const size_t bytes = (size_t)8 << 20;
        std::vector<void *> buf(ndev, (void *)0);
        for (int d = 0; d < ndev; ++d) {
            cudaSetDevice(d);
            if (cudaMalloc(&buf[d], bytes) != cudaSuccess) buf[d] = 0;
            for (int e = 0; e < ndev; ++e) {
                int can = 0;
                if (e != d && cudaDeviceCanAccessPeer(&can, d, e) == cudaSuccess && can)
                    cudaDeviceEnablePeerAccess(e, 0);
            }
        }
        cudaGetLastError();
        /* The rate picks the PME mesh, so it must be the link's, not a moment's
         * (an idle device may be in a low-power state): each link is timed in
         * GCX_PLACE_SWEEPS sweeps, the first also waking every device, keeping the
         * fastest batch of three copies. */
        for (int sweep = 0; sweep < GCX_PLACE_SWEEPS; ++sweep)
            for (int d = 0; d < ndev; ++d)
                for (int e = 0; e < ndev; ++e) {
                    if (d == e || !buf[d] || !buf[e]) continue;
                    cudaSetDevice(d);
                    cudaMemcpyPeer(buf[e], e, buf[d], d, bytes);
                    cudaDeviceSynchronize();
                    const double t0 = MPI_Wtime();
                    for (int it = 0; it < 3; ++it)
                        cudaMemcpyPeer(buf[e], e, buf[d], d, bytes);
                    cudaDeviceSynchronize();
                    const double t = (MPI_Wtime() - t0) / 3.0;
                    const double r = t > 0 ? (double)bytes / t / 1e9 : 0.0;
                    double &b = bw[(size_t)d * ndev + e];
                    if (r > b) b = r;
                }
        for (int d = 0; d < ndev; ++d) {
            cudaSetDevice(d);
            if (buf[d]) cudaFree(buf[d]);
            cudaDeviceReset();
        }
        cudaGetLastError();
        if (!all) {
            /* timing only: the ranks keep their devices */
            slow = link_floor(devs, grank, nd, bw, ndev);
            if (!(slow > 0.0)) slow = HUGE_VAL;
            cudaSetDevice(cur);
        }
    }
    if (!all) {
        MPI_Allreduce(&slow, link_gbs, 1, MPI_DOUBLE, MPI_MIN, comm);
        inter_node_floor(comm, node, rank, link_gbs);
        MPI_Comm_free(&node);
        *units = device_units(comm);
        return;
    }
    if (nr == 0) {
        const double w[3] = { box[1] / nd[1] * box[2] / nd[2],
                              box[0] / nd[0] * box[2] / nd[2],
                              box[0] / nd[0] * box[1] / nd[1] };
        for (int i = 0; i < ns; ++i) perm[i] = i;
        std::vector<int> p = perm, best = perm;
        c0 = pair_cost(perm, grank, nd, w, bw, ndev);
        c1 = c0;
        while (std::next_permutation(p.begin(), p.end())) {
            const double c = pair_cost(p, grank, nd, w, bw, ndev);
            if (c < c1) { c1 = c; best = p; }
        }
        if (c1 < 0.95 * c0) perm = best; else c1 = c0;
        slow = link_floor(perm, grank, nd, bw, ndev);
        if (!(slow > 0.0)) slow = HUGE_VAL;
    } else {
        cudaDeviceReset();
    }
    MPI_Bcast(&perm[0], ns, MPI_INT, 0, node);
    if (nr == 0) cudaSetDevice(perm[0]);
    else cudaSetDevice(perm[nr]);
    /* the slowest neighbour link measured on any node (node leaders measure) */
    MPI_Allreduce(&slow, link_gbs, 1, MPI_DOUBLE, MPI_MIN, comm);
    inter_node_floor(comm, node, rank, link_gbs);
    MPI_Comm_free(&node);
    *units = device_units(comm);

    if (rank == 0) {
        std::fprintf(stdout, "Native_Xchg> device placement (node of rank 0):");
        for (int i = 0; i < ns; ++i) std::fprintf(stdout, " %d", perm[i]);
        std::fprintf(stdout, "  link cost %.3g -> %.3g%s\n", c0, c1,
                     c1 < c0 ? "" : " (default kept)");
        std::fflush(stdout);
    }
}

extern "C" gc_status gcx_route_select(const gc_i32 mode[GCX_ROUTE_NCLASS])
{
    if (mode == 0) return GC_E_ARG;
    for (int k = 0; k < GCX_ROUTE_NCLASS; ++k)
        if (mode[k] < GCX_ROUTE_MODE_MPI || mode[k] > GCX_ROUTE_MODE_THREAD) return GC_E_ARG;
    for (int k = 0; k < GCX_ROUTE_NCLASS; ++k) probe().route_mode[k] = mode[k];
    return GC_OK;
}

extern "C" gc_status gcx_probe_release(void)
{
    ProbeArena &a = arena();
    for (size_t i = 0; i < a.mapped.size(); ++i)
        if (a.mapped[i]) cudaIpcCloseMemHandle(a.mapped[i]);
    a.mapped.clear();
    a.handle.clear();
    if (a.src)   { cudaFree(a.src);      a.src   = 0; }
    if (a.inbox) { cudaFree(a.inbox);    a.inbox = 0; }
    if (a.host)  { cudaFreeHost(a.host); a.host  = 0; }
    return GC_OK;
}

/* Each rank's board holds, per parity and source rank, 2*nwords entries: a
 * word's low and high halves, each beside its epoch in one 64-bit store, so
 * an entry is valid exactly when its epoch matches.  Epoch e uses parity
 * e & 1; e + 2 is written into e's slot only after every peer published
 * e + 1, which each does only after reading e. */
struct gcx_wsum {
    int nwords, nproc, rank;
    gc_u64 *board;                 /* this rank's, [2][nproc][2*nwords]   */
    gc_u64 **boards;               /* device [nproc]: every rank's board,
                                      mapped here                         */
    unsigned int *epoch;           /* device: the last epoch launched     */
    std::vector<void *> mapped;    /* the peers' boards, to close         */
    gc_u64 *post, *sum;            /* host: the kernel's halves, the sum's */
    gc_u64 *post_dev, *sum_dev;    /* the same, as the device sees them   */
    MPI_Comm comm;                 /* a duplicate, the thread's alone     */
    std::thread th;
    volatile bool stop;
    gcx_wsum() : nwords(0), nproc(0), rank(0), board(0), boards(0), epoch(0),
                 post(0), sum(0), post_dev(0), sum_dev(0),
                 comm(MPI_COMM_NULL), stop(false) {}
};

namespace {

/* Thread of a host-carried sum: waits until every half the kernel posted
 * carries the next epoch, adds the words over the ranks (integer MPI_SUM,
 * exact and order independent) and posts each half of the sum beside that
 * epoch.  No CUDA call.  One epoch per sum, in the order every rank runs
 * them. */
void wsum_main(gcx_wsum *w)
{
    const int m = 2 * w->nwords;
    std::vector<gc_u64> in((size_t)w->nwords), out((size_t)w->nwords);
    unsigned int next = 1;
    for (;;) {
        bool ready = true;
        for (int k = 0; k < m && ready; ++k)
            ready = (unsigned int)(__atomic_load_n(w->post + k, __ATOMIC_ACQUIRE) >> 32) == next;
        if (!ready) {
            if (w->stop) return;
            std::this_thread::yield();
            continue;
        }
        for (int j = 0; j < w->nwords; ++j)
            in[j] = (__atomic_load_n(w->post + 2 * j, __ATOMIC_RELAXED) & 0xffffffffull) |
                    (__atomic_load_n(w->post + 2 * j + 1, __ATOMIC_RELAXED) << 32);
        MPI_Request q;
        MPI_Iallreduce(&in[0], &out[0], w->nwords, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                       w->comm, &q);
        for (int done = 0; !done; ) {
            MPI_Test(&q, &done, MPI_STATUS_IGNORE);
            if (!done) std::this_thread::yield();
        }
        for (int k = 0; k < m; ++k) {
            const gc_u64 half = (k & 1) ? (out[k >> 1] >> 32) : (out[k >> 1] & 0xffffffffull);
            __atomic_store_n(w->sum + k, ((gc_u64)next << 32) | half, __ATOMIC_RELEASE);
        }
        ++next;
    }
}

void wsum_stop(gcx_wsum *w)
{
    if (!w->th.joinable()) return;
    w->stop = true;
    w->th.join();
}

/* Stops the threads of live sums at MPI_Finalize (see the transport thread). */
std::vector<gcx_wsum *> &wsum_live()
{
    static std::vector<gcx_wsum *> *v = new std::vector<gcx_wsum *>;
    return *v;
}

int wsum_at_finalize(MPI_Comm, int, void *, void *)
{
    for (size_t i = 0; i < wsum_live().size(); ++i) {
        gcx_wsum *w = wsum_live()[i];
        wsum_stop(w);
        if (w->comm != MPI_COMM_NULL) MPI_Comm_free(&w->comm);
    }
    return MPI_SUCCESS;
}

/* The host-carried sum over comm, if every rank may call MPI from a second
 * thread and no GPU hosts two ranks.  Collective; one vote for all or none. */
gcx_wsum *wsum_create_thread(MPI_Comm comm, int nwords)
{
    const Probe &pr = probe();
    int level = MPI_THREAD_SINGLE;
    MPI_Query_thread(&level);
    gcx_wsum *w = new gcx_wsum();
    MPI_Comm_size(comm, &w->nproc);
    MPI_Comm_rank(comm, &w->rank);
    w->nwords = nwords;
    const size_t bytes = (size_t)2 * nwords * sizeof(gc_u64);
    void *a = 0, *b = 0, *ad = 0, *bd = 0;
    int ok = level >= MPI_THREAD_MULTIPLE && pr.done && !pr.shared_gpu;
    if (ok && (cudaHostAlloc(&a, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
               cudaHostAlloc(&b, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
               cudaHostGetDevicePointer(&ad, a, 0) != cudaSuccess ||
               cudaHostGetDevicePointer(&bd, b, 0) != cudaSuccess ||
               cudaMalloc((void **)&w->epoch, sizeof(unsigned int)) != cudaSuccess ||
               cudaMemset(w->epoch, 0, sizeof(unsigned int)) != cudaSuccess ||
               cudaDeviceSynchronize() != cudaSuccess))
        ok = 0;
    w->post = (gc_u64 *)a; w->sum = (gc_u64 *)b;
    w->post_dev = (gc_u64 *)ad; w->sum_dev = (gc_u64 *)bd;
    int all = 0;
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, comm);
    if (!all) {
        cudaGetLastError();
        gcx_wsum_destroy(w);
        return 0;
    }
    std::memset(a, 0, bytes);
    std::memset(b, 0, bytes);
    MPI_Comm_dup(comm, &w->comm);
#ifdef __linux__
    cpu_set_t bound;
    const bool widened = widen_to_rank_cpus(&bound);
#endif
    w->th = std::thread(wsum_main, w);
#ifdef __linux__
    if (widened) sched_setaffinity(0, sizeof(bound), &bound);
#endif
    if (wsum_live().empty()) {
        int key = MPI_KEYVAL_INVALID;
        MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, wsum_at_finalize, &key, 0);
        MPI_Comm_set_attr(MPI_COMM_SELF, key, 0);
    }
    wsum_live().push_back(w);
    return w;
}

}  /* anonymous namespace */

__global__ void gcx_kern_wsum(gc_u64 *__restrict__ words,
                              const struct gcx_wsum_view v)
{
    gcx_wsum_sum(words, v);
}

extern "C" gc_status gcx_wsum_view_of(const gcx_wsum *w,
                                      struct gcx_wsum_view *v)
{
    if (w == 0 || v == 0) return GC_E_ARG;
    v->boards = w->boards; v->epoch = w->epoch;
    v->nwords = w->nwords; v->nproc = w->nproc; v->rank = w->rank;
    v->post = w->post_dev; v->sum = w->sum_dev;
    return GC_OK;
}

extern "C" gc_status gcx_wsum_create(gcx_comm comm_f, gc_i32 nwords,
                                     gc_i32 host_thread, gcx_wsum **out)
{
    if (out == 0 || nwords < 1) return GC_E_ARG;
    *out = 0;
    const Probe &pr = probe();
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)comm_f);
    if (comm == MPI_COMM_NULL) return GC_E_ARG;

    /* rank r of comm is probe rank r. */
    int n = 1, me = 0;
    MPI_Comm_size(comm, &n);
    MPI_Comm_rank(comm, &me);
    int ok = pr.done && pr.nodes == 1 && !pr.shared_gpu &&
             pr.node_size == pr.nproc && n <= pr.nproc && me == pr.rank;
    for (int r = 0; ok && r < n; ++r)
        if (r != pr.rank && !pr.peer_capable[r]) ok = 0;
    gcx_wsum *w = new gcx_wsum();
    w->nwords = nwords; w->nproc = n; w->rank = me;
    const size_t bytes = (size_t)2 * n * 2 * nwords * sizeof(gc_u64);
    cudaIpcMemHandle_t mine;
    std::memset(&mine, 0, sizeof(mine));
    if (ok && (cudaMalloc((void **)&w->board, bytes) != cudaSuccess ||
               cudaMemset(w->board, 0, bytes) != cudaSuccess ||
               cudaMalloc((void **)&w->boards,
                          n * sizeof(gc_u64 *)) != cudaSuccess ||
               cudaMalloc((void **)&w->epoch, sizeof(unsigned int)) !=
                   cudaSuccess ||
               cudaMemset(w->epoch, 0, sizeof(unsigned int)) != cudaSuccess ||
               cudaIpcGetMemHandle(&mine, w->board) != cudaSuccess))
        ok = 0;
    int all = 0;
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, comm);
    std::vector<cudaIpcMemHandle_t> h((size_t)n);
    if (all)
        MPI_Allgather(&mine, sizeof(mine), MPI_BYTE, &h[0], sizeof(mine),
                      MPI_BYTE, comm);
    std::vector<gc_u64 *> ptr((size_t)n, (gc_u64 *)0);
    w->mapped.assign((size_t)n, (void *)0);
    for (int r = 0; all && ok && r < n; ++r) {
        if (r == me) { ptr[r] = w->board; continue; }
        if (cudaIpcOpenMemHandle(&w->mapped[r], h[r],
                                 cudaIpcMemLazyEnablePeerAccess) !=
            cudaSuccess) { ok = 0; break; }
        ptr[r] = (gc_u64 *)w->mapped[r];
    }
    if (all && ok &&
        cudaMemcpy(w->boards, &ptr[0], n * sizeof(gc_u64 *),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        ok = 0;
    if (all) MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, comm);
    if (!all) {
        cudaGetLastError();            /* a refused mapping is not a fault */
        gcx_wsum_destroy(w);
        if (!host_thread) return GC_OK;   /* no device route: host reduction */
        *out = wsum_create_thread(comm, (int)nwords);
        if (me == 0 && *out) {
            std::fprintf(stdout, "Native_Xchg> sum over the ranks (%d words): "
                         "host thread\n", (int)nwords);
            std::fflush(stdout);
        }
        return GC_OK;
    }
    *out = w;
    return GC_OK;
}

extern "C" gc_status gcx_wsum_launch(gcx_wsum *w, gc_u64 *words,
                                     void *stream)
{
    if (w == 0 || words == 0) return GC_E_ARG;
    struct gcx_wsum_view v;
    gcx_wsum_view_of(w, &v);
    gcx_kern_wsum<<<1, 32, 0, (cudaStream_t)stream>>>(words, v);
    return cudaGetLastError() == cudaSuccess ? GC_OK : GC_E_DEVICE;
}

extern "C" void gcx_wsum_destroy(gcx_wsum *w)
{
    if (w == 0) return;
    wsum_stop(w);
    std::vector<gcx_wsum *> &live = wsum_live();
    live.erase(std::remove(live.begin(), live.end(), w), live.end());
    int fin = 1;
    MPI_Finalized(&fin);
    if (w->comm != MPI_COMM_NULL && !fin) MPI_Comm_free(&w->comm);
    if (w->post) cudaFreeHost(w->post);
    if (w->sum)  cudaFreeHost(w->sum);
    for (size_t r = 0; r < w->mapped.size(); ++r)
        if (w->mapped[r]) cudaIpcCloseMemHandle(w->mapped[r]);
    if (w->board)  cudaFree(w->board);
    if (w->boards) cudaFree(w->boards);
    if (w->epoch)  cudaFree(w->epoch);
    delete w;
}

namespace {

/* The transport [0] and landing [1] streams of one exchange kind's chain,
 * created on first use and kept for the process (one device per rank). */
cudaStream_t *shared_chain_streams(int op)
{
    static cudaStream_t st[GCX_OP_NKIND][2];
    return (op >= 0 && op < GCX_OP_NKIND) ? st[op] : 0;
}

/* The one timed choice of the transport: how an n-byte node-local edge moves
 * between two GPUs with an admitted peer link.  0: staged through pinned host
 * memory; 1: peer, copy engines; 2: peer, SM stores (the fused callers').
 * From the probe's fan-out readings beside compute (every rank moving to all
 * its node peers at once), flat below the small size and linear above it; an
 * edge is judged at its capacity, and peer keeps it unless slower than
 * staging by more than GCX_PEER_MARGIN.  No fixed topology rule reproduces
 * it: on six PCIe Gen5 GPUs over two sockets, four ranks across the sockets
 * ran 13% slower per step with every PCIe edge staged, while with all six
 * ranks the copy engines read 11% slower than staging at 4 MB.  The elected
 * routes are printed with each plan (Native_Xchg> plan=). */
int peer_path(const Probe &pr, gc_i64 n)
{
    const double S = (double)GCX_PROBE_SMALL, L = (double)GCX_PROBE_LARGE;
    auto at = [&](const double *t) {
        return (double)n <= S ? t[0] : t[0] + ((double)n - S) * (t[1] - t[0]) / (L - S);
    };
    const double tp = at(pr.t_peer_load), tc = at(pr.t_ce_load),
                 ts = at(pr.t_staged_load);
    const double best = tc > 0.0 && (tp <= 0.0 || tc < tp) ? tc : tp;
    if (!(best > 0.0 && ts > 0.0 && best <= GCX_PEER_MARGIN * ts)) return 0;
    return tp > 0.0 && (tc <= 0.0 || tp <= tc) ? 2 : 1;
}

/* The route this rank proposes for one edge, from the frozen probe. */
gc_i32 propose(const Plan &p, const Edge &e)
{
    const Probe &pr = probe();
    if (e.peer == p.rank)                 return GCX_ROUTE_SELF;
    if (pr.same_node[e.peer]) {
        if (pr.same_device[e.peer] && pr.peer_admitted[e.peer])
            return GCX_ROUTE_LOCAL;
        if (pr.peer_capable[e.peer] && pr.peer_admitted[e.peer] &&
            peer_path(pr, std::max(e.send_capacity, e.recv_capacity)) != 0)
            return GCX_ROUTE_PEER;
        return GCX_ROUTE_STAGED;
    }
    if (pr.mpi_device_capable)            return GCX_ROUTE_MPI_DEVICE;
    return GCX_ROUTE_STAGED;
}

/* One handshake round: every edge sends a record to its peer and receives
 * the peer's, matched by the receiving side's edge key.  shared: the peer's
 * host landing area is mapped here; sig: both areas are device-addressable
 * (shmsig). */
struct HandRec { gc_i32 route; gc_i32 key; gc_i64 recv_capacity;
                 gc_i64 inbox_offset; gc_i32 shared; gc_i32 sig; };

/* The device's address of registered, mapped host memory; 0 if it has none. */
char *host_dev_ptr(void *host)
{
    void *d = 0;
    if (cudaHostGetDevicePointer(&d, host, 0) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return (char *)d;
}

/* Map `bytes` of the POSIX shared memory object `name` and register it for
 * DMA and for device access (the shmsig words); 0 on any failure (nothing
 * left mapped). */
char *shm_map(const char *name, gc_i64 bytes, bool create)
{
    int fd = shm_open(name, create ? (O_CREAT | O_EXCL | O_RDWR) : O_RDWR, 0600);
    if (fd < 0) return 0;
    if (create && ftruncate(fd, (off_t)bytes) != 0) { close(fd); shm_unlink(name); return 0; }
    void *m = mmap(0, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { if (create) shm_unlink(name); return 0; }
    if (cudaHostRegister(m, (size_t)bytes,
                         cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
        cudaGetLastError();
        munmap(m, (size_t)bytes);
        if (create) shm_unlink(name);
        return 0;
    }
    return (char *)m;
}

void hand_round(Plan &p, int tag, const std::vector<HandRec> &mine,
                std::vector<HandRec> &theirs)
{
    size_t n = p.edge.size();
    theirs.assign(n, HandRec());
    std::vector<MPI_Request> req;
    for (size_t i = 0; i < n; ++i) {
        if (p.edge[i].peer == p.rank) { theirs[i] = mine[i]; continue; }
        MPI_Request r;
        MPI_Irecv(&theirs[i], sizeof(HandRec), MPI_BYTE, p.edge[i].peer,
                  tag + p.edge[i].key, p.comm, &r);
        req.push_back(r);
    }
    for (size_t i = 0; i < n; ++i) {
        if (p.edge[i].peer == p.rank) continue;
        MPI_Request r;
        MPI_Isend(&mine[i], sizeof(HandRec), MPI_BYTE, p.edge[i].peer,
                  tag + p.edge[i].partner, p.comm, &r);
        req.push_back(r);
    }
    if (!req.empty())
        MPI_Waitall((int)req.size(), &req[0], MPI_STATUSES_IGNORE);
}

}  /* anonymous namespace */

extern "C" gc_status gcx_plan_create(const gcx_plan_desc *desc, gcx_comm comm_f,
                                     gcx_plan **out)
{
    if (desc == 0 || out == 0) return GC_E_ARG;
    if (desc->wire_schema != GCX_WIRE_SCHEMA)          return GC_E_ABI;
    if (desc->num_edges < 0 || (desc->num_edges > 0 && desc->edge == 0))
        return GC_E_ARG;
    if (desc->op < 0 || desc->op >= GCX_OP_NKIND)      return GC_E_ARG;
    if (!probe().done)                                 return GC_E_STATE;

    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)comm_f);
    if (comm == MPI_COMM_NULL) return GC_E_ARG;

    /* Tags are bounded: a span past MPI_TAG_UB is refused, not wrapped. */
    {
        void *v = 0;
        int flag = 0;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &v, &flag);
        int ub = (flag && v) ? *(int *)v : 32767;
        if (desc->tag_base < 0 || desc->tag_base + GCX_TAG_SPAN - 1 > ub)
            return GC_E_CAPACITY;
    }

    gcx_plan *h = new gcx_plan_s();
    Plan &p = h->p;
    p.comm = comm;
    MPI_Comm_rank(comm, &p.rank);
    MPI_Comm_size(comm, &p.nproc);
    p.op           = desc->op;
    p.tag_base     = desc->tag_base;
    p.wire_schema  = desc->wire_schema;
    p.deadline_ms  = desc->deadline_ms > 0 ? desc->deadline_ms : GCX_DEADLINE_MS;

    /* Edges and landing-area byte arithmetic, checked in 64 bits before any
     * allocation. */
    gc_i64 off = 0;
    for (gc_i64 i = 0; i < desc->num_edges; ++i) {
        const gcx_edge_desc &d = desc->edge[i];
        if (d.send_capacity < 0 || d.recv_capacity < 0 ||
            d.peer < 0 || d.peer >= p.nproc) {
            gcx_plan_destroy(h); return GC_E_ARG;
        }
        Edge e;
        e.peer          = d.peer;
        e.axis          = d.axis;
        e.dir           = d.dir;
        e.send_capacity = d.send_capacity;
        e.recv_capacity = d.recv_capacity;
        e.route         = GCX_ROUTE_STAGED;
        e.peer_base     = 0;
        e.peer_inbox    = 0;
        e.ipc_open      = false;
        e.shared        = false;
        e.shared_peer   = 0;
        e.shared_peer_dev = 0;
        e.shmsig        = false;
        if (d.key >= 0 && d.partner >= 0) {
            e.key = d.key; e.partner = d.partner;
        } else if (d.axis >= 0) {
            e.key     = d.axis * 2 + (d.dir > 0 ? 1 : 0);
            e.partner = d.axis * 2 + (d.dir > 0 ? 0 : 1);
        } else {
            int n = 0;
            for (gc_i64 j = 0; j < i; ++j)
                if (p.edge[j].peer == d.peer && p.edge[j].axis < 0) ++n;
            e.key = e.partner = 6 + n;
        }
        if (e.key >= GCX_MAX_EDGE_KEY || e.partner >= GCX_MAX_EDGE_KEY) {
            gcx_plan_destroy(h); return GC_E_CAPACITY;
        }
        gc_i64 span = 0;
        if (!mul_checked(d.recv_capacity, (gc_i64)GCX_SLOTS, &span) ||
            !add_checked(sig_offset(d.recv_capacity), 16, &span)) {
            gcx_plan_destroy(h); return GC_E_OVERFLOW;
        }
        e.inbox_offset = off;
        if (!add_checked(off, span, &off)) {
            gcx_plan_destroy(h); return GC_E_OVERFLOW;
        }
        p.edge.push_back(e);
    }
    p.inbox_bytes = off;
    p.flight.assign(p.edge.size(), EdgeFlight());

    /* Round A: propose, then reconcile to the conservative common route. */
    std::vector<HandRec> mine(p.edge.size()), theirs;
    for (size_t i = 0; i < p.edge.size(); ++i) {
        mine[i].route         = propose(p, p.edge[i]);
        mine[i].key           = p.edge[i].key;
        mine[i].recv_capacity = p.edge[i].recv_capacity;
        mine[i].inbox_offset  = p.edge[i].inbox_offset;
    }
    hand_round(p, p.tag_base + GCX_TAG_OPS + 0 * GCX_MAX_EDGE_KEY, mine, theirs);
    std::vector<gc_i64> peer_off(p.edge.size(), 0);
    int capacity_mismatch = 0;
    for (size_t i = 0; i < p.edge.size(); ++i) {
        p.edge[i].route = reconcile(mine[i].route, theirs[i].route);
        if (p.edge[i].route == GCX_ROUTE_PEER) probe().peer_used[p.edge[i].peer] = 1;
        peer_off[i]     = theirs[i].inbox_offset;
        /* the peer's send capacity must equal my receive capacity */
        if (theirs[i].recv_capacity != p.edge[i].send_capacity &&
            p.edge[i].peer != p.rank) {
            capacity_mismatch = 1;
        }
    }
    /* Round A is a collective contract: every rank votes its local result
     * before any advances to the IPC exchange. */
    int any_capacity_mismatch = 0;
    int capacity_vote = MPI_Allreduce(&capacity_mismatch,
                                      &any_capacity_mismatch, 1, MPI_INT,
                                      MPI_MAX, p.comm);
    /* MPI gives no uniformity guarantee after a communicator error, so report
     * state rather than a collective refusal. */
    if (capacity_vote != MPI_SUCCESS) {
        gcx_plan_destroy(h); return GC_E_STATE;
    }
    if (any_capacity_mismatch != 0) {
        gcx_plan_destroy(h); return GC_E_MISMATCH;
    }

    /* Device landing area and IPC mappings, only for the peers this plan talks
     * to (peer access and handle opening grow with rank count). */
    bool want_inbox = false;
    for (size_t i = 0; i < p.edge.size(); ++i)
        if (p.edge[i].route == GCX_ROUTE_LOCAL ||
            p.edge[i].route == GCX_ROUTE_PEER) want_inbox = true;
    if (want_inbox && p.inbox_bytes > 0) {
        if (cudaMalloc(&p.inbox, (size_t)p.inbox_bytes) != cudaSuccess) {
            gcx_plan_destroy(h); return GC_E_NOMEM;
        }
        /* Zero the landing area and wait before the handle leaves: peers write it
         * from their own streams. */
        cudaMemset(p.inbox, 0, (size_t)p.inbox_bytes);
        cudaDeviceSynchronize();
        p.inbox_exported =
            (cudaIpcGetMemHandle(&p.inbox_handle, p.inbox) == cudaSuccess);
    }
    /* Node-local staged edge of a long-lived plan: both ranks map the host
     * slot, so the D2H writes the receiver's slot and the payload skips MPI. */
    std::vector<char> shm_cand(p.edge.size(), 0);
    bool any_cand = false;
    for (size_t i = 0; i < p.edge.size(); ++i) {
        const Edge &e = p.edge[i];
        shm_cand[i] = e.route == GCX_ROUTE_STAGED && e.peer != p.rank &&
                      probe().same_node[e.peer] &&
                      (p.op == GCX_OP_MESH || p.op == GCX_OP_HALO_COORD ||
                       p.op == GCX_OP_HALO_FORCE);
        any_cand = any_cand || shm_cand[i];
    }
    char shm_name[48] = {0};
    if (any_cand && p.inbox_bytes > 0) {
        static int seq = 0;
        std::snprintf(shm_name, sizeof shm_name, "/gcx.%d.%d", (int)getpid(), ++seq);
        p.shm_base = shm_map(shm_name, p.inbox_bytes, true);
        if (p.shm_base) p.shm_maps.push_back(std::make_pair((void *)p.shm_base, p.inbox_bytes));
        if (p.shm_base) p.shm_dev = host_dev_ptr(p.shm_base);
    }
    {
        struct IpcRec { cudaIpcMemHandle_t h; gc_i32 ok; gc_i32 shm_ok;
                        gc_i64 shm_bytes; char shm_name[48]; };
        std::vector<IpcRec> ims(p.edge.size()), ith(p.edge.size());
        for (size_t i = 0; i < p.edge.size(); ++i) {
            std::memset(&ims[i], 0, sizeof(IpcRec));
            ims[i].h  = p.inbox_handle;
            ims[i].ok = p.inbox_exported ? 1 : 0;
            ims[i].shm_ok = p.shm_base != 0;
            ims[i].shm_bytes = p.inbox_bytes;
            std::memcpy(ims[i].shm_name, shm_name, sizeof shm_name);
        }
        std::vector<MPI_Request> req;
        int base = p.tag_base + GCX_TAG_OPS + 1 * GCX_MAX_EDGE_KEY;
        for (size_t i = 0; i < p.edge.size(); ++i) {
            if (p.edge[i].peer == p.rank) { ith[i] = ims[i]; continue; }
            MPI_Request r;
            MPI_Irecv(&ith[i], sizeof(IpcRec), MPI_BYTE, p.edge[i].peer,
                      base + p.edge[i].key, p.comm, &r);
            req.push_back(r);
        }
        for (size_t i = 0; i < p.edge.size(); ++i) {
            if (p.edge[i].peer == p.rank) continue;
            MPI_Request r;
            MPI_Isend(&ims[i], sizeof(IpcRec), MPI_BYTE, p.edge[i].peer,
                      base + p.edge[i].partner, p.comm, &r);
            req.push_back(r);
        }
        if (!req.empty())
            MPI_Waitall((int)req.size(), &req[0], MPI_STATUSES_IGNORE);

        for (size_t i = 0; i < p.edge.size(); ++i) {
            Edge &e = p.edge[i];
            if (e.route != GCX_ROUTE_LOCAL && e.route != GCX_ROUTE_PEER)
                continue;
            void *base_ptr = 0;
            if (!ith[i].ok ||
                cudaIpcOpenMemHandle(&base_ptr, ith[i].h,
                                     cudaIpcMemLazyEnablePeerAccess)
                    != cudaSuccess) {
                e.route = GCX_ROUTE_STAGED;
                continue;
            }
            e.peer_base  = base_ptr;
            e.peer_inbox = (char *)base_ptr + peer_off[i];
            e.ipc_open   = true;
        }

        /* Map each candidate peer's area once (a peer can own two edges). */
        for (size_t i = 0; i < p.edge.size(); ++i) {
            if (!shm_cand[i] || !p.shm_base || !ith[i].shm_ok) continue;
            char *m = 0;
            for (size_t j = 0; j < i && !m; ++j)
                if (p.edge[j].shared_peer && p.edge[j].peer == p.edge[i].peer)
                    m = p.edge[j].shared_peer - peer_off[j];
            if (!m) {
                ith[i].shm_name[sizeof ith[i].shm_name - 1] = 0;
                m = shm_map(ith[i].shm_name, ith[i].shm_bytes, false);
                if (!m) continue;
                p.shm_maps.push_back(std::make_pair((void *)m, ith[i].shm_bytes));
            }
            p.edge[i].shared_peer = m + peer_off[i];
            char *md = host_dev_ptr(m);
            if (md) p.edge[i].shared_peer_dev = md + peer_off[i];
        }
    }

    /* Round B: the admission outcome, reconciled again. */
    for (size_t i = 0; i < p.edge.size(); ++i) {
        mine[i].route  = p.edge[i].route;
        mine[i].shared = p.edge[i].shared_peer != 0;
        mine[i].sig    = p.edge[i].shared_peer_dev != 0 && p.shm_dev != 0;
    }
    hand_round(p, p.tag_base + GCX_TAG_OPS + 2 * GCX_MAX_EDGE_KEY, mine, theirs);
    /* Every peer has mapped this rank's area, so the name can go. */
    if (shm_name[0]) shm_unlink(shm_name);
    for (size_t i = 0; i < p.edge.size(); ++i) {
        p.edge[i].shared = mine[i].shared && theirs[i].shared;
        if (!(p.edge[i].shared && mine[i].sig && theirs[i].sig))
            p.edge[i].shared_peer_dev = 0;
        gc_i32 r = reconcile(mine[i].route, theirs[i].route);
        if (r != p.edge[i].route && p.edge[i].ipc_open) {
            cudaIpcCloseMemHandle(p.edge[i].peer_base);
            p.edge[i].ipc_open   = false;
            p.edge[i].peer_base  = 0;
            p.edge[i].peer_inbox = 0;
        }
        p.edge[i].route = r;
    }

    /* Round C: confirmation; a disagreement refuses the plan. */
    for (size_t i = 0; i < p.edge.size(); ++i) mine[i].route = p.edge[i].route;
    hand_round(p, p.tag_base + GCX_TAG_OPS + 3 * GCX_MAX_EDGE_KEY, mine, theirs);
    int disagree = 0;
    for (size_t i = 0; i < p.edge.size(); ++i)
        if (theirs[i].route != p.edge[i].route) ++disagree;
    int any = 0;
    MPI_Allreduce(&disagree, &any, 1, MPI_INT, MPI_SUM, p.comm);
    if (any != 0) {
        gcx_plan_destroy(h); return GC_E_MISMATCH;
    }
    p.host_edges = false;
    for (size_t i = 0; i < p.edge.size(); ++i) {
        Edge &e = p.edge[i];
        /* PEER only: on a LOCAL edge both ranks share one GPU. */
        e.devsig = (p.op == GCX_OP_MESH || p.op == GCX_OP_HALO_COORD ||
                    p.op == GCX_OP_HALO_FORCE) && e.ipc_open &&
                   e.route == GCX_ROUTE_PEER && !probe().shared_gpu;
        /* The class's route (gcx_route_select). */
        const int cls = route_class(p.op);
        const gc_i32 mode = cls >= 0 ? probe().route_mode[cls] : GCX_ROUTE_MODE_MPI;
        /* A shared staged edge addressable on the device at both ends; not on a
         * shared GPU. */
        e.shmsig = (p.op == GCX_OP_MESH || p.op == GCX_OP_HALO_COORD ||
                    p.op == GCX_OP_HALO_FORCE) && e.route == GCX_ROUTE_STAGED &&
                   e.shared && e.shared_peer_dev != 0 &&
                   !probe().shared_gpu;
        if (e.shmsig) p.shm_edges = true;
        e.relay = cls >= 0 && mode == GCX_ROUTE_MODE_THREAD &&
                  probe().relay_ok && e.peer != p.rank && !probe().same_node[e.peer] &&
                  !e.shared && (e.route == GCX_ROUTE_MPI_DEVICE ||
                                e.route == GCX_ROUTE_STAGED);
        if (e.relay) e.devsig = false;
        if (e.peer != p.rank && !e.devsig && !e.shmsig && !e.relay)
            p.host_edges = true;
    }
    {
        bool any_relay = false;
        for (size_t i = 0; i < p.edge.size(); ++i) any_relay = any_relay || p.edge[i].relay;
        void *w = 0, *wd = 0;
        if (any_relay &&
            (cudaHostAlloc(&w, 2 * sizeof(unsigned long long), cudaHostAllocMapped | cudaHostAllocPortable)
                 != cudaSuccess ||
             cudaHostGetDevicePointer(&wd, w, 0) != cudaSuccess)) {
            if (w) cudaFreeHost(w);
            gcx_plan_destroy(h); return GC_E_DEVICE;
        }
        if (any_relay) {
            ((unsigned long long *)w)[0] = 0;   /* done: the thread's   */
            ((unsigned long long *)w)[1] = 0;   /* ready: the device's  */
            p.relay_word = (volatile unsigned long long *)w;
            p.relay_word_dev = (unsigned long long *)wd;
        }
    }

    /* Pinned host bounce for the staged route, and the transport stream. */
    gc_i64 span = 0;
    for (size_t i = 0; i < p.edge.size(); ++i) {
        gc_i64 m = p.edge[i].send_capacity > p.edge[i].recv_capacity
                 ? p.edge[i].send_capacity : p.edge[i].recv_capacity;
        if (m > span) span = m;
    }
    p.host_span = span;
    bool want_host = false;
    for (size_t i = 0; i < p.edge.size(); ++i)
        if ((p.edge[i].route == GCX_ROUTE_STAGED && !p.edge[i].shared) || p.edge[i].relay)
            want_host = true;
    if (want_host && span > 0) {
        gc_i64 total = 0;
        if (!mul_checked(span, (gc_i64)p.edge.size() * GCX_SLOTS, &total)) {
            gcx_plan_destroy(h); return GC_E_OVERFLOW;
        }
        if (cudaHostAlloc(&p.host_send, (size_t)total, cudaHostAllocDefault)
                != cudaSuccess ||
            cudaHostAlloc(&p.host_recv, (size_t)total, cudaHostAllocDefault)
                != cudaSuccess) {
            gcx_plan_destroy(h); return GC_E_NOMEM;
        }
    }
    /* Greatest priority: transport copies are on the critical path. */
    int prio_low = 0, prio_high = 0;
    cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high);
    /* Plans of one exchange kind run in sequence and are posted in the same
     * order on every rank, so they share one transport and one landing stream. */
    const bool chain = p.op == GCX_OP_MESH || p.op == GCX_OP_HALO_COORD ||
                       p.op == GCX_OP_HALO_FORCE;
    cudaStream_t *chain_st = chain ? shared_chain_streams(p.op) : 0;
    if (chain_st && !chain_st[0] &&
        cudaStreamCreateWithPriority(&chain_st[0], cudaStreamNonBlocking,
                                     prio_high) != cudaSuccess) {
        chain_st[0] = 0;
        gcx_plan_destroy(h); return GC_E_DEVICE;
    }
    if ((chain_st ? (p.transport = chain_st[0], false)
                  : cudaStreamCreateWithPriority(&p.transport, cudaStreamNonBlocking,
                                                 prio_high) != cudaSuccess) ||
        cudaEventCreateWithFlags(&p.send_ready, cudaEventDisableTiming)
            != cudaSuccess ||
        cudaEventCreateWithFlags(&p.recv_ready, cudaEventDisableTiming)
            != cudaSuccess) {
        gcx_plan_destroy(h); return GC_E_DEVICE;
    }
    p.stream_owned = true;
    p.stream_shared = chain_st != 0;
    if (p.shm_edges && chain_st && !chain_st[1] &&
        cudaStreamCreateWithPriority(&chain_st[1], cudaStreamNonBlocking,
                                     prio_high) != cudaSuccess) {
        chain_st[1] = 0;
        gcx_plan_destroy(h); return GC_E_DEVICE;
    }
    if (p.shm_edges &&
        ((chain_st ? (p.landing = chain_st[1], false)
                   : cudaStreamCreateWithPriority(&p.landing, cudaStreamNonBlocking,
                                                  prio_high) != cudaSuccess) ||
         cudaEventCreateWithFlags(&p.land_ready, cudaEventDisableTiming)
             != cudaSuccess)) {
        gcx_plan_destroy(h); return GC_E_DEVICE;
    }

    for (size_t i = 0; i < p.edge.size(); ++i)
        p.report.edges[p.edge[i].route] += 1;
    log_plan(p);
    /* A halo or mesh edge whose fixed-size moves still make the host wait;
     * named once per plan. */
    if (p.op == GCX_OP_MESH || p.op == GCX_OP_HALO_COORD || p.op == GCX_OP_HALO_FORCE)
        for (size_t i = 0; i < p.edge.size(); ++i) {
            const Edge &e = p.edge[i];
            if (e.peer == p.rank || e.devsig || e.shmsig) continue;
            std::fprintf(stderr, "Native_Xchg> rank %d plan=%s edge to %d (%s node) route=%s: "
                         "host-waited\n", p.rank, gcx_op_name(p.op), e.peer,
                         probe().same_node[e.peer] ? "same" : "other", gcx_route_name(e.route));
        }

    *out = h;
    return GC_OK;
}

extern "C" gc_status gcx_plan_destroy(gcx_plan *plan)
{
    if (plan == 0) return GC_E_ARG;
    Plan &p = plan->p;
    relay_wait(p);
    quiesce(p);
    for (size_t i = 0; i < p.edge.size(); ++i)
        if (p.edge[i].ipc_open && p.edge[i].peer_base)
            cudaIpcCloseMemHandle(p.edge[i].peer_base);
    if (p.inbox)     cudaFree(p.inbox);
    if (p.host_send) cudaFreeHost(p.host_send);
    if (p.host_recv) cudaFreeHost(p.host_recv);
    if (p.relay_word) cudaFreeHost((void *)p.relay_word);
    for (size_t i = 0; i < p.shm_maps.size(); ++i) {
        cudaHostUnregister(p.shm_maps[i].first);
        munmap(p.shm_maps[i].first, (size_t)p.shm_maps[i].second);
    }
    if (p.stream_owned) {
        if (p.send_ready) cudaEventDestroy(p.send_ready);
        if (p.recv_ready) cudaEventDestroy(p.recv_ready);
        if (p.transport && !p.stream_shared)  cudaStreamDestroy(p.transport);
        if (p.land_ready) cudaEventDestroy(p.land_ready);
        if (p.landing && !p.stream_shared)    cudaStreamDestroy(p.landing);
    }
    delete plan;
    return GC_OK;
}

extern "C" gc_status gcx_plan_device_edge(const gcx_plan *plan, gc_i64 edge,
                                          gcx_device_edge *out)
{
    if (plan == 0 || out == 0) return GC_E_ARG;
    const Plan &p = plan->p;
    if (edge < 0 || edge >= (gc_i64)p.edge.size()) return GC_E_ARG;
    std::memset(out, 0, sizeof *out);
    const Edge &e = p.edge[(size_t)edge];
    /* A fused caller stores into the peer's slots where SM stores beat the copy
     * engines, otherwise copies with them; the protocol is the same, so the two
     * ends may choose differently. */
    out->devsig = e.devsig ? 1 : 0;
    out->stores = e.devsig &&
                  peer_path(probe(), std::max(e.send_capacity, e.recv_capacity)) == 2 ? 1 : 0;
    out->send_capacity = e.send_capacity;
    out->recv_capacity = e.recv_capacity;
    if (!e.devsig) return GC_OK;
    out->peer_slots = (char *)e.peer_inbox;
    out->my_slots   = (char *)p.inbox + e.inbox_offset;
    out->peer_sig   = peer_sig(p, (size_t)edge);
    out->my_sig     = my_sig(p, (size_t)edge);
    return GC_OK;
}

extern "C" gc_status gcx_move_edges(const gcx_plan *plan, gcx_device_edge *edges,
                                    gc_i32 *fused, gc_i32 *stores)
{
    if (plan == 0 || fused == 0 || stores == 0) return GC_E_ARG;
    const Plan &p = plan->p;
    *fused = *stores = 1;
    for (size_t i = 0; i < p.edge.size(); ++i) {
        gcx_device_edge g;
        gcx_plan_device_edge(plan, (gc_i64)i, &g);
        if (edges) edges[i] = g;
        if (p.edge[i].peer == p.rank) continue;
        if (!g.devsig) *fused = 0;
        if (!g.stores) *stores = 0;
    }
    return GC_OK;
}

namespace {

/* The host side of a claim: the epoch, its slot and the slot's last user. */
gc_status move_take(Plan &p, gc_i64 epoch, gcx_move_open *o)
{
    if (p.host_edges || p.shm_edges || p.outstanding || p.faulted) return GC_E_STATE;
    if (epoch <= p.epoch) return GC_E_EPOCH;
    const int s = (int)(epoch % GCX_SLOTS);
    o->slot = s;
    o->prev = p.slot_epoch[s];
    p.epoch = epoch;
    p.slot_epoch[s] = epoch;
    p.seq_epoch = epoch;
    return GC_OK;
}

}  /* anonymous namespace */

extern "C" int gcx_move_steady(const gcx_plan *plan, gc_i64 epoch, int same_slot)
{
    if (plan == 0) return 0;
    const Plan &p = plan->p;
    const gc_i64 last = p.epoch, stride = epoch - last;
    return last > 0 && stride > 0 && p.seq_epoch == last &&
           (!same_slot || stride % GCX_SLOTS == 0) &&
           p.slot_epoch[epoch % GCX_SLOTS] == epoch - GCX_SEQ_BACK(stride);
}

extern "C" gc_status gcx_move_claim(gcx_plan *plan, gc_i64 epoch, void *stream,
                                    gcx_move_open *out)
{
    if (plan == 0 || out == 0) return GC_E_ARG;
    Plan &p = plan->p;
    std::memset(out, 0, sizeof *out);
    const gc_i64 last = p.epoch, seq = p.seq_epoch;
    const gc_status st = move_take(p, epoch, out);
    if (st != GC_OK) return st;
    /* A captured launch advances the device's epoch by the stride it sees
     * now, so it is recorded only on a steady sequence. */
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing((cudaStream_t)stream, &cap) != cudaSuccess)
        return GC_E_DEVICE;
    if (cap == cudaStreamCaptureStatusNone) return GC_OK;
    if (last <= 0 || seq != last || out->prev != epoch - GCX_SEQ_BACK(epoch - last))
        return GC_E_STATE;
    out->stride = (unsigned long long)(epoch - last);
    return GC_OK;
}

extern "C" gc_status gcx_move_replay(gcx_plan *plan, gc_i64 epoch)
{
    if (!gcx_move_steady(plan, epoch, 0)) return GC_E_STATE;
    gcx_move_open o;
    return move_take(plan->p, epoch, &o);
}

extern "C" gc_status gcx_plan_edge_identity(const gcx_plan *plan, gc_i64 edge,
                                             gc_i32 *key, gc_i32 *partner,
                                             gc_i32 *peer)
{
    if (plan == 0 || key == 0 || partner == 0 || peer == 0) return GC_E_ARG;
    if (edge < 0 || edge >= (gc_i64)plan->p.edge.size()) return GC_E_ARG;
    const gcx::Edge &e = plan->p.edge[(size_t)edge];
    *key = e.key;
    *partner = e.partner;
    *peer = e.peer;
    return GC_OK;
}

namespace {

/* Post one edge's payload receives.  Chunked above the MPI count limit, in
 * ascending 64-bit byte offset; MPI's non-overtaking guarantee matches the
 * chunks of one tag in the order both endpoints post them. */
void post_recv_payload(Plan &p, size_t e, gc_i64 bytes)
{
    if (bytes <= 0) return;
    const Edge &ed = p.edge[e];
    char *dst;
    switch (ed.route) {
    case GCX_ROUTE_STAGED:     if (ed.shared) return;   /* the sender's D2H lands it */
                               dst = host_slot(p, p.host_recv, e, p.slot);
                               break;
    case GCX_ROUTE_MPI_DEVICE: dst = (char *)p.cur_recv[e].ptr; break;
    default: return;    /* self, local and peer do not travel through MPI */
    }
    if (dst == 0) return;
    int tag = tag_of(p, ed.key, p.slot, 1);
    for (gc_i64 o = 0; o < bytes; o += GCX_MPI_CHUNK_BYTES) {
        gc_i64 n = bytes - o;
        if (n > GCX_MPI_CHUNK_BYTES) n = GCX_MPI_CHUNK_BYTES;
        MPI_Request q;
        MPI_Irecv(dst + o, (int)n, MPI_BYTE, ed.peer, tag, p.comm, &q);
        p.preq.push_back(q);
        p.report.chunks += 1;
    }
}

void post_send_payload(Plan &p, size_t e, gc_i64 bytes)
{
    if (bytes <= 0) return;
    const Edge &ed = p.edge[e];
    const char *src;
    switch (ed.route) {
    case GCX_ROUTE_STAGED:     if (ed.shared) return;
                               src = host_slot(p, p.host_send, e, p.slot);
                               break;
    case GCX_ROUTE_MPI_DEVICE: src = (const char *)p.cur_send[e].ptr; break;
    default: return;
    }
    if (src == 0) return;
    int tag = tag_of(p, ed.partner, p.slot, 1);
    for (gc_i64 o = 0; o < bytes; o += GCX_MPI_CHUNK_BYTES) {
        gc_i64 n = bytes - o;
        if (n > GCX_MPI_CHUNK_BYTES) n = GCX_MPI_CHUNK_BYTES;
        MPI_Request q;
        MPI_Isend(src + o, (int)n, MPI_BYTE, ed.peer, tag, p.comm, &q);
        p.preq.push_back(q);
        p.report.chunks += 1;
    }
}

gc_status validate(const Plan &p, size_t e)
{
    const gcx_wire_header &h = p.flight[e].recv_head;
    if (h.schema != p.wire_schema)           return GC_E_ABI;
    if (h.op     != p.op)                    return GC_E_MISMATCH;
    if (h.src    != p.edge[e].peer)          return GC_E_MISMATCH;
    if (h.dst    != p.rank)                  return GC_E_MISMATCH;
    if (h.epoch  != p.epoch)                 return GC_E_EPOCH;
    if (h.bytes  < 0)                        return GC_E_ARG;
    if (h.bytes  > p.edge[e].recv_capacity)  return GC_E_CAPACITY;
    if (p.cur_recv[e].ptr != 0 && h.bytes > p.cur_recv[e].bytes)
        return GC_E_CAPACITY;
    return GC_OK;
}

}  /* anonymous namespace */

extern "C" gc_status gcx_begin(gcx_plan *plan, const gcx_op_desc *op,
                               gcx_token **out)
{
    if (plan == 0 || out == 0) return GC_E_ARG;
    if (op == 0) return GC_E_ARG;
    Plan &p = plan->p;

    if (op->op != p.op)       return GC_E_MISMATCH;
    if (op->epoch <= p.epoch) return GC_E_EPOCH;
    /* One outstanding epoch per plan: a second begin before release is a
     * protocol error. */
    if (p.outstanding) {
        return GC_E_STATE;
    }
    if (p.faulted) {
        return GC_E_STATE;
    }
    if (!p.edge.empty() && (op->send == 0 || op->recv == 0)) return GC_E_ARG;

    /* Admit every size before any plan state moves. */
    for (size_t i = 0; i < p.edge.size(); ++i) {
        gc_i64 nb = op->send_bytes ? op->send_bytes[i] : op->send[i].bytes;
        if (nb < 0 || nb > p.edge[i].send_capacity) return GC_E_CAPACITY;
        if (op->send[i].ptr != 0 && nb > op->send[i].bytes)
            return GC_E_CAPACITY;
    }

    int slot = (int)(op->epoch % GCX_SLOTS);

    /* The transport thread owns the previous operation's buffers until it
     * publishes that epoch. */
    if (!relay_wait(p)) return GC_E_STATE;

    /* Publish what the plan still owes first: an acknowledgement not yet
     * cleared at release is owed, not in flight (ack_pending does not cover it). */
    flush_ack(p);

    /* An owed acknowledgement must leave before this call returns: the peer's
     * begin spins for it and nothing else posts it if the caller blocks elsewhere. */
    bool owed_mpi = p.host_edges;   /* a relay edge acknowledges over MPI after a variable op */
    for (size_t i = 0; i < p.edge.size() && !owed_mpi; ++i)
        owed_mpi = p.edge[i].relay && p.ack_variable;
    if (p.ack_owed && owed_mpi) {
        if (p.ack_event && cudaEventSynchronize(p.ack_event) != cudaSuccess)
            return GC_E_DEVICE;
        flush_ack(p);
    }

    /* Slot s is written only after the previous use's acknowledgement arrives;
     * this rank's own is flushed meanwhile. */
    if (p.ack_pending[slot]) {
        double t0 = MPI_Wtime();
        for (;;) {
            flush_ack(p);
            int a = 1, b = 1;
            if (!p.ack_recv[slot].empty())
                MPI_Testall((int)p.ack_recv[slot].size(), &p.ack_recv[slot][0],
                            &a, MPI_STATUSES_IGNORE);
            if (!p.ack_send[slot].empty())
                MPI_Testall((int)p.ack_send[slot].size(), &p.ack_send[slot][0],
                            &b, MPI_STATUSES_IGNORE);
            if (a && b) break;
            if ((MPI_Wtime() - t0) * 1000.0 > (double)p.deadline_ms) {
                return GC_E_STATE;
            }
        }
        p.ack_recv[slot].clear();
        p.ack_send[slot].clear();
        p.ack_pending[slot] = false;
    }

    p.phase            = PH_PRODUCER;
    p.started          = MPI_Wtime();
    p.variable         = op->variable != 0;
    p.epoch            = op->epoch;
    p.slot             = slot;
    p.out_recv_bytes   = op->recv_bytes;
    p.out_recv_records = op->recv_records;
    p.cur_send.assign(p.edge.size(), gcx_buffer());
    p.cur_recv.assign(p.edge.size(), gcx_buffer());
    p.cur_send_bytes.assign(p.edge.size(), 0);
    p.cur_send_records.assign(p.edge.size(), 0);
    for (size_t i = 0; i < p.edge.size(); ++i) {
        p.cur_send[i] = op->send[i];
        p.cur_recv[i] = op->recv[i];
        gc_i64 nb = op->send_bytes ? op->send_bytes[i] : op->send[i].bytes;
        p.cur_send_bytes[i]   = nb;
        p.cur_send_records[i] = op->send_records ? op->send_records[i] : 0;
    }

    p.req.clear();
    p.preq.clear();

    /* 1.  Receives before sends: headers and this slot's acknowledgement.
     *     Fixed-size payload receives follow in gcx_progress after the producer
     *     event, since the previous consumer may still read the caller's buffer. */
    p.ack_in[slot].assign(p.edge.size(), 0);
    p.ack_recv[slot].clear();
    for (size_t i = 0; i < p.edge.size(); ++i) {
        if (p.edge[i].peer == p.rank || quiet(p, i))
            continue;
        MPI_Request q;
        MPI_Irecv(&p.flight[i].recv_head, sizeof(gcx_wire_header), MPI_BYTE,
                  p.edge[i].peer, tag_of(p, p.edge[i].key, slot, 0),
                  p.comm, &q);
        p.req.push_back(q);
        if (p.edge[i].devsig || p.edge[i].shmsig) continue;
        MPI_Irecv(&p.ack_in[slot][i], 1, MPI_LONG_LONG, p.edge[i].peer,
                  tag_of(p, p.edge[i].key, slot, 2), p.comm, &q);
        p.ack_recv[slot].push_back(q);
    }

    /* 2.  The producer: the transport stream waits for the compute stream's
     *     event and never synchronizes the device. */
    if (op->producer_event)
        cudaStreamWaitEvent(p.transport, (cudaEvent_t)op->producer_event, 0);
    /* The transport thread's receives land from the transport stream, so they
     * also follow the previous consumer. */
    p.relay_op = false;
    for (size_t i = 0; i < p.edge.size(); ++i) p.relay_op = p.relay_op || (quiet(p, i) && p.edge[i].relay);
    if (p.relay_op && p.ack_event)
        cudaStreamWaitEvent(p.transport, p.ack_event, 0);

    /* 3.  Device dispatch: staged D2H, peer and local copies, self copy. */
    for (size_t i = 0; i < p.edge.size(); ++i) {
        Edge &e = p.edge[i];
        gc_i64 nb = p.cur_send_bytes[i];
        p.flight[i].sends_done = false;
        if (e.devsig) {
            /* the ready word is written after the payload, once the slot is free */
            if (p.slot_epoch[slot] > 0)
                k_sig_wait<<<1, 1, 0, p.transport>>>(my_sig(p, i) + 1,
                    (unsigned long long)p.slot_epoch[slot]);
            if (nb > 0)
                cudaMemcpyAsync(peer_slot(p, i, slot), p.cur_send[i].ptr,
                                (size_t)nb, cudaMemcpyDefault, p.transport);
            k_sig_set<<<1, 1, 0, p.transport>>>(peer_sig(p, i),
                                                (unsigned long long)p.epoch);
            p.flight[i].sends_done = true;
            p.report.bytes_sent += nb;
            continue;
        }
        if (e.shmsig) {
            /* devsig's protocol with the payload in the peer's host slot: the peer
             * acknowledges after its H2D; ready follows the D2H. */
            if (p.slot_epoch[slot] > 0)
                k_sig_wait<<<1, 1, 0, p.transport>>>(my_shm_sig(p, i) + 1,
                    (unsigned long long)p.slot_epoch[slot]);
            if (nb > 0)
                cudaMemcpyAsync(e.shared_peer + slot * e.send_capacity,
                                p.cur_send[i].ptr, (size_t)nb,
                                cudaMemcpyDeviceToHost, p.transport);
            k_sig_set<<<1, 1, 0, p.transport>>>(peer_shm_sig(p, i),
                                                (unsigned long long)p.epoch);
            p.flight[i].sends_done = true;
            p.report.bytes_sent += nb;
            continue;
        }
        if (e.relay && quiet(p, i)) {
            if (nb > 0)
                cudaMemcpyAsync(host_slot(p, p.host_send, i, slot), p.cur_send[i].ptr,
                                (size_t)nb, cudaMemcpyDeviceToHost, p.transport);
            p.flight[i].sends_done = true;
            p.report.bytes_sent += nb;
            continue;
        }
        if (nb == 0 || quiet(p, i)) {
            p.flight[i].sends_done = true;
            p.report.bytes_sent += nb;
            continue;
        }
        switch (e.route) {
        case GCX_ROUTE_SELF:
            cudaMemcpyAsync(p.cur_recv[i].ptr, p.cur_send[i].ptr,
                            (size_t)nb, cudaMemcpyDeviceToDevice, p.transport);
            p.flight[i].sends_done = true;
            break;
        case GCX_ROUTE_LOCAL:
            cudaMemcpyAsync(peer_slot(p, i, slot), p.cur_send[i].ptr,
                            (size_t)nb, cudaMemcpyDeviceToDevice, p.transport);
            break;
        case GCX_ROUTE_PEER:
            /* The destination is the IPC mapping of the peer's landing area, so
             * cudaMemcpyDefault resolves both ends from the pointers. */
            cudaMemcpyAsync(peer_slot(p, i, slot), p.cur_send[i].ptr,
                            (size_t)nb, cudaMemcpyDefault, p.transport);
            break;
        case GCX_ROUTE_STAGED:
            cudaMemcpyAsync(e.shared ? e.shared_peer + slot * e.send_capacity
                                     : host_slot(p, p.host_send, i, slot),
                            p.cur_send[i].ptr, (size_t)nb,
                            cudaMemcpyDeviceToHost, p.transport);
            break;
        default:
            break;     /* mpi_device sends from where it is */
        }
        p.report.bytes_sent += nb;
    }
    if (p.relay_op)     /* the staged payloads are in the host slots */
        k_sig_set<<<1, 1, 0, p.transport>>>(p.relay_word_dev + 1,
                                            (unsigned long long)p.epoch);
    cudaEventRecord(p.send_ready, p.transport);
    p.slot_epoch[slot] = p.epoch;
    if (p.relay_op) {
        RelayJob job;
        job.plan = &p;
        job.epoch = p.epoch;
        job.comm = p.comm;
        job.deadline_s = (double)p.deadline_ms * 1e-3;
        for (size_t i = 0; i < p.edge.size(); ++i) {
            if (!p.edge[i].relay) continue;
            RelayMsg g;
            g.peer = p.edge[i].peer;
            g.buf = host_slot(p, p.host_recv, i, slot);
            g.bytes = fixed_recv_bytes(p, i);
            g.tag = tag_of(p, p.edge[i].key, slot, 1);
            g.send = false;
            if (g.bytes > 0 && g.buf) job.msg.push_back(g);
            g.buf = host_slot(p, p.host_send, i, slot);
            g.bytes = p.cur_send_bytes[i];
            g.tag = tag_of(p, p.edge[i].partner, slot, 1);
            g.send = true;
            if (g.bytes > 0 && g.buf) job.msg.push_back(g);
        }
        p.relay_last = p.epoch;
        Relay &r = relay();
        {
            std::lock_guard<std::mutex> lk(r.m);
            r.q.push_back(job);
        }
        r.cv.notify_one();
    }

    /* 4.  Headers are posted by gcx_progress after send_ready: for a
     *     device-copy route the header publishes the copy's completion, and a
     *     device buffer must not reach MPI before its producer event. */
    for (size_t i = 0; i < p.edge.size(); ++i) {
        gcx_wire_header &hd = p.flight[i].send_head;
        hd.schema   = p.wire_schema;
        hd.op       = p.op;
        hd.src      = p.rank;
        hd.dst      = p.edge[i].peer;
        hd.epoch    = p.epoch;
        hd.sequence = ++p.sequence;
        hd.bytes    = p.cur_send_bytes[i];
        hd.records  = p.cur_send_records[i];
        if (p.edge[i].peer == p.rank) p.flight[i].recv_head = hd;
    }

    /* A variable-sized operation whose remote edges are all device-sequenced
     * publishes with the ready word; its header carries only the counts and
     * leaves now. */
    bool early = p.variable;
    for (size_t i = 0; i < p.edge.size() && early; ++i)
        early = p.edge[i].peer == p.rank || p.edge[i].devsig ||
                p.edge[i].shmsig;
    if (early) {
        for (size_t i = 0; i < p.edge.size(); ++i) {
            p.flight[i].sends_done = true;
            if (p.edge[i].peer == p.rank) continue;
            MPI_Request q;
            MPI_Isend(&p.flight[i].send_head, sizeof(gcx_wire_header),
                      MPI_BYTE, p.edge[i].peer,
                      tag_of(p, p.edge[i].partner, p.slot, 0), p.comm, &q);
            p.req.push_back(q);
            p.report.messages += 1;
        }
        p.phase = PH_HEADERS;
    }

    p.outstanding = true;

    gcx_token *t = new gcx_token_s();
    t->owner    = plan;
    t->epoch    = p.epoch;
    t->slot     = slot;
    t->consumed = false;
    *out = t;
    return GC_OK;
}

extern "C" gc_status gcx_progress(gcx_plan *plan)
{
    if (plan == 0) return GC_E_ARG;
    Plan &p = plan->p;
    flush_ack(p);
    if (!p.outstanding) return GC_OK;

    if (p.phase == PH_PRODUCER) {
        cudaError_t e = cudaEventQuery(p.send_ready);
        if (e == cudaErrorNotReady) return GC_OK;
        if (e != cudaSuccess)       return GC_E_DEVICE;
        if (!p.variable)
            for (size_t i = 0; i < p.edge.size(); ++i)
                if (!p.edge[i].relay)
                    post_recv_payload(p, i, fixed_recv_bytes(p, i));
        for (size_t i = 0; i < p.edge.size(); ++i) {
            if (p.edge[i].peer == p.rank || quiet(p, i)) {
                p.flight[i].sends_done = true;
                continue;
            }
            MPI_Request q;
            MPI_Isend(&p.flight[i].send_head, sizeof(gcx_wire_header),
                      MPI_BYTE, p.edge[i].peer,
                      tag_of(p, p.edge[i].partner, p.slot, 0), p.comm, &q);
            p.req.push_back(q);
            p.report.messages += 1;
            if (!p.variable) post_send_payload(p, i, p.cur_send_bytes[i]);
            p.flight[i].sends_done = true;
        }
        p.phase = p.variable ? PH_HEADERS : PH_PAYLOAD;
    }

    if (p.phase == PH_HEADERS) {
        int done = 1;
        if (!p.req.empty())
            MPI_Testall((int)p.req.size(), &p.req[0], &done,
                        MPI_STATUSES_IGNORE);
        if (!done) return GC_OK;
        p.req.clear();
        /* Counts have arrived: admit them before any payload moves (the one round
         * trip of a variable-sized operation). */
        for (size_t i = 0; i < p.edge.size(); ++i) {
            gc_status s = validate(p, i);
            if (s != GC_OK) return s;
            p.flight[i].recv_bytes   = p.flight[i].recv_head.bytes;
            p.flight[i].recv_records = p.flight[i].recv_head.records;
        }
        for (size_t i = 0; i < p.edge.size(); ++i)
            post_recv_payload(p, i, p.flight[i].recv_bytes);
        for (size_t i = 0; i < p.edge.size(); ++i) {
            post_send_payload(p, i, p.cur_send_bytes[i]);
        }
        p.phase = PH_PAYLOAD;
    }

    if (p.phase == PH_PAYLOAD) {
        int a = 1, b = 1;
        if (!p.req.empty())
            MPI_Testall((int)p.req.size(), &p.req[0], &a, MPI_STATUSES_IGNORE);
        if (!p.preq.empty())
            MPI_Testall((int)p.preq.size(), &p.preq[0], &b,
                        MPI_STATUSES_IGNORE);
        if (!a || !b) return GC_OK;
        for (size_t i = 0; i < p.edge.size(); ++i)
            if (!p.flight[i].sends_done) return GC_OK;
        p.req.clear();
        p.preq.clear();
        p.phase = PH_DONE;
    }
    return GC_OK;
}

extern "C" gc_status gcx_consume(gcx_token *token, void *consumer_stream)
{
    if (token == 0) return GC_E_ARG;
    gcx_plan *handle = token->owner;
    Plan &p = handle->p;
    if (!p.outstanding || token->epoch != p.epoch) return GC_E_EPOCH;

    /* Block once on the producer rather than poll; an all-devsig fixed-size
     * operation has nothing to wait for. */
    if (!p.host_edges && !p.variable) p.phase = PH_DONE;
    if (p.phase == PH_PRODUCER && cudaEventSynchronize(p.send_ready) != cudaSuccess)
        return GC_E_DEVICE;
    double t0 = MPI_Wtime();
    while (p.phase != PH_DONE) {
        gc_status s = gcx_progress(handle);
        if (s != GC_OK) return s;
        /* A dropped edge is a refusal, not a hang. */
        if ((MPI_Wtime() - t0) * 1000.0 > (double)p.deadline_ms)
            return GC_E_STATE;
    }

    for (size_t i = 0; i < p.edge.size(); ++i) {
        if (quiet(p, i)) {
            /* fixed-size edge: the peer sends its capacity or the receive buffer's bytes */
            p.flight[i].recv_head.bytes = fixed_recv_bytes(p, i);
            p.flight[i].recv_head.records = 0;
        } else {
            gc_status s = validate(p, i);
            if (s != GC_OK) return s;
        }
        p.flight[i].recv_bytes   = p.flight[i].recv_head.bytes;
        p.flight[i].recv_records = p.flight[i].recv_head.records;
        if (p.out_recv_bytes)   p.out_recv_bytes[i]   = p.flight[i].recv_bytes;
        if (p.out_recv_records) p.out_recv_records[i] =
                                    p.flight[i].recv_records;
        p.report.bytes_received += p.flight[i].recv_bytes;
    }

    /* Receive-ready means visible to the consumer: staged H2D and self copy
     * are ordered on the transport stream; device-copy payloads are published
     * here from the inbox. */
    cudaStream_t cs = (cudaStream_t)consumer_stream;
    /* A shmsig H2D lands in the caller's buffer, which the previous consumer
     * may still read (rule of gcx_begin). */
    if (p.shm_edges && p.ack_event)
        cudaStreamWaitEvent(p.landing, p.ack_event, 0);
    for (size_t i = 0; i < p.edge.size(); ++i) {
        gc_i64 nb = p.flight[i].recv_bytes;
        const Edge &e = p.edge[i];
        if (e.shmsig) {
            /* wait for the peer's ready word, land the payload, free the host slot */
            k_sig_wait<<<1, 1, 0, p.landing>>>(my_shm_sig(p, i),
                                               (unsigned long long)p.epoch);
            if (nb > 0)
                cudaMemcpyAsync(p.cur_recv[i].ptr,
                                p.shm_base + e.inbox_offset
                                    + p.slot * e.recv_capacity,
                                (size_t)nb, cudaMemcpyHostToDevice, p.landing);
            k_sig_set<<<1, 1, 0, p.landing>>>(peer_shm_sig(p, i) + 1,
                                                (unsigned long long)p.epoch);
            continue;
        }
        if (nb == 0 || e.route != GCX_ROUTE_STAGED || quiet(p, i)) continue;
        cudaMemcpyAsync(p.cur_recv[i].ptr,
                        e.shared ? p.shm_base + e.inbox_offset
                                       + p.slot * e.recv_capacity
                                 : host_slot(p, p.host_recv, i, p.slot),
                        (size_t)nb, cudaMemcpyHostToDevice, p.transport);
    }
    if (p.shm_edges) {
        cudaEventRecord(p.land_ready, p.landing);
        cudaStreamWaitEvent(p.transport, p.land_ready, 0);
    }
    if (p.relay_op) {
        k_sig_wait<<<1, 1, 0, p.transport>>>(p.relay_word_dev,
                                             (unsigned long long)p.epoch);
        for (size_t i = 0; i < p.edge.size(); ++i) {
            const gc_i64 nb = p.flight[i].recv_bytes;
            if (!p.edge[i].relay || nb <= 0) continue;
            cudaMemcpyAsync(p.cur_recv[i].ptr, host_slot(p, p.host_recv, i, p.slot),
                            (size_t)nb, cudaMemcpyHostToDevice, p.transport);
        }
    }
    cudaEventRecord(p.recv_ready, p.transport);
    cudaStreamWaitEvent(cs, p.recv_ready, 0);
    for (size_t i = 0; i < p.edge.size(); ++i) {
        gc_i64 nb = p.flight[i].recv_bytes;
        if (p.edge[i].devsig)
            k_sig_wait<<<1, 1, 0, cs>>>(my_sig(p, i),
                                        (unsigned long long)p.epoch);
        if (nb == 0) continue;
        /* a devsig payload is in the inbox and is published here too */
        if (p.edge[i].route == GCX_ROUTE_LOCAL ||
            p.edge[i].route == GCX_ROUTE_PEER)
            cudaMemcpyAsync(p.cur_recv[i].ptr, inbox_slot(p, i, p.slot),
                            (size_t)nb, cudaMemcpyDeviceToDevice, cs);
    }
    token->consumed = true;
    return GC_OK;
}

extern "C" gc_status gcx_release(gcx_token *token, void *consumer_done)
{
    if (token == 0) return GC_E_ARG;
    Plan &p = token->owner->p;
    if (!token->consumed) return GC_E_STATE;

    /* The acknowledgement is owed, not sent: it waits for the consumer's event
     * without a host-side device synchronization and is posted from
     * gcx_progress or the next gcx_begin that needs the slot.  A devsig edge
     * acknowledges on the device. */
    bool waited = false;
    for (size_t i = 0; i < p.edge.size(); ++i) {
        if (!p.edge[i].devsig) continue;
        if (!waited && consumer_done)
            cudaStreamWaitEvent(p.transport, (cudaEvent_t)consumer_done, 0);
        waited = true;
        k_sig_set<<<1, 1, 0, p.transport>>>(peer_sig(p, i) + 1,
                                            (unsigned long long)p.epoch);
    }
    p.ack_event = (cudaEvent_t)consumer_done;
    p.ack_slot  = token->slot;
    p.ack_owed  = true;
    p.ack_variable = p.variable;
    flush_ack(p);

    p.outstanding = false;
    p.phase = PH_IDLE;
    delete token;
    return GC_OK;
}

extern "C" gc_status gcx_abort(gcx_token *token)
{
    if (token == 0) return GC_E_ARG;
    Plan &p = token->owner->p;
    quiesce(p);
    p.faulted = true;
    delete token;
    return GC_OK;
}

extern "C" gc_status gcx_halo_create(const gcx_halo_desc *desc, gcx_comm comm,
                                     gcx_halo **out)
{
    if (desc == 0 || out == 0) return GC_E_ARG;
    gcx_halo *h = new gcx_halo_s();
    for (int a = 0; a < 3; ++a) { h->forward[a] = 0; h->reverse[a] = 0; }

    for (int pass = 0; pass < 3; ++pass) {
        for (int rev = 0; rev < 2; ++rev) {
            /* forward x, y, z; the return z, y, x along the recorded reverse route */
            int axis = rev ? 2 - pass : pass;
            gc_i64 cap = rev ? desc->force_capacity[axis]
                             : desc->coord_capacity[axis];
            gcx_edge_desc e[2];
            for (int s = 0; s < 2; ++s) {
                std::memset(&e[s], 0, sizeof(e[s]));
                e[s].peer = (s == 0) ? desc->neighbour_lower[axis]
                                     : desc->neighbour_upper[axis];
                e[s].axis = axis;
                e[s].dir  = (s == 0) ? -1 : 1;
                e[s].key = e[s].partner = -1;
                e[s].send_capacity = cap;
                e[s].recv_capacity = cap;
            }
            gcx_plan_desc pd;
            std::memset(&pd, 0, sizeof(pd));
            pd.edge = e;
            pd.num_edges = 2;
            pd.op = rev ? GCX_OP_HALO_FORCE : GCX_OP_HALO_COORD;
            pd.tag_base = desc->tag_base + (pass * 2 + rev) * GCX_TAG_SPAN;
            pd.wire_schema = GCX_WIRE_SCHEMA;
            pd.deadline_ms = desc->deadline_ms;
            gcx_plan *plan = 0;
            gc_status s = gcx_plan_create(&pd, comm, &plan);
            if (s != GC_OK) { gcx_halo_destroy(h); return s; }
            if (rev) h->reverse[pass] = plan; else h->forward[pass] = plan;
        }
    }
    *out = h;
    return GC_OK;
}

extern "C" gc_status gcx_halo_destroy(gcx_halo *halo)
{
    if (halo == 0) return GC_E_ARG;
    for (int a = 0; a < 3; ++a) {
        if (halo->forward[a]) gcx_plan_destroy(halo->forward[a]);
        if (halo->reverse[a]) gcx_plan_destroy(halo->reverse[a]);
    }
    delete halo;
    return GC_OK;
}

extern "C" gc_status gcx_halo_plan(gcx_halo *halo, gc_i32 pass, gc_i32 reverse,
                                   gcx_plan **out)
{
    if (halo == 0 || out == 0 || pass < 0 || pass > 2) return GC_E_ARG;
    *out = reverse ? halo->reverse[pass] : halo->forward[pass];
    return GC_OK;
}

extern "C" gc_status gcx_halo_forward(gcx_halo *halo, gc_i32 pass,
                                      const gcx_op_desc *op, gcx_token **out)
{
    if (halo == 0 || pass < 0 || pass > 2) return GC_E_ARG;
    return gcx_begin(halo->forward[pass], op, out);
}

extern "C" gc_status gcx_halo_reverse(gcx_halo *halo, gc_i32 pass,
                                      const gcx_op_desc *op, gcx_token **out)
{
    if (halo == 0 || pass < 0 || pass > 2) return GC_E_ARG;
    return gcx_begin(halo->reverse[pass], op, out);
}

extern "C" gc_status gcx_migration_create(const gcx_migration_desc *desc,
                                          gcx_comm comm, gcx_migration **out)
{
    if (desc == 0 || out == 0) return GC_E_ARG;
    if (desc->num_edges > 0 && desc->edge == 0) return GC_E_ARG;

    gcx_migration *m = new gcx_migration_s();
    m->payload = 0;
    gcx_plan_desc pd;
    std::memset(&pd, 0, sizeof(pd));
    pd.edge         = desc->edge;
    pd.num_edges    = desc->num_edges;
    pd.op           = GCX_OP_MIGRATE_PAYLOAD;
    pd.tag_base     = desc->tag_base + GCX_TAG_SPAN;
    pd.wire_schema  = GCX_WIRE_SCHEMA;
    pd.deadline_ms  = desc->deadline_ms;
    gc_status s = gcx_plan_create(&pd, comm, &m->payload);
    if (s != GC_OK) { delete m; return s; }
    *out = m;
    return GC_OK;
}

extern "C" gc_status gcx_migration_destroy(gcx_migration *mig)
{
    if (mig == 0) return GC_E_ARG;
    if (mig->payload) gcx_plan_destroy(mig->payload);
    delete mig;
    return GC_OK;
}

extern "C" gc_status gcx_migration_payload(gcx_migration *mig,
                                           const gcx_op_desc *op,
                                           gcx_token **out)
{
    if (mig == 0) return GC_E_ARG;
    return gcx_begin(mig->payload, op, out);
}
