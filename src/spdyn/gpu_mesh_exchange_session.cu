/*
 * gpu_mesh_exchange_session.cu : device and MPI half of the mesh exchange session.
 *
 * Allocates the packed buffers the plan's ledger sized, records the two events
 * that order the caller's compute stream against the transport's, runs the
 * pack and unpack copies, and drives gcx_begin / gcx_consume / gcx_release /
 * gcx_abort, or the move engine (gcx_move_claim) for a device-sequenced plan,
 * as gpu_core_xchg.h declares them.  The copy kernel is a generic
 * strided 3-D copy; an accumulating plan's unpack adds integer words instead
 * of storing them.
 *
 * create() runs a collective pre-flight (vote_and_create): the transport
 * answers a Round A capacity disagreement with a local return, which would
 * leave the peer waiting, so the same question is asked first with an
 * Alltoall and one disagreeing pair stops every rank.
 */
#include "gpu_mesh_exchange_session.hpp"

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace genesis_native_s4 {

namespace {

/* The triple-angle launch is the one thing a host compiler cannot read, so
 * the syntax pass defines it away; the kernel is still called with every
 * argument, so their types are still checked. */
#ifdef S4_SYNTAX_CHECK
#define S4_LAUNCH(fn, grid, blk, stm) fn
#else
#define S4_LAUNCH(fn, grid, blk, stm) fn<<<(grid), (blk), 0, (stm)>>>
#endif

const int kThreadsPerBlock = 256;
const int kBlocksCap = 8192;
/* The fused moves' copies (launch_many) run beside the step's pair kernels
 * and are bound by the peer link, not the SMs, so each copy gets a small grid
 * of kBlocksCap blocks. */
const int kFusedBlocksCap = 64;

int nblk_for(long n, int blk, int cap)
{
  long b = (n + blk - 1) / blk;
  if (b < 1) b = 1;
  if (b > cap) b = cap;
  return (int)b;
}

struct alignas(16) elem16 { unsigned long long v[2]; };
struct alignas(32) elem32 { unsigned long long v[4]; };

/* One element of a copy: stored, or -- an accumulating plan's unpack --
 * added as an integer word.  Integer adds are exact and commute, so the
 * target is the same bits whatever order the edges land in; they are
 * atomic because one launch carries several edges onto one cell. */
template <typename T>
__device__ __forceinline__ void put(T *d, const T &v, bool) { *d = v; }
__device__ __forceinline__ void put(std::uint32_t *d, std::uint32_t v, bool add)
{
  if (add) atomicAdd(d, v); else *d = v;
}
__device__ __forceinline__ void put(unsigned long long *d, unsigned long long v,
                                    bool add)
{
  if (add) atomicAdd(d, v); else *d = v;
}

/* A generic strided 3-D copy that serves every pack and unpack, with `add`
 * for an accumulating unpack.  Strides are in elements of T.  A copy's 3-D
 * index is found in 32-bit arithmetic when the count fits (64-bit division is
 * slow and bounded the fused copies). */
template <typename T>
__device__ __forceinline__ void copy_elements(
    T *__restrict__ dst, const T *__restrict__ src, long long n0, long long n1,
    long long n2, long long ds0, long long ds1, long long ds2, long long ss0,
    long long ss1, long long ss2, bool add, long long first, long long stride)
{
  const long long tot = n0 * n1 * n2;
  if (tot <= 0x7fffffffLL) {
    const unsigned u1 = (unsigned)n1, u2 = (unsigned)n2;
    for (unsigned t = (unsigned)first; t < (unsigned)tot; t += (unsigned)stride) {
      const unsigned i2 = t % u2, r = t / u2;
      const unsigned i1 = r % u1, i0 = r / u1;
      put(&dst[i0 * ds0 + i1 * ds1 + i2 * ds2],
          src[i0 * ss0 + i1 * ss1 + i2 * ss2], add);
    }
    return;
  }
  for (long long t = first; t < tot; t += stride) {
    const long long i2 = t % n2, r = t / n2;
    const long long i1 = r % n1, i0 = r / n1;
    put(&dst[i0 * ds0 + i1 * ds1 + i2 * ds2],
        src[i0 * ss0 + i1 * ss1 + i2 * ss2], add);
  }
}

template <typename T>
__global__ void k_copy3d(T *__restrict__ dst, const T *__restrict__ src,
                         long n0, long n1, long n2,
                         long ds0, long ds1, long ds2,
                         long ss0, long ss1, long ss2, bool add)
{
  copy_elements(dst, src, n0, n1, n2, ds0, ds1, ds2, ss0, ss1, ss2, add,
                (long long)blockIdx.x * blockDim.x + threadIdx.x,
                (long long)gridDim.x * blockDim.x);
}

template <typename T>
void launch(const Copy3D &c, const void *src, void *dst, bool add,
            cudaStream_t stream)
{
  const long n0 = (long)c.n[0], n1 = (long)c.n[1], n2 = (long)c.n[2];
  const int blocks = nblk_for(n0 * n1 * n2, kThreadsPerBlock, kBlocksCap);
  /* The kernel takes no offsets: the caller pre-offsets both pointers. */
  const T *s = 0;
  T *d = 0;
  copy_operands(c, src, dst, &s, &d);
  (void)blocks;   /* the syntax pass does not launch anything */
  S4_LAUNCH((k_copy3d<T>), blocks, kThreadsPerBlock, stream)(
      d, s, n0, n1, n2,
      (long)c.dst_stride[0], (long)c.dst_stride[1], (long)c.dst_stride[2],
      (long)c.src_stride[0], (long)c.src_stride[1], (long)c.src_stride[2], add);
}

/* The element kinds the session carries.  In bytes, so a real mesh moves
 * doubles and a spectral pencil moves double pairs with the same kernel.
 * `add` only with 4- or 8-byte words (admit()). */
cudaError_t launch_copy(const Copy3D &c, const void *src, void *dst,
                        count_t element_bytes, bool add, cudaStream_t stream)
{
  if (c.empty()) return cudaSuccess;
  switch (element_bytes) {
  case 4:  launch<std::uint32_t>(c, src, dst, add, stream); break;
  case 8:  launch<unsigned long long>(c, src, dst, add, stream); break;
  case 16: launch<elem16>(c, src, dst, false, stream); break;
  case 32: launch<elem32>(c, src, dst, false, stream); break;
  default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

/* The fused path: every copy of a move in one launch (blockIdx.y selects the
 * copy), and the device-sequenced edges' signal words in one launch each. */
struct CopyDesc {
  const char *src;
  char *dst;
  long long n[3], ss[3], ds[3];
  int add;                    /* an accumulating plan's copy into the target */
};

/* What a fused copy launch does once all of its blocks are done, in its last
 * block (the block that completes the count; the count is zero between
 * launches and the last block resets it).  This replaces the signal launch
 * that would follow the copy on the same stream.
 *   kTailOpen: open the move's epoch and wait for the peers' acknowledgement
 *              of the slot's previous use (k_sig_open_many);
 *   kTailSet:  publish q->epoch in the peers' acknowledgement words
 *              (k_sig_set_many) after an unpack that read this rank's slots.
 * Only a copy-engine move uses them (Impl::merged): its pack writes this
 * rank's own send buffer and its unpack reads this rank's slots, so a block
 * is done with every byte it touches once its threads pass the barrier and
 * the only system-scope ordering is the signal write's own. */
/* A copy-engine move whose ready words go as stream memory operations
 * (batch_memop) waits for the peers' ready words at the head of its unpack
 * instead (CopyTail::head_n): every block waits for them before it reads a
 * slot. */
enum { kTailNone = 0, kTailOpen = 1, kTailSet = 2 };
struct CopyTail {
  int kind;
  int n;                                  /* signal words, <= the block size */
  unsigned long long *const *f;
  unsigned *count;
  gcx_seq *q;                             /* kTailOpen: the sequence word */
  unsigned long long epoch, prev, stride; /* kTailOpen: gcx_seq_open's */
  int head_n;                             /* words every block waits on first */
  unsigned long long *const *head_f;
  unsigned long long head_v;              /* ... until each is at least this */
};

__device__ __forceinline__ void copy_head(const CopyTail &t)
{
  if ((int)threadIdx.x < t.head_n) gcx_sig_wait(t.head_f[threadIdx.x], t.head_v);
  __syncthreads();
  __threadfence();
}

__device__ __forceinline__ void copy_tail(const CopyTail &t)
{
  __shared__ int last;
  __shared__ unsigned long long wait_for;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    last = atomicAdd(t.count, 1u) == gridDim.x * gridDim.y - 1;
  }
  __syncthreads();
  if (!last) return;
  if (t.kind == kTailOpen) {
    if (threadIdx.x == 0) {
      wait_for = gcx_seq_open(t.q, t.epoch, t.prev, t.stride).prev;
      *t.count = 0;
    }
    __syncthreads();
    if ((int)threadIdx.x < t.n) gcx_sig_wait(t.f[threadIdx.x], wait_for);
  } else {
    if (threadIdx.x == 0) *t.count = 0;
    if ((int)threadIdx.x < t.n) {
      __threadfence_system();
      *(volatile unsigned long long *)t.f[threadIdx.x] =
          ((const volatile gcx_seq *)t.q)->epoch;
    }
  }
}

/* The fused path's descriptors are per slot, `per_slot` apart; the slot is
 * the open epoch's (gcx_seq).  A launch with a kTailOpen tail reads the slot
 * before its own last block opens the next epoch, so it is used only where
 * both slots' descriptors are the same (a copy-engine move). */
template <typename T>
__global__ void k_copy_many(const CopyDesc *__restrict__ d,
                            const gcx_seq *__restrict__ q, int per_slot,
                            const CopyTail t)
{
  if (t.head_n > 0) copy_head(t);
  const CopyDesc c = d[(int)(q->epoch % GCX_SLOTS) * per_slot + blockIdx.y];
  copy_elements((T *)c.dst, (const T *)c.src, c.n[0], c.n[1], c.n[2],
                c.ds[0], c.ds[1], c.ds[2], c.ss[0], c.ss[1], c.ss[2],
                c.add != 0, (long long)blockIdx.x * blockDim.x + threadIdx.x,
                (long long)gridDim.x * blockDim.x);
  if (t.kind != kTailNone) copy_tail(t);
}

/* Open the move's epoch (gcx_seq_open), then wait until every edge has
 * acknowledged the slot's previous use. */
__global__ void k_sig_open_many(unsigned long long *const *f, int n,
                                gcx_seq *q, unsigned long long epoch,
                                unsigned long long prev,
                                unsigned long long stride)
{
  __shared__ unsigned long long wait_for;
  if (threadIdx.x == 0) wait_for = gcx_seq_open(q, epoch, prev, stride).prev;
  __syncthreads();
  const int i = threadIdx.x;
  if (i < n)
    gcx_sig_wait(f[i], wait_for);
}

__global__ void k_sig_wait_many(unsigned long long *const *f, int n,
                                const gcx_seq *q)
{
  const int i = threadIdx.x;
  if (i < n)
    gcx_sig_wait(f[i], q->epoch);
}

__global__ void k_sig_set_many(unsigned long long *const *f, int n,
                               const gcx_seq *q)
{
  __threadfence_system();
  const int i = threadIdx.x;
  if (i < n) *(volatile unsigned long long *)f[i] = q->epoch;
}

/* A copy-engine move's ready words to the peers, then the wait for theirs:
 * k_sig_set_many and k_sig_wait_many in one launch. */
__global__ void k_sig_set_wait_many(unsigned long long *const *set,
                                    unsigned long long *const *wait, int n,
                                    const gcx_seq *q)
{
  __threadfence_system();
  const int i = threadIdx.x;
  if (i < n) {
    *(volatile unsigned long long *)set[i] = q->epoch;
    gcx_sig_wait(wait[i], q->epoch);
  }
}

cudaError_t launch_many(const CopyDesc *d, int count, long long most,
                        count_t element_bytes, const gcx_seq *q,
                        cudaStream_t stream, const CopyTail &t = CopyTail())
{
  if (count == 0) return cudaSuccess;
  const dim3 grid((unsigned)nblk_for((long)most, kThreadsPerBlock, kFusedBlocksCap),
                  (unsigned)count);
  switch (element_bytes) {
  case 4:  S4_LAUNCH((k_copy_many<std::uint32_t>), grid, kThreadsPerBlock, stream)(d, q, count, t); break;
  case 8:  S4_LAUNCH((k_copy_many<unsigned long long>), grid, kThreadsPerBlock, stream)(d, q, count, t); break;
  case 16: S4_LAUNCH((k_copy_many<elem16>), grid, kThreadsPerBlock, stream)(d, q, count, t); break;
  default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

/* The ready words of a copy-engine move as stream memory operations (CUDA
 * Driver API): one cuStreamBatchMemOp of 64-bit writes, each after a
 * stream-scoped system fence (CU_STREAM_WRITE_VALUE_DEFAULT), so the copies
 * before it on the stream land first.  Reached through the runtime's driver
 * entry point, so nothing links libcuda; null where the device does not
 * support 64-bit memory operations. */
typedef CUresult (*BatchMemOpFn)(CUstream, unsigned int,
                                 CUstreamBatchMemOpParams *, unsigned int);

void *driver_entry(const char *name)
{
  void *p = 0;
  cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
  if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) !=
          cudaSuccess || q != cudaDriverEntryPointSuccess)
    return 0;
  return p;
}

BatchMemOpFn batch_memop()
{
  static int state = 0;                   /* 0 unknown, 1 usable, 2 not */
  static BatchMemOpFn fn = 0;
  if (state == 0) {
    typedef CUresult (*AttrFn)(int *, CUdevice_attribute, CUdevice);
    AttrFn attr = (AttrFn)driver_entry("cuDeviceGetAttribute");
    fn = (BatchMemOpFn)driver_entry("cuStreamBatchMemOp");
    int dev = 0, ok = 0;
    if (!attr || !fn || cudaGetDevice(&dev) != cudaSuccess ||
        attr(&ok, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
             (CUdevice)dev) != CUDA_SUCCESS)
      ok = 0;
    state = ok ? 1 : 2;
    cudaGetLastError();
  }
  return state == 1 ? fn : 0;
}

std::string cuda_message(const char *what, cudaError_t e)
{
  return std::string("mesh exchange: ") + what + ": " + cudaGetErrorString(e);
}

} /* namespace */

struct MeshExchangeSession::Impl {
  gcx_plan *plan = 0;
  gcx_comm comm = 0;
  gcx_token *token = 0;
  bool created = false;
  /* A self edge whose pack and unpack meet in the same packed layout is one
   * copy, source to target, at post time.  When every other edge is empty as
   * well the move is `local` and the transport is never entered. */
  std::vector<std::vector<Copy3D> > direct;
  std::vector<char> is_direct;
  /* direct copies the caller performs itself (defer_self) */
  std::vector<char> deferred;
  bool local = false;
  /* One packed allocation per rank per direction, and the per-edge slices of
   * it the transport is handed at every post (see packed_slice_bytes). */
  void *send_base = 0;
  void *recv_base = 0;
  std::vector<gcx_buffer> send;
  std::vector<gcx_buffer> recv;
  cudaEvent_t producer = 0;
  cudaEvent_t consumer = 0;
  const void *source = 0;
  count_t source_bytes = 0;
  void *target = 0;
  count_t target_bytes = 0;
  std::string error;

  /* Fused: every remote edge is device-sequenced (gcx_plan_device_edge), so a
   * move is signal-wait, one pack launch straight into the peers' landing
   * slots, signal-set; then signal-wait, one unpack launch from this rank's
   * slots; release sets the acks.  No transport call, no host wait. */
  bool fused = false;
  /* fused with SM stores into the peers' slots (gcx_device_edge.stores);
   * otherwise the pack fills this rank's send buffer and the copy engines
   * move each edge's slice into the peer's slot */
  bool stores = false;
  /* a copy-engine fused session runs fused only while the step admits it
   * (allow_copy_engine), otherwise through gcx_begin; both paths keep the
   * plan's slot record (gcx_move_claim) */
  bool ce_on = false;
  bool live() const { return fused && (stores || ce_on); }
  bool built = false;
  std::vector<gcx_device_edge> dev;       /* per edge; devsig for remote ones */
  CopyDesc *d_desc = 0;                   /* [pack s0][pack s1][unpack s0][unpack s1] */
  int npack = 0, nunpack = 0;
  long long most_pack = 0, most_unpack = 0;
  unsigned long long **d_sig = 0;         /* [ack wait][ready set][ready wait][ack set] */
  int nsig = 0;
  gcx_seq *d_seq = 0;                     /* the open epoch, on the device */
  unsigned *d_count = 0;                  /* [pack][unpack] launch tails' counts */
  /* A copy-engine move's signal launches ride on its copy launches
   * (CopyTail), and its ready set and wait share one launch. */
  bool merged() const
  {
    return fused && !stores && npack > 0 && nunpack > 0;
  }
  /* this copy-engine move set its ready words with memory operations
   * (batch_memop), so its unpack waits for the peers' at its head */
  bool cmerge_now = false;
  std::vector<unsigned long long *> h_rdys;  /* the peers' ready words */
  cudaStream_t consumer_stream = 0;
  /* The transport plan's edges, built with the plan's admission. */
  std::vector<gcx_edge_desc> edesc;

  /* A transport refusal names the call and, for GC_E_DEVICE, the CUDA error
   * behind it (often a sticky fault from earlier work on the device). */
  void refused(const char *call, gc_status s)
  {
    error = std::string(call) + ": " + mesh_session_state_name(s);
    if (s == GC_E_DEVICE)
      error += std::string(" cuda=") + cudaGetErrorString(cudaGetLastError());
  }

  void release_buffers()
  {
    if (d_desc) { cudaFree(d_desc); d_desc = 0; }
    if (d_sig) { cudaFree(d_sig); d_sig = 0; }
    if (d_seq) { cudaFree(d_seq); d_seq = 0; }
    if (d_count) { cudaFree(d_count); d_count = 0; }
    built = false;
    send.clear();
    recv.clear();
    if (send_base) { cudaFree(send_base); send_base = 0; }
    if (recv_base) { cudaFree(recv_base); recv_base = 0; }
  }

  void release_events()
  {
    if (producer) { cudaEventDestroy(producer); producer = 0; }
    if (consumer) { cudaEventDestroy(consumer); consumer = 0; }
  }

  /* The fused path's descriptors, from the attached buffers: built once, on
   * the first post after an attach. */
  bool build_fused(const ExchangePlan &plan)
  {
    const long long eb = (long long)plan.element_bytes;
    std::vector<CopyDesc> pk[GCX_SLOTS], up[GCX_SLOTS];
    std::vector<unsigned long long *> ackw, rdys, rdyw, acks;
    most_pack = most_unpack = 0;
    const int add = plan.accumulate ? 1 : 0;
    auto desc = [](const Copy3D &d, const char *src, char *dst, int add) {
      CopyDesc c;
      c.src = src;
      c.dst = dst;
      for (int a = 0; a < 3; ++a) { c.n[a] = d.n[a]; c.ss[a] = d.src_stride[a]; c.ds[a] = d.dst_stride[a]; }
      c.add = add;
      return c;
    };
    for (std::size_t i = 0; i < plan.edge.size(); ++i) {
      const EdgePlan &e = plan.edge[i];
      for (int s = 0; s < GCX_SLOTS; ++s) {
        if (e.self) {
          if (!is_direct[i] || deferred[i]) continue;
          for (const Copy3D &d : direct[i])
            if (!d.empty())
              pk[s].push_back(desc(d, (const char *)source + (long long)d.src_offset * eb,
                                   (char *)target + (long long)d.dst_offset * eb, add));
          continue;
        }
        const gcx_device_edge &g = dev[i];
        for (const Copy3D &d : e.pack)
          if (!d.empty())
            pk[s].push_back(desc(d, (const char *)source + (long long)d.src_offset * eb,
                                 stores ? g.peer_slots + s * g.send_capacity +
                                              (long long)(d.dst_offset - e.send_offset) * eb
                                        : (char *)send_base + (long long)d.dst_offset * eb,
                                 0));
        for (const Copy3D &d : e.unpack)
          if (!d.empty())
            up[s].push_back(desc(d, g.my_slots + s * g.recv_capacity +
                                        (long long)(d.src_offset - e.receive_offset) * eb,
                                 (char *)target + (long long)d.dst_offset * eb, add));
      }
      if (!e.self) {
        ackw.push_back(dev[i].my_sig + 1);
        rdys.push_back(dev[i].peer_sig);
        rdyw.push_back(dev[i].my_sig);
        acks.push_back(dev[i].peer_sig + 1);
      }
    }
    for (std::size_t k = 0; k < pk[0].size(); ++k)
      most_pack = std::max(most_pack, pk[0][k].n[0] * pk[0][k].n[1] * pk[0][k].n[2]);
    for (std::size_t k = 0; k < up[0].size(); ++k)
      most_unpack = std::max(most_unpack, up[0][k].n[0] * up[0][k].n[1] * up[0][k].n[2]);
    npack = (int)pk[0].size();
    nunpack = (int)up[0].size();
    nsig = (int)ackw.size();
    if (nsig > kThreadsPerBlock) return false;
    std::vector<CopyDesc> all;
    for (int s = 0; s < GCX_SLOTS; ++s) all.insert(all.end(), pk[s].begin(), pk[s].end());
    for (int s = 0; s < GCX_SLOTS; ++s) all.insert(all.end(), up[s].begin(), up[s].end());
    std::vector<unsigned long long *> sig;
    sig.insert(sig.end(), ackw.begin(), ackw.end());
    sig.insert(sig.end(), rdys.begin(), rdys.end());
    sig.insert(sig.end(), rdyw.begin(), rdyw.end());
    sig.insert(sig.end(), acks.begin(), acks.end());
    h_rdys = rdys;
    if (d_desc) { cudaFree(d_desc); d_desc = 0; }
    if (d_sig) { cudaFree(d_sig); d_sig = 0; }
    if (!all.empty() &&
        (cudaMalloc(&d_desc, all.size() * sizeof(CopyDesc)) != cudaSuccess ||
         cudaMemcpy(d_desc, &all[0], all.size() * sizeof(CopyDesc),
                    cudaMemcpyHostToDevice) != cudaSuccess))
      return false;
    if (!sig.empty() &&
        (cudaMalloc(&d_sig, sig.size() * sizeof(void *)) != cudaSuccess ||
         cudaMemcpy(d_sig, &sig[0], sig.size() * sizeof(void *),
                    cudaMemcpyHostToDevice) != cudaSuccess))
      return false;
    /* kept across rebuilds of the descriptors: it carries the sequence */
    if (!d_seq &&
        (cudaMalloc(&d_seq, sizeof(gcx_seq)) != cudaSuccess ||
         cudaMemset(d_seq, 0, sizeof(gcx_seq)) != cudaSuccess))
      return false;
    if (!d_count &&
        (cudaMalloc(&d_count, 2 * sizeof(unsigned)) != cudaSuccess ||
         cudaMemset(d_count, 0, 2 * sizeof(unsigned)) != cudaSuccess))
      return false;
    built = true;
    return true;
  }

  /* slot 0's; the kernel steps to the open epoch's slot */
  const CopyDesc *pack_desc() const { return d_desc; }
  const CopyDesc *unpack_desc() const { return d_desc + GCX_SLOTS * npack; }
};

MeshExchangeSession::MeshExchangeSession()
    : impl_(new Impl()), plan_(0), epoch_(0), faulted_(false), last_(GC_OK) {}

MeshExchangeSession::~MeshExchangeSession()
{
  destroy();
  delete impl_;
  impl_ = 0;
}

namespace {

/* What a plan must satisfy before a session admits it, checked before any
 * allocation or transport call.  The plan layer's arithmetic throws rather
 * than wrapping, and create() turns a throw into a refusal. */
gc_status admit(const ExchangePlan &plan, gc_i32 tag_base)
{
  if (plan.element_bytes != 4 && plan.element_bytes != 8 &&
      plan.element_bytes != 16 && plan.element_bytes != 32)
    return GC_E_UNSUPPORTED;
  if (plan.op < 0 || plan.op >= GCX_OP_NKIND) return GC_E_ARG;
  /* An accumulating unpack adds integer words (put()). */
  if (plan.accumulate && plan.element_bytes != 4 && plan.element_bytes != 8)
    return GC_E_UNSUPPORTED;
  if (plan.edge.size() > (std::size_t)0x7fffffff) return GC_E_OVERFLOW;
  if (tag_base < 0) return GC_E_CAPACITY;
  /* The same bound the transport applies, one step earlier. */
  if ((gc_i64)tag_base + (gc_i64)GCX_TAG_SPAN - 1 > (gc_i64)0x7fffffff)
    return GC_E_CAPACITY;
  if (plan.total_send_bytes !=
          pencil::checked_mul(plan.total_send_elements, plan.element_bytes) ||
      plan.total_receive_bytes !=
          pencil::checked_mul(plan.total_receive_elements, plan.element_bytes))
    return GC_E_CAPACITY;
  for (std::size_t i = 0; i < plan.edge.size(); ++i) {
    const EdgePlan &e = plan.edge[i];
    /* A peer indexes the per-rank capacity vectors, so it must be in range
     * first; a negative caller value arrives as a large unsigned one and
     * is refused by this same bound. */
    if (e.peer > kMaxRankIndex) return GC_E_ARG;
    for (std::size_t j = i + 1; j < plan.edge.size(); ++j)
      if (plan.edge[j].peer == e.peer) return GC_E_ARG;   /* keys would collide */
    /* The transport copies a self edge's send into its receive buffer, so
     * a self edge whose banks differ in size would overrun or leave a
     * hole. */
    if (e.self && e.receive_bytes < e.send_bytes) return GC_E_MISMATCH;
    if (e.send_elements > plan.total_send_elements ||
        e.receive_elements > plan.total_receive_elements ||
        e.send_offset > plan.total_send_elements - e.send_elements ||
        e.receive_offset > plan.total_receive_elements - e.receive_elements)
      return GC_E_OVERFLOW;
    if (pencil::checked_mul(e.send_elements, plan.element_bytes) != e.send_bytes ||
        pencil::checked_mul(e.receive_elements, plan.element_bytes) != e.receive_bytes)
      return GC_E_CAPACITY;
    /* The copies must land inside the two packed buffers and the caller's pencil buffers. */
    for (const Copy3D &c : e.pack)
      if (c.dst_span() > plan.total_send_elements ||
          (!c.empty() && c.src_span() > plan.source_elements))
        return GC_E_OVERFLOW;
    for (const Copy3D &c : e.unpack)
      if (c.src_span() > plan.total_receive_elements ||
          (!c.empty() && c.dst_span() > plan.target_elements))
        return GC_E_OVERFLOW;
  }
  return GC_OK;
}

/* The transport plan's edges: plain edges, six keys being reserved for the
 * dimensional pairs.  The transport derives both key numbers from the edge
 * order, so each peer appears exactly once (admit()).  The packed buffers
 * are sized from the edges, so they must sum to the plan's own totals. */
gc_status describe(const ExchangePlan &plan, std::vector<gcx_edge_desc> *edesc)
{
  edesc->assign(plan.edge.size(), gcx_edge_desc());
  count_t send = 0, recv = 0;
  for (std::size_t i = 0; i < plan.edge.size(); ++i) {
    const EdgePlan &e = plan.edge[i];
    gcx_edge_desc &d = (*edesc)[i];
    d.peer = (gc_i32)e.peer;
    d.axis = -1;
    d.dir = 0;
    d.key = -1;
    d.partner = -1;
    d.pad0 = 0;
    d.send_capacity = to_gc_i64(e.send_bytes, "edge send capacity");
    d.recv_capacity = to_gc_i64(e.receive_bytes, "edge receive capacity");
  }
  for (std::size_t i = 0; i < plan.edge.size(); ++i) {
    send = pencil::checked_add(send, plan.edge[i].send_bytes);
    recv = pencil::checked_add(recv, plan.edge[i].receive_bytes);
  }
  return send == plan.total_send_bytes && recv == plan.total_receive_bytes
             ? GC_OK : GC_E_CAPACITY;
}

/* One collective over the communicator, before anything is created.  Every
 * rank reaches every step whatever its own verdict, so a local refusal is a
 * collective refusal.
 *
 *   1. Did any rank refuse to build its plan?  If so, none is created.
 *   2. Is every rank building the same plan of the sequence (tag span,
 *      operation, memory kind, element size)?  The edge set and capacities
 *      differ legitimately between ranks and are the next question's.
 *   3. Does every peer's declared send to this rank equal this rank's
 *      declared receive from it?  That is the transport's Round A condition,
 *      asked here with an Alltoall so ONE disagreeing pair stops EVERY rank.
 *
 * A self edge never crosses the wire; it is checked locally and admit()
 * requires it to be balanced.
 *
 * Every collective is return-checked: after a communicator error MPI gives no
 * uniformity guarantee, so a failed vote refuses with GC_E_STATE instead of
 * consuming reduction results that never happened.
 */
gc_status vote_and_create(const ExchangePlan &plan, gc_status local_status,
                          const std::vector<gcx_edge_desc> &edesc,
                          gc_i32 tag_base, gc_i64 deadline_ms, MPI_Comm comm,
                          gcx_plan **out)
{
  const bool ok = local_status == GC_OK;
  int nproc = 1;
  if (MPI_Comm_size(comm, &nproc) != MPI_SUCCESS) return GC_E_STATE;

  /* 1 and 2: the local verdict and the plan's identity, in one reduction. */
  enum { kVote = 5 };
  gc_i64 mine[kVote] = {ok ? 0 : (gc_i64)local_status, -1, -1, -1, -1};
  if (ok) {
    mine[1] = (gc_i64)tag_base;
    mine[2] = (gc_i64)plan.op;
    mine[3] = (gc_i64)(plan.memory == Memory::host ? 1 : 0);
    mine[4] = (gc_i64)plan.element_bytes;
  }
  gc_i64 lo[kVote] = {0}, hi[kVote] = {0};
  if (MPI_Allreduce(mine, lo, kVote, MPI_LONG_LONG, MPI_MIN, comm) != MPI_SUCCESS)
    return GC_E_STATE;
  if (MPI_Allreduce(mine, hi, kVote, MPI_LONG_LONG, MPI_MAX, comm) != MPI_SUCCESS)
    return GC_E_STATE;
  if (hi[0] != 0) return (gc_status)hi[0];
  for (int k = 1; k < kVote; ++k)
    if (lo[k] != hi[k]) return GC_E_MISMATCH;

  /* 3: the per-peer capacity agreement.  admit() bounds every peer; a peer
   * past this communicator is a mismatch. */
  std::vector<gc_i64> my_send((std::size_t)nproc, 0);
  std::vector<gc_i64> their_send((std::size_t)nproc, 0);
  std::vector<gc_i64> want((std::size_t)nproc, 0);
  gc_i64 mismatch = ok ? 0 : 1;
  count_t me = (count_t)nproc;
  for (std::size_t i = 0; ok && i < plan.edge.size(); ++i) {
    const EdgePlan &e = plan.edge[i];
    if (e.peer >= (count_t)nproc) { mismatch = 1; continue; }
    my_send[(std::size_t)e.peer] = (gc_i64)e.send_bytes;
    if (e.self) me = e.peer;   /* a self edge never crosses the wire */
    else want[(std::size_t)e.peer] = (gc_i64)e.receive_bytes;
  }
  if (MPI_Alltoall(&my_send[0], 1, MPI_LONG_LONG, &their_send[0], 1,
                   MPI_LONG_LONG, comm) != MPI_SUCCESS)
    return GC_E_STATE;
  for (count_t q = 0; ok && q < (count_t)nproc; ++q)
    if (q != me && their_send[(std::size_t)q] != want[(std::size_t)q]) mismatch = 1;
  /* One disagreeing pair stops every rank (the transport's own answer is local). */
  gc_i64 any = 0;
  if (MPI_Allreduce(&mismatch, &any, 1, MPI_LONG_LONG, MPI_MAX, comm) !=
      MPI_SUCCESS)
    return GC_E_STATE;
  if (any != 0) return GC_E_MISMATCH;

  /* The transport takes the communicator the Fortran side holds; the votes
   * use the C handle it resolves to, so both describe one communicator. */
  gcx_plan_desc desc;
  std::memset(&desc, 0, sizeof desc);
  desc.edge = edesc.empty() ? 0 : &edesc[0];
  desc.num_edges = (gc_i64)edesc.size();
  desc.op = plan.op;
  desc.tag_base = tag_base;
  desc.wire_schema = GCX_WIRE_SCHEMA;
  desc.deadline_ms = deadline_ms;
  return gcx_plan_create(&desc, (gcx_comm)MPI_Comm_c2f(comm), out);
}

} /* namespace */

gc_status MeshExchangeSession::create(const ExchangePlan &plan, Memory memory,
                                      gcx_comm comm, gc_i32 tag_base,
                                      gc_i64 deadline_ms)
{
  if (impl_ == 0) return last_ = GC_E_STATE;
  /* A live session is never re-created: re-entering the votes with an
   * admitted plan would admit a second plan over the first one's buffers.
   * It must not happen asymmetrically. */
  if (impl_->created) return last_ = GC_E_STATE;
  MPI_Comm c = MPI_Comm_f2c((MPI_Fint)comm);
  /* A null communicator is a caller error, not a collective refusal: no
   * collective is attempted.  All ranks must pass the identical valid handle. */
  if (c == MPI_COMM_NULL) return last_ = GC_E_ARG;
  impl_->comm = comm;

  /* A refusal here is a local verdict and must not return before the vote:
   * the other ranks wait for it in the Allreduce below.  The plan layer's
   * arithmetic throws rather than wrapping, and a throw becomes a refusal. */
  gc_status local = GC_OK;
  try {
    if (memory != Memory::device) local = GC_E_UNSUPPORTED;
    else if (faulted_) local = GC_E_STATE;
    else local = admit(plan, tag_base);
    if (local == GC_OK) {
      plan_ = &plan;
      local = describe(plan, &impl_->edesc);
    }
  } catch (const std::exception &e) {
    impl_->error = std::string("prepare: ") + e.what();
    local = GC_E_OVERFLOW;
  }

  gc_status s = vote_and_create(plan, local, impl_->edesc, tag_base,
                                deadline_ms, c, &impl_->plan);
  if (s != GC_OK) return fail(s);

  /* The packed buffers: ONE allocation per rank per direction, sliced per
   * edge at the plan's offsets.  The plan's edges tile that buffer, so the
   * slices are disjoint and cover it.
   *
   * A rank whose whole buffer is empty still gets one byte, so the transport
   * has a real pointer to compare.  A zero-length edge at the end of a
   * non-empty buffer gets a pointer one past its end, never dereferenced. */
  impl_->send.assign(plan.edge.size(), gcx_buffer());
  impl_->recv.assign(plan.edge.size(), gcx_buffer());
  gc_status alloc_status = GC_OK;
  const std::size_t send_alloc =
      (std::size_t)(plan.total_send_bytes == 0 ? 1 : plan.total_send_bytes);
  const std::size_t recv_alloc =
      (std::size_t)(plan.total_receive_bytes == 0 ? 1 : plan.total_receive_bytes);
  if (cudaMalloc(&impl_->send_base, send_alloc) != cudaSuccess ||
      cudaMalloc(&impl_->recv_base, recv_alloc) != cudaSuccess)
    alloc_status = GC_E_NOMEM;
  if (alloc_status == GC_OK)
    for (std::size_t i = 0; i < plan.edge.size(); ++i) {
      count_t off = 0, bytes = 0;
      packed_slice_bytes(plan.edge[i], plan.element_bytes, true, &off, &bytes);
      impl_->send[i].ptr = (char *)impl_->send_base + off;
      impl_->send[i].bytes = (gc_i64)plan.edge[i].send_bytes;
      packed_slice_bytes(plan.edge[i], plan.element_bytes, false, &off, &bytes);
      impl_->recv[i].ptr = (char *)impl_->recv_base + off;
      impl_->recv[i].bytes = (gc_i64)plan.edge[i].receive_bytes;
    }

  gc_i64 local_alloc = alloc_status == GC_OK ? 0 : 1;
  gc_i64 worst_alloc = 0;
  /* An allocation failure is local work after a collective admission, so it
   * is voted: a rank that cannot allocate stops every rank. */
  if (MPI_Allreduce(&local_alloc, &worst_alloc, 1, MPI_LONG_LONG, MPI_MAX,
                    c) != MPI_SUCCESS) {
    impl_->release_buffers();
    gcx_plan_destroy(impl_->plan);
    impl_->plan = 0;
    return fail(GC_E_STATE);
  }
  if (worst_alloc != 0) {
    impl_->release_buffers();
    gcx_plan_destroy(impl_->plan);
    impl_->plan = 0;
    return fail(GC_E_NOMEM);
  }

  /* Event creation is local work with the same rule: a failure stops every
   * rank through the vote below. */
  const bool events_ok =
      cudaEventCreateWithFlags(&impl_->producer, cudaEventDisableTiming) ==
          cudaSuccess &&
      cudaEventCreateWithFlags(&impl_->consumer, cudaEventDisableTiming) ==
          cudaSuccess;
  gc_i64 local_ev = events_ok ? 0 : 1, worst_ev = 0;
  if (MPI_Allreduce(&local_ev, &worst_ev, 1, MPI_LONG_LONG, MPI_MAX, c) !=
      MPI_SUCCESS) {
    impl_->release_events();
    impl_->release_buffers();
    gcx_plan_destroy(impl_->plan);
    impl_->plan = 0;
    return fail(GC_E_STATE);
  }
  if (worst_ev != 0) {
    impl_->release_events();
    impl_->release_buffers();
    gcx_plan_destroy(impl_->plan);
    impl_->plan = 0;
    return fail(GC_E_NOMEM);
  }

  impl_->direct.assign(plan.edge.size(), std::vector<Copy3D>());
  impl_->is_direct.assign(plan.edge.size(), 0);
  impl_->deferred.assign(plan.edge.size(), 0);
  impl_->local = true;
  for (std::size_t i = 0; i < plan.edge.size(); ++i) {
    const EdgePlan &e = plan.edge[i];
    /* Piece k of the pack and of the unpack must meet at the same place of
     * the packed layout, in the same shape. */
    bool same = e.self && e.pack.size() == e.unpack.size();
    for (std::size_t k = 0; k < e.pack.size() && same; ++k) {
      const Copy3D &pk = e.pack[k], &up = e.unpack[k];
      same = pk.dst_offset - e.send_offset == up.src_offset - e.receive_offset;
      for (int a = 0; a < 3 && same; ++a)
        same = pk.n[a] == up.n[a] && pk.dst_stride[a] == up.src_stride[a];
    }
    if (same) {
      for (std::size_t k = 0; k < e.pack.size(); ++k) {
        Copy3D c;
        for (int a = 0; a < 3; ++a) {
          c.n[a] = e.pack[k].n[a];
          c.src_stride[a] = e.pack[k].src_stride[a];
          c.dst_stride[a] = e.unpack[k].dst_stride[a];
        }
        c.src_offset = e.pack[k].src_offset;
        c.dst_offset = e.unpack[k].dst_offset;
        impl_->direct[i].push_back(c);
      }
      impl_->is_direct[i] = 1;
      impl_->send[i].bytes = 0;     /* the transport moves nothing to self */
    } else if (e.self || e.send_bytes != 0 || e.receive_bytes != 0) {
      impl_->local = false;
    }
  }
  impl_->fused = impl_->stores = false;
  if (!impl_->local) {
    /* The move engine's verdict on the remote edges; a self edge fuses only
     * as a direct copy or when it moves nothing. */
    gc_i32 fused = 0, stores = 0;
    impl_->dev.assign(plan.edge.size(), gcx_device_edge());
    gcx_move_edges(impl_->plan, &impl_->dev[0], &fused, &stores);
    impl_->fused = fused != 0;
    for (std::size_t i = 0; i < plan.edge.size(); ++i)
      if (plan.edge[i].self && !impl_->is_direct[i] &&
          (plan.edge[i].send_bytes || plan.edge[i].receive_bytes))
        impl_->fused = false;
    impl_->stores = stores != 0 && impl_->fused;
  }
  impl_->created = true;
  return last_ = GC_OK;
}

gc_status MeshExchangeSession::destroy()
{
  if (impl_ == 0) return GC_OK;
  if (impl_->token != 0) abort();
  if (impl_->plan != 0) {
    gcx_plan_destroy(impl_->plan);
    impl_->plan = 0;
  }
  impl_->release_events();
  impl_->release_buffers();
  impl_->created = false;
  impl_->source = 0;
  impl_->target = 0;
  return GC_OK;
}

gc_status MeshExchangeSession::attach(const void *source, count_t source_bytes,
                                      void *target, count_t target_bytes)
{
  if (!impl_->created) return last_ = GC_E_STATE;
  if (source == 0 || target == 0) return last_ = GC_E_ARG;
  /* The plan states what the two pencils must hold; a caller that hands over
   * something smaller is refused here rather than read past. */
  if (source_bytes < plan_->source_bytes || target_bytes < plan_->target_bytes)
    return last_ = GC_E_CAPACITY;
  impl_->source = source;
  impl_->source_bytes = source_bytes;
  impl_->target = target;
  impl_->target_bytes = target_bytes;
  impl_->built = false;
  return last_ = GC_OK;
}

gc_status MeshExchangeSession::begin_epoch(gc_i64 epoch, void *stream)
{
  if (!impl_->created) return last_ = GC_E_STATE;
  if (impl_->source == 0 || impl_->target == 0) return last_ = GC_E_STATE;
  cudaStream_t cs = (cudaStream_t)stream;
  if (cs == 0) return last_ = GC_E_ARG;
  if (faulted_) return last_ = GC_E_STATE;
  if (epoch <= epoch_) return last_ = GC_E_EPOCH;
  epoch_ = epoch;

  const ExchangePlan &plan = *plan_;
  if (impl_->live()) {
    if (!impl_->built && !impl_->build_fused(plan)) {
      impl_->error = "fused descriptors";
      return fail(GC_E_NOMEM);
    }
    const int n = impl_->nsig;
    /* The engine claims the epoch and its slot, and checks a captured move
     * against the sequence its launches will advance on the device. */
    gcx_move_open o;
    gc_status s = gcx_move_claim(impl_->plan, epoch, stream, &o);
    if (s != GC_OK) {
      impl_->refused("gcx_move_claim", s);
      return fail(s);
    }
    const int slot = o.slot;
    const unsigned long long stride = o.stride;
    cudaError_t e = cudaSuccess;
    /* Merged (copy engines): the pack opens the epoch in its last block, as
     * its slices leave only after it. */
    const bool merge = impl_->merged() && n > 0;
    CopyTail pt = CopyTail();
    if (merge) {
      pt.kind = kTailOpen;
      pt.n = n;
      pt.f = impl_->d_sig;
      pt.count = impl_->d_count;
      pt.q = impl_->d_seq;
      pt.epoch = (unsigned long long)epoch;
      pt.prev = (unsigned long long)o.prev;
      pt.stride = stride;
    }
    if (!merge)
      S4_LAUNCH(k_sig_open_many, 1, kThreadsPerBlock, cs)(
          impl_->d_sig, n, impl_->d_seq, (unsigned long long)epoch,
          (unsigned long long)o.prev, stride);
    e = cudaGetLastError();
    if (e == cudaSuccess)
      e = launch_many(impl_->pack_desc(), impl_->npack, impl_->most_pack,
                      plan.element_bytes, impl_->d_seq, cs, pt);
    /* the copy engines move each packed slice into its peer's slot */
    for (std::size_t i = 0; i < plan.edge.size() && e == cudaSuccess &&
                            !impl_->stores; ++i) {
      const gcx_device_edge &g = impl_->dev[i];
      const gcx_buffer &b = impl_->send[i];
      if (plan.edge[i].self || b.bytes <= 0) continue;
      if (b.bytes > g.send_capacity) {
        impl_->error = "packed slice larger than the peer's slot";
        return fail(GC_E_CAPACITY);
      }
      e = cudaMemcpyAsync(g.peer_slots + (gc_i64)slot * g.send_capacity, b.ptr,
                          (std::size_t)b.bytes, cudaMemcpyDefault, cs);
    }
    if (e == cudaSuccess && n > 0 && !merge) {
      S4_LAUNCH(k_sig_set_many, 1, kThreadsPerBlock, cs)(
          impl_->d_sig + n, n, impl_->d_seq);
      e = cudaGetLastError();
    } else if (e == cudaSuccess && merge && stride == 0 && batch_memop() &&
               (int)impl_->h_rdys.size() == n && n < 256) {
      /* the ready words as memory operations after the copies; the unpack
       * waits for the peers' (wait()) */
      CUstreamBatchMemOpParams op[256];
      std::memset(op, 0, sizeof(CUstreamBatchMemOpParams) * (std::size_t)n);
      for (int i = 0; i < n; ++i) {
        op[i].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
        op[i].writeValue.address = (CUdeviceptr)impl_->h_rdys[(std::size_t)i];
        op[i].writeValue.value64 = (cuuint64_t)epoch;
        op[i].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      }
      if (batch_memop()((CUstream)cs, (unsigned)n, op, 0) != CUDA_SUCCESS)
        e = cudaErrorUnknown;
      impl_->cmerge_now = true;
    } else if (e == cudaSuccess && merge) {
      /* the ready words, then the wait for the peers' (wait() skips it) */
      S4_LAUNCH(k_sig_set_wait_many, 1, kThreadsPerBlock, cs)(
          impl_->d_sig + n, impl_->d_sig + 2 * n, n, impl_->d_seq);
      e = cudaGetLastError();
    }
    if (e != cudaSuccess) {
      impl_->error = cuda_message("fused pack", e);
      return fail(GC_E_DEVICE);
    }
    return last_ = GC_OK;
  }
  for (std::size_t i = 0; i < plan.edge.size(); ++i) {
    /* The copy addresses the rank's whole packed buffer and lands on its own
     * edge's slice through the copy's dst_offset.  The slice pointer is the
     * transport's only: passing it here too would apply the offset twice. */
    const bool direct = impl_->is_direct[i] != 0;
    if (direct && impl_->deferred[i]) continue;
    for (const Copy3D &c : direct ? impl_->direct[i] : plan.edge[i].pack) {
      const cudaError_t e = direct
          ? launch_copy(c, impl_->source, impl_->target, plan.element_bytes,
                        plan.accumulate, cs)
          : launch_copy(c, impl_->source, impl_->send_base, plan.element_bytes,
                        false, cs);
      if (e != cudaSuccess) {
        impl_->error = cuda_message("pack", e);
        return fail(GC_E_DEVICE);
      }
    }
  }
  /* Recorded after the last pack, before the transport is told to wait. */
  if (const cudaError_t e = cudaEventRecord(impl_->producer, cs)) {
    impl_->error = cuda_message("producer event", e);
    return fail(GC_E_DEVICE);
  }
  if (impl_->local) return last_ = GC_OK;

  /* Post: the transport is told the buffers are visible, and it posts its
   * receives, its headers and its dispatch. */
  const std::size_t n = plan.edge.size();

  gcx_op_desc op;
  std::memset(&op, 0, sizeof op);
  op.epoch = epoch;
  op.op = plan.op;
  /* A fixed-size schedule: the capacities are exact, so the plan pays no
   * count round trip.  See gcx_op_desc.variable in the ABI header. */
  op.variable = 0;
  op.send = n ? &impl_->send[0] : 0;
  op.recv = n ? &impl_->recv[0] : 0;
  op.send_bytes = 0;      /* the buffer's own bytes ARE the schedule's */
  op.send_records = 0;
  op.recv_bytes = 0;
  op.recv_records = 0;
  /* The transport stream is ordered after this event, so it is the pack that
   * publishes the send buffers -- never an implicit device sync. */
  op.producer_event = (void *)impl_->producer;

  const gc_status s = gcx_begin(impl_->plan, &op, &impl_->token);
  if (s != GC_OK) {
    impl_->refused("gcx_begin", s);
    impl_->token = 0;    /* a refused begin returns no token */
    return fail(s);
  }
  return last_ = GC_OK;
}

gc_status MeshExchangeSession::wait(void *stream)
{
  cudaStream_t cs = (cudaStream_t)stream;
  if (impl_->live()) {
    const int n = impl_->nsig;
    const bool merge = impl_->merged() && n > 0;
    if (n > 0 && !merge)
      S4_LAUNCH(k_sig_wait_many, 1, kThreadsPerBlock, cs)(
          impl_->d_sig + 2 * n, n, impl_->d_seq);
    cudaError_t e = cudaGetLastError();
    /* merged: the unpack's last block sets the acknowledgements */
    CopyTail ut = CopyTail();
    if (merge) {
      ut.kind = kTailSet;
      ut.n = n;
      ut.f = impl_->d_sig + 3 * n;
      ut.count = impl_->d_count + 1;
      ut.q = impl_->d_seq;
    }
    if (merge && impl_->cmerge_now) {
      /* every block waits for the peers' ready words before it reads */
      ut.head_n = n;
      ut.head_f = impl_->d_sig + 2 * n;
      ut.head_v = (unsigned long long)epoch_;
    }
    if (e == cudaSuccess)
      e = launch_many(impl_->unpack_desc(), impl_->nunpack,
                      impl_->most_unpack, plan_->element_bytes,
                      impl_->d_seq, cs, ut);
    if (e != cudaSuccess) {
      impl_->error = cuda_message("fused unpack", e);
      return fail(GC_E_DEVICE);
    }
    impl_->consumer_stream = cs;
    return last_ = GC_OK;
  }

  /* gcx_consume drives the transport to completion under the plan's own
   * deadline, validates every header, and leaves `cs` ordered after the
   * payload.  It is the only place a peer's message can be refused. */
  const gc_status s = impl_->local
      ? (cudaStreamWaitEvent(cs, impl_->producer, 0) == cudaSuccess ? GC_OK : GC_E_DEVICE)
      : gcx_consume(impl_->token, stream);
  if (s != GC_OK) {
    impl_->refused(impl_->local ? "local wait" : "gcx_consume", s);
    /* A refused operation leaves messages the peer will never match.  Drop
     * them, stop the session, and leave the decision to the caller. */
    gcx_abort(impl_->token);
    impl_->token = 0;
    return fail(s);
  }

  const ExchangePlan &plan = *plan_;
  for (std::size_t i = 0; i < plan.edge.size(); ++i) {
    if (impl_->is_direct[i]) continue;
    /* As in the pack: the rank's whole receive buffer is the source and the
     * edge's slice is reached through the copy's src_offset. */
    for (const Copy3D &c : plan.edge[i].unpack) {
      const cudaError_t e = launch_copy(c, impl_->recv_base, impl_->target,
                                        plan.element_bytes, plan.accumulate, cs);
      if (e != cudaSuccess) {
        impl_->error = cuda_message("unpack", e);
        return fail(GC_E_DEVICE);
      }
    }
  }
  /* The acknowledgement may not leave before the consumer is done reading. */
  if (const cudaError_t e = cudaEventRecord(impl_->consumer, cs)) {
    impl_->error = cuda_message("consumer event", e);
    return fail(GC_E_DEVICE);
  }
  return last_ = GC_OK;
}

gc_status MeshExchangeSession::release()
{
  if (impl_->local) return last_ = GC_OK;
  if (impl_->live()) {
    const int n = impl_->nsig;
    if (n > 0 && !impl_->merged()) {
      S4_LAUNCH(k_sig_set_many, 1, kThreadsPerBlock, impl_->consumer_stream)(
          impl_->d_sig + 3 * n, n, impl_->d_seq);
      if (cudaGetLastError() != cudaSuccess) return fail(GC_E_DEVICE);
    }
    return last_ = GC_OK;
  }
  if (impl_->token == 0) return last_ = GC_E_STATE;
  const gc_status s = gcx_release(impl_->token, (void *)impl_->consumer);
  impl_->token = 0;
  if (s != GC_OK) {
    impl_->refused("gcx_release", s);
    return fail(s);
  }
  return last_ = GC_OK;
}

bool MeshExchangeSession::defer_self(std::vector<Copy3D> *copies,
                                     std::size_t most)
{
  if (!impl_->created || epoch_ != 0 || !plan_->accumulate) return false;
  const ExchangePlan &plan = *plan_;
  for (std::size_t i = 0; i < plan.edge.size(); ++i) {
    if (!plan.edge[i].self || !impl_->is_direct[i] || impl_->deferred[i]) continue;
    if (impl_->direct[i].size() > most) return false;
    impl_->deferred[i] = 1;
    impl_->built = false;
    if (copies) *copies = impl_->direct[i];
    return true;
  }
  return false;
}

/* begin_epoch, wait and release on one stream, with nothing between them, so
 * a copy-engine move's signal launches ride on its copies (Impl::merged). */
gc_status MeshExchangeSession::move(gc_i64 epoch, void *stream)
{
  gc_status s = begin_epoch(epoch, stream);
  if (s == GC_OK) s = wait(stream);
  if (s == GC_OK) s = release();
  impl_->cmerge_now = false;
  return s;
}

bool MeshExchangeSession::steady(gc_i64 epoch) const
{
  if (!impl_->created || faulted_) return false;
  if (impl_->local) return true;
  return impl_->live() && impl_->built &&
         gcx_move_steady(impl_->plan, epoch, 0) != 0;
}

bool MeshExchangeSession::copy_engine_candidate() const
{
  return impl_->created && impl_->fused && !impl_->stores;
}

void MeshExchangeSession::allow_copy_engine(bool on)
{
  impl_->ce_on = on;
}

void MeshExchangeSession::graph_capability(bool *fused, bool *stores) const
{
  *fused = impl_->created && (impl_->local || impl_->fused);
  *stores = impl_->created && (impl_->local || impl_->stores);
}

bool MeshExchangeSession::slot_bound() const
{
  return impl_->created && impl_->live() && !impl_->stores;
}

/* A replayed step's move: the host side of move(), the engine's claim of the
 * epoch, without its launches, which the step's graph holds.  Only a steady
 * sequence is replayed. */
gc_status MeshExchangeSession::replay_epoch(gc_i64 epoch)
{
  if (!steady(epoch)) return last_ = GC_E_STATE;
  if (epoch <= epoch_) return fail(GC_E_EPOCH);
  epoch_ = epoch;
  if (impl_->live() && gcx_move_replay(impl_->plan, epoch) != GC_OK)
    return fail(GC_E_STATE);
  return last_ = GC_OK;
}

void MeshExchangeSession::abort()
{
  if (impl_->token != 0) {
    gcx_abort(impl_->token);
    impl_->token = 0;
  }
  faulted_ = true;
  last_ = GC_OK;
}

const char *MeshExchangeSession::last_error() const
{
  return impl_ == 0 ? "" : impl_->error.c_str();
}

} /* namespace genesis_native_s4 */
