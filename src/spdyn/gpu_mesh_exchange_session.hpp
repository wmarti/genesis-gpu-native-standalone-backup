/*
 * gpu_mesh_exchange_session.hpp : transport half of the mesh exchange.
 *
 * One session owns one exchange plan for its whole life: the GCX plan and its
 * edges, the packed send and receive buffers, the two events that order the
 * compute stream against the transport stream, and the epoch the buffers
 * hold.  A caller drives it with one call per exchange, move(e, stream):
 * the epoch opens, the buffers are packed and posted, the received payload
 * is unpacked on `stream`, and the peers are told the slot may be reused.
 *
 * A device-sequenced plan moves without the transport: the plan's move
 * engine (gcx_move_claim in gpu_core_xchg.h) keeps the epochs and slots, and
 * the session launches the copies and signal words.  Every refusal faults
 * the session, so a caller that ignores the error cannot enter the next
 * epoch with messages outstanding.
 *
 * Collective refusal: the transport answers a Round A capacity disagreement
 * with a local return, which would leave the peer waiting.  `create` therefore
 * asks the Round A question first with an Alltoall over the communicator, so
 * one disagreeing pair stops every rank (see vote_and_create in the .cu).
 *
 * The session owns the packed buffers only; the pencil buffers and their
 * layout belong to the caller and gpu_layout_schedule.hpp.  `gcx_*` is used as the
 * ABI declares it: the session adds buffer ownership and pack/unpack
 * dispatch, not routes, tags, slots or acknowledgement.
 */
#ifndef GENESIS_NATIVE_S4_MESH_EXCHANGE_SESSION_HPP
#define GENESIS_NATIVE_S4_MESH_EXCHANGE_SESSION_HPP

#include "gpu_core_xchg.h"

#include "gpu_mesh_exchange_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace genesis_native_s4 {

/* The largest rank index a plan may name: a peer has to survive narrowing to
 * the descriptor's gc_i32 peer field and index the per-rank capacity
 * vectors. */
const count_t kMaxRankIndex = (count_t)0x7fffffff;

/* count_t is unsigned, so a caller's -1 arrives near UINT64_MAX and
 * kMaxRankIndex refuses it; a written-out `e.peer < 0` would be dead code
 * (an always-false unsigned comparison under -Wextra/-Werror).  If count_t
 * becomes signed this assertion fails and the bound needs the `< 0` test. */
static_assert(!std::numeric_limits<count_t>::is_signed,
              "count_t is unsigned; admit()'s peer bound relies on it "
              "and must gain an explicit negative-peer test if that changes");

/* Where one edge's packed bytes live inside this rank's packed buffer.
 *
 * There is ONE packed send buffer and ONE packed receive buffer per rank.
 * The plan's edges tile them in order (checked by the plan layer), so the
 * slices are disjoint and cover the buffer exactly.  A separate allocation
 * per edge would make every offset past the first address memory outside its
 * own edge.  The session allocates from this function and the transport is
 * handed its result, so the rule has one statement. */
inline void packed_slice_bytes(const EdgePlan &e, count_t element_bytes,
                               bool sending, count_t *offset_bytes,
                               count_t *bytes)
{
  *offset_bytes = pencil::checked_mul(
      sending ? e.send_offset : e.receive_offset, element_bytes);
  *bytes = sending ? e.send_bytes : e.receive_bytes;
}

/* The copy kernel takes no offsets: the caller pre-offsets BOTH pointers,
 * in elements, or a copy with a non-zero src_offset/dst_offset lands at the
 * buffer's origin instead of at the patch.  The device launch uses this
 * function and the host gate checks it. */
template <typename T>
inline void copy_operands(const Copy3D &c, const void *src, void *dst,
                          const T **src_out, T **dst_out)
{
  *src_out = reinterpret_cast<const T *>(src) +
             (std::ptrdiff_t)c.src_offset;
  *dst_out = reinterpret_cast<T *>(dst) + (std::ptrdiff_t)c.dst_offset;
}

class MeshExchangeSession {
public:
  MeshExchangeSession();
  ~MeshExchangeSession();
  MeshExchangeSession(const MeshExchangeSession &) = delete;
  MeshExchangeSession &operator=(const MeshExchangeSession &) = delete;

  /* Collective over `comm`.  Every rank votes, in order: its local verdict,
   * the identity of the plan it is creating, and the per-peer capacity
   * agreement Round A would check.  A failure of any is returned on every
   * rank.  On success the session owns its packed buffers and two events.
   *
   * Every rank calls create exactly once per session object, with the same
   * valid communicator and the same plan of the sequence.  A null
   * communicator is a local E_ARG; a second create on a live session is
   * E_STATE; neither may happen asymmetrically.
   *
   * LIFETIME: the plan must outlive the session; its edge geometry is read
   * on every exchange and never copied, so pass a named object. */
  gc_status create(const ExchangePlan &plan, Memory memory, gcx_comm comm,
                   gc_i32 tag_base, gc_i64 deadline_ms);

  /* Collective over `comm`; safe to call twice, and safe after a failure. */
  gc_status destroy();

  /* The caller's two pencil buffers: `source` is read by every pack,
   * `target` written by every unpack.  The byte counts the plan requires are
   * checked before either pointer is kept. */
  gc_status attach(const void *source, count_t source_bytes, void *target,
                   count_t target_bytes);

  /* One exchange on `stream`: the epoch opens (epochs increase), the packs
   * run and the payload is posted, the received payload is unpacked, and the
   * peers are told the slot may be reused.  A refusal faults the session. */
  gc_status move(gc_i64 epoch, void *stream);
  /* An accumulating plan only, after create() and before the first move: the
   * self edge's copies (source -> target, offsets in elements) are left to
   * the caller, who adds them where it reads the target.  Returns false and
   * changes nothing when there is no such edge or it has more than `most`
   * copies. */
  bool defer_self(std::vector<Copy3D> *copies, std::size_t most);
  /* A step captured in a CUDA graph and replayed: `steady` says whether the
   * move opening `epoch` may be captured (its launches then advance the epoch
   * on the device); `replay_epoch` is that move's host side at replay. */
  bool steady(gc_i64 epoch) const;
  gc_status replay_epoch(gc_i64 epoch);
  /* A fused move whose copy engines address the peers' slots from the host:
   * a captured step replays only on epochs of the same slot (epoch %
   * GCX_SLOTS). */
  bool slot_bound() const;
  /* A fused move that copies with the copy engines (not SM stores) runs
   * fused only while allowed, otherwise through the transport; call
   * between moves. */
  bool copy_engine_candidate() const;
  void allow_copy_engine(bool on);
  /* Fixed at create: whether the move can run fused (or moves nothing
   * through the transport), and whether it then stores into the peers'
   * slots (else it needs the copy engines' admission). */
  void graph_capability(bool *fused, bool *stores) const;

  /* The device-side detail of the last refusal, or an empty string. */
  const char *last_error() const;

private:
  gc_status begin_epoch(gc_i64 epoch, void *stream);   /* pack + post */
  gc_status wait(void *stream);                        /* unpack */
  gc_status release();
  void abort();
  gc_status fail(gc_status s)
  {
    faulted_ = true;
    return last_ = s;
  }

  struct Impl;
  Impl *impl_;
  const ExchangePlan *plan_;
  gc_i64 epoch_;          /* the last epoch opened */
  bool faulted_;          /* a refusal stops the session */
  gc_status last_;
};

inline const char *mesh_session_state_name(gc_status s)
{
  switch (s) {
  case GC_OK:            return "GC_OK";
  case GC_E_ARG:         return "GC_E_ARG";
  case GC_E_ABI:         return "GC_E_ABI";
  case GC_E_CAPACITY:    return "GC_E_CAPACITY";
  case GC_E_UNSUPPORTED: return "GC_E_UNSUPPORTED";
  case GC_E_DEVICE:      return "GC_E_DEVICE";
  case GC_E_NOMEM:       return "GC_E_NOMEM";
  case GC_E_OVERFLOW:    return "GC_E_OVERFLOW";
  case GC_E_MISMATCH:    return "GC_E_MISMATCH";
  case GC_E_EPOCH:       return "GC_E_EPOCH";
  case GC_E_OWNER:       return "GC_E_OWNER";
  case GC_E_ENDPOINT:    return "GC_E_ENDPOINT";
  case GC_E_ARITY:       return "GC_E_ARITY";
  case GC_E_STATE:       return "GC_E_STATE";
  }
  return "GC_E_?";
}

} /* namespace genesis_native_s4 */

#endif /* GENESIS_NATIVE_S4_MESH_EXCHANGE_SESSION_HPP */
