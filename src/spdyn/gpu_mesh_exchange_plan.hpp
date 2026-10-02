/*
 * gpu_mesh_exchange_plan.hpp : layout-to-transport half of the mesh exchange.
 *
 * A reciprocal-space pencil decomposition moves data three times: between the
 * brick partition (spread/gather) and the X-real pencil, and twice between
 * the three pencil orientations.  This header turns one such move, a
 * PencilLayout schedule, into the edge set and per-edge copy geometry the
 * shared exchange ABI consumes.  It is free of MPI, CUDA, GENESIS state and
 * allocation.
 *
 * It owns the plan's edges (with the byte capacities the transport checks),
 * the pack/unpack geometry of every edge as strided 3-D copies over the
 * caller's own pencil buffers, and the checked byte/count/offset arithmetic.
 * The transport, FFT and solve are not decided here.
 *
 * Pencil storage (fastest axis last):
 *   X complex: [z][y][x], x extent Nxh, y extent Ny, z local;
 *   Y complex: [z][x][y], x extent Nxh, y extent Ny, z local;
 *   Z complex: [y][x][z], z extent Nz, x extent Nxh, y local.
 * The layout's `local_offset` is a canonical [z][y][x] offset: the physical
 * offset for the X pencils, the brick and the X-real buffers, but not for Y
 * and Z.  The Y and Z offsets are derived from the layout's BlockSplits and
 * cross-checked against its element count for each region.
 */
#ifndef GENESIS_NATIVE_S4_MESH_EXCHANGE_PLAN_HPP
#define GENESIS_NATIVE_S4_MESH_EXCHANGE_PLAN_HPP

#include "gpu_core_xchg.h"

#include "gpu_layout_schedule.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace genesis_native_s4 {

namespace pencil = genesis_native_pencil;
using count_t = pencil::count_t;

/* `gc_i64` is what the ABI carries: every value reaching a descriptor is
 * converted once, with its range checked. */
inline gc_i64 to_gc_i64(count_t value, const char *what)
{
  if (value > static_cast<count_t>(std::numeric_limits<gc_i64>::max()))
    throw std::overflow_error(std::string("mesh exchange: ") + what + " exceeds gc_i64");
  return static_cast<gc_i64>(value);
}

/* Which memory the plan's packed buffers live in; the session accepts
 * device memory only. */
enum class Memory { device, host };

/* A strided 3-D copy: for i0 in [0,n0), i1 in [0,n1), i2 in [0,n2),
 *
 *   dst[dst_offset + i0*dst_stride0 + i1*dst_stride1 + i2*dst_stride2]
 *     = src[src_offset + i0*src_stride0 + i1*src_stride1 + i2*src_stride2]
 *
 * in elements, i0 slowest.  A copy whose largest touched offset does not fit
 * in the buffer it names is refused by span(). */
struct Copy3D {
  count_t n[3] = {0, 0, 0};
  count_t src_stride[3] = {0, 0, 0};
  count_t dst_stride[3] = {0, 0, 0};
  count_t src_offset = 0;
  count_t dst_offset = 0;

  count_t elements() const
  {
    return pencil::checked_mul(pencil::checked_mul(n[0], n[1]), n[2]);
  }

  bool empty() const { return n[0] == 0 || n[1] == 0 || n[2] == 0; }

  /* One past the last element this copy touches, on the source side. */
  count_t src_span() const
  {
    if (empty())
      return src_offset;
    count_t last = 0;
    for (int a = 0; a < 3; ++a)
      last = pencil::checked_add(last,
                                 pencil::checked_mul(n[a] - 1, src_stride[a]));
    return pencil::checked_add(src_offset, pencil::checked_add(last, 1));
  }

  count_t dst_span() const
  {
    if (empty())
      return dst_offset;
    count_t last = 0;
    for (int a = 0; a < 3; ++a)
      last = pencil::checked_add(last,
                                 pencil::checked_mul(n[a] - 1, dst_stride[a]));
    return pencil::checked_add(dst_offset, pencil::checked_add(last, 1));
  }
};

/* One directed edge of a fixed-size schedule: the bytes this rank sends to
 * `peer` and receives from it, and the copies that fill and drain this
 * rank's packed buffers, one per piece (a transpose edge is one box, a brick
 * edge up to eight because the brick may wrap).  The pieces of an edge lie
 * back to back in its packed slice, so a packed offset is a plain element
 * index and `send_offset + send_elements` is the send buffer's length. */
struct EdgePlan {
  count_t peer = 0;
  bool self = false;
  count_t send_elements = 0;
  count_t receive_elements = 0;
  count_t send_bytes = 0;      /* the edge's send capacity    */
  count_t receive_bytes = 0;   /* the edge's receive capacity */
  count_t send_offset = 0;
  count_t receive_offset = 0;
  std::vector<Copy3D> pack;    /* caller's source buffer -> packed send buffer */
  std::vector<Copy3D> unpack;  /* packed receive buffer -> caller's target buffer */
};

struct ExchangePlan {
  gc_i32 op = GCX_OP_MESH;
  Memory memory = Memory::device;
  count_t element_bytes = 0;
  /* The unpack ADDS each element into the target instead of storing it: the
   * elements are integer words of fixed-point sums, which add exactly in any
   * order, and several edges land on one target cell (brick -> X-real). */
  bool accumulate = false;
  /* What the caller's two pencil buffers must hold: the layout's own region
   * sizes. */
  count_t source_elements = 0;
  count_t target_elements = 0;
  count_t source_bytes = 0;
  count_t target_bytes = 0;
  std::vector<EdgePlan> edge;
  count_t total_send_elements = 0;
  count_t total_receive_elements = 0;
  count_t total_send_bytes = 0;
  count_t total_receive_bytes = 0;
  count_t active_peer_count = 0;

  count_t max_send_bytes() const
  {
    count_t m = 0;
    for (const EdgePlan &e : edge)
      m = e.send_bytes > m ? e.send_bytes : m;
    return m;
  }

  count_t max_receive_bytes() const
  {
    count_t m = 0;
    for (const EdgePlan &e : edge)
      m = e.receive_bytes > m ? e.receive_bytes : m;
    return m;
  }

  bool has_self_edge() const
  {
    for (const EdgePlan &e : edge)
      if (e.self) return true;
    return false;
  }

  bool has_zero_length_edge() const
  {
    for (const EdgePlan &e : edge)
      if (e.send_elements == 0 || e.receive_elements == 0) return true;
    return false;
  }
};

inline count_t part_offset(const pencil::BlockSplit &s, count_t part)
{
  if (part >= s.offset.size())
    throw std::out_of_range("mesh exchange: block split part");
  return s.offset[static_cast<std::size_t>(part)];
}

inline count_t part_length(const pencil::BlockSplit &s, count_t part)
{
  if (part >= s.length.size())
    throw std::out_of_range("mesh exchange: block split part");
  return s.length[static_cast<std::size_t>(part)];
}

/* The region of a canonical [z][y][x] buffer that one copy moves, as
 * strides over that buffer.  Used only where the canonical order is also the
 * physical order. */
inline void canonical_strides(const pencil::Region &region, count_t row,
                              count_t *stride)
{
  stride[0] = pencil::checked_mul(region.ny, row);
  stride[1] = row;
  stride[2] = 1;
}

/* Build the exchange plan of one pencil transpose.  The send and receive
 * counts are the layout schedule's, and the geometry is checked element for
 * element against it. */
inline ExchangePlan build_transpose_plan(const pencil::PencilLayout &layout,
                                         count_t rank,
                                         pencil::Transpose transpose,
                                         pencil::Direction direction,
                                         count_t element_bytes,
                                         Memory memory)
{
  if (element_bytes == 0)
    throw std::invalid_argument("mesh exchange: element size must be positive");
  const pencil::TransposeSchedule s =
      layout.schedule(rank, transpose, direction, element_bytes);
  ExchangePlan p;
  p.memory = memory;
  p.element_bytes = element_bytes;
  const pencil::ProcessGrid::Coord c = layout.grid().coord(layout.grid_rank(rank));
  const count_t nxh = layout.nx_half();
  const count_t ny = layout.mesh().ny;
  const count_t nz = layout.mesh().nz;

  const pencil::Region x_reg = layout.x_complex(rank);
  const pencil::Region y_reg = layout.y_complex(rank);
  const pencil::Region z_reg = layout.z_complex(rank);
  const bool forward = direction == pencil::Direction::forward;
  const bool x_to_y = transpose == pencil::Transpose::x_to_y;
  p.source_elements = forward ? (x_to_y ? x_reg.elements() : y_reg.elements())
                              : (x_to_y ? y_reg.elements() : z_reg.elements());
  p.target_elements = forward ? (x_to_y ? y_reg.elements() : z_reg.elements())
                              : (x_to_y ? x_reg.elements() : y_reg.elements());
  p.source_bytes = pencil::checked_mul(p.source_elements, element_bytes);
  p.target_bytes = pencil::checked_mul(p.target_elements, element_bytes);

  p.edge.reserve(s.segment.size());
  for (std::size_t i = 0; i < s.segment.size(); ++i) {
    const pencil::Segment &seg = s.segment[i];
    const count_t m = static_cast<count_t>(i);
    EdgePlan e;
    e.peer = seg.peer;
    e.self = seg.peer == rank;
    e.send_elements = seg.send_elements;
    e.receive_elements = seg.receive_elements;
    e.send_bytes = seg.send_bytes;
    e.receive_bytes = seg.receive_bytes;
    e.send_offset = seg.send_offset;
    e.receive_offset = seg.receive_offset;
    Copy3D pk, up;

    if (x_to_y) {
      /* Peer m owns rows yA[m] of both pencils; this rank owns columns
       * xhA[a] of the X complex and rows yA[a] of the Y complex. */
      const count_t nxa_m = part_length(layout.xh_by_a(), m);
      const count_t nya_m = part_length(layout.y_by_a(), m);
      const count_t nxh_a = part_length(layout.xh_by_a(), c.a);
      const count_t nya_a = part_length(layout.y_by_a(), c.a);
      const count_t nzb = part_length(layout.z_by_b(), c.b);

      if (forward) {
        /* X complex [z][y][x] -> packed send buffer */
        pk.n[0] = nzb;
        pk.n[1] = nya_a;
        pk.n[2] = nxa_m;
        canonical_strides(pencil::Region{0, 0, 0, nxh, nya_a, nzb}, nxh,
                          pk.src_stride);
        pk.src_offset = part_offset(layout.xh_by_a(), m);
        canonical_strides(pencil::Region{0, 0, 0, nxa_m, nya_a, nzb}, nxa_m,
                          pk.dst_stride);
        pk.dst_offset = seg.send_offset;
        /* packed receive buffer -> Y complex [z][x][y] */
        up.n[0] = nzb;
        up.n[1] = nya_m;
        up.n[2] = nxh_a;
        canonical_strides(pencil::Region{0, 0, 0, nxh_a, nya_m, nzb}, nxh_a,
                          up.src_stride);
        up.src_offset = seg.receive_offset;
        up.dst_stride[0] = pencil::checked_mul(nxh_a, ny);
        up.dst_stride[1] = 1;
        up.dst_stride[2] = ny;
        up.dst_offset = part_offset(layout.y_by_a(), m);
      } else {
        /* Y complex [z][x][y] -> packed send buffer */
        pk.n[0] = nzb;
        pk.n[1] = nya_m;
        pk.n[2] = nxh_a;
        pk.src_stride[0] = pencil::checked_mul(nxh_a, ny);
        pk.src_stride[1] = 1;
        pk.src_stride[2] = ny;
        pk.src_offset = part_offset(layout.y_by_a(), m);
        canonical_strides(pencil::Region{0, 0, 0, nxh_a, nya_m, nzb}, nxh_a,
                          pk.dst_stride);
        pk.dst_offset = seg.send_offset;
        /* packed receive buffer -> X complex [z][y][x] */
        up.n[0] = nzb;
        up.n[1] = nya_a;
        up.n[2] = nxa_m;
        canonical_strides(pencil::Region{0, 0, 0, nxa_m, nya_a, nzb}, nxa_m,
                          up.src_stride);
        up.src_offset = seg.receive_offset;
        canonical_strides(pencil::Region{0, 0, 0, nxh, nya_a, nzb}, nxh,
                          up.dst_stride);
        up.dst_offset = part_offset(layout.xh_by_a(), m);
      }
    } else {
      /* Peer m owns columns zB[m] of both pencils; this rank owns rows
       * yB[b] of the Y complex and columns zB[b] of the Z complex. */
      const count_t nyb_m = part_length(layout.y_by_b(), m);
      const count_t nzb_m = part_length(layout.z_by_b(), m);
      const count_t nxh_a = part_length(layout.xh_by_a(), c.a);
      const count_t nyb_b = part_length(layout.y_by_b(), c.b);
      const count_t nzb = part_length(layout.z_by_b(), c.b);

      if (forward) {
        /* Y complex [z][x][y] -> packed send buffer */
        pk.n[0] = nzb;
        pk.n[1] = nxh_a;
        pk.n[2] = nyb_m;
        pk.src_stride[0] = pencil::checked_mul(nxh_a, ny);
        pk.src_stride[1] = ny;
        pk.src_stride[2] = 1;
        pk.src_offset = part_offset(layout.y_by_b(), m);
        canonical_strides(pencil::Region{0, 0, 0, nyb_m, nxh_a, nzb}, nyb_m,
                          pk.dst_stride);
        pk.dst_offset = seg.send_offset;
        /* packed receive buffer -> Z complex [y][x][z] */
        up.n[0] = nzb_m;
        up.n[1] = nxh_a;
        up.n[2] = nyb_b;
        canonical_strides(pencil::Region{0, 0, 0, nyb_b, nxh_a, nzb_m}, nyb_b,
                          up.src_stride);
        up.src_offset = seg.receive_offset;
        up.dst_stride[0] = 1;
        up.dst_stride[1] = nz;
        up.dst_stride[2] = pencil::checked_mul(nxh_a, nz);
        up.dst_offset = part_offset(layout.z_by_b(), m);
      } else {
        /* Z complex [y][x][z] -> packed send buffer */
        pk.n[0] = nzb_m;
        pk.n[1] = nxh_a;
        pk.n[2] = nyb_b;
        pk.src_stride[0] = 1;
        pk.src_stride[1] = nz;
        pk.src_stride[2] = pencil::checked_mul(nxh_a, nz);
        pk.src_offset = part_offset(layout.z_by_b(), m);
        canonical_strides(pencil::Region{0, 0, 0, nyb_b, nxh_a, nzb_m}, nyb_b,
                          pk.dst_stride);
        pk.dst_offset = seg.send_offset;
        /* packed receive buffer -> Y complex [z][x][y] */
        up.n[0] = nzb;
        up.n[1] = nxh_a;
        up.n[2] = nyb_m;
        canonical_strides(pencil::Region{0, 0, 0, nyb_m, nxh_a, nzb}, nyb_m,
                          up.src_stride);
        up.src_offset = seg.receive_offset;
        up.dst_stride[0] = pencil::checked_mul(nxh_a, ny);
        up.dst_stride[1] = ny;
        up.dst_stride[2] = 1;
        up.dst_offset = part_offset(layout.y_by_b(), m);
      }
    }

    /* The geometry must move exactly what the schedule says the edge
     * carries, and the packed buffers must tile [0, total) with no gap or
     * overlap. */
    if (pk.elements() != e.send_elements ||
        up.elements() != e.receive_elements)
      throw std::logic_error(
          "mesh exchange: transpose geometry disagrees with the layout schedule");
    if (pk.dst_offset != p.total_send_elements ||
        up.src_offset != p.total_receive_elements)
      throw std::logic_error(
          "mesh exchange: packed offsets do not tile the send/receive buffer");
    e.pack.push_back(pk);
    e.unpack.push_back(up);

    p.total_send_elements =
        pencil::checked_add(p.total_send_elements, e.send_elements);
    p.total_receive_elements =
        pencil::checked_add(p.total_receive_elements, e.receive_elements);
    p.total_send_bytes = pencil::checked_add(p.total_send_bytes, e.send_bytes);
    p.total_receive_bytes =
        pencil::checked_add(p.total_receive_bytes, e.receive_bytes);
    if (!e.self && (e.send_elements != 0 || e.receive_elements != 0))
      ++p.active_peer_count;
    p.edge.push_back(e);
  }

  if (p.total_send_elements != p.source_elements ||
      p.total_receive_elements != p.target_elements)
    throw std::logic_error(
        "mesh exchange: transpose plan does not conserve the layout region");
  return p;
}

/* Build the exchange plan between the ranks' bricks and this rank's X-real
 * pencil.  `bricks[r]` is rank r's brick, a box of the periodic mesh that may
 * wrap (gpu_layout_schedule.hpp, brick_pieces); neighbouring bricks overlap.
 * Both buffers are canonical [z][y][x].
 *
 * Forward (brick -> X real) moves the words of the spread's fixed-point sums
 * and ADDS them on arrival (`accumulate`): a pencil cell receives one
 * contribution from every brick that holds it.  Reverse (X real -> brick)
 * copies each pencil cell to every brick that holds it. */
inline ExchangePlan build_brick_plan(const pencil::PencilLayout &layout,
                                     const std::vector<pencil::Region> &bricks,
                                     count_t rank,
                                     pencil::Direction direction,
                                     count_t element_bytes,
                                     Memory memory)
{
  if (element_bytes == 0)
    throw std::invalid_argument("mesh exchange: element size must be positive");
  const bool forward = direction == pencil::Direction::forward;
  if (forward && element_bytes != 4 && element_bytes != 8)
    throw std::invalid_argument("mesh exchange: brick words must be 4 or 8 bytes");
  const pencil::BrickSchedule s =
      layout.brick_schedule(rank, bricks, direction, element_bytes);

  const pencil::Region &brick = bricks[static_cast<std::size_t>(rank)];
  const pencil::Region x_reg = layout.x_real(rank);

  ExchangePlan p;
  p.memory = memory;
  p.element_bytes = element_bytes;
  p.accumulate = forward;
  p.source_elements = forward ? brick.elements() : x_reg.elements();
  p.target_elements = forward ? x_reg.elements() : brick.elements();
  p.source_bytes = pencil::checked_mul(p.source_elements, element_bytes);
  p.target_bytes = pencil::checked_mul(p.target_elements, element_bytes);

  count_t brick_stride[3];
  canonical_strides(brick, brick.nx, brick_stride);
  count_t real_stride[3];
  canonical_strides(x_reg, layout.mesh().nx, real_stride);
  const count_t *src_stride = forward ? brick_stride : real_stride;
  const count_t *dst_stride = forward ? real_stride : brick_stride;

  /* One copy per piece, packed back to back from the edge's offset; a piece
   * is the same mesh box at both ends, so the packed layouts agree. */
  auto copies = [](const std::vector<pencil::BrickPatch> &patch,
                   count_t packed, const count_t *stride, bool packing) {
    std::vector<Copy3D> out;
    for (const pencil::BrickPatch &q : patch) {
      Copy3D c;
      c.n[0] = q.global.nz;
      c.n[1] = q.global.ny;
      c.n[2] = q.global.nx;
      count_t *local = packing ? c.src_stride : c.dst_stride;
      count_t *wire = packing ? c.dst_stride : c.src_stride;
      for (int a = 0; a < 3; ++a) local[a] = stride[a];
      canonical_strides(q.global, q.global.nx, wire);
      (packing ? c.src_offset : c.dst_offset) = q.local_offset;
      (packing ? c.dst_offset : c.src_offset) = packed;
      packed = pencil::checked_add(packed, q.elements());
      out.push_back(c);
    }
    return out;
  };

  p.edge.reserve(s.segment.size());
  for (std::size_t i = 0; i < s.segment.size(); ++i) {
    const pencil::BrickSegment &seg = s.segment[i];
    EdgePlan e;
    e.peer = seg.peer;
    e.self = seg.peer == rank;
    e.send_elements = seg.send_elements;
    e.receive_elements = seg.receive_elements;
    e.send_bytes = seg.send_bytes;
    e.receive_bytes = seg.receive_bytes;
    e.send_offset = seg.send_offset;
    e.receive_offset = seg.receive_offset;
    e.pack = copies(seg.send_patch, seg.send_offset, src_stride, true);
    e.unpack = copies(seg.receive_patch, seg.receive_offset, dst_stride, false);

    if (e.send_offset != p.total_send_elements ||
        e.receive_offset != p.total_receive_elements)
      throw std::logic_error(
          "mesh exchange: packed offsets do not tile the send/receive buffer");
    p.total_send_elements =
        pencil::checked_add(p.total_send_elements, e.send_elements);
    p.total_receive_elements =
        pencil::checked_add(p.total_receive_elements, e.receive_elements);
    p.total_send_bytes = pencil::checked_add(p.total_send_bytes, e.send_bytes);
    p.total_receive_bytes =
        pencil::checked_add(p.total_receive_bytes, e.receive_bytes);
    if (!e.self && (e.send_elements != 0 || e.receive_elements != 0))
      ++p.active_peer_count;
    p.edge.push_back(e);
  }

  /* The brick side moves each cell once; every copy stays inside its
   * buffer. */
  for (const EdgePlan &e : p.edge) {
    for (const Copy3D &c : e.pack)
      if (c.src_span() > p.source_elements ||
          c.dst_span() > p.total_send_elements)
        throw std::logic_error("mesh exchange: brick pack leaves its buffers");
    for (const Copy3D &c : e.unpack)
      if (c.dst_span() > p.target_elements ||
          c.src_span() > p.total_receive_elements)
        throw std::logic_error("mesh exchange: brick unpack leaves its buffers");
  }
  return p;
}

} /* namespace genesis_native_s4 */

#endif /* GENESIS_NATIVE_S4_MESH_EXCHANGE_PLAN_HPP */
