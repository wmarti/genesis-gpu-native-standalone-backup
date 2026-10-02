/*
 * Pure arithmetic for the pencil layout: block splits, rank mapping, pencil
 * origins and the two row/column transpose schedules.  It owns no MPI, CUDA
 * or GENESIS state; a transport adapter consumes the checked element counts
 * and offsets.
 */
#ifndef GENESIS_NATIVE_PENCIL_LAYOUT_SCHEDULE_HPP
#define GENESIS_NATIVE_PENCIL_LAYOUT_SCHEDULE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace genesis_native_pencil {

using count_t = std::uint64_t;

inline count_t checked_add(count_t a, count_t b)
{
  if (a > std::numeric_limits<count_t>::max() - b)
    throw std::overflow_error("pencil layout addition overflow");
  return a + b;
}

inline count_t checked_mul(count_t a, count_t b)
{
  if (a != 0 && b > std::numeric_limits<count_t>::max() / a)
    throw std::overflow_error("pencil layout multiplication overflow");
  return a * b;
}

struct BlockSplit {
  count_t extent = 0;
  std::vector<count_t> offset;
  std::vector<count_t> length;

  static BlockSplit make(count_t extent, count_t parts)
  {
    if (parts == 0)
      throw std::invalid_argument("pencil split requires at least one part");

    BlockSplit result;
    result.extent = extent;
    result.offset.assign(static_cast<std::size_t>(parts), 0);
    result.length.assign(static_cast<std::size_t>(parts), 0);
    const count_t base = extent / parts;
    const count_t remainder = extent % parts;
    count_t cursor = 0;
    for (count_t part = 0; part < parts; ++part) {
      result.offset[static_cast<std::size_t>(part)] = cursor;
      const count_t n = checked_add(base, part < remainder ? 1 : 0);
      result.length[static_cast<std::size_t>(part)] = n;
      cursor = checked_add(cursor, n);
    }
    if (cursor != extent)
      throw std::logic_error("pencil split does not conserve its extent");
    return result;
  }

  count_t end(count_t part) const
  {
    if (part >= length.size())
      throw std::out_of_range("pencil split part");
    return checked_add(offset[static_cast<std::size_t>(part)],
                       length[static_cast<std::size_t>(part)]);
  }
};

struct MeshShape {
  count_t nx = 0;
  count_t ny = 0;
  count_t nz = 0;
};

struct ProcessGrid {
  count_t pa = 0;
  count_t pb = 0;

  count_t size() const
  {
    if (pa == 0 || pb == 0)
      throw std::invalid_argument("pencil process-grid dimensions must be positive");
    return checked_mul(pa, pb);
  }

  struct Coord {
    count_t a = 0;
    count_t b = 0;
  };

  Coord coord(count_t rank) const
  {
    const count_t n = size();
    if (rank >= n)
      throw std::out_of_range("pencil rank");
    return Coord{rank / pb, rank % pb};
  }

  count_t rank(count_t a, count_t b) const
  {
    if (a >= pa || b >= pb)
      throw std::out_of_range("pencil process-grid coordinate");
    return checked_add(checked_mul(a, pb), b);
  }
};

struct Region {
  count_t x0 = 0;
  count_t y0 = 0;
  count_t z0 = 0;
  count_t nx = 0;
  count_t ny = 0;
  count_t nz = 0;

  count_t elements() const
  {
    return checked_mul(checked_mul(nx, ny), nz);
  }
};

inline bool contains(const Region &outer, const Region &inner)
{
  return inner.x0 >= outer.x0 && inner.y0 >= outer.y0 && inner.z0 >= outer.z0 &&
         checked_add(inner.x0, inner.nx) <= checked_add(outer.x0, outer.nx) &&
         checked_add(inner.y0, inner.ny) <= checked_add(outer.y0, outer.ny) &&
         checked_add(inner.z0, inner.nz) <= checked_add(outer.z0, outer.nz);
}

inline count_t local_offset(const Region &local, const Region &patch)
{
  if (patch.elements() == 0)
    return 0;
  if (!contains(local, patch))
    throw std::logic_error("pencil patch is outside its local region");
  const count_t z = patch.z0 - local.z0;
  const count_t y = patch.y0 - local.y0;
  const count_t x = patch.x0 - local.x0;
  return checked_add(
      checked_mul(checked_add(checked_mul(z, local.ny), y), local.nx), x);
}

enum class Transpose { x_to_y, y_to_z };
enum class Direction { forward, reverse };

struct Segment {
  count_t peer = 0;
  count_t send_offset = 0;
  count_t receive_offset = 0;
  count_t send_elements = 0;
  count_t receive_elements = 0;
  count_t send_bytes = 0;
  count_t receive_bytes = 0;
};

struct TransposeSchedule {
  Transpose transpose = Transpose::x_to_y;
  Direction direction = Direction::forward;
  count_t element_bytes = 0;
  std::vector<Segment> segment;
  count_t total_send_elements = 0;
  count_t total_receive_elements = 0;
  count_t total_send_bytes = 0;
  count_t total_receive_bytes = 0;
  count_t active_peer_count = 0;

  std::vector<count_t> active_peers(count_t self) const
  {
    std::vector<count_t> result;
    for (const Segment &s : segment) {
      if (s.peer != self && (s.send_elements != 0 || s.receive_elements != 0))
        result.push_back(s.peer);
    }
    return result;
  }
};

/* One piece of a brick <-> X-real move: a box of the mesh that lies in no
 * wrap (`global`), and its first cell's offset in the caller's buffer on
 * this side, which is canonical [z][y][x] over that buffer's own box. */
struct BrickPatch {
  Region global;
  count_t local_offset = 0;

  count_t elements() const { return global.elements(); }
};

struct BrickSegment {
  count_t peer = 0;
  // Packed-buffer offsets.  An edge carries its pieces back to back, in the
  // order brick_pieces gives them, and both ends derive that order from the
  // same two regions.
  count_t send_offset = 0;
  count_t receive_offset = 0;
  count_t send_elements = 0;
  count_t receive_elements = 0;
  count_t send_bytes = 0;
  count_t receive_bytes = 0;
  std::vector<BrickPatch> send_patch;
  std::vector<BrickPatch> receive_patch;
};

struct BrickSchedule {
  Direction direction = Direction::forward;
  count_t element_bytes = 0;
  std::vector<BrickSegment> segment;
  count_t total_send_elements = 0;
  count_t total_receive_elements = 0;
  count_t total_send_bytes = 0;
  count_t total_receive_bytes = 0;
  count_t active_peer_count = 0;

  std::vector<count_t> active_peers(count_t self) const
  {
    std::vector<count_t> result;
    for (const BrickSegment &s : segment) {
      if (s.peer != self && (s.send_elements != 0 || s.receive_elements != 0))
        result.push_back(s.peer);
    }
    return result;
  }
};

/* A brick is a box of the periodic mesh: cells (x0 + i) mod Nx for i in
 * [0, nx), and alike in y and z, stored [z][y][x] with x fastest over its
 * own extents.  x0 < Nx and nx <= Nx, so a brick may run past the mesh edge
 * and wrap, but never holds a cell twice.  Its intersection with a box that
 * does not wrap is at most two runs per axis; brick_pieces lists them, z
 * outermost, each with its first cell in the brick's own coordinates. */
struct BrickPiece {
  Region global;
  count_t lx = 0, ly = 0, lz = 0;
};

inline std::vector<BrickPiece> brick_pieces(const MeshShape &mesh,
                                            const Region &brick,
                                            const Region &box)
{
  struct Run { count_t g0, l0, n; };
  const count_t N[3] = {mesh.nx, mesh.ny, mesh.nz};
  const count_t b0[3] = {brick.x0, brick.y0, brick.z0};
  const count_t bn[3] = {brick.nx, brick.ny, brick.nz};
  const count_t t0[3] = {box.x0, box.y0, box.z0};
  const count_t tn[3] = {box.nx, box.ny, box.nz};
  Run run[3][2];
  int nrun[3] = {0, 0, 0};
  for (int a = 0; a < 3; ++a) {
    if (b0[a] >= N[a] || bn[a] > N[a])
      throw std::invalid_argument("brick is not a box of the periodic mesh");
    /* The brick's stretch up to the mesh edge, at local 0, and its wrapped
     * rest from 0, at local N - b0. */
    const count_t first = std::min(bn[a], N[a] - b0[a]);
    const count_t s0[2] = {b0[a], 0}, sl[2] = {0, first};
    const count_t sn[2] = {first, bn[a] - first};
    for (int s = 0; s < 2; ++s) {
      const count_t lo = std::max(s0[s], t0[a]);
      const count_t hi = std::min(checked_add(s0[s], sn[s]),
                                  checked_add(t0[a], tn[a]));
      if (sn[s] != 0 && hi > lo)
        run[a][nrun[a]++] = Run{lo, sl[s] + (lo - s0[s]), hi - lo};
    }
  }
  std::vector<BrickPiece> out;
  for (int z = 0; z < nrun[2]; ++z)
    for (int y = 0; y < nrun[1]; ++y)
      for (int x = 0; x < nrun[0]; ++x) {
        BrickPiece p;
        p.global = Region{run[0][x].g0, run[1][y].g0, run[2][z].g0,
                          run[0][x].n, run[1][y].n, run[2][z].n};
        p.lx = run[0][x].l0;
        p.ly = run[1][y].l0;
        p.lz = run[2][z].l0;
        out.push_back(p);
      }
  return out;
}

class PencilLayout {
public:
  PencilLayout(MeshShape mesh, ProcessGrid grid, count_t world_size)
      : mesh_(validate_inputs(mesh, grid, world_size)), grid_(grid),
        world_size_(world_size),
        y_by_a_(BlockSplit::make(mesh.ny, grid.pa)),
        z_by_b_(BlockSplit::make(mesh.nz, grid.pb)),
        xh_by_a_(BlockSplit::make(x_half(mesh.nx), grid.pa)),
        y_by_b_(BlockSplit::make(mesh.ny, grid.pb))
  {}

  const MeshShape &mesh() const { return mesh_; }
  const ProcessGrid &grid() const { return grid_; }
  count_t world_size() const { return world_size_; }
  count_t nx_half() const { return x_half(mesh_.nx); }
  /* Node-major numbering (set_node_major): the grid's ranks run `local` per
   * node, node by node, and grid position a * pb + b is (rank on the node)
   * * nodes + node, so a = the rank on its node and b = its node (pa =
   * local, pb = nodes).  Otherwise the grid position is the rank. */
  void set_node_major(count_t local) { node_local_ = local; }
  count_t grid_rank(count_t rank) const
  {
    if (node_local_ == 0) return rank;
    const count_t nodes = grid_.size() / node_local_;
    return (rank % node_local_) * nodes + rank / node_local_;
  }
  /* The rank at grid position g (grid_rank's inverse). */
  count_t rank_at(count_t g) const
  {
    if (node_local_ == 0) return g;
    const count_t nodes = grid_.size() / node_local_;
    return (g % nodes) * node_local_ + g / nodes;
  }

  Region x_real(count_t rank) const
  {
    const ProcessGrid::Coord c = grid_.coord(grid_rank(rank));
    return Region{0, y_by_a_.offset[static_cast<std::size_t>(c.a)],
                  z_by_b_.offset[static_cast<std::size_t>(c.b)], mesh_.nx,
                  y_by_a_.length[static_cast<std::size_t>(c.a)],
                  z_by_b_.length[static_cast<std::size_t>(c.b)]};
  }

  Region x_complex(count_t rank) const
  {
    Region r = x_real(rank);
    r.nx = nx_half();
    return r;
  }

  Region y_complex(count_t rank) const
  {
    const ProcessGrid::Coord c = grid_.coord(grid_rank(rank));
    return Region{xh_by_a_.offset[static_cast<std::size_t>(c.a)], 0,
                  z_by_b_.offset[static_cast<std::size_t>(c.b)],
                  xh_by_a_.length[static_cast<std::size_t>(c.a)], mesh_.ny,
                  z_by_b_.length[static_cast<std::size_t>(c.b)]};
  }

  Region z_complex(count_t rank) const
  {
    const ProcessGrid::Coord c = grid_.coord(grid_rank(rank));
    return Region{xh_by_a_.offset[static_cast<std::size_t>(c.a)],
                  y_by_b_.offset[static_cast<std::size_t>(c.b)], 0,
                  xh_by_a_.length[static_cast<std::size_t>(c.a)],
                  y_by_b_.length[static_cast<std::size_t>(c.b)], mesh_.nz};
  }

  const BlockSplit &y_by_a() const { return y_by_a_; }
  const BlockSplit &z_by_b() const { return z_by_b_; }
  const BlockSplit &xh_by_a() const { return xh_by_a_; }
  const BlockSplit &y_by_b() const { return y_by_b_; }

  TransposeSchedule schedule(count_t rank, Transpose transpose,
                             Direction direction, count_t element_bytes) const
  {
    if (element_bytes == 0)
      throw std::invalid_argument("pencil element size must be positive");
    TransposeSchedule result;
    result.transpose = transpose;
    result.direction = direction;
    result.element_bytes = element_bytes;
    const ProcessGrid::Coord c = grid_.coord(grid_rank(rank));
    const count_t peer_count = transpose == Transpose::x_to_y ? grid_.pa : grid_.pb;
    result.segment.reserve(static_cast<std::size_t>(peer_count));

    for (count_t part = 0; part < peer_count; ++part) {
      count_t peer = 0;
      count_t forward_send = 0;
      count_t forward_receive = 0;
      if (transpose == Transpose::x_to_y) {
        peer = rank_at(grid_.rank(part, c.b));
        forward_send = checked_mul(
            checked_mul(z_by_b_.length[static_cast<std::size_t>(c.b)],
                        y_by_a_.length[static_cast<std::size_t>(c.a)]),
            xh_by_a_.length[static_cast<std::size_t>(part)]);
        forward_receive = checked_mul(
            checked_mul(z_by_b_.length[static_cast<std::size_t>(c.b)],
                        y_by_a_.length[static_cast<std::size_t>(part)]),
            xh_by_a_.length[static_cast<std::size_t>(c.a)]);
      } else {
        peer = rank_at(grid_.rank(c.a, part));
        forward_send = checked_mul(
            checked_mul(z_by_b_.length[static_cast<std::size_t>(c.b)],
                        xh_by_a_.length[static_cast<std::size_t>(c.a)]),
            y_by_b_.length[static_cast<std::size_t>(part)]);
        forward_receive = checked_mul(
            checked_mul(z_by_b_.length[static_cast<std::size_t>(part)],
                        xh_by_a_.length[static_cast<std::size_t>(c.a)]),
            y_by_b_.length[static_cast<std::size_t>(c.b)]);
      }

      Segment s;
      s.peer = peer;
      s.send_elements = direction == Direction::forward ? forward_send : forward_receive;
      s.receive_elements = direction == Direction::forward ? forward_receive : forward_send;
      s.send_bytes = checked_mul(s.send_elements, element_bytes);
      s.receive_bytes = checked_mul(s.receive_elements, element_bytes);
      s.send_offset = result.total_send_elements;
      s.receive_offset = result.total_receive_elements;
      result.total_send_elements = checked_add(result.total_send_elements, s.send_elements);
      result.total_receive_elements = checked_add(result.total_receive_elements, s.receive_elements);
      result.total_send_bytes = checked_add(result.total_send_bytes, s.send_bytes);
      result.total_receive_bytes = checked_add(result.total_receive_bytes, s.receive_bytes);
      if (peer != rank && (s.send_elements != 0 || s.receive_elements != 0))
        ++result.active_peer_count;
      result.segment.push_back(s);
    }

    const count_t forward_source = transpose == Transpose::x_to_y
        ? x_complex(rank).elements() : y_complex(rank).elements();
    const count_t forward_target = transpose == Transpose::x_to_y
        ? y_complex(rank).elements() : z_complex(rank).elements();
    const count_t expected_source = direction == Direction::forward
        ? forward_source : forward_target;
    const count_t expected_target = direction == Direction::forward
        ? forward_target : forward_source;
    const count_t expected_source_bytes = checked_mul(expected_source, element_bytes);
    const count_t expected_target_bytes = checked_mul(expected_target, element_bytes);
    if (result.total_send_elements != expected_source ||
        result.total_receive_elements != expected_target ||
        result.total_send_bytes != expected_source_bytes ||
        result.total_receive_bytes != expected_target_bytes)
      throw std::logic_error("pencil transpose schedule violates conservation");
    return result;
  }

  BrickSchedule brick_schedule(count_t rank, const std::vector<Region> &bricks,
                               Direction direction, count_t element_bytes) const
  {
    // Bricks of neighbouring ranks overlap; the X-real pencils partition the
    // mesh.  Forward, a brick sends each cell to the one pencil holding it and
    // a pencil receives a cell from every brick that holds it (the caller sums
    // them).  Reverse, a pencil sends each cell to every brick that holds it.
    // A peer with no piece either way gets no segment (self always does).
    if (element_bytes == 0)
      throw std::invalid_argument("pencil element size must be positive");
    if (bricks.size() != world_size_)
      throw std::invalid_argument("brick count does not match communicator");

    const Region local_x = x_real(rank);
    const bool forward = direction == Direction::forward;
    BrickSchedule result;
    result.direction = direction;
    result.element_bytes = element_bytes;
    result.segment.reserve(static_cast<std::size_t>(world_size_));

    // One side of an edge: the pieces of `brick` in `box`, with offsets in
    // this rank's brick buffer (in_brick) or in its X-real pencil.
    auto patches = [&](const Region &brick, const Region &box, bool in_brick) {
      std::vector<BrickPatch> out;
      const Region &mine = bricks[static_cast<std::size_t>(rank)];
      for (const BrickPiece &p : brick_pieces(mesh_, brick, box)) {
        BrickPatch q;
        q.global = p.global;
        q.local_offset = in_brick
            ? checked_add(checked_mul(checked_add(checked_mul(p.lz, mine.ny),
                                                  p.ly), mine.nx), p.lx)
            : local_offset(local_x, p.global);
        out.push_back(q);
      }
      return out;
    };
    auto total = [](const std::vector<BrickPatch> &v) {
      count_t n = 0;
      for (const BrickPatch &p : v) n = checked_add(n, p.elements());
      return n;
    };

    const Region &local_brick = bricks[static_cast<std::size_t>(rank)];
    for (count_t peer = 0; peer < world_size_; ++peer) {
      const Region &peer_brick = bricks[static_cast<std::size_t>(peer)];
      const Region peer_x = x_real(peer);
      BrickSegment s;
      s.peer = peer;
      s.send_patch = forward ? patches(local_brick, peer_x, true)
                             : patches(peer_brick, local_x, false);
      s.receive_patch = forward ? patches(peer_brick, local_x, false)
                                : patches(local_brick, peer_x, true);
      s.send_elements = total(s.send_patch);
      s.receive_elements = total(s.receive_patch);
      if (peer != rank && s.send_elements == 0 && s.receive_elements == 0)
        continue;
      s.send_bytes = checked_mul(s.send_elements, element_bytes);
      s.receive_bytes = checked_mul(s.receive_elements, element_bytes);
      s.send_offset = result.total_send_elements;
      s.receive_offset = result.total_receive_elements;
      result.total_send_elements = checked_add(result.total_send_elements, s.send_elements);
      result.total_receive_elements = checked_add(result.total_receive_elements, s.receive_elements);
      result.total_send_bytes = checked_add(result.total_send_bytes, s.send_bytes);
      result.total_receive_bytes = checked_add(result.total_receive_bytes, s.receive_bytes);
      if (peer != rank)
        ++result.active_peer_count;
      result.segment.push_back(std::move(s));
    }

    // The pencils partition the mesh, so the brick side moves each of its
    // cells exactly once; the pencil side moves one copy per brick holding
    // the cell, which the pieces above count by construction.
    const count_t brick_side = forward ? result.total_send_elements
                                       : result.total_receive_elements;
    if (brick_side != local_brick.elements())
      throw std::logic_error("brick schedule violates conservation");
    return result;
  }

private:

  static MeshShape validate_inputs(MeshShape mesh, const ProcessGrid &grid,
                                   count_t world_size)
  {
    if (mesh.nx == 0 || mesh.ny == 0 || mesh.nz == 0)
      throw std::invalid_argument("pencil mesh dimensions must be positive");
    if (world_size == 0 || grid.size() != world_size)
      throw std::invalid_argument("pencil grid size does not match communicator");
    return mesh;
  }

  static count_t x_half(count_t nx) {
    return checked_add(nx / 2, 1);
  }

  MeshShape mesh_;
  ProcessGrid grid_;
  count_t world_size_;
  count_t node_local_ = 0;
  BlockSplit y_by_a_;
  BlockSplit z_by_b_;
  BlockSplit xh_by_a_;
  BlockSplit y_by_b_;
};

} // namespace genesis_native_pencil

#endif
