// Native inner loop for fpsgen.data.hard_poisson.  This deliberately performs
// the same deterministic greedy 3D spatial-hash acceptance test as the Python
// reference, without materialising a dense point-pair matrix.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

struct Cell {
  int64_t x, y, z;
  bool operator==(const Cell& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct CellHash {
  static uint64_t mix(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }
  size_t operator()(const Cell& cell) const {
    return static_cast<size_t>(mix(static_cast<uint64_t>(cell.x)) ^
                               (mix(static_cast<uint64_t>(cell.y)) << 1) ^
                               (mix(static_cast<uint64_t>(cell.z)) << 7));
  }
};

extern "C" int hard_poisson_select_native(
    const float* xyz, int64_t point_count, double radius, uint64_t seed,
    int64_t max_accept, int64_t* output_indices, int64_t* selected_count,
    int64_t* scanned_count) {
  if (xyz == nullptr || output_indices == nullptr || selected_count == nullptr ||
      scanned_count == nullptr || point_count <= 0 || radius <= 0 ||
      max_accept <= 0) {
    return 1;
  }
  std::vector<int64_t> order(point_count);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 generator(seed);
  std::shuffle(order.begin(), order.end(), generator);
  std::unordered_map<Cell, std::vector<int64_t>, CellHash> accepted_by_cell;
  accepted_by_cell.reserve(static_cast<size_t>(max_accept * 2));
  const double radius_sq = radius * radius;
  int64_t selected = 0;
  int64_t scanned = 0;
  for (const int64_t index : order) {
    ++scanned;
    const float* point = xyz + 3 * index;
    const Cell cell{static_cast<int64_t>(std::floor(point[0] / radius)),
                    static_cast<int64_t>(std::floor(point[1] / radius)),
                    static_cast<int64_t>(std::floor(point[2] / radius))};
    bool allowed = true;
    for (int dx = -1; dx <= 1 && allowed; ++dx) {
      for (int dy = -1; dy <= 1 && allowed; ++dy) {
        for (int dz = -1; dz <= 1 && allowed; ++dz) {
          const auto found = accepted_by_cell.find(Cell{cell.x + dx, cell.y + dy, cell.z + dz});
          if (found == accepted_by_cell.end()) continue;
          for (const int64_t other : found->second) {
            const float* neighbour = xyz + 3 * other;
            const double delta_x = static_cast<double>(neighbour[0]) - point[0];
            const double delta_y = static_cast<double>(neighbour[1]) - point[1];
            const double delta_z = static_cast<double>(neighbour[2]) - point[2];
            if (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z < radius_sq) {
              allowed = false;
              break;
            }
          }
        }
      }
    }
    if (!allowed) continue;
    output_indices[selected++] = index;
    accepted_by_cell[cell].push_back(index);
    if (selected >= max_accept) break;
  }
  *selected_count = selected;
  *scanned_count = scanned;
  return 0;
}
