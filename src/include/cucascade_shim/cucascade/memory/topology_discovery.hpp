/*
 * RasterDB cucascade::memory::topology_discovery shim.
 */
#pragma once
#include "common.hpp"
#include <vector>
#include <string>

namespace cucascade {
namespace memory {

struct system_topology_info {
    int32_t num_gpus{0};
    int32_t num_numa_nodes{1};
    std::vector<int32_t> gpu_numa_affinity;
};

inline system_topology_info discover_topology() {
    return system_topology_info{0, 1, {}};
}

} // namespace memory
} // namespace cucascade
