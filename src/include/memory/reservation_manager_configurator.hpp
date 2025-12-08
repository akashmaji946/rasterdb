/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "memory/common.hpp"
#include "memory/memory_reservation_manager.hpp"
#include "memory/memory_space.hpp"
#include "memory/topology_discovery.hpp"

#include <variant>
#include <vector>

namespace sirius {
namespace memory {

//===----------------------------------------------------------------------===//
// memory_reservation_manager
//===----------------------------------------------------------------------===//

class reservation_manager_config_builder {
 public:
  using builder_reference   = reservation_manager_config_builder;
  using memory_space_config = memory_reservation_manager::memory_space_config;

  /// either set gpu ids or number of gpus
  builder_reference& set_number_of_gpus(std::size_t n_gpus);
  builder_reference& set_gpu_ids(std::vector<int> gpu_ids);

  // either set space capacity or set a ratio of gpu total capacity
  builder_reference& set_gpu_usage_limit(std::size_t bytes);
  builder_reference& set_usage_limit_ratio_per_gpu(double fraction);

  // either set host ids or create as many host tiers as numa nodes of gpus create by this builder
  builder_reference& set_numa_ids(std::vector<int> numa_ids);
  builder_reference& bind_cpu_tier_to_gpus();

  /// set capacity per host tier
  builder_reference& set_capacity_per_numa_node(std::size_t bytes);

  // set ratio of space capacity used for reservation in gpus
  builder_reference& set_reservation_limit_ratio_per_gpu(double fraction);
  // set ratio of space capacity used for reservation in cpus
  builder_reference& set_reservation_limit_ratio_per_numa_node(double fraction);

  // set the function that takes in the device id and create gpu memory resource
  builder_reference& set_gpu_memory_resource_factory(DeviceMemoryResourceFactoryFn mr_fn);

  // set the function that takes in the numa node id and create cpu memory resource
  builder_reference& set_cpu_memory_resource_factory(DeviceMemoryResourceFactoryFn mr_fn);

  // don't do topoly checking
  builder_reference& ignore_topology();

  std::vector<memory_space_config> build(const system_topology_info& topology) const;

  std::vector<memory_space_config> build() const;

 private:
  std::vector<memory_space_config> build(const system_topology_info* topology) const;

  std::vector<int> extract_gpu_ids(const system_topology_info* topology) const;
  std::vector<std::pair<size_t, size_t>> extract_gpu_memory_thresholds(
    const std::vector<int>& gpus_ids, const system_topology_info* topology) const;
  std::vector<int> extract_host_ids(const std::vector<int>& gpu_ids,
                                    const system_topology_info* topology) const;

  bool ignore_topology_{false};
  std::variant<std::size_t, std::vector<int>> n_gpus_or_gpu_ids_{1UL};
  std::variant<std::size_t, double> gpu_usage_limit_or_ratio_{
    static_cast<std::size_t>(1UL << 30)};          // uses 1GB of gpu memory
  double gpu_reservation_limit_ratio_{0.75};       // limit to 75% of GPU usagel limit
  std::size_t capacity_per_numa_node_{8UL << 30};  // 8GB per NUMA node by default
  std::variant<std::monostate, std::vector<int>> auto_binding_or_numa_ids_{};
  double cpu_reservation_limit_ratio_{0.75};  // 75% limit per NUMA node by default
  DeviceMemoryResourceFactoryFn gpu_mr_fn_ = make_default_gpu_memory_resource;
  DeviceMemoryResourceFactoryFn cpu_mr_fn_ = make_default_host_memory_resource;
};

}  // namespace memory
}  // namespace sirius
