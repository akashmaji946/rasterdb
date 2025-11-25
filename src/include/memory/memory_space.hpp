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

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <variant>

// RMM includes for memory resource management
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace sirius {
namespace memory {

// Forward declaration
struct reservation;
struct reservation_aware_resource_adaptor;
struct fixed_size_host_memory_resource;

/**
 * memory_space represents a specific memory location identified by a tier and device ID.
 * It manages memory reservations within that space and owns allocator resources.
 *
 * Each memory_space:
 * - Has a fixed memory limit
 * - Tracks active reservations
 * - Provides thread-safe reservation management
 * - Owns one or more RMM memory allocators
 */
class memory_space {
 public:
  /**
   * Construct a memory_space with the given parameters.
   *
   * @param tier The memory tier (GPU, HOST, DISK)
   * @param device_id The device identifier within the tier
   * @param memory_limit Maximum memory capacity in bytes
   * @param allocator RMM memory allocator (must be non-empty)
   * @param capacity Total memory capacity in bytes (optional [default: device capacity])
   */
  memory_space(Tier tier,
               int device_id,
               size_t memory_limit,
               std::unique_ptr<rmm::mr::device_memory_resource> allocator,
               std::optional<std::size_t> capacity = std::nullopt);

  // Disable copy/move to ensure stable addresses for reservations
  memory_space(const memory_space&)            = delete;
  memory_space& operator=(const memory_space&) = delete;
  memory_space(memory_space&&)                 = delete;
  memory_space& operator=(memory_space&&)      = delete;

  ~memory_space();

  // Comparison operators
  bool operator==(const memory_space& other) const;
  bool operator!=(const memory_space& other) const;

  // Basic properties
  Tier get_tier() const;
  int get_device_id() const;

  // Reservation management - these are the core methods that do the actual work
  std::unique_ptr<reservation> request_reservation(size_t size);

  // State queries
  size_t get_available_memory(rmm::cuda_stream_view stream) const;
  size_t get_available_memory() const;
  size_t get_total_reserved_memory() const;
  size_t get_max_memory() const noexcept;

  // Allocator management
  rmm::mr::device_memory_resource* get_default_allocator() const noexcept;

  // Utility methods
  std::string to_string() const;

 protected:
  const Tier _tier;
  const int _device_id;
  const size_t _memory_limit;
  const size_t _capacity;
  using reserving_adaptor_type = std::variant<std::monostate,
                                              std::unique_ptr<reservation_aware_resource_adaptor>,
                                              std::unique_ptr<fixed_size_host_memory_resource>>;

  mutable std::mutex _reservation_mutex;
  std::condition_variable _reservation_cv;
  bool _reservation_release{false};

  void notify_release_of_reservation();

  // Memory resources owned by this memory_space
  std::unique_ptr<rmm::mr::device_memory_resource> _allocator;
  reserving_adaptor_type _reservation_allocator;
};

/**
 * Hash function for memory_space to enable use in unordered containers.
 * Hash is based on tier and device_id combination.
 */
struct memory_space_hash {
  size_t operator()(const memory_space& ms) const;
};

}  // namespace memory
}  // namespace sirius
