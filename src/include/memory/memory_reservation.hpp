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
#include "memory/memory_space.hpp"

#include <rmm/cuda_device.hpp>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// RMM includes for memory resource management
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace sirius {
namespace memory {

// Forward declarations
class memory_reservation_manager;
class reservation_aware_resource_adaptor;
struct reservation;

//===----------------------------------------------------------------------===//
// Reservation Limit Policy Interface
//===----------------------------------------------------------------------===//

/**
 * @brief Base class for reservation limit policies that control behavior when stream reservations
 * are exceeded.
 *
 * Reservation limit policies are pluggable strategies that determine what happens when an
 * allocation would cause a stream's memory usage to exceed its reservation limit.
 */
class reservation_limit_policy {
 public:
  virtual ~reservation_limit_policy() = default;

  /**
   * @brief Handle an allocation that would exceed the stream's reservation.
   *
   * This method is called when an allocation would cause the current allocated bytes
   * plus the new allocation to exceed the stream's reservation. The policy can:
   * 1. Allow the allocation to proceed (ignore policy)
   * 2. Increase the reservation and allow the allocation (increase policy)
   * 3. Throw an exception to prevent the allocation (fail policy)
   *
   * @param adaptor Reference to the tracking resource adaptor
   * @param stream The stream that would exceed its reservation
   * @param requested_bytes Number of bytes being requested
   * @param current_allocated Number of bytes currently allocated
   * @param reserved_bytes Pointer to the reservation object
   * @throws rmm::out_of_memory if the policy decides to reject the allocation
   */
  virtual void handle_over_reservation(reservation_aware_resource_adaptor& adaptor,
                                       rmm::cuda_stream_view stream,
                                       std::size_t requested_bytes,
                                       std::size_t current_allocated,
                                       reservation* reserved_bytes) = 0;

  /**
   * @brief Get a human-readable name for this policy.
   * @return Policy name string
   */
  virtual std::string get_policy_name() const = 0;
};

/**
 * @brief Ignore policy - allows allocations to proceed even if they exceed reservations.
 *
 * This policy simply ignores reservation limits and allows all allocations to proceed.
 * It's useful for soft reservations where you want to track usage but not enforce limits.
 */
class ignore_reservation_limit_policy : public reservation_limit_policy {
 public:
  ignore_reservation_limit_policy();

  void handle_over_reservation(reservation_aware_resource_adaptor& adaptor,
                               rmm::cuda_stream_view stream,
                               std::size_t requested_bytes,
                               std::size_t current_allocated,
                               reservation* reserved_bytes) final;

  std::string get_policy_name() const override;
};

/**
 * @brief Fail policy - throws an exception when reservations are exceeded.
 *
 * This policy enforces strict reservation limits by throwing rmm::out_of_memory
 * when an allocation would exceed the stream's reservation.
 */
class fail_reservation_limit_policy : public reservation_limit_policy {
 public:
  fail_reservation_limit_policy();

  void handle_over_reservation(reservation_aware_resource_adaptor& adaptor,
                               rmm::cuda_stream_view stream,
                               std::size_t requested_bytes,
                               std::size_t current_allocated,
                               reservation* reserved_bytes) final;

  std::string get_policy_name() const override;
};

/**
 * @brief remaining policy - automatically reserves remaining memory up to a limit.
 *
 * This policy implements a best effort policy that reserves the stream's reservation upto the
 * specified limit.
 */
class increase_reservation_limit_policy : public reservation_limit_policy {
 public:
  increase_reservation_limit_policy();

  /**
   * @brief Constructs an increase policy with the specified padding factor.
   */
  explicit increase_reservation_limit_policy(double padding_factor,
                                             bool allow_beyond_limit = false);

  void handle_over_reservation(reservation_aware_resource_adaptor& adaptor,
                               rmm::cuda_stream_view stream,
                               std::size_t requested_bytes,
                               std::size_t current_allocated,
                               reservation* reserved_bytes) override;

  std::string get_policy_name() const override;

 private:
  double _padding_factor{1.25f};               ///< Padding factor when increasing reservations
  bool allow_reservation_beyond_limit{false};  ///< Allow reservation beyond limit
};

std::unique_ptr<reservation_limit_policy> make_default_reservation_limit_policy();

//===----------------------------------------------------------------------===//
// Reservation Request Strategies
//===----------------------------------------------------------------------===//

/**
 * Request reservation in any memory space within a tier, with optional device preference.
 * If preferred_device_id is specified, that device will be tried first.
 */
struct any_memory_space_in_tier_with_preference {
  Tier tier;
  std::optional<size_t> preferred_device_id;  // Optional preferred device within tier

  explicit any_memory_space_in_tier_with_preference(Tier t,
                                                    std::optional<size_t> device_id = std::nullopt)
    : tier(t), preferred_device_id(device_id)
  {
  }
};

/**
 * Request reservation in any memory space within a specific tier.
 */
struct any_memory_space_in_tier {
  Tier tier;
  explicit any_memory_space_in_tier(Tier t) : tier(t) {}
};

/**
 * Request reservation in memory spaces across multiple tiers, ordered by preference.
 * The first available tier in the list will be selected.
 */
struct any_memory_space_in_tiers {
  std::vector<Tier> tiers;  // Ordered by preference
  explicit any_memory_space_in_tiers(std::vector<Tier> t) : tiers(std::move(t)) {}
};

/**
 * Variant type for different reservation request strategies.
 * Supports three main approaches:
 * 1. Specific tier with optional device preference
 * 2. Any device in a specific tier
 * 3. Any device across multiple tiers (ordered by preference)
 */
using reservation_request = std::variant<any_memory_space_in_tier_with_preference,
                                         any_memory_space_in_tier,
                                         any_memory_space_in_tiers>;

//===----------------------------------------------------------------------===//
// Reservation
//===----------------------------------------------------------------------===//

/**
 * Represents a memory reservation in a specific memory space.
 * Contains only the essential identifying information (tier, device_id, size).
 * The actual memory_space can be obtained through the memory_reservation_manager.
 */
struct reservation {
  memory_space_id space_id;
  std::size_t size;
  std::atomic<int64_t> allocated_bytes{0};

  friend class reservation_aware_resource_adaptor;

  static std::unique_ptr<reservation> create(memory_space_id id,
                                             std::size_t size,
                                             std::function<void(reservation*)> release_callback)
  {
    return std::unique_ptr<reservation>(new reservation(id, size, std::move(release_callback)));
  }

  //===----------------------------------------------------------------------===//
  // Reservation Size Management
  //===----------------------------------------------------------------------===//

  std::size_t get_available_memory() const noexcept
  {
    auto current_bytes = allocated_bytes.load();
    return size > current_bytes ? size - current_bytes : 0UL;
  }

  /**
   * @brief Attempts to grow this reservation to a new larger size.
   * @param new_size The new size for the reservation (must be larger than current size)
   * @return true if the reservation was successfully grown, false otherwise
   */
  bool grow_to(size_t new_size);

  /**
   * @brief Attempts to grow this reservation by additional bytes.
   * @param additional_bytes Number of bytes to add to the current reservation
   * @return true if the reservation was successfully grown, false otherwise
   */
  bool grow_by(size_t additional_bytes);

  /**
   * @brief Attempts to shrink this reservation to a new smaller size.
   * @param new_size The new size for the reservation (must be smaller than current size)
   * @return true if the reservation was successfully shrunk, false otherwise
   */
  bool shrink_to(size_t new_size);

  /**
   * @brief Attempts to shrink this reservation by removing bytes.
   * @param bytes_to_remove Number of bytes to remove from the current reservation
   * @return true if the reservation was successfully shrunk, false otherwise
   */
  bool shrink_by(size_t bytes_to_remove);

  // Disable copy/move to prevent issues with memory_space tracking
  reservation(const reservation&)            = delete;
  reservation& operator=(const reservation&) = delete;
  reservation(reservation&&)                 = delete;
  reservation& operator=(reservation&&)      = delete;

  ~reservation()
  {
    try {
      if (_release_callback) { _release_callback(this); }
    } catch (...) {
    }
  }

 private:
  reservation(memory_space_id id,
              std::size_t size,
              std::function<void(reservation*)> release_callback);

  std::function<void(reservation*)> _release_callback;
};

//===----------------------------------------------------------------------===//
// memory_reservation_manager
//===----------------------------------------------------------------------===//

/**
 * Central manager for memory reservations across multiple memory spaces.
 * Implements singleton pattern and coordinates reservation requests using
 * different strategies (specific space, tier-based, multi-tier fallback).
 */
class memory_reservation_manager {
 public:
  //===----------------------------------------------------------------------===//
  // Configuration and Initialization
  //===----------------------------------------------------------------------===//

  /**
   * Configuration for a single memory_space.
   * Contains all parameters needed to create a memory_space instance.
   */
  struct memory_space_config {
    Tier tier;
    int device_id;
    size_t memory_limit;
    std::optional<std::size_t>
      memory_capacity;  // Optional total capacity, defaults to device capacity
    std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators;

    // Constructor - allocators must be explicitly provided
    memory_space_config(Tier t,
                        int dev_id,
                        size_t mem_limit,
                        std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocs);

    // Constructor - allocators must be explicitly provided
    memory_space_config(Tier t,
                        int dev_id,
                        size_t mem_limit,
                        size_t mem_capacity,
                        std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocs);

    // Move constructor
    memory_space_config(memory_space_config&&)            = default;
    memory_space_config& operator=(memory_space_config&&) = default;

    // Delete copy constructor/assignment since allocators contain unique_ptr
    memory_space_config(const memory_space_config&)            = delete;
    memory_space_config& operator=(const memory_space_config&) = delete;
  };

  /**
   * Initialize the singleton instance with the given memory space configurations.
   * Must be called before get_instance() can be used.
   */
  static void initialize(std::vector<memory_space_config> configs);

  /**
   * Test-only: Reset the singleton so tests can reinitialize with different configs.
   * Not thread-safe; intended only for unit tests.
   */
  static void reset_for_testing();

  /**
   * Get the singleton instance.
   * Throws if initialize() has not been called first.
   */
  static memory_reservation_manager& get_instance();

  // Disable copy/move for singleton
  memory_reservation_manager(const memory_reservation_manager&)            = delete;
  memory_reservation_manager& operator=(const memory_reservation_manager&) = delete;
  memory_reservation_manager(memory_reservation_manager&&)                 = delete;
  memory_reservation_manager& operator=(memory_reservation_manager&&)      = delete;

  // Public destructor (required for std::unique_ptr)
  ~memory_reservation_manager() = default;

  //===----------------------------------------------------------------------===//
  // Reservation Interface
  //===----------------------------------------------------------------------===//

  /**
   * Main reservation interface using strategy pattern.
   * Supports different reservation strategies through the reservation_request variant.
   */
  std::unique_ptr<reservation> request_reservation(const reservation_request& request, size_t size);

  //===----------------------------------------------------------------------===//
  // memory_space Access and Queries
  //===----------------------------------------------------------------------===//

  /**
   * Get a specific memory_space by tier and device ID.
   * Returns nullptr if no such space exists.
   */
  const memory_space* get_memory_space(Tier tier, int32_t device_id) const;

  /**
   * Get all memory_spaces for a specific tier.
   * Returns empty vector if no spaces exist for that tier.
   */
  std::vector<const memory_space*> get_memory_spaces_for_tier(Tier tier) const;
  std::vector<memory_space*> get_memory_spaces_for_tier(Tier tier);

  /**
   * Get all memory_spaces managed by this instance.
   */
  std::vector<const memory_space*> get_all_memory_spaces() const noexcept;
  std::vector<memory_space*> get_all_memory_spaces() noexcept;

  //===----------------------------------------------------------------------===//
  // Aggregated Queries
  //===----------------------------------------------------------------------===//

  // Tier-level aggregations
  size_t get_available_memory_for_tier(Tier tier) const;
  size_t get_total_reserved_memory_for_tier(Tier tier) const;
  size_t get_active_reservation_count_for_tier(Tier tier) const;

  // System-wide aggregations
  size_t get_total_available_memory() const;
  size_t get_total_reserved_memory() const;
  size_t get_active_reservation_count() const;

 private:
  /**
   * Private constructor - use initialize() and get_instance() instead.
   */
  explicit memory_reservation_manager(std::vector<memory_space_config> configs);

  // Singleton state
  static std::unique_ptr<memory_reservation_manager> _instance;
  static std::once_flag _initialized;
  static bool _allow_reinitialize_for_tests;

  // Storage for memory_space instances (owned by the manager)
  std::vector<std::unique_ptr<memory_space>> _memory_spaces;

  // Fast lookups
  std::unordered_map<memory_space_id, const memory_space*> _memory_space_lookup;
  std::unordered_map<Tier, std::vector<const memory_space*>> _tier_to_memory_spaces;

  // Helper method: attempts to select a space and immediately make a reservation
  // Returns a reservation when successful, or std::nullopt if none can satisfy the request
  std::optional<std::unique_ptr<reservation>> select_memory_space_and_make_reservation(
    const reservation_request& request, size_t size);

  void build_lookup_tables();

  // Synchronization for cross-space waiting when no memory_space can currently satisfy a request
  mutable std::mutex _wait_mutex;
  std::condition_variable _wait_cv;
};

}  // namespace memory
}  // namespace sirius
