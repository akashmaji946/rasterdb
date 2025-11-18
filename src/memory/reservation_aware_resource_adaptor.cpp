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

#include "memory/reservation_aware_resource_adaptor.hpp"

#include "aligned.hpp"
#include "memory/memory_reservation.hpp"

#include <rmm/detail/error.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <set>

namespace sirius {
namespace memory {

using reservation_state = reservation_aware_resource_adaptor::reservation_state;

namespace {

struct per_stream_reservation_state_container
  : public reservation_aware_resource_adaptor::reservation_state_container {
  mutable std::mutex mutex;
  std::unordered_map<cudaStream_t, std::unique_ptr<reservation_state>> stream_stats_map;

  per_stream_reservation_state_container() = default;

  void reset_stream_state(rmm::cuda_stream_view stream) override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it == stream_stats_map.end()) { return; }
    auto* res_pnt = it->second->memory_reservation.get();
    stream_stats_map.erase(stream.value());
  }

  void set_stream_state(rmm::cuda_stream_view stream,
                        std::unique_ptr<reservation> reservation,
                        std::unique_ptr<reservation_limit_policy> policy,
                        std::unique_ptr<oom_handling_policy> oom_policy) override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it != stream_stats_map.end()) {
      throw rmm::logic_error("Stream already has reservation state set");
    }

    auto stats                       = std::make_unique<reservation_state>();
    stats->memory_reservation        = std::move(reservation);
    stats->reservation_policy        = std::move(policy);
    stats->oom_policy                = std::move(oom_policy);
    stream_stats_map[stream.value()] = std::move(stats);
  }

  reservation_aware_resource_adaptor::reservation_state* get_stream_state(
    rmm::cuda_stream_view stream) override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it == stream_stats_map.end()) { return nullptr; }
    return it->second.get();
  }

  const reservation_aware_resource_adaptor::reservation_state* get_stream_state(
    rmm::cuda_stream_view stream) const override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it == stream_stats_map.end()) { return nullptr; }
    return it->second.get();
  }
};

struct per_thread_reservation_state_container
  : public reservation_aware_resource_adaptor::reservation_state_container {
  static inline thread_local std::unique_ptr<reservation_state> thread_reservation_state;

  void reset_stream_state(rmm::cuda_stream_view stream) override
  {
    if (thread_reservation_state) {
      auto* res_pnt = thread_reservation_state->memory_reservation.get();
      thread_reservation_state.reset();
    }
  }

  void set_stream_state(rmm::cuda_stream_view stream,
                        std::unique_ptr<reservation> reservation,
                        std::unique_ptr<reservation_limit_policy> policy,
                        std::unique_ptr<oom_handling_policy> oom_policy) override
  {
    if (thread_reservation_state) {
      throw rmm::logic_error("Thread already has reservation state set");
    }
    thread_reservation_state                     = std::make_unique<reservation_state>();
    thread_reservation_state->memory_reservation = std::move(reservation);
    thread_reservation_state->reservation_policy = std::move(policy);
    thread_reservation_state->oom_policy         = std::move(oom_policy);
  }

  reservation_aware_resource_adaptor::reservation_state* get_stream_state(
    rmm::cuda_stream_view stream) override
  {
    return thread_reservation_state.get();
  }

  const reservation_aware_resource_adaptor::reservation_state* get_stream_state(
    rmm::cuda_stream_view stream) const override
  {
    return thread_reservation_state.get();
  }
};

}  // namespace

std::size_t
reservation_aware_resource_adaptor::reservation_state::check_reservation_and_handle_overflow(
  reservation_aware_resource_adaptor& adaptor,
  std::size_t allocation_size,
  rmm::cuda_stream_view stream)
{
  auto tracking_size           = rmm::align_up(allocation_size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto& allocated_bytes        = this->memory_reservation->allocated_bytes;
  auto updated_allocated_bytes = allocated_bytes.fetch_add(tracking_size) + tracking_size;
  // check if we are within reservation
  if (updated_allocated_bytes > memory_reservation->size) {
    std::lock_guard lock(_arbitration_mutex);
    updated_allocated_bytes = allocated_bytes.fetch_sub(tracking_size) - tracking_size;
    reservation_policy->handle_over_reservation(
      adaptor, stream, allocation_size, allocated_bytes, memory_reservation.get());
    updated_allocated_bytes = allocated_bytes.fetch_add(tracking_size) + tracking_size;
    if (updated_allocated_bytes > memory_reservation->size) {
      return updated_allocated_bytes - memory_reservation->size;
    }
  }
  return 0UL;
}

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  rmm::device_async_resource_ref upstream,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> default_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope)
  : _upstream(std::move(upstream)),
    _capacity(capacity),
    _memory_limit(capacity),
    _reservation_states([&]() -> std::unique_ptr<reservation_state_container> {
      if (tracking_scope == AllocationTrackingScope::PER_STREAM) {
        return std::make_unique<per_stream_reservation_state_container>();
      } else {
        return std::make_unique<per_thread_reservation_state_container>();
      }
    }()),
    _default_reservation_policy(default_reservation_policy
                                  ? std::move(default_reservation_policy)
                                  : make_default_reservation_limit_policy()),
    _default_oom_policy(default_oom_policy ? std::move(default_oom_policy)
                                           : make_default_oom_policy())
{
}

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  rmm::device_async_resource_ref upstream,
  std::size_t capacity,
  std::size_t memory_limit,
  std::unique_ptr<reservation_limit_policy> default_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope)
  : _upstream(std::move(upstream)),
    _capacity(capacity),
    _memory_limit(memory_limit),
    _reservation_states([&]() -> std::unique_ptr<reservation_state_container> {
      if (tracking_scope == AllocationTrackingScope::PER_STREAM) {
        return std::make_unique<per_stream_reservation_state_container>();
      } else {
        return std::make_unique<per_thread_reservation_state_container>();
      }
    }()),
    _default_reservation_policy(default_reservation_policy
                                  ? std::move(default_reservation_policy)
                                  : make_default_reservation_limit_policy()),
    _default_oom_policy(default_oom_policy ? std::move(default_oom_policy)
                                           : make_default_oom_policy())
{
}

rmm::device_async_resource_ref reservation_aware_resource_adaptor::get_upstream_resource()
  const noexcept
{
  return _upstream;
}

std::size_t reservation_aware_resource_adaptor::get_allocated_bytes(
  rmm::cuda_stream_view stream) const
{
  const auto* stats = _reservation_states->get_stream_state(stream);
  return stats ? stats->memory_reservation->allocated_bytes.load() : 0;
}

std::size_t reservation_aware_resource_adaptor::get_peak_allocated_bytes(
  rmm::cuda_stream_view stream) const
{
  const auto* stats = _reservation_states->get_stream_state(stream);
  return stats ? stats->peak_allocated_bytes.load() : 0;
}

std::size_t reservation_aware_resource_adaptor::get_total_allocated_bytes() const
{
  return _total_allocated_bytes.load();
}

std::size_t reservation_aware_resource_adaptor::get_peak_total_allocated_bytes() const
{
  return _peak_total_allocated_bytes.load();
}

void reservation_aware_resource_adaptor::reset_peak_allocated_bytes(rmm::cuda_stream_view stream)
{
  auto* stats = _reservation_states->get_stream_state(stream);
  if (stats) { stats->peak_allocated_bytes.store(0); }
}

std::size_t reservation_aware_resource_adaptor::get_total_reserved_bytes() const
{
  std::lock_guard lock(reservation_mutex);
  std::size_t total = 0;
  for (const auto& res : reservation_views) {
    total += res->size;
  }
  return total;
}

bool reservation_aware_resource_adaptor::is_stream_tracked(rmm::cuda_stream_view stream) const
{
  return _reservation_states->get_stream_state(stream) != nullptr;
}

bool reservation_aware_resource_adaptor::set_stream_reservation(
  rmm::cuda_stream_view stream,
  std::unique_ptr<reservation> reserved_bytes,
  std::unique_ptr<reservation_limit_policy> stream_reservation_policy,
  std::unique_ptr<oom_handling_policy> stream_oom_policy)
{
  auto* stats = _reservation_states->get_stream_state(stream);
  if (stats) { return false; }

  if (!stream_reservation_policy) {
    stream_reservation_policy = make_default_reservation_limit_policy();
  }

  if (!stream_oom_policy) { stream_oom_policy = make_default_oom_policy(); }

  _reservation_states->set_stream_state(stream,
                                        std::move(reserved_bytes),
                                        std::move(stream_reservation_policy),
                                        std::move(stream_oom_policy));

  return true;
}

std::unique_ptr<reservation> reservation_aware_resource_adaptor::reserve(std::size_t size_bytes)
{
  std::lock_guard lock(reservation_mutex);
  if (do_reserve(size_bytes, _memory_limit)) {
    auto res =
      reservation::create(tier, device_id.value(), size_bytes, [this, size_bytes](reservation* r) {
        this->do_release_reservation(r);
      });
    reservation_views.insert(res.get());
    return res;
  } else {
    return nullptr;
  }
}

void* reservation_aware_resource_adaptor::do_allocate(std::size_t bytes,
                                                      rmm::cuda_stream_view stream)
{
  auto* reservation_state = _reservation_states->get_stream_state(stream);
  if (reservation_state != nullptr) {
    return do_allocate_managed(bytes, reservation_state, stream);
  } else {
    return do_allocate_managed(bytes, stream);
  }
}

void* reservation_aware_resource_adaptor::do_allocate_managed(std::size_t bytes,
                                                              rmm::cuda_stream_view stream)
{
  auto tracking_size = rmm::align_up(bytes, 256);
  try {
    return do_allocate_unmanaged(bytes, tracking_size, stream);
  } catch (...) {
    return _default_oom_policy->handle_oom(
      bytes,
      stream,
      std::current_exception(),
      std::bind(&reservation_aware_resource_adaptor::do_allocate_unmanaged,
                this,
                std::placeholders::_1,
                tracking_size,
                std::placeholders::_2));
  }
}

void* reservation_aware_resource_adaptor::do_allocate_managed(std::size_t bytes,
                                                              reservation_state* state,
                                                              rmm::cuda_stream_view stream)
{
  auto tracking_size = state->check_reservation_and_handle_overflow(*this, bytes, stream);
  try {
    return do_allocate_unmanaged(bytes, tracking_size, stream);
  } catch (...) {
    try {
      return state->oom_policy->handle_oom(
        bytes,
        stream,
        std::current_exception(),
        std::bind(&reservation_aware_resource_adaptor::do_allocate_unmanaged,
                  this,
                  std::placeholders::_1,
                  tracking_size,
                  std::placeholders::_2));
    } catch (...) {
      state->memory_reservation->allocated_bytes.fetch_sub(tracking_size);
      throw;
    }
  }
}

void* reservation_aware_resource_adaptor::do_allocate_unmanaged(std::size_t allocation_bytes,
                                                                std::size_t tracking_bytes,
                                                                rmm::cuda_stream_view stream)
{
  auto new_allocation_size = _total_allocated_bytes.fetch_add(tracking_bytes) + tracking_bytes;
  if (new_allocation_size <= _capacity) {
    auto peak_allocated = _peak_total_allocated_bytes.load();
    while (
      new_allocation_size > peak_allocated &&
      !_peak_total_allocated_bytes.compare_exchange_weak(peak_allocated, new_allocation_size)) {}

    try {
      return _upstream.allocate_async(allocation_bytes, stream);
    } catch (std::exception& e) {
      _total_allocated_bytes.fetch_sub(tracking_bytes);
      throw sirius_out_of_memory(e.what(), allocation_bytes, _total_allocated_bytes.load());
    }
  } else {
    _total_allocated_bytes.fetch_sub(tracking_bytes);
    throw sirius_out_of_memory(
      "not enough capacity to allocate memory", allocation_bytes, _total_allocated_bytes.load());
  }
}

void reservation_aware_resource_adaptor::do_deallocate(void* ptr,
                                                       std::size_t bytes,
                                                       rmm::cuda_stream_view stream) noexcept
{
  auto* reservation_state = _reservation_states->get_stream_state(stream);
  if (reservation_state != nullptr) {
    _upstream.deallocate_async(ptr, bytes, stream);
    return;
  }
  _upstream.deallocate_async(ptr, bytes, stream);
}

bool reservation_aware_resource_adaptor::do_is_equal(
  const rmm::mr::device_memory_resource& other) const noexcept
{
  // Check if it's the same type
  const auto* other_adaptor = dynamic_cast<const reservation_aware_resource_adaptor*>(&other);
  if (other_adaptor == nullptr) { return false; }

  return _upstream == other_adaptor->get_upstream_resource();
}

bool reservation_aware_resource_adaptor::do_reserve(std::size_t size_bytes, std::size_t limit_bytes)
{
  auto new_allocated_bytes = _total_allocated_bytes.fetch_add(size_bytes) + size_bytes;
  if (new_allocated_bytes <= limit_bytes) {
    return true;
  } else {
    _total_allocated_bytes.fetch_sub(size_bytes);
    return false;
  }
}

void reservation_aware_resource_adaptor::do_release_reservation(reservation* reservation)
{
  std::lock_guard lock(reservation_mutex);
  reservation_views.erase(reservation);
  if (reservation->size > reservation->allocated_bytes.load()) {
    auto released_bytes = reservation->size - reservation->allocated_bytes.load();
    _total_allocated_bytes.fetch_sub(released_bytes);
  }
}

}  // namespace memory
}  // namespace sirius
