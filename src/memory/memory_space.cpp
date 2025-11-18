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

#include "memory/memory_space.hpp"

#include "memory/memory_reservation.hpp"
#include "memory/oom_handling_policy.hpp"
#include "memory/reservation_aware_resource_adaptor.hpp"

#include <rmm/cuda_device.hpp>

#include <optional>
#include <sstream>
#include <stdexcept>

namespace sirius {
namespace memory {

//===----------------------------------------------------------------------===//
// memory_space Implementation
//===----------------------------------------------------------------------===//

memory_space::memory_space(Tier tier,
                           int device_id,
                           size_t memory_limit,
                           std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators,
                           std::optional<std::size_t> capacity)
  : _tier(tier),
    _device_id(device_id),
    _memory_limit(memory_limit),
    _capacity([&]() {
      if (capacity.has_value()) {
        return capacity.value();
      } else if (tier == Tier::GPU) {
        // Query device for total memory capacity
        rmm::cuda_set_device_raii guard(rmm::cuda_device_id{device_id});
        return rmm::available_device_memory().second;
      } else {
        return std::numeric_limits<std::size_t>::max();
      }
    }()),
    _allocator(std::move(allocators[0])),
    _reserving_adaptor(std::make_unique<reservation_aware_resource_adaptor>(
      *_allocator,
      _capacity,
      _memory_limit,
      make_default_reservation_limit_policy(),
      make_default_oom_policy(),
      reservation_aware_resource_adaptor::AllocationTrackingScope::PER_THREAD))
{
  if (memory_limit == 0) { throw std::invalid_argument("Memory limit must be greater than 0"); }
  if (!_allocator) { throw std::invalid_argument("At least one allocator must be provided"); }
}

memory_space::~memory_space() = default;

bool memory_space::operator==(const memory_space& other) const
{
  return _tier == other._tier && _device_id == other._device_id;
}

bool memory_space::operator!=(const memory_space& other) const { return !(*this == other); }

Tier memory_space::get_tier() const { return _tier; }

int memory_space::get_device_id() const { return _device_id; }

std::unique_ptr<reservation> memory_space::request_reservation(size_t size)
{
  return std::unique_ptr<reservation>(nullptr);
}

void memory_space::release_reservation(std::unique_ptr<reservation> res) {}

bool memory_space::shrink_reservation(reservation* res, size_t new_size) { return true; }

bool memory_space::grow_reservation(reservation* res, size_t new_size) { return true; }

size_t memory_space::get_available_memory() const { return 0; }

size_t memory_space::get_total_reserved_memory() const { return 0; }

size_t memory_space::get_max_memory() const { return _memory_limit; }

size_t memory_space::get_active_reservation_count() const { return 0; }

rmm::device_async_resource_ref memory_space::get_default_allocator() const noexcept
{
  return *_allocator;
}

bool memory_space::can_reserve(size_t size) const { return 0; }

std::string memory_space::to_string() const
{
  std::ostringstream oss;
  oss << "memory_space(tier=";
  switch (_tier) {
    case Tier::GPU: oss << "GPU"; break;
    case Tier::HOST: oss << "HOST"; break;
    case Tier::DISK: oss << "DISK"; break;
    default: oss << "UNKNOWN"; break;
  }
  oss << ", device_id=" << _device_id << ", limit=" << _memory_limit << ")";
  return oss.str();
}

void memory_space::wait_for_memory(size_t size, std::unique_lock<std::mutex>& lock) {}

bool memory_space::validate_reservation(const reservation* res) const
{
  return res && res->tier == _tier && res->device_id == _device_id;
}

//===----------------------------------------------------------------------===//
// memory_space_hash Implementation
//===----------------------------------------------------------------------===//

size_t memory_space_hash::operator()(const memory_space& ms) const
{
  return std::hash<int>{}(static_cast<int>(ms.get_tier())) ^
         (std::hash<size_t>{}(ms.get_device_id()) << 1);
}

}  // namespace memory
}  // namespace sirius
