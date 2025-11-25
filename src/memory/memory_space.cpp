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

#include "memory/common.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"
#include "memory/memory_reservation.hpp"
#include "memory/reservation_aware_resource_adaptor.hpp"

#include <rmm/cuda_device.hpp>

#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <variant>

namespace sirius {
namespace memory {

//===----------------------------------------------------------------------===//
// memory_space Implementation
//===----------------------------------------------------------------------===//

memory_space::memory_space(Tier tier,
                           int device_id,
                           size_t memory_limit,
                           std::unique_ptr<rmm::mr::device_memory_resource> allocator,
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
    _allocator(std::move(allocator))
{
  if (memory_limit == 0) { throw std::invalid_argument("Memory limit must be greater than 0"); }
  if (!_allocator) { throw std::invalid_argument("At least one allocator must be provided"); }
  if (tier == Tier::GPU) {
    _reservation_allocator = std::make_unique<reservation_aware_resource_adaptor>(
      _tier, _device_id, *_allocator, _memory_limit, _capacity);
  } else if (tier == Tier::HOST) {
    _reservation_allocator = std::make_unique<fixed_size_host_memory_resource>(
      _device_id, *_allocator, _memory_limit, _capacity);
  } else {
    _reservation_allocator = std::monostate{};
  }
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
  return std::visit(sirius::overloaded{
                      [&](auto others) { return std::unique_ptr<reservation>(nullptr); },
                      [&](std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                        return mr->reserve(size, [this] { this->notify_release_of_reservation(); });
                      },
                      [&](std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                        return mr->reserve(size, [this] { this->notify_release_of_reservation(); });
                      }},
                    _reservation_allocator);
}

void memory_space::notify_release_of_reservation()
{
  std::lock_guard lock(_reservation_mutex);
  _reservation_release = true;
  _reservation_cv.notify_one();
}

size_t memory_space::get_available_memory(rmm::cuda_stream_view stream) const
{
  return std::visit(
    sirius::overloaded{[&](auto others) { return _memory_limit; },
                       [&](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                         return mr->get_available_memory(stream);
                       },
                       [&](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                         return mr->get_available_memory();
                       }},
    _reservation_allocator);
}

size_t memory_space::get_available_memory() const
{
  return std::visit(
    sirius::overloaded{[&](auto others) { return _memory_limit; },
                       [](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                         return mr->get_available_memory();
                       },
                       [](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                         return mr->get_available_memory();
                       }},
    _reservation_allocator);
}

size_t memory_space::get_total_reserved_memory() const
{
  return std::visit(
    sirius::overloaded{[&](auto others) { return 0UL; },
                       [](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                         return mr->get_total_reserved_bytes();
                       },
                       [](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                         return mr->get_total_reserved_bytes();
                       }},
    _reservation_allocator);
}

size_t memory_space::get_max_memory() const noexcept { return _memory_limit; }

rmm::mr::device_memory_resource* memory_space::get_default_allocator() const noexcept
{
  return _allocator.get();
}

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
