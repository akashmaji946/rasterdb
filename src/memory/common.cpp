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

#include "memory/common.hpp"

namespace sirius {

namespace memory {

const char* MemoryErrorCategory::name() const noexcept { return "SiriusMemorySystem"; }

std::string MemoryErrorCategory::message(int ev) const
{
  switch (static_cast<MemoryError>(ev)) {
    case MemoryError::SUCCESS: return "Success";
    case MemoryError::ALLOCATION_FAILED: return "System allocation failed";
    case MemoryError::LIMIT_EXCEEDED: return "Reservation limit exceeded";
    case MemoryError::POOL_EXHAUSTED: return "Internal memory pool exhausted";
    default: return "Unknown memory error";
  }
}

const MemoryErrorCategory& memory_category()
{
  static const MemoryErrorCategory instance;
  return instance;
}

std::error_code make_error_code(MemoryError e)
{
  return std::error_code(static_cast<int>(e), memory_category());
}

sirius_out_of_memory::sirius_out_of_memory(std::string_view message,
                                           std::size_t requested_bytes,
                                           std::size_t global_usage)
  : rmm::out_of_memory(message.data()), requested_bytes(requested_bytes), global_usage(global_usage)
{
}

}  // namespace memory
}  // namespace sirius

namespace std {
size_t hash<std::pair<sirius::memory::Tier, size_t>>::operator()(
  const std::pair<sirius::memory::Tier, size_t>& p) const
{
  return std::hash<int>{}(static_cast<int>(p.first)) ^ (std::hash<size_t>{}(p.second) << 1);
}
}  // namespace std
