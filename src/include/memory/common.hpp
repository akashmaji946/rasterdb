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

#include <cstddef>
#include <memory>

namespace sirius {
namespace memory {

// Enum to represent different memory tiers
enum class Tier {
    GPU,
    HOST,
    DISK,
    SIZE //value = size of the enum, allows code to be more dynamic
};

/**
 * @brief Interface to represent allocated memory within a specific tier
 * 
 * All the underlying implementations should be RAII-compliant in that
 * they should release the memory to the allocator when they go out of scope.
 */
class IAllocatedMemory { 
public:
    /**
     * @brief Get the tier of memory that this allocation resides in
     * 
     * @return Tier The memory tier
     */
    virtual Tier getTier() const = 0;

    /**
     * @brief Get the size of the allocated memory in bytes
     * 
     * @return std::size_t Size of the allocation in bytes
     */
    virtual std::size_t getAllocatedBytes() const = 0;
};

/**
 * @brief Interface for memory allocators managing different memory tiers.
 * 
 * The primary purpose of implementing this abstraction ourselves is that RMM doesn't have a generic representation for 
 * a memory allocator and instead has two base classes in device_memory_resource and host_memory_resource.
 * 
 */
class IMemoryAllocator { 
public:
    /**
     * @brief Get the tier of memory that this allocator manages
     * 
     * @return Tier The memory tier
     */
    virtual Tier getTier() const = 0;

    /**
     * @brief Allocates the requested number of bytes
     * 
     * @param total_bytes Total size in bytes to allocate
     * @return std::unique_ptr<IAllocatedMemory> A pointer to allocated memory. See IAllocatedMemory for more details.
     * @throws std::bad_alloc If the allocation fails
     */
    virtual std::unique_ptr<IAllocatedMemory> allocate_memory(size_t total_bytes) = 0;
};

} // namespace memory
} // namespace sirius
