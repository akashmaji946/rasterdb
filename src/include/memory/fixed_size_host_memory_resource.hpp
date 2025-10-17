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

#include <rmm/mr/host/host_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/aligned.hpp>
#include <rmm/detail/nvtx/ranges.hpp>
#include "helper/helper.hpp"
#include "memory/common.hpp"

#include <cstddef>
#include <memory>
#include <algorithm>

namespace sirius {

using sirius::memory::Tier;
using sirius::memory::IAllocatedMemory;
using sirius::memory::IMemoryAllocator;

/**
 * @brief Simple RAII wrapper representing an multiple block allocations made by the fixed_size_host_memory_resource
 */
class multiple_blocks_allocation : public IAllocatedMemory {
public:
    /**
     * @brief Construct a new multiple blocks allocation object
     * 
     * @param b The vector of blocks allocated
     * @param m The memory resource used for allocation/deallocation
     * @param bs The size of each block in bytes
     */
    multiple_blocks_allocation(sirius::vector<void*> b, rmm::mr::host_memory_resource* m, std::size_t bs)
        : blocks(std::move(b)), mr(m), block_size(bs) {}

    /** 
     * @brief Destructor for the multiple_blocks_allocation 
     * 
     * Returns all of the allocated blocks back to the allocator
     */
    ~multiple_blocks_allocation() {
        for (void* ptr : blocks) {
            mr->deallocate(ptr, block_size);
        }
    }

    // Disable copy to prevent double deallocation
    multiple_blocks_allocation(const multiple_blocks_allocation&) = delete;
    multiple_blocks_allocation& operator=(const multiple_blocks_allocation&) = delete;

    // Move semantics
    multiple_blocks_allocation(multiple_blocks_allocation&& other) noexcept
        : blocks(std::move(other.blocks)), mr(other.mr), block_size(other.block_size) {
        other.blocks.clear();
    }

    multiple_blocks_allocation& operator=(multiple_blocks_allocation&& other) noexcept {
        if (this != &other) {
            for (void* ptr : blocks) {
                mr->deallocate(ptr, block_size);
            }
            blocks = std::move(other.blocks);
            mr = other.mr;
            block_size = other.block_size;
            other.blocks.clear();
        }
        return *this;
    }

    /**
     * @brief Get the number of blocks allocated
     * 
     * @return std::size_t Number of blocks
     */
    std::size_t size() const noexcept { return blocks.size(); }

    /**
     * @brief Access a specific block by index
     * 
     * @param i Index of the block to access
     * @return void* Pointer to the block at index i
     */
    void* getBlock(std::size_t i) const { return blocks[i]; }

    /**
     * @brief Get the tier of memory that this allocation resides in
     * 
     * @return Tier The memory tier
     */
    Tier getTier() const override { return Tier::HOST; }

    /**
     * @brief Get the size of the allocated memory in bytes
     * 
     * @return std::size_t Size of the allocation in bytes
     */
    std::size_t getAllocatedBytes() const override { return blocks.size() * block_size; }

    /**
     * @brief Get the size of each block in bytes
     * 
     * @return std::size_t Size of each block
     */
    size_t getBlockSize() const noexcept { return block_size; }

private:
    sirius::vector<void*> blocks; // The vector of allocated blocks
    rmm::mr::host_memory_resource* mr;  // The memory resource used for allocation
    std::size_t block_size; // The size of each block in bytes
};

/**
 * @brief A host memory resource that allocates fixed-size blocks using pinned host memory as upstream.
 *
 * This memory resource pre-allocates a pool of fixed-size blocks from the pinned host memory
 * resource and manages them in a free list. Allocations are limited to the configured block size.
 * 
 * The pool is allocated as a single large allocation from the upstream resource and then split
 * into individual blocks for efficient memory management and reduced allocation overhead.
 * 
 * When the pool is exhausted, it automatically expands by allocating additional blocks from
 * the upstream resource, making it suitable for workloads with varying memory requirements.
 */
class fixed_size_host_memory_resource : public rmm::mr::host_memory_resource, public IMemoryAllocator {
public:
    friend class multiple_blocks_allocation; 

    static constexpr std::size_t default_block_size = 1 << 20;  ///< Default block size (1MB)
    static constexpr std::size_t default_pool_size = 128;       ///< Default number of blocks in pool<
    static constexpr std::size_t default_initial_number_pools = 4; ///< Default number of pools to pre-allocate

    /**
     * @brief Construct a new fixed-size host memory resource.
     *
     * @param block_size Size of each block in bytes
     * @param pool_size Number of blocks to pre-allocate
     * @param initial_pools Number of pools to pre-allocate
     */
    explicit fixed_size_host_memory_resource(
        std::size_t block_size = default_block_size,
        std::size_t pool_size = default_pool_size,
        std::size_t initial_pools = default_initial_number_pools);

    /**
     * @brief Construct with custom upstream resource.
     *
     * @param upstream_mr Upstream memory resource to use
     * @param block_size Size of each block in bytes
     * @param pool_size Number of blocks to pre-allocate
     * @param initial_pools Number of pools to pre-allocate
     */
    explicit fixed_size_host_memory_resource(
        sirius::unique_ptr<rmm::mr::host_memory_resource> upstream_mr,
        std::size_t block_size = default_block_size,
        std::size_t pool_size = default_pool_size,
        std::size_t initial_pools = default_initial_number_pools);

    // Disable copy and move
    fixed_size_host_memory_resource(const fixed_size_host_memory_resource&) = delete;
    fixed_size_host_memory_resource& operator=(const fixed_size_host_memory_resource&) = delete;
    fixed_size_host_memory_resource(fixed_size_host_memory_resource&&) = delete;
    fixed_size_host_memory_resource& operator=(fixed_size_host_memory_resource&&) = delete;

    /**
     * @brief Destructor - frees all allocated blocks.
     */
    ~fixed_size_host_memory_resource() override;

    /**
     * @brief Get the block size.
     *
     * @return std::size_t The size of each block in bytes
     */
    [[nodiscard]] std::size_t get_block_size() const noexcept;

    /**
     * @brief Get the number of free blocks.
     *
     * @return std::size_t Number of available blocks
     */
    [[nodiscard]] std::size_t get_free_blocks() const noexcept;

    /**
     * @brief Get the total number of blocks in the pool.
     *
     * @return std::size_t Total number of blocks
     */
    [[nodiscard]] std::size_t get_total_blocks() const noexcept;

    /**
     * @brief Get the upstream memory resource.
     *
     * @return rmm::mr::host_memory_resource* Pointer to upstream resource (nullptr if using pinned host)
     */
    [[nodiscard]] rmm::mr::host_memory_resource* get_upstream_resource() const noexcept;

    /**
     * @brief Allocate multiple blocks to satisfy a large allocation request.
     *
     * This method allocates the minimum number of blocks needed to satisfy the requested size.
     * The blocks are returned as a RAII wrapper that automatically deallocates all blocks
     * when it goes out of scope, preventing memory leaks.
     *
     * @param total_bytes Total size in bytes to allocate across multiple blocks
     * @return multiple_blocks_allocation RAII wrapper for the allocated blocks
     * @throws std::bad_alloc if insufficient blocks are available or upstream allocation fails
     */
    [[nodiscard]] sirius::unique_ptr<multiple_blocks_allocation> allocate_multiple_blocks(std::size_t total_bytes);

    // IMemoryAllocator interface implementation
    Tier getTier() const override { return Tier::HOST; }
    std::unique_ptr<IAllocatedMemory> allocate_memory(size_t total_bytes) override {
        return allocate_multiple_blocks(total_bytes);
    }

protected:
    /**
     * @brief Allocate memory of the specified size.
     *
     * @param bytes Size in bytes (must be <= block_size_)
     * @param alignment Alignment (ignored, uses block alignment)
     * @return void* Pointer to allocated memory
     * @throws std::bad_alloc if allocation size exceeds block size or upstream allocation fails
     */
    void* do_allocate(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) override;

    /**
     * @brief Deallocate memory.
     *
     * @param ptr Pointer to deallocate
     * @param bytes Size in bytes (must be <= block_size_)
     * @param alignment Alignment (ignored)
     */
    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) override;

    /**
     * @brief Check if this resource is equal to another.
     *
     * @param other Other resource to compare
     * @return bool True if equal
     */
    [[nodiscard]] bool do_is_equal(const rmm::mr::host_memory_resource& other) const noexcept override;

private:
    /**
     * @brief Expand the pool by allocating more blocks from upstream.
     * 
     * Allocates a new chunk of blocks and adds them to the free list.
     */
    void expand_pool();

    std::size_t block_size_;                                    ///< Size of each block
    std::size_t pool_size_;                                     ///< Number of blocks in pool
    sirius::unique_ptr<rmm::mr::host_memory_resource> upstream_mr_; ///< Upstream memory resource (optional)
    sirius::vector<void*> allocated_blocks_;                       ///< All allocated blocks
    sirius::vector<void*> free_blocks_;                           ///< Currently free blocks
    mutable sirius::mutex mutex_;                                 ///< Mutex for thread safety
};

} // namespace sirius