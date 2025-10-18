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

#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>
#include <memory>
#include <thread>
#include <array>
#include "common.hpp"

namespace sirius {
namespace memory {

class MemoryReservationManager; // Forward declaration to resolve circular dependency

/**
 * @brief A RAII wrapper representing a memory reservation in a specific tier.
 * 
 * This mean that the owner of a reservation doesn't need to call release reservation but the reservation automatically
 * gets released when the reservation goes out of scope.
 */
class Reservation { 
public:   
    friend class MemoryReservationManager;

    Tier tier; // The tier of memory for the reservation
    size_t size; // The size of the reservation in bytes

    /**
     *  @brief Construct a new Reservation object
     * 
     *  @param t The memory tier of the reservation
     *  @param s The size of the reservation in bytes 
     *  @param manager Pointer to the MemoryReservationManager managing this reservation
     */
    Reservation(Tier t, size_t s, MemoryReservationManager* manager);

    /** 
     * @brief Destructor for the Reservation object
     * 
     * Releases the reservation back to the MemoryReservationManager when the object goes out of scope.
     */
    ~Reservation();

    // Disable copy to prevent double deallocation
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;

    // Enable move semantics
    Reservation(Reservation&& other) noexcept;
    Reservation& operator=(Reservation&& other) noexcept;

private:
    MemoryReservationManager* manager_;
};

/**
 * @brief Singleton class to manage memory reservations across different tiers.
 * 
 * The reservation mamanger is primarily used to track the memory usage of currently
 * running tasks to prevent over-allocation by scheduling too many memory-intensive tasks at once.
 * 
 * Any Task Executor must ensure that is able to acquire a reservation for a task,
 * based on its estimated memory usage, before scheduling it for execution. 
 * 
 * TODO: The MemoryReservationManager should communicate with the Memory Allocator to determine
 * the amount of memory that has already been allocated as well as the reservations for the running
 * task before granting new reservations at every tier.
 */
class MemoryReservationManager {
public:
    friend class Reservation; 

    /**
     * @brief Initialize the singleton instance with tier limits.
     * 
     * This method must be called before getInstance() and only once.
     * 
     * @param tier_limits An array specifying the memory limit for each tier.
     */
    static void initialize(const std::array<size_t, static_cast<size_t>(Tier::SIZE)>& tier_limits);
    
    /**
     * @brief Get the singleton instance of MemoryReservationManager.
     * 
     * @return Reference to the singleton instance.
     * @throws std::runtime_error if the instance has not been initialized.
     */
    static MemoryReservationManager& getInstance();
    
    // Disable copy and move constructors and assignment operators
    MemoryReservationManager(const MemoryReservationManager&) = delete;
    MemoryReservationManager& operator=(const MemoryReservationManager&) = delete;
    MemoryReservationManager(MemoryReservationManager&&) = delete;
    MemoryReservationManager& operator=(MemoryReservationManager&&) = delete;
    
    /**
     * @brief Request a memory reservation of the specified size in the given tier.
     * 
     * If sufficient memory is not available, this call will block until the requested memory is freed.
     * 
     * @param tier The memory tier to reserve from.
     * @param size The size of memory to reserve in bytes.
     * @return A unique_ptr to a Reservation object representing the reservation.
     */
    std::unique_ptr<Reservation> requestReservation(Tier tier, size_t size);
    
    /** 
     * @brief Attempt to shrink an existing reservation to a new size.
     * 
     * @param reservation Pointer to the existing Reservation object.
     * @param new_size The desired new size in bytes.
     * @return true if the reservation was successfully shrunk, false otherwise.
    */
    bool shrinkReservation(Reservation* reservation, size_t new_size);

    /**
     * @brief Attempt to grow an existing reservation to a new size.
     * 
     * @param reservation Pointer to the existing Reservation object.
     * @param new_size The desired new size in bytes.
     * @return true if the reservation was successfully grown, false otherwise.
     */
    bool growReservation(Reservation* reservation, size_t new_size);
    
    /**
     * @brief Get the amount of available memory in the specified tier.
     * 
     * @param tier The memory tier to query.
     * @return The amount of available memory in bytes.
     */
    size_t getAvailableMemory(Tier tier) const;

    /**
     * @brief Get the amount of reserved memory in the specified tier.
     * 
     * @param tier The memory tier to query.
     * 
     * @return The amount of reserved memory in bytes.
     */
    size_t getTotalReservedMemory(Tier tier) const;

    /**
     * @brief Get the maximum reservation limit for the specified tier.
     * 
     * @param tier The memory tier to query.
     * 
     * @return The maximum reservation limit in bytes.
     */
    size_t getMaxReservation(Tier tier) const;
    
    /**
     * @brief Get the number of active reservations in the specified tier.
     * 
     * @param tier The memory tier to query.
     * 
     * @return The number of active reservations.
     */
    size_t getActiveReservationCount(Tier tier) const;
    
public:
    /**
     * @brief Construct a new Memory Reservation Manager object
     * 
     * @param tier_limits An array specifying the memory limit for each tier.
     */
    explicit MemoryReservationManager(const std::array<size_t, static_cast<size_t>(Tier::SIZE)>& tier_limits);
    
    /**
     * @brief Destructor for the MemoryReservationManager
     */
    ~MemoryReservationManager() = default;

private:
    
    static std::unique_ptr<MemoryReservationManager> instance_; // A singleton instance of the manager
    static std::once_flag initialized_; // Flag to ensure single initialization
    
    /**
     * @brief Struct to hold information about each memory tier.
     */
    struct TierInfo {
        const size_t limit; // The maximum limit for this tier in bytes
        std::atomic<size_t> total_reserved{0}; // The total amount of memory currently reserved in bytes
        std::atomic<size_t> active_count{0}; // The number of active reservations in this tier
        
        /**
         * @brief Constructor for TierInfo
         * 
         * @param l The maximum limit for this tier in bytes
         */
        TierInfo(size_t l) : limit(l) {}
    };
    
    mutable std::mutex mutex_; // Mutex to protect shared data
    std::condition_variable cv_; // Condition variable to notify waiting threads about memory availability
    
    TierInfo tier_info_[static_cast<size_t>(Tier::SIZE)]; //dynamic size of the enum since SIZE is the last value
    
    /**
     * @brief Get the index corresponding to the given tier.
     * 
     * @param tier The memory tier.
     * @return The index of the tier in the tier_info_ array.
     * @throws std::invalid_argument if the tier is invalid.
     */
    size_t getTierIndex(Tier tier) const;

    /**
     * @brief Check if a reservation of the specified size can be made in the given tier.
     * 
     * @param tier The memory tier to check.
     * @param size The size of memory to reserve in bytes.
     * @return true if the reservation can be made, false otherwise.
     */
    bool canReserve(Tier tier, size_t size) const;

    /**
     * @brief Wait until sufficient memory is available to make a reservation.
     * 
     * This method will block until the requested memory can be reserved. It checks the memory availability
     * in a loop and waits on the condition variable if memory is not available.
     * 
     * @param tier The memory tier to reserve from.
     * @param size The size of memory to reserve in bytes.
     * @param lock A unique_lock that is used to wait on the conditional variable
     */
    void waitForMemory(Tier tier, size_t size, std::unique_lock<std::mutex>& lock);

    /**
     * @brief Releases the specified amount of memory from the given tier.
     * 
     * This method also notifies any threads waiting for memory reservations to retry thier reservations. 
     * 
     * @param tier The memory tier to release from.
     * @param size The size of memory to release in bytes.
     */
    void release_memory(Tier tier, size_t size);
};

} // namespace memory
} // namespace sirius