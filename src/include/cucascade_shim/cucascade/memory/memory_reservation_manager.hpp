/*
 * RasterDB cucascade::memory::memory_reservation_manager shim.
 */
#pragma once
#include "common.hpp"
#include "config.hpp"
#include "memory_space.hpp"
#include "memory_reservation.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace cucascade {
namespace memory {

// Forward declarations
class reservation_aware_resource_adaptor;

struct reservation_request_strategy {
    explicit reservation_request_strategy(bool strong_ordering) : _strong_ordering(strong_ordering) {}
    virtual ~reservation_request_strategy() = default;
    virtual std::vector<memory_space*> get_candidates(class memory_reservation_manager&) const = 0;
    [[nodiscard]] bool has_strong_ordering() const noexcept { return _strong_ordering; }
protected:
    bool _strong_ordering{false};
};

struct any_memory_space_in_tier : reservation_request_strategy {
    Tier tier;
    int32_t preferred_device_id{-1};
    any_memory_space_in_tier(Tier t, int32_t dev = -1)
        : reservation_request_strategy(false), tier(t), preferred_device_id(dev) {}
    std::vector<memory_space*> get_candidates(memory_reservation_manager&) const override { return {}; }
};

struct specific_memory_space : reservation_request_strategy {
    memory_space_id target;
    specific_memory_space(memory_space_id id)
        : reservation_request_strategy(true), target(id) {}
    std::vector<memory_space*> get_candidates(memory_reservation_manager&) const override { return {}; }
};

class memory_reservation_manager {
public:
    memory_reservation_manager() = default;
    ~memory_reservation_manager() = default;

    std::optional<reservation> try_reserve(std::size_t, const reservation_request_strategy&) {
        return std::nullopt;
    }
    reservation reserve(std::size_t bytes, const reservation_request_strategy& s) {
        return reservation{};
    }
    void release(reservation&&) {}

    memory_space* get_memory_space(memory_space_id) { return nullptr; }
    std::span<memory_space*> get_all_memory_spaces() { return {}; }
    std::span<memory_space*> get_all_memory_spaces(Tier) { return {}; }

    bool has_memory_space(memory_space_id) const { return false; }
    std::size_t total_capacity(Tier) const { return 0; }
    std::size_t total_available(Tier) const { return 0; }
};

} // namespace memory
} // namespace cucascade
