/*
 * RasterDB cucascade::memory::reservation_manager_configurator shim.
 */
#pragma once
#include "memory_reservation_manager.hpp"
#include "config.hpp"

namespace cucascade {
namespace memory {

class reservation_manager_configurator {
public:
    static std::unique_ptr<memory_reservation_manager> create(const memory_space_config&) {
        return std::make_unique<memory_reservation_manager>();
    }
};

} // namespace memory
} // namespace cucascade
