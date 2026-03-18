/*
 * RasterDB cucascade::memory::reservation_aware_resource_adaptor shim.
 */
#pragma once
#include "common.hpp"
#include "memory_reservation.hpp"

namespace cucascade {
namespace memory {

class reservation_aware_resource_adaptor : public device_memory_resource {
public:
    reservation_aware_resource_adaptor() = default;
    explicit reservation_aware_resource_adaptor(reservation&) {}
    ~reservation_aware_resource_adaptor() override = default;
};

} // namespace memory
} // namespace cucascade
