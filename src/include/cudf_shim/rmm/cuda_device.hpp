/* RasterDB rmm compatibility shim — device management stubs */
#pragma once
#include "../types.hpp"
#include "../column.hpp"

// cuda_device_id and cuda_set_device_raii are defined in column.hpp
// This header just provides the include path compatibility.

namespace rmm {
inline cuda_device_id get_current_cuda_device() { return cuda_device_id{0}; }
} // namespace rmm
