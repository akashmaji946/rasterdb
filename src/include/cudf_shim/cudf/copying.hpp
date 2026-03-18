/*
 * RasterDB cudf::copying compatibility shim.
 */
#pragma once
#include "../types.hpp"
#include "../column.hpp"
#include <memory>
#include <vector>
#include <stdexcept>

namespace cudf {

inline std::unique_ptr<table> gather(table_view const&, column_view const&,
    out_of_bounds_policy = out_of_bounds_policy::DONT_CHECK) {
    throw std::runtime_error("cudf::gather not implemented — use rasterdf");
}

inline std::unique_ptr<table> scatter(table_view const&, column_view const&, table_view const&) {
    throw std::runtime_error("cudf::scatter not implemented — use rasterdf");
}

inline std::unique_ptr<column> copy_if_else(column_view const&, column_view const&, column_view const&) {
    throw std::runtime_error("cudf::copy_if_else not implemented — use rasterdf");
}

inline std::unique_ptr<column> allocate_like(column_view const& cv, mask_state ms = mask_state::UNALLOCATED) {
    return make_empty_column(cv.type());
}

inline std::unique_ptr<column> allocate_like(column_view const& cv, size_type size, mask_state ms = mask_state::UNALLOCATED) {
    return std::make_unique<column>(cv.type(), size, rmm::device_buffer(size * cudf::size_of(cv.type())));
}

// host_span and device_span are defined in column.hpp

} // namespace cudf
