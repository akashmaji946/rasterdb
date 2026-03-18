/*
 * RasterDB cudf::sorting compatibility shim.
 */
#pragma once
#include "../types.hpp"
#include "../column.hpp"
#include <memory>
#include <vector>
#include <stdexcept>

namespace cudf {

inline std::unique_ptr<column> sorted_order(table_view const&,
    std::vector<order> const& = {}, std::vector<null_order> const& = {}) {
    throw std::runtime_error("cudf::sorted_order not implemented — use rasterdf");
}

inline std::unique_ptr<table> sort(table_view const&,
    std::vector<order> const& = {}, std::vector<null_order> const& = {}) {
    throw std::runtime_error("cudf::sort not implemented — use rasterdf");
}

inline std::unique_ptr<table> sort_by_key(table_view const&, table_view const&,
    std::vector<order> const& = {}, std::vector<null_order> const& = {}) {
    throw std::runtime_error("cudf::sort_by_key not implemented — use rasterdf");
}

inline bool is_sorted(table_view const&,
    std::vector<order> const& = {}, std::vector<null_order> const& = {}) {
    return false;
}

inline std::unique_ptr<table> stable_sort_by_key(table_view const&, table_view const&,
    std::vector<order> const& = {}, std::vector<null_order> const& = {}) {
    throw std::runtime_error("cudf::stable_sort_by_key not implemented — use rasterdf");
}

} // namespace cudf
