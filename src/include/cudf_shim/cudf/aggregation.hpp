/*
 * RasterDB cudf::aggregation compatibility shim.
 */
#pragma once
#include "../types.hpp"
#include "../column.hpp"

#include <memory>
#include <vector>
#include <stdexcept>

namespace cudf {

class aggregation {
public:
    using Kind = aggregation_kind;
    virtual ~aggregation() = default;
    aggregation_kind kind{aggregation_kind::SUM};
    virtual std::unique_ptr<aggregation> clone() const { return std::make_unique<aggregation>(*this); }
};

class groupby_aggregation : public aggregation {};
class reduce_aggregation : public aggregation {};
class scan_aggregation : public aggregation {};
class rolling_aggregation : public aggregation {};
class segmented_reduce_aggregation : public aggregation {};

// Factory functions
inline std::unique_ptr<aggregation> make_sum_aggregation() {
    auto a = std::make_unique<aggregation>(); a->kind = aggregation_kind::SUM; return a;
}
inline std::unique_ptr<aggregation> make_min_aggregation() {
    auto a = std::make_unique<aggregation>(); a->kind = aggregation_kind::MIN; return a;
}
inline std::unique_ptr<aggregation> make_max_aggregation() {
    auto a = std::make_unique<aggregation>(); a->kind = aggregation_kind::MAX; return a;
}
inline std::unique_ptr<aggregation> make_count_aggregation(null_policy = null_policy::EXCLUDE) {
    auto a = std::make_unique<aggregation>(); a->kind = aggregation_kind::COUNT_VALID; return a;
}
inline std::unique_ptr<aggregation> make_mean_aggregation() {
    auto a = std::make_unique<aggregation>(); a->kind = aggregation_kind::MEAN; return a;
}
inline std::unique_ptr<aggregation> make_any_aggregation() {
    auto a = std::make_unique<aggregation>(); a->kind = aggregation_kind::ANY; return a;
}
inline std::unique_ptr<aggregation> make_all_aggregation() {
    auto a = std::make_unique<aggregation>(); a->kind = aggregation_kind::ALL; return a;
}
inline std::unique_ptr<aggregation> make_nunique_aggregation(null_policy = null_policy::EXCLUDE) {
    auto a = std::make_unique<aggregation>(); a->kind = aggregation_kind::NUNIQUE; return a;
}
inline std::unique_ptr<aggregation> make_nth_element_aggregation(size_type n, null_policy = null_policy::INCLUDE) {
    auto a = std::make_unique<aggregation>(); a->kind = aggregation_kind::NTH_ELEMENT; return a;
}

// Groupby
class groupby {
public:
    struct aggregation_request {
        column_view values;
        std::vector<std::unique_ptr<groupby_aggregation>> aggregations;
    };
    struct aggregation_result {
        std::vector<std::unique_ptr<column>> results;
    };

    groupby(table_view const& keys, null_policy include_null = null_policy::EXCLUDE,
            sorted keys_are_sorted = sorted::NO,
            std::vector<order> const& col_order = {},
            std::vector<null_order> const& null_prec = {})
        : _keys(keys) {}

    std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
    aggregate(std::vector<aggregation_request>&& requests) {
        throw std::runtime_error("cudf::groupby not implemented in RasterDB — use rasterdf");
    }

private:
    table_view _keys;
};

// Reduction
inline std::unique_ptr<scalar> reduce(column_view const&, reduce_aggregation const&,
                                       data_type = data_type{type_id::EMPTY}) {
    throw std::runtime_error("cudf::reduce not implemented in RasterDB — use rasterdf");
}

} // namespace cudf
