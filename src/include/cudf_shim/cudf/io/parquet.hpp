/*
 * RasterDB cudf::io::parquet compatibility shim.
 */
#pragma once
#include "../../types.hpp"
#include "../../column.hpp"
#include "datasource.hpp"
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <functional>
#include <optional>

namespace cudf {
namespace io {

// Forward declaration
struct parquet_reader_options_builder;

// Parquet reader options
struct parquet_reader_options {
    source_info source;
    std::vector<std::string> columns;
    int64_t skip_rows{0};
    int64_t num_rows{-1};
    bool use_pandas_metadata{false};

    static parquet_reader_options_builder builder(source_info src);
};

struct parquet_reader_options_builder {
    parquet_reader_options opts;
    parquet_reader_options_builder(source_info src) { opts.source = std::move(src); }
    parquet_reader_options_builder& columns(std::vector<std::string> c) { opts.columns = std::move(c); return *this; }
    parquet_reader_options_builder& skip_rows(int64_t n) { opts.skip_rows = n; return *this; }
    parquet_reader_options_builder& num_rows(int64_t n) { opts.num_rows = n; return *this; }
    parquet_reader_options build() { return opts; }
    operator parquet_reader_options() { return opts; }
};

inline parquet_reader_options_builder parquet_reader_options::builder(source_info src) {
    return parquet_reader_options_builder(std::move(src));
}

// Parquet reader
class parquet_reader {
public:
    parquet_reader(parquet_reader_options const&) {}
    table_with_metadata read() {
        throw std::runtime_error("cudf::io::parquet_reader not implemented — use rasterdf");
    }
    bool has_next() { return false; }
    table_with_metadata read_chunk() {
        throw std::runtime_error("cudf::io::parquet_reader not implemented — use rasterdf");
    }
};

inline table_with_metadata read_parquet(parquet_reader_options const&) {
    throw std::runtime_error("cudf::io::read_parquet not implemented — use rasterdf");
}

// Parquet metadata / schema stubs
struct parquet_metadata {
    struct row_group_info {
        int64_t num_rows{0};
    };
    std::vector<row_group_info> row_groups;
    int64_t num_rows() const {
        int64_t total = 0;
        for (auto& rg : row_groups) total += rg.num_rows;
        return total;
    }
};

struct parquet_column_schema {
    std::string name;
    cudf::data_type type{};
};

namespace parquet {
    using reader = parquet_reader;
    using reader_options = parquet_reader_options;
}

} // namespace io
} // namespace cudf
