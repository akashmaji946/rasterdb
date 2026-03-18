/*
 * RasterDB cudf::io compatibility shim.
 */
#pragma once
#include "../../types.hpp"
#include "../../column.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace io {

class datasource {
public:
    virtual ~datasource() = default;
    struct buffer {
        const uint8_t* data_{nullptr};
        size_t size_{0};
        const uint8_t* data() const { return data_; }
        size_t size() const { return size_; }
    };
    virtual std::unique_ptr<buffer> host_read(size_t offset, size_t size) = 0;
    virtual size_t host_read(size_t offset, size_t size, uint8_t* dst) = 0;
    virtual size_t size() const = 0;
};

struct source_info {
    std::vector<std::string> filepaths;
    source_info() = default;
    source_info(std::string const& path) : filepaths{path} {}
    source_info(std::vector<std::string> const& paths) : filepaths(paths) {}
};

struct table_with_metadata {
    std::unique_ptr<cudf::table> tbl;
    struct column_name_info {
        std::string name;
        std::vector<column_name_info> children;
    };
    std::vector<column_name_info> metadata;
};

struct table_input_metadata {
    struct column_in_metadata {
        std::string name;
        bool nullable{true};
    };
    std::vector<column_in_metadata> column_metadata;
};

// datasource::create static factory
inline std::unique_ptr<datasource> create(std::string const&) { return nullptr; }

// text namespace — byte_range_info
namespace text {
struct byte_range_info {
    std::size_t offset{0};
    std::size_t size{0};
    byte_range_info() = default;
    byte_range_info(std::size_t o, std::size_t s) : offset(o), size(s) {}
    std::size_t end() const { return offset + size; }
};
} // namespace text

// parquet experimental stubs
namespace parquet {
namespace experimental {
class hybrid_scan_reader {};
} // namespace experimental
struct FileMetaData {};
} // namespace parquet

} // namespace io
} // namespace cudf
