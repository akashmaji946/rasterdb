/*
 * RasterDB cucascade::memory::host_table shim.
 */
#pragma once
#include "common.hpp"
#include "memory_space.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace cucascade {
namespace memory {

class host_table {
public:
    host_table() = default;
    host_table(std::size_t num_rows, std::size_t num_cols) : _num_rows(num_rows), _num_cols(num_cols) {}
    ~host_table() = default;

    std::size_t num_rows() const { return _num_rows; }
    std::size_t num_columns() const { return _num_cols; }

private:
    std::size_t _num_rows{0};
    std::size_t _num_cols{0};
};

} // namespace memory
} // namespace cucascade
