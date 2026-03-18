/*
 * RasterDB cucascade::host_data_representation shim.
 */
#pragma once
#include "common.hpp"
#include "../memory/common.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace cucascade {

class host_data_representation : public idata_representation {
public:
    host_data_representation() = default;
    ~host_data_representation() override = default;

    std::size_t num_rows() const override { return _num_rows; }
    std::size_t num_columns() const override { return _num_cols; }
    std::size_t size_bytes() const override { return _size_bytes; }
    memory::memory_space_id location() const override {
        return memory::memory_space_id{memory::Tier::HOST, 0};
    }

    void set_num_rows(std::size_t n) { _num_rows = n; }
    void set_num_columns(std::size_t n) { _num_cols = n; }

private:
    std::size_t _num_rows{0};
    std::size_t _num_cols{0};
    std::size_t _size_bytes{0};
};

} // namespace cucascade
