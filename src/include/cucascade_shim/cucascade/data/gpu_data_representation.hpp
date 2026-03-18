/*
 * RasterDB cucascade::gpu_table_representation shim.
 * GPU data now managed by rasterdf (Vulkan + VMA).
 */
#pragma once
#include "common.hpp"
#include "../memory/common.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace cucascade {

class gpu_table_representation : public idata_representation {
public:
    gpu_table_representation() = default;
    template <typename TablePtr>
    gpu_table_representation(TablePtr&&, memory::memory_space&) {}
    ~gpu_table_representation() override = default;

    std::size_t num_rows() const override { return _num_rows; }
    std::size_t num_columns() const override { return _num_cols; }
    std::size_t size_bytes() const override { return _size_bytes; }
    memory::memory_space_id location() const override {
        return memory::memory_space_id{memory::Tier::GPU, 0};
    }

    void set_num_rows(std::size_t n) { _num_rows = n; }
    void set_num_columns(std::size_t n) { _num_cols = n; }

    // Column data pointers (host-side pointers to GPU buffers)
    struct gpu_column {
        void* data{nullptr};
        std::size_t size{0};
        int32_t type_id{0};
    };

    std::vector<gpu_column>& columns() { return _columns; }
    const std::vector<gpu_column>& columns() const { return _columns; }

private:
    std::size_t _num_rows{0};
    std::size_t _num_cols{0};
    std::size_t _size_bytes{0};
    std::vector<gpu_column> _columns;
};

} // namespace cucascade
