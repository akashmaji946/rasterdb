/*
 * Copyright 2025, RasterDB Contributors.
 *
 * host_parquet_representation — STUBBED for RasterDB.
 * The cudf hybrid_scan_reader has been removed (CUDA-only).
 * Parquet reading is handled by DuckDB's built-in parquet reader
 * or rasterdf's I/O layer.
 */

#pragma once

#include <cucascade/data/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>
#include <stdexcept>
#include <cstddef>

namespace rasterdb {

/**
 * @brief Stub host_parquet_representation for RasterDB.
 *
 * The original cudf hybrid_scan_reader based implementation has been removed.
 * All methods throw — parquet reading goes through DuckDB CPU path or rasterdf.
 */
class host_parquet_representation : public cucascade::idata_representation {
 public:
  host_parquet_representation() = default;

  std::size_t num_rows() const override { return 0; }
  std::size_t num_columns() const override { return 0; }
  std::size_t size_bytes() const override { return 0; }
  cucascade::memory::memory_space_id location() const override {
    return cucascade::memory::memory_space_id{cucascade::memory::Tier::HOST, 0};
  }

  std::size_t get_size_in_bytes() const { return 0; }
  std::size_t get_uncompressed_size_in_bytes() const { return 0; }

  std::unique_ptr<cucascade::idata_representation> clone() {
    throw std::runtime_error("host_parquet_representation: CUDA hybrid scan removed in RasterDB");
  }
  std::unique_ptr<cucascade::idata_representation> shallow_clone() {
    throw std::runtime_error("host_parquet_representation: CUDA hybrid scan removed in RasterDB");
  }

  std::vector<cudf::size_type> const& get_row_group_indices() const { return _empty_rg; }
  std::vector<cudf::size_type>& get_row_group_indices() { return _empty_rg; }

  cudf::host_span<cudf::size_type const> get_rg_span() const {
    return cudf::host_span<cudf::size_type const>(_empty_rg.data(), _empty_rg.size());
  }

 private:
  std::vector<cudf::size_type> _empty_rg;
};

}  // namespace rasterdb
