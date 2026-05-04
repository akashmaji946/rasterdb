/*
 * Copyright 2025, RasterDB Contributors.
 * GPU Types — bridges DuckDB LogicalType ↔ rasterdf data_type.
 */

#pragma once

#include <rasterdf/core/types.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/exception.hpp>

#include <cstddef>
#include <rasterdf/execution/constants.hpp>

namespace rasterdb {
namespace gpu {

/// Convert DuckDB LogicalType → rasterdf data_type.
/// Throws NotImplementedException for unsupported types (triggers CPU fallback).
inline rasterdf::data_type to_rdf_type(const duckdb::LogicalType& type) {
  switch (type.id()) {
    case duckdb::LogicalTypeId::TINYINT:   return {rasterdf::type_id::INT8};
    case duckdb::LogicalTypeId::SMALLINT:  return {rasterdf::type_id::INT16};
    case duckdb::LogicalTypeId::INTEGER:   return {rasterdf::type_id::INT32};
    case duckdb::LogicalTypeId::BIGINT:    return {rasterdf::type_id::INT64};
    case duckdb::LogicalTypeId::FLOAT:     return {rasterdf::type_id::FLOAT32};
    case duckdb::LogicalTypeId::DOUBLE:    return {rasterdf::type_id::FLOAT64};
    case duckdb::LogicalTypeId::BOOLEAN:   return {rasterdf::type_id::BOOL8};
    case duckdb::LogicalTypeId::DATE:      return {rasterdf::type_id::TIMESTAMP_DAYS};
    case duckdb::LogicalTypeId::TIMESTAMP: return {rasterdf::type_id::TIMESTAMP_MICROSECONDS};
    case duckdb::LogicalTypeId::DECIMAL:   return {rasterdf::type_id::FLOAT32};  // treat DECIMAL as float for GPU
    case duckdb::LogicalTypeId::HUGEINT:   return {rasterdf::type_id::INT64};    // best-effort for large int intermediates
    default:
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported type %s — falling back to CPU",
        type.ToString().c_str());
  }
}

/// Size in bytes of a single element for the given rasterdf type.
inline size_t rdf_type_size(rasterdf::type_id tid) {
  switch (tid) {
    case rasterdf::type_id::INT8:
    case rasterdf::type_id::BOOL8:                return 1;
    case rasterdf::type_id::INT16:                return 2;
    case rasterdf::type_id::INT32:
    case rasterdf::type_id::FLOAT32:
    case rasterdf::type_id::TIMESTAMP_DAYS:       return 4;
    case rasterdf::type_id::INT64:
    case rasterdf::type_id::FLOAT64:
    case rasterdf::type_id::TIMESTAMP_SECONDS:
    case rasterdf::type_id::TIMESTAMP_MILLISECONDS:
    case rasterdf::type_id::TIMESTAMP_MICROSECONDS:
    case rasterdf::type_id::TIMESTAMP_NANOSECONDS: return 8;
    default:
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported type_id %d for size computation",
        static_cast<int>(tid));
  }
}

/// Returns the dispatcher type_id code used in push constants (0=int32, 1=float32).
/// For types that don't have dedicated shaders yet, throws.

inline int32_t rdf_shader_type_id(rasterdf::type_id tid) {
  switch (tid) {
    case rasterdf::type_id::INT32:
    case rasterdf::type_id::TIMESTAMP_DAYS:   return static_cast<int32_t>(rasterdf::ShaderTypeId::INT32);
    case rasterdf::type_id::FLOAT32:          return static_cast<int32_t>(rasterdf::ShaderTypeId::FLOAT32);
    case rasterdf::type_id::INT64:            return static_cast<int32_t>(rasterdf::ShaderTypeId::INT64);
    case rasterdf::type_id::FLOAT64:          return static_cast<int32_t>(rasterdf::ShaderTypeId::FLOAT64);
    default:
      throw duckdb::NotImplementedException(
        "RasterDB GPU: type_id %d not yet supported in Vulkan shaders — falling back to CPU",
        static_cast<int>(tid));
  }
}

} // namespace gpu
} // namespace rasterdb
