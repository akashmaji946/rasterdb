/*
 * Copyright 2026, RasterDB Contributors.
 * GPU Types — bridges DuckDB LogicalType ↔ rasterdf data_type.
 */

#pragma once

#include <rasterdf/core/types.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/types/hugeint.hpp>
#include <duckdb/common/types/vector.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
    case duckdb::LogicalTypeId::DECIMAL: {
      auto width = duckdb::DecimalType::GetWidth(type);
      auto scale = static_cast<int32_t>(duckdb::DecimalType::GetScale(type));
      if (width <= 9) {
        // DuckDB stores DECIMAL(1..4) as INT16. The Vulkan operator path
        // deliberately widens those values to the DECIMAL32 representation.
        return {rasterdf::type_id::INT32, scale};
      }
      if (width <= 18) {
        return {rasterdf::type_id::INT64, scale};
      }
      throw duckdb::NotImplementedException(
        "RasterDB GPU: DECIMAL width %d requires INT128 support — falling back to CPU",
        static_cast<int>(width));
    }
    case duckdb::LogicalTypeId::HUGEINT:   return {rasterdf::type_id::INT64};    // best-effort for large int intermediates
    case duckdb::LogicalTypeId::VARCHAR:   return {rasterdf::type_id::STRING};
    default:
      throw duckdb::NotImplementedException(
        "RasterDB GPU: unsupported type %s — falling back to CPU",
        type.ToString().c_str());
  }
}

inline bool is_decimal_type(const duckdb::LogicalType& type) {
  return type.id() == duckdb::LogicalTypeId::DECIMAL;
}

inline bool is_narrow_decimal_type(const duckdb::LogicalType& type) {
  return is_decimal_type(type) && type.InternalType() == duckdb::PhysicalType::INT16;
}

inline size_t rdf_type_size(rasterdf::type_id tid);

/// Copy a flat DuckDB vector into RasterDF physical storage.
/// DECIMAL(1..4, scale) is widened from DuckDB INT16 storage to the DECIMAL32
/// representation used by RasterDB operators. All other supported values are
/// already byte-compatible with their GPU representation.
inline size_t copy_duckdb_vector_to_rdf(const duckdb::Vector& vector,
                                        size_t count,
                                        const duckdb::LogicalType& source_type,
                                        rasterdf::data_type target_type,
                                        uint8_t* dst) {
  if (is_narrow_decimal_type(source_type)) {
    auto src = reinterpret_cast<const int16_t*>(vector.GetData());
    auto out = reinterpret_cast<int32_t*>(dst);
    for (size_t row = 0; row < count; row++) {
      out[row] = static_cast<int32_t>(src[row]);
    }
    return count * sizeof(int32_t);
  }
  size_t bytes = count * rdf_type_size(target_type.id);
  std::memcpy(dst, vector.GetData(), bytes);
  return bytes;
}

/// Materialize a GPU decimal payload into DuckDB's requested physical storage.
/// Returns false for non-decimal output types, allowing callers to continue
/// with ordinary numeric conversion handling.
inline bool copy_rdf_decimal_to_duckdb(const uint8_t* src,
                                       rasterdf::type_id source_type,
                                       const duckdb::LogicalType& target_type,
                                       size_t count,
                                       uint8_t* dst) {
  if (!is_decimal_type(target_type)) {
    return false;
  }

  switch (target_type.InternalType()) {
    case duckdb::PhysicalType::INT16: {
      if (source_type != rasterdf::type_id::INT32) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: DECIMAL16 result requires DECIMAL32 physical input");
      }
      auto input = reinterpret_cast<const int32_t*>(src);
      auto output = reinterpret_cast<int16_t*>(dst);
      for (size_t row = 0; row < count; row++) {
        if (input[row] < std::numeric_limits<int16_t>::min() ||
            input[row] > std::numeric_limits<int16_t>::max()) {
          throw duckdb::OutOfRangeException("RasterDB GPU: DECIMAL16 result overflow");
        }
        output[row] = static_cast<int16_t>(input[row]);
      }
      return true;
    }
    case duckdb::PhysicalType::INT32: {
      if (source_type != rasterdf::type_id::INT32) {
        throw duckdb::NotImplementedException(
          "RasterDB GPU: DECIMAL32 result requires DECIMAL32 physical input");
      }
      std::memcpy(dst, src, count * sizeof(int32_t));
      return true;
    }
    case duckdb::PhysicalType::INT64: {
      if (source_type == rasterdf::type_id::INT64) {
        std::memcpy(dst, src, count * sizeof(int64_t));
        return true;
      }
      if (source_type == rasterdf::type_id::INT32) {
        auto input = reinterpret_cast<const int32_t*>(src);
        auto output = reinterpret_cast<int64_t*>(dst);
        for (size_t row = 0; row < count; row++) {
          output[row] = static_cast<int64_t>(input[row]);
        }
        return true;
      }
      break;
    }
    case duckdb::PhysicalType::INT128: {
      if (source_type == rasterdf::type_id::INT32 ||
          source_type == rasterdf::type_id::INT64) {
        for (size_t row = 0; row < count; row++) {
          int64_t value = source_type == rasterdf::type_id::INT64
                            ? reinterpret_cast<const int64_t*>(src)[row]
                            : static_cast<int64_t>(reinterpret_cast<const int32_t*>(src)[row]);
          duckdb::hugeint_t wide;
          wide.lower = static_cast<uint64_t>(value);
          wide.upper = value < 0 ? -1 : 0;
          std::memcpy(dst + row * sizeof(duckdb::hugeint_t), &wide, sizeof(duckdb::hugeint_t));
        }
        return true;
      }
      break;
    }
    default:
      break;
  }
  throw duckdb::NotImplementedException(
    "RasterDB GPU: unsupported decimal materialization from type_id %d to %s",
    static_cast<int>(source_type), target_type.ToString().c_str());
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
    case rasterdf::type_id::STRING:               return 0; // variable-width; use offsets+chars
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
