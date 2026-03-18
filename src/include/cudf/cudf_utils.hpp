/*
 * Copyright 2025, RasterDB Contributors.
 *
 * cudf_utils.hpp — Type bridge between DuckDB and cudf shim types.
 * Originally used real cudf headers; now uses RasterDB's cudf compatibility shim.
 */

#pragma once

// Use our cudf compatibility shim (not real cudf)
#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/sorting.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/round.hpp>
#include <cudf/unary.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/reduction/distinct_count.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <duckdb/common/exception.hpp>
#include <duckdb/common/types.hpp>

#include <memory>
#include <vector>

// Compatibility: define version macros so existing #if guards compile
#ifndef CUDF_VERSION_MAJOR
#define CUDF_VERSION_MAJOR 99
#endif
#ifndef CUDF_VERSION_MINOR
#define CUDF_VERSION_MINOR 99
#endif
#define CUDF_VERSION_NUM (CUDF_VERSION_MAJOR * 100 + CUDF_VERSION_MINOR)

namespace duckdb {

inline bool IsCudfTypeDecimal(const cudf::data_type& type)
{
  return type.id() == cudf::type_id::DECIMAL32 || type.id() == cudf::type_id::DECIMAL64 ||
         type.id() == cudf::type_id::DECIMAL128;
}

inline int GetCudfDecimalTypeSize(const cudf::data_type& type)
{
  if (type.id() == cudf::type_id::DECIMAL32) { return sizeof(int32_t); }
  if (type.id() == cudf::type_id::DECIMAL64) { return sizeof(int64_t); }
  if (type.id() == cudf::type_id::DECIMAL128) { return sizeof(__int128_t); }
  throw InternalException("Non decimal cudf type called in `GetCudfDecimalTypeSize`: %d",
                          static_cast<int>(type.id()));
}

/**
 * @brief Type switch from duckdb LogicalType to cudf data_type
 */
inline cudf::data_type GetCudfType(const LogicalType& logical_type)
{
  switch (logical_type.id()) {
    case LogicalTypeId::TINYINT: return cudf::data_type(cudf::type_id::INT8);
    case LogicalTypeId::SMALLINT: return cudf::data_type(cudf::type_id::INT16);
    case LogicalTypeId::INTEGER: return cudf::data_type(cudf::type_id::INT32);
    case LogicalTypeId::BIGINT:
    case LogicalTypeId::HUGEINT:
      return cudf::data_type(cudf::type_id::INT64);
    case LogicalTypeId::UTINYINT: return cudf::data_type(cudf::type_id::UINT8);
    case LogicalTypeId::USMALLINT: return cudf::data_type(cudf::type_id::UINT16);
    case LogicalTypeId::UINTEGER: return cudf::data_type(cudf::type_id::UINT32);
    case LogicalTypeId::UBIGINT:
    case LogicalTypeId::UHUGEINT: return cudf::data_type(cudf::type_id::UINT64);
    case LogicalTypeId::FLOAT: return cudf::data_type(cudf::type_id::FLOAT32);
    case LogicalTypeId::DOUBLE: return cudf::data_type(cudf::type_id::FLOAT64);
    case LogicalTypeId::BOOLEAN: return cudf::data_type(cudf::type_id::BOOL8);
    case LogicalTypeId::DATE: return cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
    case LogicalTypeId::TIMESTAMP_SEC: return cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS);
    case LogicalTypeId::TIMESTAMP_MS: return cudf::data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
    case LogicalTypeId::TIMESTAMP: return cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS);
    case LogicalTypeId::TIMESTAMP_NS: return cudf::data_type(cudf::type_id::TIMESTAMP_NANOSECONDS);
    case LogicalTypeId::VARCHAR: return cudf::data_type(cudf::type_id::STRING);
    case LogicalTypeId::STRUCT: return cudf::data_type(cudf::type_id::STRUCT);
    case LogicalTypeId::DECIMAL: {
      switch (logical_type.InternalType()) {
        case PhysicalType::INT32:
          return cudf::data_type(cudf::type_id::DECIMAL32, -DecimalType::GetScale(logical_type));
        case PhysicalType::INT64:
          return cudf::data_type(cudf::type_id::DECIMAL64, -DecimalType::GetScale(logical_type));
        case PhysicalType::INT128:
          return cudf::data_type(cudf::type_id::DECIMAL128, -DecimalType::GetScale(logical_type));
        default:
          throw InvalidInputException("GetCudfType: Unsupported duckdb decimal physical type: %d",
                                      static_cast<int>(logical_type.InternalType()));
      }
    }
    default:
      throw InvalidInputException("GetCudfType: Unsupported duckdb type: %d",
                                  static_cast<int>(logical_type.id()));
  }
}

inline std::unique_ptr<cudf::table> make_empty_like(cudf::table_view input)
{
  std::vector<std::unique_ptr<cudf::column>> empty_cols;
  empty_cols.reserve(input.num_columns());
  for (cudf::size_type col_idx = 0; col_idx < input.num_columns(); ++col_idx) {
    empty_cols.push_back(cudf::make_empty_column(input.column(col_idx).type()));
  }
  return std::make_unique<cudf::table>(std::move(empty_cols));
}

}  // namespace duckdb
