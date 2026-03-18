/*
 * Copyright 2025, RasterDB Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op/rasterdb_physical_operator_type.hpp"

namespace rasterdb::op {

// LCOV_EXCL_START
std::string RasterdbPhysicalOperatorToString(RasterDBPhysicalOperatorType type)
{
  switch (type) {
    case RasterDBPhysicalOperatorType::TABLE_SCAN: return "TABLE_SCAN";
    case RasterDBPhysicalOperatorType::DUMMY_SCAN: return "DUMMY_SCAN";
    case RasterDBPhysicalOperatorType::CHUNK_SCAN: return "CHUNK_SCAN";
    case RasterDBPhysicalOperatorType::COLUMN_DATA_SCAN: return "COLUMN_DATA_SCAN";
    case RasterDBPhysicalOperatorType::DELIM_SCAN: return "DELIM_SCAN";
    case RasterDBPhysicalOperatorType::ORDER_BY: return "ORDER_BY";
    case RasterDBPhysicalOperatorType::LIMIT: return "LIMIT";
    case RasterDBPhysicalOperatorType::LIMIT_PERCENT: return "LIMIT_PERCENT";
    case RasterDBPhysicalOperatorType::STREAMING_LIMIT: return "STREAMING_LIMIT";
    case RasterDBPhysicalOperatorType::RESERVOIR_SAMPLE: return "RESERVOIR_SAMPLE";
    case RasterDBPhysicalOperatorType::STREAMING_SAMPLE: return "STREAMING_SAMPLE";
    case RasterDBPhysicalOperatorType::TOP_N: return "TOP_N";
    case RasterDBPhysicalOperatorType::WINDOW: return "WINDOW";
    case RasterDBPhysicalOperatorType::STREAMING_WINDOW: return "STREAMING_WINDOW";
    case RasterDBPhysicalOperatorType::UNNEST: return "UNNEST";
    case RasterDBPhysicalOperatorType::UNGROUPED_AGGREGATE: return "UNGROUPED_AGGREGATE";
    case RasterDBPhysicalOperatorType::HASH_GROUP_BY: return "HASH_GROUP_BY";
    case RasterDBPhysicalOperatorType::PERFECT_HASH_GROUP_BY: return "PERFECT_HASH_GROUP_BY";
    case RasterDBPhysicalOperatorType::PARTITIONED_AGGREGATE: return "PARTITIONED_AGGREGATE";
    case RasterDBPhysicalOperatorType::FILTER: return "FILTER";
    case RasterDBPhysicalOperatorType::PROJECTION: return "PROJECTION";
    case RasterDBPhysicalOperatorType::COPY_TO_FILE: return "COPY_TO_FILE";
    case RasterDBPhysicalOperatorType::BATCH_COPY_TO_FILE: return "BATCH_COPY_TO_FILE";
    case RasterDBPhysicalOperatorType::LEFT_DELIM_JOIN: return "LEFT_DELIM_JOIN";
    case RasterDBPhysicalOperatorType::RIGHT_DELIM_JOIN: return "RIGHT_DELIM_JOIN";
    case RasterDBPhysicalOperatorType::BLOCKWISE_NL_JOIN: return "BLOCKWISE_NL_JOIN";
    case RasterDBPhysicalOperatorType::NESTED_LOOP_JOIN: return "NESTED_LOOP_JOIN";
    case RasterDBPhysicalOperatorType::HASH_JOIN: return "HASH_JOIN";
    case RasterDBPhysicalOperatorType::PIECEWISE_MERGE_JOIN: return "PIECEWISE_MERGE_JOIN";
    case RasterDBPhysicalOperatorType::IE_JOIN: return "IE_JOIN";
    case RasterDBPhysicalOperatorType::ASOF_JOIN: return "ASOF_JOIN";
    case RasterDBPhysicalOperatorType::CROSS_PRODUCT: return "CROSS_PRODUCT";
    case RasterDBPhysicalOperatorType::POSITIONAL_JOIN: return "POSITIONAL_JOIN";
    case RasterDBPhysicalOperatorType::POSITIONAL_SCAN: return "POSITIONAL_SCAN";
    case RasterDBPhysicalOperatorType::UNION: return "UNION";
    case RasterDBPhysicalOperatorType::INSERT: return "INSERT";
    case RasterDBPhysicalOperatorType::BATCH_INSERT: return "BATCH_INSERT";
    case RasterDBPhysicalOperatorType::DELETE_OPERATOR: return "DELETE";
    case RasterDBPhysicalOperatorType::UPDATE: return "UPDATE";
    case RasterDBPhysicalOperatorType::MERGE_INTO: return "MERGE_INTO";
    case RasterDBPhysicalOperatorType::EMPTY_RESULT: return "EMPTY_RESULT";
    case RasterDBPhysicalOperatorType::CREATE_TABLE: return "CREATE_TABLE";
    case RasterDBPhysicalOperatorType::CREATE_TABLE_AS: return "CREATE_TABLE_AS";
    case RasterDBPhysicalOperatorType::BATCH_CREATE_TABLE_AS: return "BATCH_CREATE_TABLE_AS";
    case RasterDBPhysicalOperatorType::CREATE_INDEX: return "CREATE_INDEX";
    case RasterDBPhysicalOperatorType::EXPLAIN: return "EXPLAIN";
    case RasterDBPhysicalOperatorType::EXPLAIN_ANALYZE: return "EXPLAIN_ANALYZE";
    case RasterDBPhysicalOperatorType::EXECUTE: return "EXECUTE";
    case RasterDBPhysicalOperatorType::VACUUM: return "VACUUM";
    case RasterDBPhysicalOperatorType::RECURSIVE_CTE: return "REC_CTE";
    case RasterDBPhysicalOperatorType::RECURSIVE_KEY_CTE: return "REC_KEY_CTE";
    case RasterDBPhysicalOperatorType::CTE: return "CTE";
    case RasterDBPhysicalOperatorType::RECURSIVE_CTE_SCAN: return "REC_CTE_SCAN";
    case RasterDBPhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN: return "REC_REC_CTE_SCAN";
    case RasterDBPhysicalOperatorType::CTE_SCAN: return "CTE_SCAN";
    case RasterDBPhysicalOperatorType::EXPRESSION_SCAN: return "EXPRESSION_SCAN";
    case RasterDBPhysicalOperatorType::ALTER: return "ALTER";
    case RasterDBPhysicalOperatorType::CREATE_SEQUENCE: return "CREATE_SEQUENCE";
    case RasterDBPhysicalOperatorType::CREATE_VIEW: return "CREATE_VIEW";
    case RasterDBPhysicalOperatorType::CREATE_SCHEMA: return "CREATE_SCHEMA";
    case RasterDBPhysicalOperatorType::CREATE_MACRO: return "CREATE_MACRO";
    case RasterDBPhysicalOperatorType::CREATE_SECRET: return "CREATE_SECRET";
    case RasterDBPhysicalOperatorType::DROP: return "DROP";
    case RasterDBPhysicalOperatorType::PRAGMA: return "PRAGMA";
    case RasterDBPhysicalOperatorType::TRANSACTION: return "TRANSACTION";
    case RasterDBPhysicalOperatorType::PREPARE: return "PREPARE";
    case RasterDBPhysicalOperatorType::EXPORT: return "EXPORT";
    case RasterDBPhysicalOperatorType::SET: return "SET";
    case RasterDBPhysicalOperatorType::SET_VARIABLE: return "SET_VARIABLE";
    case RasterDBPhysicalOperatorType::RESET: return "RESET";
    case RasterDBPhysicalOperatorType::LOAD: return "LOAD";
    case RasterDBPhysicalOperatorType::INOUT_FUNCTION: return "INOUT_FUNCTION";
    case RasterDBPhysicalOperatorType::CREATE_TYPE: return "CREATE_TYPE";
    case RasterDBPhysicalOperatorType::ATTACH: return "ATTACH";
    case RasterDBPhysicalOperatorType::DETACH: return "DETACH";
    case RasterDBPhysicalOperatorType::RESULT_COLLECTOR: return "RESULT_COLLECTOR";
    case RasterDBPhysicalOperatorType::EXTENSION: return "EXTENSION";
    case RasterDBPhysicalOperatorType::PIVOT: return "PIVOT";
    case RasterDBPhysicalOperatorType::COPY_DATABASE: return "COPY_DATABASE";
    case RasterDBPhysicalOperatorType::VERIFY_VECTOR: return "VERIFY_VECTOR";
    case RasterDBPhysicalOperatorType::UPDATE_EXTENSIONS: return "UPDATE_EXTENSIONS";
    case RasterDBPhysicalOperatorType::PARTITION: return "PARTITION";
    case RasterDBPhysicalOperatorType::CONCAT: return "CONCAT";
    case RasterDBPhysicalOperatorType::MERGE_SORT: return "MERGE_SORT";
    case RasterDBPhysicalOperatorType::MERGE_GROUP_BY: return "MERGE_GROUP_BY";
    case RasterDBPhysicalOperatorType::MERGE_TOP_N: return "MERGE_TOP_N";
    case RasterDBPhysicalOperatorType::MERGE_AGGREGATE: return "MERGE_AGGREGATE";
    case RasterDBPhysicalOperatorType::SORT_PARTITION: return "SORT_PARTITION";
    case RasterDBPhysicalOperatorType::SORT_SAMPLE: return "SORT_SAMPLE";
    case RasterDBPhysicalOperatorType::DUCKDB_SCAN: return "DUCKDB_SCAN";
    case RasterDBPhysicalOperatorType::PARQUET_SCAN: return "PARQUET_SCAN";
    case RasterDBPhysicalOperatorType::INVALID: break;
  }
  return "INVALID";
}
// LCOV_EXCL_STOP

}  // namespace rasterdb::op
