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

/*
 * MINIMAL RasterDB DuckDB Extension — bare-minimum skeleton.
 * All cudf/rmm/cucascade/CUDA dependencies removed.
 * GPU operators will be added with Vulkan/rasterdf implementations.
 */

#include "duckdb/main/database.hpp"
#define DUCKDB_EXTENSION_MAIN

#include "config.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "log/logging.hpp"
#include "rasterdb_extension.hpp"
#include "util/segfault_backtrace.hpp"

#include <cstdlib>

namespace duckdb {

bool RasterdbExtension::buffer_is_initialized = false;

// =========================================================================
// gpu_execution — forwards query to DuckDB CPU for now.
// Will be replaced with Vulkan/rasterdf GPU execution path.
// =========================================================================

struct RasterDBQueryData : public TableFunctionData {
  unique_ptr<QueryResult> res;
  unique_ptr<Connection> conn;
  string query;
  bool finished = false;
};

static unique_ptr<FunctionData> GPUExecutionBind(ClientContext& context,
                                                  TableFunctionBindInput& input,
                                                  vector<LogicalType>& return_types,
                                                  vector<string>& names)
{
  auto result  = make_uniq<RasterDBQueryData>();
  result->conn = make_uniq<Connection>(*context.db);
  result->query = input.inputs[0].ToString();
  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_execution cannot be called with a NULL parameter");
  }

  // Parse just for column names/types
  Parser parser(context.GetParserOptions());
  parser.ParseQuery(result->query);
  Planner planner(context);
  planner.CreatePlan(std::move(parser.statements[0]));

  for (auto& col : planner.names) { names.emplace_back(col); }
  for (auto& type : planner.types) { return_types.emplace_back(type); }

  return std::move(result);
}

static void GPUExecutionFunction(ClientContext& context,
                                  TableFunctionInput& data_p,
                                  DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<RasterDBQueryData>();
  if (data.finished) { return; }

  if (!data.res) {
    // TODO: Replace with Vulkan/rasterdf GPU execution
    RASTERDB_LOG_INFO("RasterDB: forwarding query to DuckDB CPU (GPU ops not yet implemented)");
    data.res = data.conn->Query(data.query);
  }

  auto chunk = data.res->Fetch();
  if (!chunk) {
    output.SetCardinality(0);
    return;
  }
  output.Reference(*chunk);
}

// =========================================================================
// Config setters — minimal set for the extension skeleton
// =========================================================================

static void SetEnableDuckdbFallback(ClientContext&, SetScope, Value& parameter)
{
  Config::ENABLE_DUCKDB_FALLBACK = BooleanValue::Get(parameter);
}

// =========================================================================
// Extension registration
// =========================================================================

void RasterdbExtension::RegisterGPUFunctions(DatabaseInstance& instance)
{
  auto transaction = CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = Catalog::GetSystemCatalog(instance);

  // gpu_execution(query) — the main entry point
  TableFunction gpu_execution("gpu_execution",
                              {LogicalType::VARCHAR},
                              GPUExecutionFunction,
                              GPUExecutionBind);
  CreateTableFunctionInfo gpu_execution_info(gpu_execution);
  catalog.CreateTableFunction(transaction, gpu_execution_info);
}

void RasterdbExtension::InitialGPUConfigs(DBConfig& config)
{
  config.AddExtensionOption(
    "enable_duckdb_fallback",
    "Whether to fall back to DuckDB CPU execution on GPU error",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(Config::ENABLE_DUCKDB_FALLBACK),
    SetEnableDuckdbFallback);
}

static void LoadInternal(ExtensionLoader& loader)
{
  rasterdb::util::install_segfault_backtrace_handler();

  auto& db     = loader.GetDatabaseInstance();
  auto& config = DBConfig::GetConfig(db);
  RasterdbExtension::InitialGPUConfigs(config);
  RasterdbExtension::RegisterGPUFunctions(db);

  RASTERDB_LOG_INFO("RasterDB extension loaded (minimal skeleton — GPU ops pending)");
}

void RasterdbExtension::Load(ExtensionLoader& loader) { LoadInternal(loader); }

std::string RasterdbExtension::Name() { return "rasterdb"; }

std::string RasterdbExtension::Version() const
{
#ifdef EXT_VERSION_RASTERDB
  return EXT_VERSION_RASTERDB;
#else
  return "0.1.0-dev";
#endif
}

}  // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(rasterdb, loader) { duckdb::LoadInternal(loader); }
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
