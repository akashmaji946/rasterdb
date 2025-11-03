// catch2
#include <catch.hpp>

// sirius
#include <data/data_repository.hpp>
#include <scan/duckdb_physical_table_scan.hpp>
#include <scan/duckdb_scan_task.hpp>
#include <scan/duckdb_scan_task_executor.hpp>
#include <sirius_context.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/common/named_parameter_map.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parallel/thread_context.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>

// standard library
#include <iostream>
#include <memory>
#include <string>

using idx_t = duckdb::idx_t;
using namespace sirius;

static constexpr size_t LINEITEM_CARDINALITY = 6001215;
static constexpr std::string_view LINEITEM_TBL_PATH =
  "/mnt/wiscdb/kevin/sirius/test_datasets/tpch-dbgen/s1/lineitem.tbl";

// Get the schema (column names and types) of a table
static void resolve_schema(duckdb::ClientContext& ctx,
                           std::string const& table,
                           std::vector<std::string>& names_out,
                           std::vector<duckdb::LogicalType>& types_out)
{
  duckdb::Connection con(*ctx.db);
  auto info = con.Query("PRAGMA table_info('" + table + "');");
  REQUIRE(info);
  REQUIRE(!info->HasError());
  idx_t const n = info->RowCount();
  names_out.reserve(n);
  types_out.reserve(n);
  std::cout << "\nABOUT TO RESOLVE SCHEMA FOR TABLE: " << table << "\n";
  for (idx_t i = 0; i < n; ++i) {
    auto name_val     = info->GetValue(1, i);
    auto type_str_val = info->GetValue(2, i);
    auto name         = name_val.GetValue<std::string>();
    auto type_str     = type_str_val.GetValue<std::string>();
    names_out.push_back(name);
    types_out.push_back(duckdb::TransformStringToLogicalType(type_str));
  }
}

// Make a PhysicalTableScan operator for the given table
static std::unique_ptr<duckdb::DuckDBPhysicalTableScan> make_physical_table_scan(
  std::string const& table_name, duckdb::ClientContext& ctx)
{
  // Resolve the schema
  duckdb::vector<std::string> column_names;
  duckdb::vector<duckdb::LogicalType> column_types;
  resolve_schema(ctx, table_name, column_names, column_types);
  duckdb::vector<duckdb::Value> dummy_values;
  duckdb::named_parameter_map_t dummy_params;
  duckdb::TableFunctionRef dummy_tf_ref;

  std::cout << "\nResolved schema for table: " << table_name << "\n";

  // Bind the table scan
  auto& entry =
    duckdb::Catalog::GetEntry<duckdb::TableCatalogEntry>(ctx, "main", DEFAULT_SCHEMA, table_name);
  duckdb::unique_ptr<duckdb::FunctionData> bind_data;
  auto tf = entry.GetScanFunction(ctx, bind_data);

  std::cout << "\nMetadata for physical table scan ready: " << table_name << "\n";

  // Build column/projection vectors
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.reserve(column_types.size());
  for (idx_t i = 0; i < column_types.size(); ++i) {
    column_ids.emplace_back(i);
  }
  duckdb::vector<idx_t> projection_ids;  // empty = all columns

  // No filters
  duckdb::unique_ptr<duckdb::TableFilterSet> dummy_filters = nullptr;
  duckdb::ExtraOperatorInfo dummy_extra_info;
  duckdb::vector<duckdb::Value> dummy_pts_params;

  // Build and return the scan operator
  return std::make_unique<duckdb::DuckDBPhysicalTableScan>(column_types,
                                                           tf,
                                                           std::move(bind_data),
                                                           column_types,
                                                           column_ids,
                                                           projection_ids,
                                                           column_names,
                                                           std::move(dummy_filters),
                                                           LINEITEM_CARDINALITY,
                                                           dummy_extra_info,
                                                           dummy_pts_params);
}

// Load the lineitem table from the .tbl file
static void load_lineitem_table(duckdb::Connection& con, const std::string& path_to_tbl)
{
  auto probe = con.Query(
    "SELECT COUNT(*) FROM read_csv('" + std::string(LINEITEM_TBL_PATH) +
    "', "
    "delim='|', header=false, quote='', escape='', auto_detect=false, dateformat='%Y-%m-%d', "
    "columns={"
    "l_orderkey:'BIGINT', l_partkey:'BIGINT', l_suppkey:'BIGINT', l_linenumber:'INTEGER', "
    "l_quantity:'DECIMAL(15,2)', l_extendedprice:'DECIMAL(15,2)', l_discount:'DECIMAL(15,2)', "
    "l_tax:'DECIMAL(15,2)', "
    "l_returnflag:'VARCHAR', l_linestatus:'VARCHAR', l_shipdate:'DATE', l_commitdate:'DATE', "
    "l_receiptdate:'DATE', "
    "l_shipinstruct:'VARCHAR', l_shipmode:'VARCHAR', l_comment:'VARCHAR', dummy:'VARCHAR'});");
  REQUIRE(probe);
  REQUIRE_FALSE(probe->HasError());
  std::cout << "Rows parsed (probe): " << probe->GetValue<int64_t>(0, 0) << "\n";

  auto res1 = con.Query("DROP TABLE IF EXISTS lineitem;");
  REQUIRE(res1);
  REQUIRE_FALSE(res1->HasError());

  // read_csv with explicit schema + an extra 'dummy' column to consume the trailing '|'
  // std::string sql =
  //   "CREATE TABLE lineitem AS "
  //   "SELECT "
  //   "  l_orderkey::BIGINT, l_partkey::BIGINT, l_suppkey::BIGINT, l_linenumber::INTEGER, "
  //   "  l_quantity::DECIMAL(15,2), l_extendedprice::DECIMAL(15,2), l_discount::DECIMAL(15,2), "
  //   "l_tax::DECIMAL(15,2), "
  //   "  l_returnflag::VARCHAR, l_linestatus::VARCHAR, "
  //   "  l_shipdate::DATE, l_commitdate::DATE, l_receiptdate::DATE, "
  //   "  l_shipinstruct::VARCHAR, l_shipmode::VARCHAR, l_comment::VARCHAR "
  //   "FROM read_csv('" +
  //   path_to_tbl +
  //   "', "
  //   "  delim='|', header=false, quote='', escape='', auto_detect=false, dateformat='%Y-%m-%d', "
  //   "  columns={"
  //   "    l_orderkey:'BIGINT', l_partkey:'BIGINT', l_suppkey:'BIGINT', l_linenumber:'INTEGER', "
  //   "    l_quantity:'DECIMAL(15,2)', l_extendedprice:'DECIMAL(15,2)', l_discount:'DECIMAL(15,2)', "
  //   "l_tax:'DECIMAL(15,2)', "
  //   "    l_returnflag:'VARCHAR', l_linestatus:'VARCHAR', l_shipdate:'DATE', l_commitdate:'DATE', "
  //   "l_receiptdate:'DATE', "
  //   "    l_shipinstruct:'VARCHAR', l_shipmode:'VARCHAR', l_comment:'VARCHAR', dummy:'VARCHAR' "
  //   "});";
  // auto res2 = con.Query(sql);
  // REQUIRE(res2);
  // if (res2->HasError()) std::cerr << res2->GetErrorObject().Message() << "\n";
  // REQUIRE_FALSE(res2->HasError());

  // auto res1 = con.Query("DROP TABLE IF EXISTS lineitem;");
  // REQUIRE(res1);
  // REQUIRE(!res1->HasError());

  // // clang-format off
  // auto res2 = con.Query("CREATE TABLE lineitem ( l_orderkey BIGINT NOT NULL, \
  //                                                l_partkey BIGINT NOT NULL, \
  //                                                l_suppkey BIGINT NOT NULL, \
  //                                                l_linenumber INTEGER NOT NULL, \
  //                                                l_quantity DECIMAL(15, 2) NOT NULL, \
  //                                                l_extendedprice DECIMAL(15, 2) NOT NULL, \
  //                                                l_discount DECIMAL(15, 2) NOT NULL, \
  //                                                l_tax DECIMAL(15, 2) NOT NULL, \
  //                                                l_returnflag CHAR(1) NOT NULL, \
  //                                                l_linestatus CHAR(1) NOT NULL, \
  //                                                l_shipdate DATE NOT NULL, \
  //                                                l_commitdate DATE NOT NULL, \
  //                                                l_receiptdate DATE NOT NULL, \
  //                                                l_shipinstruct CHAR(25) NOT NULL, \
  //                                                l_shipmode CHAR(10) NOT NULL, \
  //                                                l_comment VARCHAR(44) NOT NULL );");
  // // clang-format on
  // REQUIRE(res2);
  // REQUIRE(!res2->HasError());

  // std::cout << "\nABOUT TO COPY LINEITEM" << std::endl;
  // auto res3 =
  //   con.Query("COPY lineitem FROM '" + path_to_tbl + "' WITH (HEADER false, DELIMITER '|');");
  // std::cout << "\nCOPIED LINEITEM" << std::endl;
  // REQUIRE(res3);
  // if (res3->HasError()) { std::cout << res3->GetErrorObject().Message() << std::endl; }
  // REQUIRE(!res3->HasError());
}

// CreaTe a staging table with the same schema as the reference table
static void create_staging_table_like(duckdb::Connection& con,
                                      std::string const& ref_table,
                                      std::string const& stage_table)
{
  auto res1 = con.Query("DROP TABLE IF EXISTS " + stage_table + ";");
  REQUIRE(res1);
  REQUIRE(!res1->HasError());
  auto res2 =
    con.Query("CREATE TABLE " + stage_table + " AS SELECT * FROM " + ref_table + " WHERE 0=1;");
  REQUIRE(res2);
  REQUIRE(!res2->HasError());
}

// Helper function to check validity bit
static inline bool is_valid(const uint8_t* mask, uint64_t row_idx)
{
  return !mask || (((mask[row_idx / 8] >> (row_idx % 8)) & 1U) != 0);
}

// Reconstruct the table from the data batches in the repository.
// This reconstructor requires columns to be NOT NULL, as is true for TPC-H.
static void evict_data_repository_batches(duckdb::Connection& con,
                                          sirius::DataRepository& repo,
                                          std::string const& stage_table)
{
  duckdb::Appender app(con, stage_table);

  duckdb::LogicalType f;

  while (true) {
    auto batch = repo.EvictDataBatch(/*pipeline_id=*/0);
    if (!batch) break;

    auto const& types   = batch->types;
    auto& data_ptrs     = batch->data_ptrs;
    auto& mask_ptrs     = batch->mask_ptrs;
    auto& offset_ptrs   = batch->offset_ptrs;
    auto const num_rows = batch->num_rows;

    for (idx_t i = 0; i < num_rows; ++i) {
      app.BeginRow();
      for (idx_t col = 0; col < types.size(); ++col) {
        // Require NOT NULL
        if (mask_ptrs[col]) { REQUIRE(is_valid(mask_ptrs[col], i)); }

        // Type switch (does not currently handle all types)
        const auto& t = types[col];
        switch (t.id()) {
          case duckdb::LogicalTypeId::CHAR:  // Fallthrough
          case duckdb::LogicalTypeId::VARCHAR: {
            auto const beg  = offset_ptrs[col][i];
            auto const end  = offset_ptrs[col][i + 1];
            auto const* ptr = reinterpret_cast<const char*>(data_ptrs[col] + beg);
            auto const len  = end - beg;
            app.Append<duckdb::string_t>(std::string(ptr, len));
            break;
          }
          case duckdb::LogicalTypeId::INTEGER: {
            auto const* base = reinterpret_cast<const int32_t*>(data_ptrs[col]);
            app.Append<int32_t>(base[i]);
            break;
          }
          case duckdb::LogicalTypeId::BIGINT: {
            auto const* base = reinterpret_cast<const int64_t*>(data_ptrs[col]);
            app.Append<int64_t>(base[i]);
            break;
          }
          case duckdb::LogicalTypeId::DECIMAL: {
            auto width = duckdb::DecimalType::GetWidth(t);
            auto scale = duckdb::DecimalType::GetScale(t);
            // Determine internal storage type based on width
            switch (t.InternalType()) {
              case duckdb::PhysicalType::INT16: {
                auto const* base = reinterpret_cast<const int16_t*>(data_ptrs[col]);
                app.Append(duckdb::Value::DECIMAL(base[i], width, scale));
                break;
              }
              case duckdb::PhysicalType::INT32: {
                auto const* base = reinterpret_cast<const int32_t*>(data_ptrs[col]);
                app.Append(duckdb::Value::DECIMAL(base[i], width, scale));
                break;
              }
              case duckdb::PhysicalType::INT64: {
                auto const* base = reinterpret_cast<const int64_t*>(data_ptrs[col]);
                app.Append(duckdb::Value::DECIMAL(base[i], width, scale));
                break;
              }
              case duckdb::PhysicalType::INT128: {
                auto const* base = reinterpret_cast<const duckdb::hugeint_t*>(data_ptrs[col]);
                app.Append(duckdb::Value::DECIMAL(base[i], width, scale));
                break;
              }
              default: FAIL("Unsupported decimal internal type");
            }
            break;
          }
          case duckdb::LogicalTypeId::DATE: {
            auto const* base = reinterpret_cast<const int32_t*>(data_ptrs[col]);
            app.Append<duckdb::date_t>(duckdb::date_t(base[i]));
            break;
          }
          default: FAIL("Type not handled in staging appender.");
        }
      }
      app.EndRow();
    }
    // batch buffers are owned by repo/batch; you’re moving them out later as needed
  }
  app.Close();
}

static void validate_tables_equal(duckdb::Connection& con,
                                  std::string const& ref_table,
                                  std::string const& stage_table)
{
  auto cnt_ref = con.Query("SELECT COUNT(*) FROM " + ref_table + ";");
  auto cnt_stg = con.Query("SELECT COUNT(*) FROM " + stage_table + ";");
  REQUIRE(cnt_ref);
  REQUIRE(!cnt_ref->HasError());
  REQUIRE(cnt_stg);
  REQUIRE(!cnt_stg->HasError());

  auto ref_n = cnt_ref->GetValue<int64_t>(0, 0);
  auto stg_n = cnt_stg->GetValue<int64_t>(0, 0);
  REQUIRE(ref_n == stg_n);

  // Differences present in ref but not in stage
  // clang-format off
  auto missing = con.Query(
    "SELECT COUNT(*) \
       FROM ( SELECT * \
                FROM " + ref_table + " \
                  EXCEPT ALL SELECT * \
                               FROM " + stage_table + ");");
  // clang-format on
  REQUIRE(missing);
  REQUIRE(!missing->HasError());
  auto missing_n = missing->GetValue<int64_t>(0, 0);

  // Differences present in stage but not in ref
  // clang-format off
  auto extra = con.Query(
    "SELECT COUNT(*) \
       FROM ( SELECT * \
                FROM " + stage_table + " \
                  EXCEPT ALL SELECT * \
                               FROM " + ref_table + ");");
  // clang-format on
  REQUIRE(extra);
  REQUIRE(!extra->HasError());
  auto extra_n = extra->GetValue<int64_t>(0, 0);

  if (missing_n != 0 || extra_n != 0) {
    // Dump a few rows to help debugging
    // clang-format off
    auto diff1 = con.Query("SELECT * \
                              FROM " + ref_table + " \
                                EXCEPT ALL SELECT * \
                                             FROM " + stage_table + " \
                                             LIMIT 10;");
    auto diff2 = con.Query("SELECT * \
                              FROM " + stage_table + " \
                                EXCEPT ALL SELECT * \
                                             FROM " + ref_table + " \
                                             LIMIT 10;");
    // clang-format on
    std::cout << "REFERENCE TABLE: " << ref_table << "\n";
    std::cout << "MISSING:\n";
    diff1->Print();
    std::cout << "EXTRA:\n";
    diff2->Print();
  }
  REQUIRE(missing_n == 0);
  REQUIRE(extra_n == 0);
}

TEST_CASE("Compare custom scan vs DuckDB internal scan on lineitem", "[duckdb_scan_lineitem]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  // Create and load lineitem table
  std::cout << "\nAbout to load lineitem table into DuckDB" << std::endl;
  load_lineitem_table(con, std::string(LINEITEM_TBL_PATH));
  std::cout << "\nLoaded lineitem table into DuckDB" << std::endl;

  // Make the physical table scan operator for lineitem
  // auto op_ptr = make_physical_table_scan("lineitem", *con.context);
  // REQUIRE(op_ptr != nullptr);
  // auto& op = *op_ptr;

  //   duckdb::ThreadContext thread(*con.context);
  // duckdb::ExecutionContext exec_ctx(*con.context, thread, nullptr);

  //===----------Sirius Setup----------===//
  // // Create sirius context, extract the scan executor, data repository, message queue
  // auto& sirius_context  = SiriusContext::GetInstance();
  // auto& scan_executor   = sirius_context.GetDuckDBScanTaskExecutor();
  // auto& data_repository = sirius_context.GetDataRepository();
  // auto& message_queue   = sirius_context.GetTaskCreator().GetTaskCompletionMessageQueue();

  // // Create global state
  // uint64_t pipeline_id = 0;
  // auto gstate =
  //   sirius::make_shared<parallel::DuckDBScanTaskGlobalState>(pipeline_id, *con.context, op);
  // REQUIRE(gstate->MaxThreads() == 1);

  // Create local state
  // auto lstate = sirius::make_unique<parallel::DuckDBScanTaskLocalState>(
  //  sirius_context.GetTaskCreator().GetTaskCompletionMessageQueue(), *gstate, exec_ctx, op);
  // auto another_lstate = sirius::make_unique<parallel::DuckDBScanTaskLocalState>(
  //   sirius_context.GetTaskCreator().GetTaskCompletionMessageQueue(), *gstate, exec_ctx, op);

  // Create the scan task
  // uint64_t task_id = 0;
  // auto task = sirius::make_unique<parallel::DuckDBScanTask>(task_id, std::move(lstate), gstate);
  // uint64_t another_task_id = 1;
  // auto another_task        = sirius::make_unique<parallel::DuckDBScanTask>(
  //   another_task_id, std::move(another_lstate), gstate);

  // Run through executor with its own queue
  // scan_executor.Schedule(std::move(task));
  // scan_executor.Schedule(std::move(another_task));
  // scan_executor.Start();
  // scan_executor.Wait();
  // scan_executor.Stop();

  // REQUIRE(gstate->IsSourceDrained());

  // // --- Reassemble into staging table with identical schema to lineitem ---
  // const std::string stage_table = "sirius_lineitem";
  // CreateStageLike(con, stage_table, ref_table);
  // AppendBatchesTo(con, repo, stage_table);

  // // --- Validate equality against DuckDB’s internal scan (the table itself) ---
  // ValidateEqual(con, ref_table, stage_table);
}