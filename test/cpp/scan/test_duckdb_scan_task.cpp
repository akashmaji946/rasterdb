/*
 * Tests for DuckDBScanTask integration
 */

// catch2
#include "catch.hpp"

// sirius
#include <data/data_repository.hpp>
#include <data/simple_data_repository_level.hpp>
#include <scan/duckdb_scan_task_executor_new.hpp>
#include <scan/duckdb_scan_task_new.hpp>

// duckdb
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parallel/thread_context.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>

// standard library
#include <atomic>

using namespace sirius;

namespace
{

struct TestScanBindData : public duckdb::FunctionData
{
  idx_t total_rows;
  explicit TestScanBindData(idx_t total)
      : total_rows(total)
  {}
  bool Equals(const FunctionData& other) const override
  {
    return true;
  }
  duckdb::unique_ptr<FunctionData> Copy() const override
  {
    return duckdb::make_uniq<TestScanBindData>(total_rows);
  }
};

struct TestScanGlobalState : public duckdb::GlobalTableFunctionState
{
  std::atomic<idx_t> offset{0};
  idx_t MaxThreads() const override
  {
    return 1;
  }
};

struct TestScanLocalState : public duckdb::LocalTableFunctionState
{};

static duckdb::unique_ptr<duckdb::FunctionData>
TestScanBind(duckdb::ClientContext& context,
             duckdb::TableFunctionBindInput& input,
             duckdb::vector<duckdb::LogicalType>& return_types,
             duckdb::vector<std::string>& names)
{
  return_types     = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::VARCHAR};
  names            = {"i", "s"};
  idx_t total_rows = 10; // fixed small dataset
  return duckdb::make_uniq<TestScanBindData>(total_rows);
}

static duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
TestScanInitGlobal(duckdb::ClientContext& context, duckdb::TableFunctionInitInput& input)
{
  auto state = duckdb::make_uniq<TestScanGlobalState>();
  state->offset.store(0);
  return std::move(state);
}

static duckdb::unique_ptr<duckdb::LocalTableFunctionState>
TestScanInitLocal(duckdb::ExecutionContext& context,
                  duckdb::TableFunctionInitInput& input,
                  duckdb::GlobalTableFunctionState* gstate)
{
  return duckdb::make_uniq<TestScanLocalState>();
}

static void TestScanFunc(duckdb::ClientContext& context,
                         duckdb::TableFunctionInput& data,
                         duckdb::DataChunk& output)
{
  auto& bind = data.bind_data->Cast<TestScanBindData>();
  auto& g    = data.global_state->Cast<TestScanGlobalState>();

  auto start = g.offset.load();
  if (start >= bind.total_rows)
  {
    output.SetCardinality(0);
    return;
  }

  idx_t remaining  = bind.total_rows - start;
  idx_t this_chunk = std::min<idx_t>(remaining, (start % 3 == 0) ? 1 : ((start % 3 == 1) ? 2 : 5));

  output.SetCardinality(this_chunk);

  // Column 0: INTEGER
  auto icol = duckdb::FlatVector::GetData<int32_t>(output.data[0]);
  for (idx_t i = 0; i < this_chunk; ++i)
  {
    icol[i] = static_cast<int32_t>(start + i);
  }

  // Column 1: VARCHAR
  auto& validity = duckdb::FlatVector::Validity(output.data[1]);
  validity.SetAllValid(this_chunk);
  auto svec = duckdb::FlatVector::GetData<duckdb::string_t>(output.data[1]);
  for (idx_t i = 0; i < this_chunk; ++i)
  {
    idx_t row = start + i;
    if (row == 1 || row == 7)
    {
      validity.SetInvalid(i);
    }
    else
    {
      std::string str = std::string("s") + std::to_string(row);
      svec[i]         = duckdb::StringVector::AddString(output.data[1], str);
    }
  }

  g.offset.store(start + this_chunk);
}

duckdb::TableFunction MakeTestScanFunction()
{
  duckdb::TableFunction tf({}, TestScanFunc, TestScanBind, TestScanInitGlobal, TestScanInitLocal);
  tf.projection_pushdown = true;
  tf.filter_pushdown     = true;
  return tf;
}

} // namespace

TEST_CASE("DuckDBScanTask drains source and completes", "[duckdb_scan_task]")
{
  // Set up in-memory DuckDB database with test table
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  {
    auto res = con.Query("CREATE TABLE t(i INTEGER, s VARCHAR)");
    REQUIRE(res);
    REQUIRE(!res->HasError());
  }
  {
    // clang-format off
    auto res = con.Query("INSERT INTO t VALUES "
                         "(0, 's0'),"
                         "(1, NULL),"
                         "(2, 's2')," 
                         "(3, 's3'),"
                         "(4, 's4'),"
                         "(5, 's5'),"
                         "(6, 's6'),"
                         "(7, NULL),"
                         "(8, 's8'),"
                         "(9, 's9')");
    // clang-format on
    REQUIRE(res);
    REQUIRE(!res->HasError());
  }

  // Create the data repository
  DataRepository data_repository;
  data_repository.AddNewLevel(0, sirius::make_unique<SimpleDataRepositoryLevel>());

  // Create the scan task executor
  parallel::TaskExecutorConfig config{1, false};
  parallel::DuckDBScanExecutor scan_executor(config);

  // Create the message queue
  TaskCompletionMessageQueue message_queue{};

  // Create the duckdb contexts
  duckdb::ThreadContext thread(*con.context);
  duckdb::ExecutionContext exec_ctx(*con.context, thread, nullptr);

  // Creat the PhysicalTableScan operator
  auto tf                                            = MakeTestScanFunction();
  duckdb::vector<duckdb::LogicalType> returned_types = {duckdb::LogicalType::INTEGER,
                                                        duckdb::LogicalType::VARCHAR};
  duckdb::vector<duckdb::ColumnIndex> column_ids = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  duckdb::vector<idx_t> projection_ids           = {};
  duckdb::vector<string> names                   = {"i", "s"};
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  duckdb::ExtraOperatorInfo extra_info;
  duckdb::vector<duckdb::Value> params;
  // Bind using our bind function
  duckdb::vector<duckdb::LogicalType> dummy_types;
  duckdb::vector<string> dummy_names;
  duckdb::vector<duckdb::Value> in_vals;
  duckdb::named_parameter_map_t named;
  duckdb::vector<duckdb::LogicalType> input_tbl_types;
  duckdb::vector<string> input_tbl_names;
  duckdb::TableFunctionRef ref;
  duckdb::TableFunctionBindInput
    bind_input(in_vals, named, input_tbl_types, input_tbl_names, nullptr, nullptr, tf, ref);
  auto bind_data = tf.bind(*con.context, bind_input, dummy_types, dummy_names);
  duckdb::PhysicalTableScan op(returned_types,
                               tf,
                               std::move(bind_data),
                               returned_types,
                               column_ids,
                               projection_ids,
                               names,
                               std::move(table_filters),
                               /*estimated_cardinality*/ 10,
                               extra_info,
                               params);

  // Create global state
  uint64_t pipeline_id = 0;
  auto gstate =
    sirius::make_shared<parallel::DuckDBScanTaskGlobalState>(pipeline_id, *con.context, op);
  REQUIRE(gstate->MaxThreads() == 1);

  // Create local state
  auto lstate = sirius::make_unique<parallel::DuckDBScanTaskLocalState>(data_repository,
                                                                        message_queue,
                                                                        scan_executor,
                                                                        *gstate,
                                                                        exec_ctx,
                                                                        op);

  // Create the scan task
  uint64_t task_id = 0;
  auto task = sirius::make_unique<parallel::DuckDBScanTask>(task_id, std::move(lstate), gstate);

  // Run through executor with its own queue
  scan_executor.Start();
  scan_executor.Schedule(std::move(task));
  scan_executor.Wait();
  scan_executor.Stop();

  // Verify source drained
  REQUIRE(gstate->IsSourceDrained());
}

// struct NoOpTask : public ITask
// {
//   NoOpTask()
//       : ITask(nullptr, nullptr)
//   {}
//   void Execute() override
//   {}
// };

// TEST_CASE("DuckDBScanTaskQueue blocking and close", "[duckdb_scan_task]")
// {
//   auto queue = std::make_unique<DuckDBScanTaskQueue>();
//   queue->Open();

//   std::atomic<bool> got_task{false};
//   std::thread consumer([&]() {
//     auto task = queue->Pull();
//     if (task)
//       got_task.store(true);
//   });

//   std::this_thread::sleep_for(50ms);
//   queue->Push(std::make_unique<NoOpTask>());
//   consumer.join();
//   REQUIRE(got_task.load());

//   std::thread consumer2([&]() {
//     auto task = queue->Pull();
//     REQUIRE(task == nullptr);
//   });
//   std::this_thread::sleep_for(50ms);
//   queue->Close();
//   consumer2.join();
// }
