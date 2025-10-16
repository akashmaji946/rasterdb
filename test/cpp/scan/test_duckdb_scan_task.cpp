/*
 * Tests for DuckDBScanTask integration
 */

#include "catch.hpp"

#include "duckdb.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

#include "scan/duckdb_scan_task_new.hpp"
#include "parallel/task_executor.hpp"

#include <atomic>
#include <thread>
#include <chrono>

using namespace duckdb;
using namespace sirius::parallel;
using namespace std::chrono_literals;

namespace {

struct TestScanBindData : public FunctionData {
	idx_t total_rows;
	explicit TestScanBindData(idx_t total) : total_rows(total) {}
	bool Equals(const FunctionData &other) const override { return true; }
	unique_ptr<FunctionData> Copy() const override { return make_uniq<TestScanBindData>(total_rows); }
};

struct TestScanGlobalState : public GlobalTableFunctionState {
	std::atomic<idx_t> offset{0};
	idx_t MaxThreads() const override { return 1; }
};

struct TestScanLocalState : public LocalTableFunctionState {
};

static unique_ptr<FunctionData> TestScanBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {
	return_types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	names = {"i", "s"};
	idx_t total_rows = 10; // fixed small dataset
	return make_uniq<TestScanBindData>(total_rows);
}

static unique_ptr<GlobalTableFunctionState> TestScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<TestScanGlobalState>();
	state->offset.store(0);
	return std::move(state);
}

static unique_ptr<LocalTableFunctionState> TestScanInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                             GlobalTableFunctionState *gstate) {
	return make_uniq<TestScanLocalState>();
}

static void TestScanFunc(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<TestScanBindData>();
	auto &g = data.global_state->Cast<TestScanGlobalState>();

	auto start = g.offset.load();
	if (start >= bind.total_rows) {
		output.SetCardinality(0);
		return;
	}
	// Vary chunk sizes to exercise aligned/unaligned validity writes
	idx_t remaining = bind.total_rows - start;
	idx_t this_chunk = std::min<idx_t>(remaining, (start % 3 == 0) ? 1 : ((start % 3 == 1) ? 2 : 5));
	output.Initialize(Allocator::DefaultAllocator(), {LogicalType::INTEGER, LogicalType::VARCHAR});
	output.SetCardinality(this_chunk);

	auto icol = FlatVector::GetData<int32_t>(output.data[0]);
	for (idx_t i = 0; i < this_chunk; ++i) {
		icol[i] = static_cast<int32_t>(start + i);
	}

	auto &validity = FlatVector::Validity(output.data[1]);
	auto svec = FlatVector::GetData<string_t>(output.data[1]);
	for (idx_t i = 0; i < this_chunk; ++i) {
		idx_t row = start + i;
		bool is_null = (row == 1 || row == 7);
		if (is_null) {
			validity.SetInvalid(i);
		} else {
			std::string str = std::string("s") + std::to_string(row);
			svec[i] = StringVector::AddString(output.data[1], str);
		}
	}
	g.offset.store(start + this_chunk);
}

TableFunction MakeTestScanFunction() {
	TableFunction tf({}, TestScanFunc, TestScanBind, TestScanInitGlobal, TestScanInitLocal);
	tf.projection_pushdown = true;
	tf.filter_pushdown = true;
	return tf;
}

} // namespace

TEST_CASE("DuckDBScanTask drains source and completes", "[duckdb_scan_task]") {
	DuckDB db(nullptr);
	Connection con(db);
	{
		auto res = con.Query("CREATE TABLE t(i INTEGER, s VARCHAR)");
		REQUIRE(res && !res->HasError());
	}
	{
		auto res = con.Query("INSERT INTO t VALUES (0, 's0'), (1, NULL), (2, 's2'), (3, 's3'), (4, 's4'), (5, 's5'), (6, 's6'), (7, NULL), (8, 's8'), (9, 's9')");
		REQUIRE(res && !res->HasError());
	}

	ThreadContext thread(*con.context);
	ExecutionContext exec_ctx(*con.context, thread, nullptr);

	auto tf = MakeTestScanFunction();
	// Prepare fields for PhysicalTableScan
	vector<LogicalType> returned_types = {LogicalType::INTEGER, LogicalType::VARCHAR};
	vector<ColumnIndex> column_ids = {ColumnIndex(0), ColumnIndex(1)};
	vector<idx_t> projection_ids = {};
	vector<string> names = {"i", "s"};
	unique_ptr<TableFilterSet> table_filters;
	ExtraOperatorInfo extra_info;
	vector<Value> params;
	// Bind using our bind function
	vector<LogicalType> dummy_types;
	vector<string> dummy_names;
	vector<Value> in_vals;
	named_parameter_map_t named;
	vector<LogicalType> input_tbl_types;
	vector<string> input_tbl_names;
	TableFunctionRef ref;
	TableFunctionBindInput bind_input(in_vals, named, input_tbl_types, input_tbl_names, nullptr, nullptr, tf, ref);
	auto bind_data = tf.bind(*con.context, bind_input, dummy_types, dummy_names);

	PhysicalTableScan op(returned_types, tf, std::move(bind_data), returned_types, column_ids, projection_ids, names,
	                     std::move(table_filters), /*estimated_cardinality*/ 10, extra_info, params);

	auto gstate = std::make_shared<DuckDBScanTaskGlobalState>(*con.context, op);
	REQUIRE(gstate->MaxThreads() == 1);

	// Provide a queue for potential requeueing inside the task (not consumed by executor)
	auto local_queue = std::make_shared<DuckDBScanTaskQueue>();
	local_queue->Open();

	auto local_state = sirius::make_unique<DuckDBScanTaskLocalState>(local_queue, exec_ctx, *gstate, op);
	auto task = sirius::make_unique<DuckDBScanTask>(std::move(local_state), gstate);

	// Run through executor with its own queue
	auto exec_queue = std::make_unique<DuckDBScanTaskQueue>();
	TaskExecutorConfig config{1, false};
	ITaskExecutor executor(std::move(exec_queue), config);
	executor.Start();
	executor.Schedule(std::move(task));
	executor.Wait();
	executor.Stop();

	// Verify source drained
	REQUIRE(gstate->IsSourceDrained());
}

struct NoOpTask : public ITask {
	NoOpTask() : ITask(nullptr, nullptr) {}
	void Execute() override {}
};

TEST_CASE("DuckDBScanTaskQueue blocking and close", "[duckdb_scan_task]") {
	auto queue = std::make_unique<DuckDBScanTaskQueue>();
	queue->Open();

	std::atomic<bool> got_task{false};
	std::thread consumer([&]() {
		auto task = queue->Pull();
		if (task) got_task.store(true);
	});

	std::this_thread::sleep_for(50ms);
	queue->Push(std::make_unique<NoOpTask>());
	consumer.join();
	REQUIRE(got_task.load());

	std::thread consumer2([&]() {
		auto task = queue->Pull();
		REQUIRE(task == nullptr);
	});
	std::this_thread::sleep_for(50ms);
	queue->Close();
	consumer2.join();
}
