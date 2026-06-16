# RasterDB Performance Design Guide

This document proposes a Sirius-like redesign for RasterDB performance on numeric TPC-H queries. The focus is `INT32`, `INT64`, `FLOAT32`, and `FLOAT64` columns, especially joins, filters, projections, grouped aggregates, order-by, top-n, and limit. VARCHAR support can remain available, but it should not drive the first performance refactor.

The goal is to move RasterDB from a recursive logical-plan executor into an optimized physical-plan and pipelined GPU engine similar to Sirius.

## 1. Current RasterDB Architecture

RasterDB executes queries through `gpu_execution` in `src/rasterdb_extension.cpp`.

Current flow:

```text
gpu_execution(sql)
  -> parse SQL
  -> DuckDB Planner::CreatePlan
  -> use planner.plan directly
  -> ColumnBindingResolver
  -> gpu_executor::execute(logical_plan)
  -> recursive execute_operator
  -> one full gpu_table result
  -> bulk download final result
```

The hot path currently uses:

```cpp
Planner planner(context);
planner.CreatePlan(std::move(parser.statements[0]));
auto& plan = *planner.plan;
ColumnBindingResolver resolver;
resolver.VisitOperator(plan);
```

This means RasterDB often executes a raw unoptimized logical plan. Sirius, by contrast, runs DuckDB's optimizer before physical plan generation.

The recursive RasterDB executor lives under `src/gpu/executor`:

- `gpu_executor_core.cpp`
- `gpu_executor_scan.cpp`
- `gpu_executor_filter.cpp`
- `gpu_executor_projection.cpp`
- `gpu_executor_join.cpp`
- `gpu_executor_aggregate.cpp`
- `gpu_executor_grouped_aggregate.cpp`
- `gpu_executor_order.cpp`
- `gpu_executor_limit.cpp`

`gpu_executor::execute_operator` directly dispatches DuckDB logical operators to implementation functions and each function returns a fully materialized `gpu_table`.

## 2. Main Performance Problems

### 2.1 Unoptimized Plans

Using the raw planner plan causes TPC-H slowdowns because DuckDB has not yet applied important rewrites:

- Filter pushdown into scans.
- Join reordering.
- Predicate movement into join conditions.
- Projection pruning.
- Limit and top-n pushdown.
- Removal of unnecessary intermediate projections.

For join-heavy TPC-H queries, this can keep selective filters above a large join tree. The executor then joins millions of rows before applying filters that Sirius applies much earlier.

### 2.2 Full Materialization Between Every Operator

The current recursive executor fully materializes each child before the parent starts. For example, a join does this:

```text
execute left child fully
execute right child fully
hash join
materialize joined payload columns
return full gpu_table
```

This increases:

- GPU memory traffic.
- Temporary allocation pressure.
- Gather/scatter cost.
- Join output width.
- Group-by input size.
- Sort input size.

### 2.3 No Physical Planning Layer

RasterDB executes DuckDB logical operators directly. Optimized DuckDB plans require more physical-plan awareness:

- `LogicalGet::table_filters`
- `LogicalGet::projection_ids`
- `LogicalJoin::left_projection_map`
- `LogicalJoin::right_projection_map`
- `LOGICAL_TOP_N`
- `LOGICAL_EMPTY_RESULT`
- multi-condition joins
- reordered joins
- build/probe side decisions
- local and merge aggregate stages

Without a physical planning layer, these details become fragile patches in executor functions.

### 2.4 No Pipeline Breaker Model

TPC-H performance depends on separating streaming operators from pipeline breakers.

Streaming operators:

- Scan
- Filter
- Projection
- Expression evaluation
- Simple limit

Pipeline breakers:

- Hash join build side
- Grouped aggregate
- Ungrouped aggregate
- Sort/order-by
- Top-n
- Final result collection

RasterDB currently has no source/operator/sink distinction, so it cannot perform local aggregate + merge aggregate, local top-n + merge top-n, or proper build/probe join scheduling.

### 2.5 Scan and Transfer Costs

`execute_get` scans DuckDB chunks on CPU and flattens them into GPU-visible memory. This is practical, but it must be aggressively optimized:

- scan only needed columns,
- pass table filters into DuckDB scan,
- use cached numeric columns safely,
- avoid full-table staging when filters or limits are available,
- produce batches instead of one giant table.

## 3. Sirius Architecture to Mirror

Sirius has three important layers RasterDB should copy conceptually.

### 3.1 Optimized Plan Extraction

Sirius `ExtractPlan` performs:

```cpp
Planner planner(context);
planner.CreatePlan(std::move(parser.statements[0]));
plan = std::move(planner.plan);

if (context.config.enable_optimizer) {
  Optimizer optimizer(*planner.binder, context);
  plan = optimizer.Optimize(std::move(plan));
}

plan->ResolveOperatorTypes();

ColumnBindingResolver resolver;
ColumnBindingResolver::Verify(*plan);
resolver.VisitOperator(*plan);
```

Sirius also disables a small set of optimizers that complicate GPU execution:

- `IN_CLAUSE`
- `COMPRESSED_MATERIALIZATION`
- `COLUMN_LIFETIME` in debug builds

RasterDB should use the same sequence and fix compatibility issues instead of avoiding the optimizer.

### 3.2 Physical Plan Generation

Sirius creates GPU physical operators from the optimized logical plan. This is done before execution. The physical operator carries all execution-specific metadata, such as output types, cardinality, join conditions, projection maps, table filters, and operator role.

RasterDB should add an equivalent `RasterPhysicalPlanGenerator` rather than executing DuckDB logical operators directly.

### 3.3 Pipeline Execution

Sirius builds pipelines with:

```text
source -> operators -> sink
```

Its operator base supports source/operator/sink roles. Sink operators create child pipelines and dependencies. This is how Sirius naturally supports join build/probe, local aggregation, merge aggregation, local sort, merge sort, and result collection.

RasterDB should implement a simpler synchronous version first, then add asynchronous scheduling later.

## 4. Target RasterDB Architecture

The target architecture should be:

```text
gpu_execution(sql)
  -> ExtractOptimizedPlan
  -> RasterPhysicalPlanGenerator
  -> RasterPipelineBuilder
  -> RasterPipelineExecutor
  -> RasterResultCollector
  -> DuckDB output chunks
```

The existing recursive `gpu_executor` can remain as a fallback/debug path, but the performance path should use physical operators and pipelines.

Recommended new directories:

```text
src/gpu/physical/
  raster_physical_operator.hpp
  raster_physical_plan_generator.hpp/.cpp
  operators/
    raster_physical_scan.hpp/.cpp
    raster_physical_filter.hpp/.cpp
    raster_physical_projection.hpp/.cpp
    raster_physical_hash_join.hpp/.cpp
    raster_physical_nested_loop_join.hpp/.cpp
    raster_physical_grouped_aggregate.hpp/.cpp
    raster_physical_grouped_aggregate_merge.hpp/.cpp
    raster_physical_ungrouped_aggregate.hpp/.cpp
    raster_physical_ungrouped_aggregate_merge.hpp/.cpp
    raster_physical_order.hpp/.cpp
    raster_physical_merge_sort.hpp/.cpp
    raster_physical_top_n.hpp/.cpp
    raster_physical_top_n_merge.hpp/.cpp
    raster_physical_limit.hpp/.cpp
    raster_physical_result_collector.hpp/.cpp

src/gpu/pipeline/
  raster_pipeline.hpp/.cpp
  raster_meta_pipeline.hpp/.cpp
  raster_pipeline_executor.hpp/.cpp
  raster_operator_data.hpp
```

## 5. Optimized Plan Extraction

Add a helper similar to Sirius:

```cpp
unique_ptr<LogicalOperator> RasterDBQueryData::ExtractPlan(ClientContext& context);
```

It should:

1. Save `context.config` and disabled optimizer state.
2. Enable optimizer.
3. Disable only the small Sirius-compatible set:
   - `OptimizerType::IN_CLAUSE`
   - `OptimizerType::COMPRESSED_MATERIALIZATION`
   - `OptimizerType::COLUMN_LIFETIME` in debug builds
4. Parse and plan SQL.
5. Run `Optimizer::Optimize`.
6. Run `ResolveOperatorTypes`.
7. Run `ColumnBindingResolver::Verify`.
8. Run `ColumnBindingResolver::VisitOperator`.
9. Restore context config.

Do not disable filter pushdown, join order optimization, top-n optimization, projection pruning, or limit pushdown. These are required for competitive TPC-H performance.

Add debug plan logs:

```text
RDB_PLAN_RAW
RDB_PLAN_OPT
RDB_PHYSICAL_PLAN
RDB_PIPELINES
```

These should be comparable to Sirius `SIR_PLAN`.

## 6. Required Optimized-Plan Compatibility

Before optimizer-enabled execution can be the default, RasterDB must support these DuckDB plan features:

- `LogicalGet::table_filters`
- `LogicalGet::projection_ids`
- `LogicalGet::GetColumnIds()`
- `LogicalJoin::left_projection_map`
- `LogicalJoin::right_projection_map`
- multi-condition `LogicalComparisonJoin`
- reordered joins
- `LOGICAL_TOP_N`
- `LOGICAL_EMPTY_RESULT`
- `LOGICAL_LIMIT`
- `BoundCastExpression` around join/filter keys
- `COMPARE_NOT_DISTINCT_FROM` where possible
- semi/anti joins if adapted TPC-H queries require them

Correctness rules:

- Never silently truncate scans.
- Never ignore join projection maps.
- Never assume unoptimized column layout.
- Never compare join predicates against stale column indices.
- Always keep physical operator output types aligned with DuckDB `op.types`.

## 7. Physical Operator Model

Introduce a base class:

```cpp
class raster_physical_operator {
public:
  RasterPhysicalOperatorType type;
  duckdb::vector<duckdb::LogicalType> types;
  duckdb::idx_t estimated_cardinality;
  std::vector<std::unique_ptr<raster_physical_operator>> children;

  virtual bool is_source() const;
  virtual bool is_sink() const;

  virtual std::unique_ptr<raster_operator_data> get_data(raster_execution_context& ctx);
  virtual std::unique_ptr<raster_operator_data> execute(raster_execution_context& ctx,
                                                        const raster_operator_data& input);
  virtual void sink(raster_execution_context& ctx,
                    const raster_operator_data& input);
  virtual std::unique_ptr<raster_operator_data> finalize(raster_execution_context& ctx);

  virtual void build_pipelines(raster_pipeline& current,
                               raster_meta_pipeline& meta);
};
```

Introduce operator data:

```cpp
struct raster_gpu_batch {
  std::shared_ptr<gpu_table> table;
  idx_t batch_id;
  idx_t row_count;
};

struct raster_operator_data {
  std::vector<std::shared_ptr<raster_gpu_batch>> batches;
};
```

Initially every source can return a single batch. The API should still support multiple batches so scan, local aggregation, and join probe can later become streaming.

## 8. Pipeline Model

A pipeline should contain:

```cpp
struct raster_pipeline {
  raster_physical_operator* source;
  std::vector<raster_physical_operator*> operators;
  raster_physical_operator* sink;
  std::vector<raster_pipeline*> dependencies;
};
```

Start with a synchronous executor:

```text
for pipeline in topological_order(pipelines):
  data = pipeline.source->get_data(ctx)
  for op in pipeline.operators:
    data = op->execute(ctx, data)
  if pipeline.sink:
    pipeline.sink->sink(ctx, data)
```

This is much simpler than Sirius's full asynchronous runtime but creates the same architectural boundaries.

Classify operators as:

Sources:

- table scan
- hash join probe source after build is ready
- aggregate merge source
- sort merge source

Regular operators:

- filter
- projection
- expression evaluation
- simple limit

Sinks / pipeline breakers:

- hash join build
- grouped aggregate local state
- ungrouped aggregate local state
- order-by local sort
- top-n local top-k
- result collector

## 9. Scan Design

The physical scan operator should own:

```text
function
bind_data
returned_types
column_ids
projection_ids
names
table_filters
estimated_cardinality
```

Design requirements:

- Use optimized `column_ids` and `projection_ids`.
- Pass `table_filters` into DuckDB scan when valid.
- Remap table filter column ids if DuckDB expects scan-vector-relative indices.
- Produce only columns needed by the optimized plan.
- Use numeric column cache for repeated dimension columns.
- Never trust underestimated cardinality if it can cause truncation.

The first implementation can still scan through DuckDB table functions. The performance improvement comes from fewer scanned columns, pushed filters, and smaller downstream intermediates.

Later improvement:

```cpp
bool raster_physical_scan::get_next_batch(raster_gpu_batch& out);
```

This allows:

```text
scan batch -> filter -> projection -> join probe -> local aggregate
```

instead of full-table materialization.

## 10. Join Design

The existing Simple Garuda hash join should remain the core equality join primitive. The surrounding execution architecture must change.

Physical hash join should store:

```text
conditions
join_type
left_projection_map
right_projection_map
estimated_cardinality
build_side
probe_side
```

Planner rules:

1. Reorder equality conditions before non-equality conditions.
2. Prefer hash join when at least one equality condition exists.
3. Use all equality conditions if multi-key join is implemented.
4. Otherwise hash on the best equality key and post-filter remaining predicates.
5. Use pure nested-loop only if one side is tiny and the candidate pair count is bounded.
6. Apply `left_projection_map` and `right_projection_map` exactly.

Join output must be:

```text
left[left_projection_map] + right[right_projection_map]
```

If a map is empty, output all columns from that side.

For TPC-H, prioritize:

- inner equi-join,
- mixed equi + inequality join,
- tiny non-equi join,
- semi join,
- anti join.

Build/probe side should be chosen by:

```text
estimated_cardinality * projected_row_width
```

Small dimension tables should normally be the build side.

## 11. Group-By and Aggregate Design

Split current monolithic group-by into:

```text
raster_physical_grouped_aggregate
raster_physical_grouped_aggregate_merge
raster_physical_ungrouped_aggregate
raster_physical_ungrouped_aggregate_merge
```

Execution shape:

```text
scan/filter/project/join pipeline
  -> local grouped aggregate sink
  -> grouped aggregate merge
  -> final projection/order
```

Benefits:

- smaller intermediate tables,
- lower peak memory,
- batch-wise execution,
- less data sent into final sort/order,
- easier parity with Sirius.

For numeric TPC-H:

- single `INT32` group key should be direct,
- single `FLOAT32` group key can use raw bits for equality,
- multi-column group-by should eventually use tuple-key hashing,
- composite packed keys are acceptable only as a temporary fast path with no collision/overflow risk,
- avoid CPU downloads for surrogate key mapping in hot paths.

Common aggregate expressions such as:

```sql
l_extendedprice * (1.0 - l_discount)
```

should be fused into aggregate input generation or into a projection immediately before local aggregate.

## 12. Sort, Limit, and Top-N Design

Support `LOGICAL_TOP_N` directly. With DuckDB optimizer enabled, many `ORDER BY ... LIMIT` patterns become top-n.

Implement:

```text
raster_physical_top_n
raster_physical_top_n_merge
```

For each batch:

1. Keep only `offset + limit` rows.
2. Merge partial top-n outputs.
3. Apply final offset/limit.

For full order-by:

```text
local sort per batch -> merge sort
```

This mirrors Sirius's `ORDER_BY` and merge-sort split and avoids full sort when top-n is sufficient.

## 13. Expression, Filter, and Projection Fusion

RasterDB should avoid materializing separate tables for:

```text
projection -> filter -> projection -> join -> projection -> aggregate
```

Introduce a physical expression layer:

```text
raster_expression_program
raster_expression_compiler
raster_expression_executor
```

Use it to fuse:

- scan-time filters,
- scan-time projections,
- filter + projection after scan,
- join post-filter + projection maps,
- aggregate value expressions.

Filter priority:

1. DuckDB `TableFilterSet` into scan.
2. GPU scan-side predicate while loading batches.
3. Standalone GPU filter only when necessary.

Projection priority:

1. Use DuckDB optimizer projection pruning.
2. Preserve join projection maps.
3. Add a physical projection-pruning pass from root required columns downward.

## 14. Memory Management

Current query execution resets workspace once and then all recursive operators allocate from it. A pipelined executor should have explicit lifetimes:

```text
operator scratch memory  -> released after operator
pipeline memory          -> released after pipeline
query memory             -> released after query
cached memory            -> survives across queries when safe
```

Add a `raster_execution_context`:

```cpp
struct raster_execution_context {
  gpu_context& gpu;
  duckdb::ClientContext& duckdb_context;
  raster_memory_context& memory;
  raster_query_profile& profile;
};
```

Rules:

- Do not download intermediate columns to CPU in hot paths.
- Do not allocate oversized staging buffers based on bad estimates.
- If an estimate is wrong, grow/retry or fail safely; never truncate.
- Keep build-side hash tables alive only until dependent probe pipelines finish.
- Keep final result alive only until DuckDB consumes it.

Longer-term optimization:

- Carry selection vectors through joins and filters.
- Materialize payload columns only when a later operator actually needs them.

## 15. TPC-H Priorities

Highest-impact implementation order:

1. Enable optimizer using the Sirius planning sequence.
2. Fix optimized-plan compatibility in current executor.
3. Add physical plan generator.
4. Add physical scan and hash join.
5. Add join projection-map support everywhere.
6. Add local + merge grouped aggregate.
7. Add `LOGICAL_TOP_N`.
8. Add synchronous pipeline executor.
9. Add batch scans.
10. Add async overlap only after correctness.

Query-specific priorities:

Q1:

- push date filter into scan,
- scan only needed lineitem columns,
- local grouped aggregate,
- merge aggregate.

Q3/Q5/Q7/Q9/Q10:

- optimized join order,
- dimension filter pushdown,
- build hash tables on small sides,
- project join outputs tightly,
- group after reduced join result.

Q7:

- `lineitem` date filter must be a scan filter,
- `nation` name filters must be scan filters,
- `n1.n_name_id <> n2.n_name_id` should be handled as tiny nested-loop or mixed join,
- join projection maps must be exact,
- group-by should run only on reduced rows.

## 16. Benchmarking and Validation

For each TPC-H query, log:

```text
raw plan
optimized logical plan
physical plan
pipelines
operator input rows
operator output rows
input bytes
output bytes
kernel time
CPU scan time
GPU upload/download time
allocation bytes
```

For joins, additionally log:

```text
build rows
probe rows
key columns
match count
post-filter count
projection output width
hash table bytes
```

For group-by:

```text
input rows
group count
key columns
aggregate count
local aggregate time
merge aggregate time
```

For sort/top-n:

```text
input rows
kept rows
key count
local sort/top-n time
merge time
```

Benchmark table format:

```text
Query | Sirius plan | RasterDB plan | Sirius time | RasterDB time | Main delta
```

Every slowdown should be traceable to one of:

- worse logical plan,
- extra scanned columns,
- missing filter pushdown,
- full sort instead of top-n,
- larger join payload,
- wrong build side,
- missing local aggregate,
- CPU downloads in hot path,
- excess materialization.

## 17. Migration Roadmap

### Phase 0: Baseline and Logging

- Save per-query RasterDB and Sirius logs.
- Add raw/optimized/physical/pipeline plan dumps.
- Record row counts and bytes at every operator.
- Verify all output against DuckDB CPU.

### Phase 1: Optimizer Compatibility in Current Executor

- Use the Sirius optimizer sequence.
- Support table filters in scan.
- Support join projection maps.
- Support `LOGICAL_TOP_N` or lower it to order+limit.
- Support `LOGICAL_EMPTY_RESULT`.
- Support multi-condition joins.

Success criterion:

- All target numeric adapted TPC-H queries run on GPU with optimizer enabled.

### Phase 2: Physical Plan Generator

- Add `RasterPhysicalPlanGenerator`.
- Convert optimized logical operators into physical operators.
- Keep execution initially simple.
- Reuse existing kernels behind physical operators.

Success criterion:

- Physical plan metadata is correct and visible in logs.

### Phase 3: Synchronous Pipeline Executor

- Add source/operator/sink roles.
- Add pipeline builder.
- Add dependency-ordered executor.
- Make join build, group-by, order, top-n, and result collector pipeline breakers.

Success criterion:

- Same results as recursive executor with clearer lifetimes and smaller intermediates.

### Phase 4: Local + Merge Operators

- Implement grouped aggregate + merge.
- Implement ungrouped aggregate + merge.
- Implement order + merge sort.
- Implement top-n + top-n merge.

Success criterion:

- Q1 and Q7 avoid unnecessary large intermediates.

### Phase 5: Batch-Aware Execution

- Make scans produce batches.
- Push batches through filter/projection/join probe.
- Accumulate local aggregates per batch.

Success criterion:

- Lower peak memory and better large-table performance.

### Phase 6: Advanced Scheduling

- Add CPU scan / GPU overlap.
- Add independent pipeline concurrency.
- Add cross-query dimension caches and hash-table caches.

Do this only after the synchronous pipeline design is correct and benchmarked.

## 18. Summary

The most important change is architectural: RasterDB should stop treating DuckDB logical plans as an executable tree of fully materialized operators. Instead, it should follow Sirius:

```text
optimized logical plan
  -> GPU physical plan
  -> source/operator/sink pipelines
  -> local + merge pipeline breakers
  -> final result collector
```

For TPC-H numeric performance, the highest-value work is:

- run DuckDB optimizer,
- preserve optimized plan semantics,
- push filters into scans,
- prune projections aggressively,
- use hash join with correct build/probe and projection maps,
- split group-by and sort/top-n into local and merge phases,
- move from full-table recursion to batch-aware pipelines.
