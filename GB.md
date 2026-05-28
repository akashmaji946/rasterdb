# RasterDB / RasterDF GroupBy Redesign Plan

## Goal

Build a new compute-shader groupby path that supports any practical number of
GROUP BY keys and many aggregate outputs in one grouped aggregation pipeline,
without changing the existing working GFXm, simple hash aggregate, decimal, or
composite-key paths.

The target is Sirius/cuDF-like behavior:

- No pre-materialized composite key for multi-column GROUP BY.
- No one-pass-per-aggregate hash table rebuild.
- One tuple-key hash table build/probe pass for all requested aggregates.
- Compact output extraction once for all keys and all aggregate columns.
- Dynamic hash table sizing based on estimated unique groups, not just row count.
- A safe opt-in integration path from RasterDB so current correctness stays intact.

## Current State

RasterDB currently calls RasterDF groupby through
`rasterdb/src/gpu/executor/gpu_executor_grouped_aggregate.cpp`.

The working paths are:

- Single INT32 key: direct hash/groupby path.
- Single STRING key: string hash to INT32, then groupby.
- Multi-key GROUP BY: RasterDB creates an INT32 or INT64 composite key, then
  decomposes the grouped key after aggregation.
- GFXm aggregate path: fast for some cases but performs one aggregate at a time.
- Compute hash aggregate path: has a fused wrapper for multiple requests but
  still uses a single scalar key column.
- Decimal d32/d64 aggregates: supported through the existing decimal hash
  aggregate shaders for selected operators.

The main bottlenecks for Q1-like queries are:

- `groupby_composite_key` materialization before aggregation.
- Multiple independent aggregate passes in the GFXm path.
- Extraction/reordering overhead for each aggregate result.
- Hash table sizing tied mostly to input rows or fixed hints instead of unique
  group count.

RasterDF already exposes the right public shape in
`rasterdf/include/rasterdf/groupby.hpp`:

- `rasterdf::groupby(table_view keys, ...)`
- `aggregate(vector<aggregation_request>)`
- `groupby_multi_key(...)`

But `groupby_multi_key` is currently a stub. The new implementation should land
there as a new engine, while preserving all older paths.

## Design Overview

### What Sirius/cuDF Actually Does

Sirius does not implement its own low-level groupby kernel for the fast path.
It materializes Raster/Sirius columns as cuDF column views, builds a
`cudf::table_view` for the group keys, groups all aggregations into
`cudf::groupby::aggregation_request` objects, and calls
`cudf::groupby(...).aggregate(requests)`.

Relevant local references:

- `sirius/src/cuda/cudf/cudf_groupby.cu`: builds `keys_table`, request vectors,
  and calls cuDF groupby.
- `sirius/src/operator/gpu_physical_grouped_aggregate.cpp`: measures the four
  visible stages: materialize group keys, materialize aggregate columns,
  `cudf_groupby`, and combine results.
- `rasterdf/cudf/cpp/src/groupby/groupby.cu`: cuDF chooses hash groupby when
  requests are compatible, otherwise sort groupby.
- `rasterdf/cudf/cpp/src/groupby/hash/compute_single_pass_aggs.cuh`: cuDF's
  important fast path. It computes row-to-group mappings, tries a
  shared-memory/local aggregation path, and falls back to global memory only
  when the workload is not compatible.

The performance difference for Q1 comes from that middle step. Our current
tuple shader does:

```text
for every row:
  find global hash slot
  atomic update final aggregate state for that global slot
```

For Q1 this means roughly 59M rows hammer only four final groups. The count can
survive that, but floating/double CAS accumulation becomes both slow and risky
under extreme contention.

cuDF's compatible hash groupby does closer to:

```text
1. Build global key identity set.
2. Map each row to a group id or a block-local rank.
3. For low-cardinality fixed-width work, aggregate inside block/local memory.
4. Merge local results into compact final result columns.
5. Finalize compound aggregates such as mean from sum/count.
```

The important knobs from cuDF are:

- block size: 128 rows
- shared/local cardinality threshold: 128 groups per block
- hash table load factor around 50%
- fallback to global-memory aggregation when shared/local aggregation is not
  compatible
- grouping multiple aggregations for the same input column into one request

This means the RasterDF redesign should not stop at a "tuple global hash table".
That path is useful for medium/high-cardinality multi-key groupby, but it is the
wrong default for Q1-style low-cardinality hot groups.

### RasterDF Equivalent

Add a new compute-only engine:

```text
rasterdf/include/rasterdf/groupby_tuple.hpp
rasterdf/src/operators/groupby_tuple.cpp
rasterdf/shaders/groupby_tuple/*.comp
```

The engine stores full tuple keys in hash table slots instead of compressing
keys into one composite integer. It uses a row-wise tuple hash plus equality
check across all key columns.

The high-level tuple/global pipeline:

```text
1. Plan aggregate requests
2. Estimate unique group count and choose hash table size
3. Allocate tuple-key hash table and aggregate state columns
4. Initialize validity / slot metadata / aggregate states
5. Build aggregate table in one pass over input rows
6. Read back unique group count only
7. Extract keys + all aggregate outputs in one pass over occupied slots
8. Optional stable CPU/GPU sort for deterministic output order
```

This becomes the new preferred path for:

- 2+ key GROUP BY.
- 1-key GROUP BY with many aggregates when the fused tuple path is faster.
- Medium/high-cardinality grouped aggregations with multiple
  SUM/COUNT/AVG/MIN/MAX outputs.

For Q1/Q5-style low-cardinality grouped aggregations, add a second
cuDF-inspired local-combine path:

```text
1. Estimate key ranges/cardinality.
2. If fixed-width keys and estimated groups <= local threshold, use dense/local path.
3. Build row -> dense group id mapping.
4. Aggregate per workgroup/tile into partial state buffers.
5. Reduce partial states into final compact groups.
6. Extract key columns and aggregate outputs once.
```

The existing tuple global hash path should be skipped for very large inputs with
very small estimated group counts, because that is exactly the hot-key
contention shape that cuDF avoids.

The existing GFXm/simple hash paths remain available and can be selected by
flags or by planner heuristics.

## New Internal API

Add a new RasterDF operator-level API:

```cpp
namespace rasterdf {

struct tuple_groupby_options {
  uint32_t estimated_groups = 0;
  uint32_t max_load_factor_percent = 55;
  bool deterministic_output = true;
  bool force_two_pass_sizing = false;
};

aggregation_result groupby_tuple_hash(
    table_view const& keys,
    std::vector<aggregation_request> requests,
    context& ctx,
    execution::dispatcher& disp,
    memory_resource* mr,
    tuple_groupby_options options = {});

}
```

Then change only the stub path:

```cpp
aggregation_result groupby_multi_key(...) {
  return groupby_tuple_hash(keys, std::move(requests), ctx, disp, mr, options);
}
```

Keep `groupby::aggregate()` behavior unchanged for the old 1-key path until the
new engine has correctness and performance coverage.

## Tuple Key Model

Each hash table slot stores:

```text
slot_state[slot]    uint32   0 = empty, 1 = occupied
slot_hash[slot]     uint64   full row tuple hash
slot_row[slot]      uint32   first row index that owns this tuple
```

The slot does not initially duplicate every key column. Instead, `slot_row`
points back to the first input row for the group, and tuple equality compares
candidate row keys against the stored row keys.

This saves memory and avoids writing N key columns into the hash table during
aggregation. Extraction later gathers output key values from `slot_row`.

Equality check:

```text
equal_tuple(candidate_row, stored_row):
  for each key column:
    compare key_col[k][candidate_row] == key_col[k][stored_row]
```

Supported key dtypes in the first implementation:

- INT16, INT32, INT64
- FLOAT32, FLOAT64 using bit-normalized equality for grouping
- DECIMAL16/32/64 as fixed-width scaled integers
- STRING/VARCHAR by comparing offsets/chars or, initially, by using existing
  string hashes plus optional exact verification

D128 support should be added later as two 64-bit words.

## Tuple Hashing

Create new shader helpers under:

```text
rasterdf/shaders/groupby_tuple/tuple_hash_common.glsl
```

Hash rules:

- Compute a 64-bit hash per row.
- Mix each column with type-specific normalization.
- Include the key column index in the mix to reduce accidental symmetry.
- For strings, hash bytes from offsets/chars with the same string hash seed used
  elsewhere, but widened to 64-bit.
- For decimals, hash the scaled integer representation directly.

Suggested mixing:

```text
h = seed
for each key column:
  x = normalized_value_hash(column_value)
  h = mix64(h ^ (x + column_index_constant))
```

Use one canonical hash routine for:

- cardinality estimation
- aggregate build
- extract/debug validation

This prevents a painful class of bugs where sizing and aggregation disagree.

## Smart Table Sizing

The new engine should not blindly allocate based on input rows. It should select
table size using one of three modes.

### Mode A: Planner Hints

If RasterDB has reliable cardinality or dictionary information, pass it through
`tuple_groupby_options.estimated_groups`.

Examples:

- TPC-H Q1 has tiny key domains.
- Dictionary encoded columns can often expose domain size.
- Filters on small dimension keys can reduce expected groups heavily.

Initial RasterDB integration can hardcode no hints; the API should still be
ready for them.

### Mode B: GPU Cardinality Sketch

Add a lightweight compute shader:

```text
groupby_tuple/tuple_cardinality_sketch.comp
```

It computes row tuple hashes and fills a small bitset or HyperLogLog-style
register array. The CPU reads only the small sketch, estimates unique groups,
and picks table size.

Recommended first version:

- 16K or 64K registers/bitset buckets.
- One pass over rows.
- Conservative estimate with upper clamp.
- If estimate is uncertain, over-allocate rather than rehash.

This pass is much cheaper than materializing composite keys or doing a full
groupby with an oversized table.

### Mode C: Fallback Sizing

If no hint/sketch is available:

```text
estimated_groups = min(input_rows, max(1024, input_rows / 3))
table_size = next_power_of_two(estimated_groups * 100 / load_factor)
```

For low-cardinality TPC-H groupbys, the sketch should quickly replace this.

## Aggregate State Layout

Represent aggregate outputs as state columns attached to the hash table.

```cpp
enum class tuple_agg_kind {
  COUNT,
  SUM_I64,
  SUM_F64,
  MIN_I32,
  MAX_I32,
  MIN_I64,
  MAX_I64,
  MIN_F32,
  MAX_F32,
  MIN_F64,
  MAX_F64,
  AVG_I64,
  AVG_F64,
  DECIMAL_SUM_I64,
  DECIMAL_MIN_I64,
  DECIMAL_MAX_I64
};

struct tuple_agg_spec {
  tuple_agg_kind kind;
  type_id input_type;
  type_id output_type;
  uint32_t input_value_column;
  uint32_t state_offset;
};
```

State buffers:

- One state buffer per aggregate output is easiest and keeps shader indexing
  simple.
- A later optimization can pack states into a structure-of-arrays bundle.
- AVG should be represented as SUM + COUNT internally and extracted as AVG.
- `COUNT(*)` should not require a values column.

The first production target should support:

- `COUNT(*)`, `COUNT(col)`
- `SUM(int32/int64/float32/float64/decimal32/decimal64)`
- `MIN/MAX(int32/int64/float32/float64/decimal32/decimal64)`
- `AVG(int32/int64/float32/float64/decimal32/decimal64)`

## Build Shader

New shader:

```text
groupby_tuple/tuple_hash_groupby_build.comp
```

Responsibilities:

- Iterate input rows.
- Compute tuple hash.
- Probe the hash table.
- Insert new tuple slot with `slot_row = row`.
- Update all aggregate states for that slot.

Pseudo-flow:

```text
for row in input:
  h = tuple_hash(row)
  slot = h & (table_size - 1)

  for probe in probe_limit:
    state = slot_state[slot]

    if state == occupied:
      if slot_hash[slot] == h and equal_tuple(row, slot_row[slot]):
        update_aggregates(slot, row)
        return

    if state == empty:
      if atomicCAS(slot_state[slot], empty, locked) succeeds:
        slot_hash[slot] = h
        slot_row[slot] = row
        initialize aggregate states for this row
        memoryBarrierBuffer()
        slot_state[slot] = occupied
        atomicAdd(unique_count, 1)
        return

    slot = next_probe(slot)

  overflow_counter++
```

Use a state machine rather than using `EMPTY_KEY` sentinel. That avoids sentinel
collisions and works for every dtype.

Slot states:

```text
0 = empty
1 = locked/inserting
2 = occupied
```

Readers that see `locked` should continue probing or spin briefly. Keep the
first version simple and conservative.

## Avoiding Atomic Hotspots

cuDF-like performance comes from reducing global atomics and avoiding repeated
full-table passes. The first tuple path should support two build modes.

### Direct Global Mode

Every row updates the global hash table directly.

Good for:

- high cardinality
- simple implementation
- correctness baseline

### Workgroup Local Combine Mode

Each workgroup uses a shared-memory mini hash table, then flushes one partial
aggregate per local group to global memory.

Good for:

- low cardinality
- Q1-like workloads
- repeated keys

This is similar to the current hash aggregate shaders but generalized to tuple
keys and many aggregate outputs.

Shader names:

```text
groupby_tuple/tuple_hash_groupby_build_global.comp
groupby_tuple/tuple_hash_groupby_build_local.comp
```

The engine can choose:

```text
if estimated_groups <= 4096 or estimated_groups / rows is tiny:
  use local combine
else:
  use global
```

## Extract Shader

New shader:

```text
groupby_tuple/tuple_hash_groupby_extract.comp
```

Responsibilities:

- Scan hash table slots.
- For each occupied slot:
  - assign compact output index
  - gather every key column from `slot_row`
  - write every aggregate output
  - finalize AVG as SUM / COUNT

Output order can initially be unspecified. RasterDB can sort on CPU for tiny
group counts, as it already does. Later, a GPU sort/gather path can provide
deterministic output for larger group counts.

Important improvement over the current fused wrapper:

- Extract keys once.
- Extract all aggregate outputs in the same pass.
- Avoid per-aggregate dummy key extraction and CPU-side reordering.

## Handling Any Number Of Keys And Aggregates

Push constants are too small for arbitrary key/aggregate metadata. Use metadata
buffers.

```cpp
struct tuple_key_desc {
  VkDeviceAddress data0;
  VkDeviceAddress data1;      // strings: offsets
  VkDeviceAddress data2;      // strings: chars
  uint32_t type_id;
  uint32_t scale;
  uint32_t flags;
  uint32_t reserved;
};

struct tuple_agg_desc {
  VkDeviceAddress values;
  VkDeviceAddress state;
  VkDeviceAddress aux_state;  // AVG count, optional
  uint32_t kind;
  uint32_t input_type_id;
  uint32_t output_type_id;
  uint32_t flags;
};
```

Push constants only carry addresses and sizes:

```cpp
struct tuple_groupby_build_pc {
  VkDeviceAddress key_descs;
  VkDeviceAddress agg_descs;
  VkDeviceAddress slot_state;
  VkDeviceAddress slot_hash;
  VkDeviceAddress slot_row;
  VkDeviceAddress unique_count;
  VkDeviceAddress overflow_count;
  uint32_t num_rows;
  uint32_t num_keys;
  uint32_t num_aggs;
  uint32_t table_size;
};
```

This lets the same shader support Q1, Q5, decimal tests, Python RasterDF, and
future workloads without generating one shader per arity.

Practical first limits:

- Up to 8 key columns.
- Up to 16 aggregate outputs.
- More can be enabled later by raising constants after testing register pressure.

## Decimal Support

Decimal d16/d32/d64 can work in the first tuple path by treating them as scaled
integers:

- d16/d32: stored and hashed as INT32.
- d64: stored and hashed as INT64.
- SUM d64: use INT64 accumulator initially, with overflow detection flag.
- AVG d64: SUM + COUNT, finalize to FLOAT64 or fixed-point depending on RasterDB
  result type policy.

d128 should not be forced into this first path. Add it later as:

```text
low64/high64 key compare
two-limb hash
two-limb min/max compare
two-limb sum with carry, or explicit unsupported for SUM until ready
```

## String Support

Phase 1 can support string keys in two tiers:

1. Hash-only mode for current RasterDB behavior parity.
2. Exact string equality mode for correctness under hash collisions.

Exact mode compares:

```text
len_a == len_b
memcmp(chars_a + off_a, chars_b + off_b, len)
```

Because string compares are expensive, add a string policy:

```text
STRING_HASH_ONLY_FAST
STRING_HASH_THEN_EXACT
```

Default should be exact for public correctness. Benchmark can enable hash-only
for controlled dictionary encoded workloads.

## RasterDB Integration

Add a new feature flag in:

```text
rasterdb/src/gpu/executor/gpu_executor_grouped_aggregate.cpp
```

```cpp
static constexpr bool USE_TUPLE_COMPUTE_GROUPBY = true;
```

Selection logic:

```text
if USE_TUPLE_COMPUTE_GROUPBY
   and query aggregate set is supported
   and key dtypes are supported:
     call rasterdf::groupby_tuple_hash directly with table_view of original keys
else:
     use current GFXm/composite/compute paths unchanged
```

Do not remove the current composite key path. It remains the fallback for any
unsupported tuple case.

RasterDB output reconstruction becomes simpler:

- The tuple engine returns one output key column per GROUP BY key.
- No mixed-radix decomposition.
- No surrogate ID mapping.
- No string hash key reverse lookup if exact string key extraction is supported.

During rollout, keep the old reconstruction code active for fallback only.

## Dispatcher And Build Integration

Add new dispatch methods only; do not alter existing methods:

```cpp
dispatch_tuple_groupby_cardinality_sketch(...)
dispatch_tuple_groupby_init(...)
dispatch_tuple_groupby_build_global(...)
dispatch_tuple_groupby_build_local(...)
dispatch_tuple_groupby_extract(...)
```

Add new pipeline handles:

```cpp
_tuple_groupby_cardinality_sketch_pipeline
_tuple_groupby_init_pipeline
_tuple_groupby_build_global_pipeline
_tuple_groupby_build_local_pipeline
_tuple_groupby_extract_pipeline
```

Add shader compilation entries under new names:

```text
groupby_tuple/tuple_cardinality_sketch
groupby_tuple/tuple_groupby_init
groupby_tuple/tuple_hash_groupby_build_global
groupby_tuple/tuple_hash_groupby_build_local
groupby_tuple/tuple_hash_groupby_extract
```

This keeps all current shader names and pipeline handles untouched.

## Performance Plan

Add DEBUG timers under `[Tuple GB]`:

```text
[Tuple GB] plan_requests
[Tuple GB] cardinality_sketch
[Tuple GB] allocate
[Tuple GB] init
[Tuple GB] build_global or build_local
[Tuple GB] readback_unique_count
[Tuple GB] extract_all
[Tuple GB] output_sort
[Tuple GB] total
```

Track counters:

```text
rows
estimated_groups
actual_groups
table_size
load_factor
overflow_count
num_keys
num_aggs
local_combine_enabled
```

For Q1 SF10, the desired shape is:

```text
projection/filter: unchanged initially
groupby_composite_key: removed
groupby build+aggregate: one pass
extract: one pass
aggregate total: target < 40 ms after local combine
```

The first correctness version may be slower than GFXm. The performance version
requires local combine and extract-all.

## Implementation Phases

### Phase 0: Non-invasive scaffolding

- Add `groupby_tuple.hpp`.
- Add `groupby_tuple.cpp` with feature-gated entry point.
- Add tuple metadata structs.
- Add dispatcher stubs and shader compile entries.
- Add `USE_TUPLE_COMPUTE_GROUPBY = false` by default until tests pass.
- No behavior change.

### Phase 1: Global tuple hash, COUNT only

- Implement tuple hash and tuple equality for INT32 keys.
- Implement `COUNT(*)`.
- Support 1-4 key columns.
- Add extraction of key columns from `slot_row`.
- Add tests:
  - one key count
  - two key count
  - three key count
  - high cardinality
  - repeated tiny cardinality

Acceptance:

- Matches DuckDB for multi-key COUNT.
- No composite key path used when flag is enabled.

### Phase 2: SUM/MIN/MAX/AVG for INT32/FLOAT32

- Add aggregate descriptor buffer.
- Update all aggregate states in one build pass.
- Implement extract-all for keys plus aggregate outputs.
- AVG uses SUM + COUNT.
- Add Q1-like test:
  - two INT32 keys
  - two SUM outputs
  - one COUNT output

Acceptance:

- Q1 adapted result matches CPU/Sirius.
- `[Tuple GB] groupby_composite_key` no longer appears.

### Phase 3: cuDF-like local combine mode

This is the Sirius/cuDF-equivalent phase and should be prioritized before using
the tuple global path for Q1.

Add new shaders under `rasterdf/shaders/groupby_tuple/` without changing the
existing tuple or GFXm shaders:

```text
tuple_groupby_dense_map.comp
tuple_groupby_dense_partial.comp
tuple_groupby_dense_reduce.comp
tuple_groupby_dense_extract.comp
```

Implementation shape:

- `dense_map`: compute dense group id for each row from fixed-width key ranges.
- `dense_partial`: each workgroup/tile updates a private partial table shaped
  `[num_workgroups][num_groups][num_aggs]`.
- `dense_reduce`: one group/aggregate reducer merges all workgroup partials into
  final states.
- `dense_extract`: writes compact keys and aggregate outputs.

Initial scope:

- fixed-width keys only
- INT32 key ranges first, then INT64/FLOAT32/FLOAT64 by typed normalization
- SUM, COUNT, MIN, MAX, AVG
- no strings
- no nulls

Planner rule:

```text
if rows >= 1M
   and fixed_width_keys
   and estimated_groups <= 4096
   and aggregate values are fixed-width:
       use dense/local groupby
else:
       use tuple global hash or existing fallback
```

Why not directly use workgroup shared memory for all state? GLSL shared arrays
need static sizing, while `num_groups * num_aggs * dtype_size` is dynamic. The
first Vulkan version should use per-workgroup global partial buffers because it
is simpler and safe. Once correct, replace the hot inner loop with a
shared-memory mini table for `num_groups <= 128`, matching cuDF's threshold.

Acceptance:

- Low-cardinality Q1 groupby time is close to Sirius/cuDF groupby time.
- No correctness regressions for high-cardinality tests.

### Phase 4: Better cardinality estimation

- Add tuple cardinality sketch shader.
- Use planner hints when available, otherwise sketch.
- Size table from estimated unique groups.
- Log estimate vs actual groups.

Acceptance:

- Avoids huge hash tables for tiny group counts.
- Avoids rehash/overflow for high group counts.

### Phase 5: Wider dtype coverage

- INT64/FLOAT64 keys and values.
- DECIMAL32/DECIMAL64 keys and values.
- STRING exact tuple keys.
- Null handling policy, if/when nullable columns are exposed.

Acceptance:

- Existing decimal groupby tests pass through tuple path where supported.
- String groupby tests match CPU exactly.

### Phase 6: RasterDB default rollout

- Enable tuple path for supported aggregate plans.
- Keep fallback for unsupported dtypes/aggs.
- Add per-query A/B environment flag:

```text
RDB_GROUPBY_ENGINE=auto|tuple|gfxm|compute|legacy
```

Acceptance:

- Q1 and Q5 are correct.
- Q1 groupby is materially faster than current GFXm multi-pass path.
- Fallback path still handles unsupported cases.

## Test Matrix

Add RasterDF C++ tests:

- `groupby_tuple_count_int32`
- `groupby_tuple_sum_int32`
- `groupby_tuple_sum_float32`
- `groupby_tuple_min_max`
- `groupby_tuple_avg`
- `groupby_tuple_many_keys`
- `groupby_tuple_many_aggs`
- `groupby_tuple_high_cardinality`
- `groupby_tuple_low_cardinality_hot_keys`

Add RasterDB SQL tests:

- Q1-style groupby with 2 keys and 3 aggregates.
- 1-key groupby fallback comparison.
- 3-key groupby.
- Decimal d32/d64 key groupby.
- String key groupby.

For every test:

- Compare against DuckDB CPU.
- Run with `RDB_GROUPBY_ENGINE=tuple`.
- Run with `RDB_GROUPBY_ENGINE=legacy`.

## Risks And Mitigations

### Risk: tuple equality with arbitrary types gets complicated

Mitigation:

- Phase by dtype.
- Keep unsupported dtypes on old path.
- Centralize key descriptor hashing/equality helpers.

### Risk: too many aggregate variants in one shader create register pressure

Mitigation:

- Use metadata descriptors but keep per-kind helper functions small.
- Add build shader variants later:
  - numeric only
  - decimal only
  - string keys

### Risk: global atomics are slow for low-cardinality groupby

Mitigation:

- Local combine mode is mandatory for Sirius-like Q1 performance.
- Use cardinality estimate to choose it aggressively.

### Risk: hash table sizing estimate is wrong

Mitigation:

- Track `overflow_count`.
- If overflow is nonzero, rerun once with doubled table size.
- Log estimate vs actual group count in DEBUG mode.

### Risk: output order differs

Mitigation:

- Keep deterministic CPU sort for small group counts initially.
- Later add GPU sort/gather for larger outputs.

## Why This Is Closer To cuDF/Sirius

cuDF-style groupby does not first encode arbitrary key tuples into ad hoc
composite integers. It treats rows as key tuples, hashes those tuples, builds a
grouping hash table, and applies aggregate operators to the group state. The
important performance properties are:

- Tuple keys are handled directly.
- Multiple aggregates share grouping work.
- Group count estimation avoids wasteful allocation.
- Local/block aggregation reduces global atomic pressure.
- Extraction materializes all result columns together.

This plan moves RasterDB/RasterDF toward that architecture while leaving every
currently working path available as fallback.

## First PR Scope

The smallest useful PR should include:

- New files only under `groupby_tuple`.
- Dispatcher/pipeline registration for new tuple shaders.
- `groupby_tuple_hash()` with INT32 keys and `COUNT(*)`.
- RasterDB feature flag defaulting off.
- Tests for 2-key and 3-key count.

The second PR should add:

- SUM/MIN/MAX/AVG for INT32/FLOAT32.
- Extract-all.
- Q1-style benchmark and DEBUG timers.

The third PR should add:

- Local combine mode.
- Cardinality sketch.
- Auto selection in RasterDB.
