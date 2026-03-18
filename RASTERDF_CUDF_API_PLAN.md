# RasterDF Reorganization Plan: cuDF-Compatible API Layer

## Goal
Reorganize rasterdf to mirror cuDF's API structure so that rasterdb (DuckDB extension)
can do near 1:1 API replacements from Sirius's cuDF calls → rasterdf calls.

---

## 1. Current State Analysis

### Sirius Architecture (cuDF-based)
```
src/cuda/cudf/           ← cuDF wrapper layer (6 files)
  cudf_join.cu             cudf::hash_join, cudf::distinct_hash_join
  cudf_aggregate.cu        cudf::reduce (SUM/MIN/MAX/MEAN/NUNIQUE)
  cudf_groupby.cu          cudf::groupby::groupby
  cudf_orderby.cu          cudf::sort_by_key, cudf::sorted_order
  cudf_duplicate_elimination.cu  cudf::groupby::get_groups
  cudf_utils.cu            type conversions, null mask creation

src/cuda/operator/       ← Custom CUDA kernels (14 files)
  hash_join_inner.cu, hash_join_right.cu, hash_join_single.cu
  nested_loop_join.cu, materialize.cu, comparison_expression.cu
  strings_matching.cu, substring.cu, etc.

src/include/cudf/cudf_utils.hpp  ← GetCudfType(), IsCudfTypeDecimal()
src/include/gpu_columns.hpp      ← GPUColumn, GPUIntermediateRelation, DataWrapper
src/include/gpu_buffer_manager.hpp ← GPUBufferManager (RMM pool_memory_resource)
```

**Sirius Call Pattern:**
```
Operator → cudf_hash_inner_join() → GPUColumn::convertToCudfColumn() →
  cudf::hash_join(table_view, table_view) → extract cudf::column → setFromCudfColumn()
```

All cuDF APIs used by Sirius operate on `cudf::column_view` / `cudf::table_view`.
Memory is managed via `rmm::mr::pool_memory_resource`.

### rasterdf Current State
```
include/rasterdf/
  core/     ← types, column, column_view, table, table_view, device_buffer, context
  execution/← dispatcher.hpp (22KB header, 57KB impl — ALL operations in one class)
  memory/   ← memory_manager, pool_memory_resource, vulkan_memory_resource
  
src/
  core/     ← VkEngine, VulkanDevice, context, device_buffer, column, table
  execution/← dispatcher.cpp (monolithic 57KB)
  memory/   ← memory_manager, pool_memory_resource, vulkan_memory_resource
```

**rasterdf has low-level primitives but NO high-level API layer.**
Everything goes through `dispatcher.dispatch_*()` with raw push constants.

### rasterdf Available Operations (via dispatcher)
| Category | Operations | Types |
|----------|-----------|-------|
| Reduction | sum, min, max | int32, float32 |
| Filter | compare (gt/lt/ge/le/eq/ne) + stream compaction | int32, float32 |
| Binary ops | add, sub, mul, div (col-col, col-scalar) | int32, float32 |
| Unary ops | negate, abs, etc. | int32, float32 |
| Radix sort | full 8-pass sort with payload | int32, float32 |
| GroupBy (sort) | find_boundaries, segment_sum/count/min/max, extract_keys | int32 |
| GroupBy (hash) | hash_aggregate_sum/count/min/max/mean, extract | int32 |
| Join (hash) | build_count, build_insert, probe_count, probe_write | int32 |
| Prefix scan | 3-pass (local + global + add) | int32 |
| Gather | by mask, by indices | int32 |

### What's MISSING vs cuDF
- **High-level API functions** (sort_by_key, inner_join, reduce, groupby — all currently manual push constant wiring)
- **INT64/FLOAT64** shader variants for most operations
- **VARCHAR/STRING** support (shaders + column representation)
- **DECIMAL** support
- **Null mask** handling in operations (cudf::bitmask_type equivalent)
- **cast()** between types
- **DISTINCT** / stream compaction as a high-level API
- **Left/Right/Outer** join variants
- **Multi-column** sort and join
- **make_empty_column()** factory
- **Scalar** type with proper reduce-to-scalar support

---

## 2. Target API Structure (mirroring cuDF)

### New Header Layout
```
include/rasterdf/
├── core/                          (KEEP — no changes needed)
│   ├── types.hpp                    rasterdf::type_id, rasterdf::data_type
│   ├── column.hpp                   rasterdf::column (owning)
│   ├── column_view.hpp              rasterdf::column_view (non-owning)
│   ├── table.hpp                    rasterdf::table (owning)
│   ├── table_view.hpp               rasterdf::table_view (non-owning)
│   ├── device_buffer.hpp            rasterdf::device_buffer
│   ├── scalar.hpp                   rasterdf::scalar
│   └── context.hpp                  rasterdf::context
│
├── column/                        (NEW — mirrors cudf/column/)
│   ├── column_factories.hpp         make_empty_column(), make_numeric_column()
│   └── column_view.hpp              (re-export from core/)
│
├── sorting.hpp                    (NEW — mirrors cudf/sorting.hpp)
│   Functions:
│     std::unique_ptr<column> sorted_order(table_view input, 
│       vector<order> column_order, vector<null_order> null_precedence, mr*)
│     std::unique_ptr<table> sort_by_key(table_view values, table_view keys,
│       vector<order> column_order, vector<null_order> null_precedence, mr*)
│     std::unique_ptr<table> sort(table_view input, 
│       vector<order> column_order, mr*)
│
├── join.hpp                       (NEW — mirrors cudf/join.hpp)
│   Classes/Functions:
│     class hash_join {
│       hash_join(table_view build, null_equality);
│       pair<unique_ptr<column>, unique_ptr<column>> inner_join(table_view probe, mr*);
│       pair<unique_ptr<column>, unique_ptr<column>> left_join(table_view probe, mr*);
│     };
│     // Free functions:
│     inner_join(table_view left, table_view right, ...);
│     left_join(table_view left, table_view right, ...);
│
├── groupby.hpp                    (NEW — mirrors cudf/groupby.hpp)
│   Classes:
│     class groupby {
│       groupby(table_view keys);
│       struct aggregate_result { unique_ptr<table> keys; vector<unique_ptr<column>> values; };
│       aggregate_result aggregate(vector<aggregation_request> requests, mr*);
│       groups_result get_groups(mr*);  // for DISTINCT
│     };
│     struct aggregation_request {
│       column_view values;
│       vector<unique_ptr<aggregation>> aggregations;
│     };
│
├── reduction.hpp                  (NEW — mirrors cudf/reduction.hpp)
│   Functions:
│     std::unique_ptr<scalar> reduce(column_view col, reduce_aggregation agg,
│       data_type output_type, mr*);
│
├── aggregation.hpp                (NEW — mirrors cudf/aggregation.hpp)
│   Classes:
│     class aggregation { Kind kind; };
│     class reduce_aggregation : public aggregation {};
│     class groupby_aggregation : public aggregation {};
│     // Factories:
│     make_sum_aggregation(), make_min_aggregation(), make_max_aggregation(),
│     make_mean_aggregation(), make_count_aggregation()
│
├── copying.hpp                    (NEW — mirrors cudf/copying.hpp)
│   Functions:
│     std::unique_ptr<table> gather(table_view source, column_view gather_map, mr*);
│     std::unique_ptr<table> scatter(table_view source, column_view scatter_map, 
│       table_view target, mr*);
│     std::unique_ptr<column> empty_like(column_view input);
│
├── stream_compaction.hpp          (NEW — mirrors cudf/stream_compaction.hpp)
│   Functions:
│     std::unique_ptr<table> apply_boolean_mask(table_view input, 
│       column_view boolean_mask, mr*);
│     std::unique_ptr<table> distinct(table_view input, 
│       vector<size_type> keys, mr*);
│
├── unary.hpp                      (NEW — mirrors cudf/unary.hpp)
│   Functions:
│     std::unique_ptr<column> cast(column_view input, data_type out_type, mr*);
│     std::unique_ptr<column> unary_operation(column_view input, unary_op op, mr*);
│
├── binary.hpp                     (NEW — mirrors cudf/binaryop.hpp)  
│   Functions:
│     std::unique_ptr<column> binary_operation(column_view lhs, column_view rhs,
│       binary_op op, data_type output_type, mr*);
│     std::unique_ptr<column> binary_operation(column_view lhs, scalar rhs,
│       binary_op op, data_type output_type, mr*);
│
├── filling.hpp                    (NEW)
│   Functions:
│     void fill(mutable_column_view dest, size_type begin, size_type end, scalar value);
│
├── types.hpp                      (NEW — re-export + enums)
│   Enums:
│     enum class order { ASCENDING, DESCENDING };
│     enum class null_order { BEFORE, AFTER };
│     enum class null_equality { EQUAL, UNEQUAL };
│     enum class binary_op { ADD, SUB, MUL, DIV, ... };
│     enum class unary_op { NEGATE, ABS, ... };
│
├── execution/                     (KEEP — internal, not called by rasterdb directly)
│   ├── dispatcher.hpp
│   └── constants.hpp
│
└── memory/                        (KEEP as-is, aliased as vmm/)
    ├── memory_manager.hpp           → aliased: rasterdf::vmm::memory_manager
    ├── memory_resource.hpp          → aliased: rasterdf::vmm::memory_resource  
    ├── pool_memory_resource.hpp     → aliased: rasterdf::vmm::pool_memory_resource
    └── vulkan_memory_resource.hpp
```

### New Source Layout
```
src/
├── core/           (KEEP as-is)
├── execution/      (KEEP dispatcher.cpp as-is — it's the backend)
├── memory/         (KEEP as-is)
├── sorting.cpp     (NEW — implements sorting.hpp using dispatcher radix sort)
├── join.cpp        (NEW — implements join.hpp using dispatcher hash join)
├── groupby.cpp     (NEW — implements groupby.hpp using dispatcher hash/sort groupby)
├── reduction.cpp   (NEW — implements reduction.hpp using dispatcher sum/min/max)
├── copying.cpp     (NEW — implements copying.hpp using dispatcher gather)
├── stream_compaction.cpp (NEW — implements stream_compaction.hpp using prefix scan + gather)
├── unary.cpp       (NEW — implements unary.hpp)
├── binary.cpp      (NEW — implements binary.hpp)
└── column_factories.cpp (NEW)
```

---

## 3. API Mapping: cuDF → rasterdf

### 3.1 Types (cudf_utils.hpp → rasterdf/types.hpp)
```
cudf::type_id::INT32          → rasterdf::type_id::INT32
cudf::type_id::INT64          → rasterdf::type_id::INT64
cudf::type_id::FLOAT32        → rasterdf::type_id::FLOAT32
cudf::type_id::FLOAT64        → rasterdf::type_id::FLOAT64
cudf::type_id::BOOL8          → rasterdf::type_id::BOOL8
cudf::type_id::STRING         → rasterdf::type_id::STRING
cudf::type_id::TIMESTAMP_*    → rasterdf::type_id::TIMESTAMP_*
cudf::data_type               → rasterdf::data_type
cudf::column_view             → rasterdf::column_view
cudf::table_view              → rasterdf::table_view
cudf::column                  → rasterdf::column
cudf::table                   → rasterdf::table
cudf::scalar                  → rasterdf::scalar
cudf::bitmask_type            → rasterdf::bitmask_type (uint32_t)
cudf::size_type               → rasterdf::size_type (int32_t)  [already exists]
```

### 3.2 Memory (rmm → rasterdf::vmm)
```
rmm::mr::device_memory_resource     → rasterdf::memory_resource       [exists]
rmm::mr::pool_memory_resource       → rasterdf::pool_memory_resource  [exists]
rmm::mr::cuda_memory_resource       → rasterdf::vulkan_memory_resource [exists]
rmm::device_buffer                  → rasterdf::device_buffer         [exists]
GPUBufferManager::GetInstance().mr   → rasterdf::memory_manager::workspace_resource()
cudf::set_current_device_resource   → (not needed — pass mr* explicitly)
```

### 3.3 Reduction (cudf_aggregate.cu → rasterdf/reduction.hpp)
```cpp
// Sirius:
cudf::reduce(col_view, cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
             output_type, stream, mr);

// rasterdf target:
rasterdf::reduce(col_view, rasterdf::make_sum_aggregation<rasterdf::reduce_aggregation>(),
                 output_type, mr);
```

### 3.4 Sorting (cudf_orderby.cu → rasterdf/sorting.hpp)
```cpp
// Sirius:
auto sorted = cudf::sort_by_key(values_table, keys_table, column_orders, null_precedence);

// rasterdf target:
auto sorted = rasterdf::sort_by_key(values_table, keys_table, column_orders, null_precedence, mr);
```

### 3.5 GroupBy (cudf_groupby.cu → rasterdf/groupby.hpp)
```cpp
// Sirius:
cudf::groupby::groupby grpby_obj(keys_table);
auto [keys, results] = grpby_obj.aggregate(requests);

// rasterdf target:
rasterdf::groupby grpby_obj(keys_table);
auto result = grpby_obj.aggregate(requests, mr);
```

### 3.6 Join (cudf_join.cu → rasterdf/join.hpp)
```cpp
// Sirius:
auto hash_table = cudf::distinct_hash_join(build_table, cudf::null_equality::EQUAL);
auto [left_indices, right_indices] = hash_table.inner_join(probe_table);

// rasterdf target:
rasterdf::hash_join hash_table(build_table, rasterdf::null_equality::EQUAL);
auto [left_indices, right_indices] = hash_table.inner_join(probe_table, mr);
```

### 3.7 Stream Compaction (cudf → rasterdf/stream_compaction.hpp)
```cpp
// Sirius:
auto result = cudf::distinct(table, key_columns, null_equality);

// rasterdf target:
auto result = rasterdf::distinct(table, key_columns, null_equality, mr);
```

### 3.8 Copying (cudf → rasterdf/copying.hpp)
```cpp
// Sirius:
auto result = cudf::gather(source_table, gather_map);

// rasterdf target:
auto result = rasterdf::gather(source_table, gather_map, mr);
```

### 3.9 Type Casting (cudf → rasterdf/unary.hpp)
```cpp
// Sirius:
auto casted = cudf::cast(input_col, target_type, stream, mr);

// rasterdf target:
auto casted = rasterdf::cast(input_col, target_type, mr);
```

---

## 4. rasterdb API Mapping (Sirius operator → RasterDB operator)

### 4.1 GPUColumn → rasterdf::column (already partially done)
```
Sirius GPUColumn::convertToCudfColumn()  →  (not needed, rasterdf::column IS the native type)
Sirius GPUColumn::setFromCudfColumn()    →  (not needed)
Sirius GPUBufferManager::customCudaMalloc →  rasterdf::device_buffer(mr, size)
Sirius GPUBufferManager::customCudaFree   →  device_buffer destructor
Sirius GPUColumnType                      →  rasterdf::data_type
Sirius GPUIntermediateRelation            →  rasterdf::table (or rasterdb::gpu_table)
```

### 4.2 Operator Files (src/cuda/cudf/ → src/gpu/rdf/)
```
cudf_aggregate.cu       → rdf_aggregate.cpp    (calls rasterdf::reduce)
cudf_groupby.cu         → rdf_groupby.cpp      (calls rasterdf::groupby)
cudf_orderby.cu         → rdf_orderby.cpp      (calls rasterdf::sort_by_key)
cudf_join.cu            → rdf_join.cpp         (calls rasterdf::hash_join)
cudf_duplicate_elimination.cu → rdf_dedup.cpp  (calls rasterdf::distinct)
cudf_utils.cu           → rdf_utils.cpp        (type conversions)
```

---

## 5. Implementation Phases

### Phase 1: Core API Headers + Enums (no shader changes)
**Effort: ~2 days**

1. Create `rasterdf/types.hpp` top-level with `order`, `null_order`, `null_equality`, `binary_op`, `unary_op`, `bitmask_type` enums
2. Create `rasterdf/aggregation.hpp` with aggregation class hierarchy + factory functions
3. Create `rasterdf/column/column_factories.hpp` with `make_empty_column()`, `make_numeric_column()`
4. Add `rasterdf::bitmask_type` = `uint32_t` typedef and null mask utilities

### Phase 2: High-Level API — Reduction + Binary/Unary (wraps existing dispatcher)
**Effort: ~2 days**

1. Create `rasterdf/reduction.hpp` + `src/reduction.cpp`
   - `rasterdf::reduce()` wraps `dispatcher::dispatch_sum/min/max` + readback to scalar
   - Supports: SUM, MIN, MAX, COUNT for int32/float32
2. Create `rasterdf/binary.hpp` + `src/binary.cpp`
   - `rasterdf::binary_operation()` wraps `dispatcher::dispatch_binary_op`
3. Create `rasterdf/unary.hpp` + `src/unary.cpp`
   - `rasterdf::unary_operation()` wraps `dispatcher::dispatch_unary_op`
   - `rasterdf::cast()` — placeholder, initially identity for same-type

### Phase 3: High-Level API — Sorting + Copying + Stream Compaction
**Effort: ~3 days**

1. Create `rasterdf/sorting.hpp` + `src/sorting.cpp`
   - `rasterdf::sort()` wraps batched radix sort dispatcher
   - `rasterdf::sort_by_key()` — sort values by separate key columns
   - `rasterdf::sorted_order()` — return permutation indices only
   - Initially: single-column int32/float32 sort
2. Create `rasterdf/copying.hpp` + `src/copying.cpp`
   - `rasterdf::gather()` wraps `dispatcher::dispatch_gather_indices`
3. Create `rasterdf/stream_compaction.hpp` + `src/stream_compaction.cpp`
   - `rasterdf::apply_boolean_mask()` wraps prefix_scan + gather

### Phase 4: High-Level API — GroupBy
**Effort: ~3 days**

1. Create `rasterdf/groupby.hpp` + `src/groupby.cpp`
   - `rasterdf::groupby` class with `aggregate()` method
   - Internally uses hash-based groupby (dispatch_hash_aggregate_*)
   - Supports: SUM, COUNT, MIN, MAX, MEAN for int32
   - `get_groups()` for DISTINCT

### Phase 5: High-Level API — Join
**Effort: ~3 days**

1. Create `rasterdf/join.hpp` + `src/join.cpp`
   - `rasterdf::hash_join` class
   - `inner_join()` wraps build_count → prefix_scan → build_insert → probe_count → prefix_scan → probe_write
   - Returns `pair<unique_ptr<column>, unique_ptr<column>>` (left/right index columns)
2. Free functions: `rasterdf::inner_join(left, right, left_on, right_on, mr)`

### Phase 6: Extended Type Support — INT64/FLOAT64 Shaders
**Effort: ~4 days**

1. Duplicate int32 shaders for int64 (compare, binary_op, reduction, filter, sort, groupby, join)
2. Duplicate float32 shaders for float64
3. Update dispatcher to select shader by type_id
4. Update all high-level APIs to support int64/float64

### Phase 7: Null Mask Support
**Effort: ~3 days**

1. Add `rasterdf::bitmask_type` (uint32_t) to column
2. Create null mask shaders: create_null_mask, set_null_mask, combine_masks
3. Update reduction/groupby/join to propagate null masks
4. Add `null_count()` computation shader

### Phase 8: Multi-Column Operations
**Effort: ~3 days**

1. Multi-column sort (composite key radix sort or cuDF-style sorted_order → gather)
2. Multi-column join (composite key hashing)
3. Multi-column groupby (composite key hashing)

### Phase 9: VARCHAR/STRING Support (Future)
**Effort: ~5+ days**

1. String column representation (offsets + chars device buffers)
2. String comparison shaders
3. String hash shaders (for join/groupby)
4. String gather/scatter

---

## 6. rasterdb Refactoring (After rasterdf API Layer)

Once the high-level API layer is in place, rasterdb's gpu_executor.cpp simplifies dramatically:

### Current (push constant wiring in gpu_executor.cpp):
```cpp
// ~50 lines to do a filter
compare_push_constants pc{};
pc.input_addr = col.address();
pc.output_addr = result.address();
// ... set 6 more fields ...
disp.dispatch_compare(pc);
// then 30 lines for prefix scan + gather
```

### Target (one-liner):
```cpp
auto filtered = rasterdf::apply_boolean_mask(input_table.view(), mask_col.view(), mr);
```

### rasterdb Operator File Structure (matching Sirius):
```
rasterdb/src/gpu/
├── gpu_context.hpp/.cpp         (KEEP — wraps rasterdf context)
├── gpu_types.hpp                (KEEP — DuckDB ↔ rasterdf type conversion)
├── gpu_table.hpp/.cpp           (KEEP — simplified, wraps rasterdf::table)
├── gpu_executor.hpp/.cpp        (REFACTOR — use high-level rasterdf APIs)
├── rdf/                         (NEW — mirrors Sirius src/cuda/cudf/)
│   ├── rdf_aggregate.cpp          calls rasterdf::reduce()
│   ├── rdf_groupby.cpp            calls rasterdf::groupby
│   ├── rdf_orderby.cpp            calls rasterdf::sort_by_key()
│   ├── rdf_join.cpp               calls rasterdf::hash_join
│   ├── rdf_dedup.cpp              calls rasterdf::distinct()
│   └── rdf_utils.cpp              type conversions, null mask utils
└── operator/                    (NEW — mirrors Sirius src/operator/)
    ├── rasterdb_physical_filter.cpp
    ├── rasterdb_physical_projection.cpp
    ├── rasterdb_physical_hash_join.cpp
    ├── rasterdb_physical_order.cpp
    ├── rasterdb_physical_grouped_aggregate.cpp
    ├── rasterdb_physical_ungrouped_aggregate.cpp
    ├── rasterdb_physical_limit.cpp
    └── rasterdb_physical_table_scan.cpp
```

---

## 7. Priority Order (What Enables Most Queries)

| Priority | Component | Enables |
|----------|-----------|---------|
| P0 | reduction.hpp + sorting.hpp + stream_compaction.hpp | Clean up existing working queries |
| P1 | groupby.hpp | GROUP BY queries |
| P2 | join.hpp | JOIN queries |
| P3 | INT64/FLOAT64 shaders | BIGINT/DOUBLE columns |
| P4 | Multi-column sort/join/groupby | Complex queries |
| P5 | Null mask support | Correct NULL handling |
| P6 | VARCHAR/STRING | String columns |
| P7 | DECIMAL | Decimal arithmetic |

---

## 8. Naming Convention Summary

| cuDF | rasterdf | rmm → vmm |
|------|----------|-----------|
| `cudf::` | `rasterdf::` | `rmm::mr::` → `rasterdf::` |
| `cudf::column` | `rasterdf::column` | `rmm::device_buffer` → `rasterdf::device_buffer` |
| `cudf::table_view` | `rasterdf::table_view` | `rmm::mr::pool_memory_resource` → `rasterdf::pool_memory_resource` |
| `cudf::reduce()` | `rasterdf::reduce()` | `cudf::set_current_device_resource(mr)` → pass `mr*` param |
| `cudf::sort_by_key()` | `rasterdf::sort_by_key()` | |
| `cudf::hash_join` | `rasterdf::hash_join` | |
| `cudf::groupby::groupby` | `rasterdf::groupby` | |
| `cudf::gather()` | `rasterdf::gather()` | |
| `cudf::cast()` | `rasterdf::cast()` | |
| `cudf::distinct()` | `rasterdf::distinct()` | |
| `cudf::bitmask_type` | `rasterdf::bitmask_type` | |

---

## 9. Key Design Decisions

1. **Memory resource passed explicitly** — No global `set_current_device_resource()`. Every API takes `memory_resource* mr` as the last parameter (defaulting to workspace pool). This is cleaner than cuDF's thread-local approach.

2. **Dispatcher stays internal** — The dispatcher and push constants are implementation details. High-level APIs hide them completely. rasterdb never touches push constants directly.

3. **column_view stores VkDeviceAddress** — Unlike cuDF which stores raw pointers, rasterdf::column_view stores VkDeviceAddress. This is already the case and should be preserved.

4. **Scalar readback for reduce** — `rasterdf::reduce()` returns `unique_ptr<scalar>` which downloads the single result value from GPU to host. This matches cuDF's interface.

5. **table owns columns** — `rasterdf::table` takes ownership of `unique_ptr<column>` vectors, exactly like cuDF.

6. **Incremental type support** — Start with int32/float32 (already have shaders), add int64/float64 by duplicating shaders with type specialization. STRING/DECIMAL come last.
