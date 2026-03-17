# RasterDB Architecture: Migrating from Sirius (cuDF) to RasterDF (Vulkan)

## 1. Executive Summary

**RasterDB** is a fork of **Sirius** — a GPU-native SQL engine built as a DuckDB extension.
Sirius uses **NVIDIA CUDA + cuDF + RMM** for GPU execution. RasterDB replaces this entire
GPU backend with **rasterdf**, a Vulkan-based GPU DataFrame library.

The migration preserves:
- DuckDB extension architecture (planner, pipeline, operators)
- Query plan generation logic
- Pipeline execution framework
- Fallback-to-CPU strategy

The migration replaces:
- All CUDA kernel launches → Vulkan compute shader dispatches (via rasterdf `dispatcher`)
- cuDF API calls → rasterdf C++ API calls
- RMM memory management → rasterdf `memory_manager` (VMA-based)
- `cudf::bitmask_type*` null masks → rasterdf null mask buffers
- cuCascade GPU memory tiers → rasterdf pool memory resources

---

## 2. Sirius Architecture (Reference)

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   DuckDB Core Engine                     │
├─────────────────────────────────────────────────────────┤
│              sirius_extension.cpp                        │
│  Registers: gpu_buffer_init, gpu_processing,            │
│             gpu_execution table functions                │
├─────────────────────────────────────────────────────────┤
│            Query Planner (src/planner/)                  │
│  sirius_physical_plan_generator.cpp                     │
│  sirius_plan_{filter,aggregate,join,order,...}.cpp       │
├─────────────────────────────────────────────────────────┤
│         Engine & Pipeline (src/pipeline/)                │
│  sirius_engine.cpp, sirius_pipeline.cpp                 │
│  pipeline_executor.cpp, task_request.cpp                │
├─────────────────────────────────────────────────────────┤
│        Sirius Operators (src/op/)                       │
│  sirius_physical_{filter,hash_join,order,...}.cpp        │
├─────────────────────────────────────────────────────────┤
│        GPU Operators (src/operator/)                    │
│  gpu_physical_{filter,hash_join,order,...}.cpp           │
├──────────────────────┬──────────────────────────────────┤
│  cuDF Wrappers       │  Custom CUDA Kernels             │
│  src/cuda/cudf/      │  src/cuda/operator/              │
│  cudf_join.cu        │  hash_join_inner.cu              │
│  cudf_aggregate.cu   │  materialize.cu                  │
│  cudf_groupby.cu     │  comparison_expression.cu        │
│  cudf_orderby.cu     │  nested_loop_join.cu             │
│  cudf_utils.cu       │  strings_matching.cu             │
├──────────────────────┴──────────────────────────────────┤
│        GPU Memory (gpu_buffer_manager.cpp)              │
│  RMM pool_memory_resource, cuCascade tiers              │
├─────────────────────────────────────────────────────────┤
│        GPU Columns (gpu_columns.cpp)                    │
│  GPUColumn, DataWrapper, convertToCudfColumn()          │
│  cudf::bitmask_type* validity masks                     │
└─────────────────────────────────────────────────────────┘
```

### 2.2 cuDF API Surface Used by Sirius

| Sirius File | cuDF APIs Called |
|---|---|
| `cudf_join.cu` | `cudf::hash_join`, `cudf::distinct_hash_join`, `cudf::conditional_inner_join`, `cudf::mixed_inner_join`, `cudf::cast` |
| `cudf_aggregate.cu` | `cudf::reduce` (MIN, MAX, MEAN, SUM, NUNIQUE), `cudf::make_*_aggregation` |
| `cudf_groupby.cu` | `cudf::groupby::groupby`, `cudf::distinct` (stream compaction) |
| `cudf_orderby.cu` | `cudf::sorted_order`, `cudf::sort`, CUB radix sort |
| `cudf_duplicate_elimination.cu` | `cudf::distinct` |
| `cudf_utils.cu` | `cudf::table_view`, column construction utilities |
| `gpu_columns.cpp` | `cudf::column_view`, `cudf::bitmask_type`, `cudf::make_empty_column` |
| `gpu_buffer_manager.cpp` | `rmm::mr::pool_memory_resource`, `rmm::mr::cuda_memory_resource` |
| Expression executors | `cudf::cast`, `cudf::unary_cast` |

### 2.3 Key Types Mapping

| Sirius / cuDF | rasterdf Equivalent | Status |
|---|---|---|
| `cudf::data_type` | `rasterdf::data_type` | ✅ Exists |
| `cudf::type_id` | `rasterdf::type_id` | ✅ Exists (needs UINT*, DECIMAL) |
| `cudf::column` | `rasterdf::column` | ✅ Exists |
| `cudf::column_view` | `rasterdf::column_view` | ✅ Exists |
| `cudf::mutable_column_view` | `rasterdf::mutable_column_view` | ✅ Exists |
| `cudf::table` | `rasterdf::table` | ✅ Exists |
| `cudf::table_view` | `rasterdf::table_view` | ✅ Exists |
| `cudf::scalar` | `rasterdf::scalar` | ⚠️ Partial (needs full impl) |
| `cudf::bitmask_type` (`uint32_t`) | `rasterdf::bitmask_type` | ❌ Needs creation |
| `cudf::mask_state` | `rasterdf::mask_state` | ❌ Needs creation |
| `rmm::device_buffer` | `rasterdf::device_buffer` | ✅ Exists (VMA-based) |
| `rmm::mr::device_memory_resource` | `rasterdf::memory_resource` | ✅ Exists |
| `rmm::mr::pool_memory_resource` | `rasterdf::pool_memory_resource` | ✅ Exists |

---

## 3. RasterDB Target Architecture

### 3.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   DuckDB Core Engine                     │
├─────────────────────────────────────────────────────────┤
│              rasterdb_extension.cpp                      │
│  Registers: gpu_buffer_init, gpu_processing,            │
│             gpu_execution table functions                │
├─────────────────────────────────────────────────────────┤
│            Query Planner (src/planner/)                  │
│  rasterdb_physical_plan_generator.cpp                   │
│  rasterdb_plan_{filter,aggregate,join,order,...}.cpp     │
│  [REUSED FROM SIRIUS — minimal changes]                 │
├─────────────────────────────────────────────────────────┤
│         Engine & Pipeline (src/pipeline/)                │
│  rasterdb_engine.cpp, rasterdb_pipeline.cpp             │
│  [REUSED FROM SIRIUS — minimal changes]                 │
├─────────────────────────────────────────────────────────┤
│        RasterDB Operators (src/op/)                     │
│  rasterdb_physical_{filter,hash_join,order,...}.cpp      │
│  [REUSED FROM SIRIUS — minimal changes]                 │
├─────────────────────────────────────────────────────────┤
│        GPU Operators (src/operator/)                    │
│  gpu_physical_{filter,hash_join,order,...}.cpp           │
│  [Changed: calls rasterdf instead of cuDF]              │
├──────────────────────┬──────────────────────────────────┤
│  rasterdf Wrappers   │  rasterdf Dispatch Layer         │
│  src/gpu/rasterdf/   │  src/gpu/operator/               │
│  rdf_join.cpp        │  hash_join_inner.cpp             │
│  rdf_aggregate.cpp   │  materialize.cpp                 │
│  rdf_groupby.cpp     │  comparison_expression.cpp       │
│  rdf_orderby.cpp     │  nested_loop_join.cpp            │
│  rdf_utils.cpp       │  (dispatch Vulkan shaders)       │
├──────────────────────┴──────────────────────────────────┤
│        GPU Memory (gpu_buffer_manager.cpp)              │
│  rasterdf::memory_manager (VMA pool resources)          │
├─────────────────────────────────────────────────────────┤
│        GPU Columns (gpu_columns.cpp)                    │
│  GPUColumn, DataWrapper, convertToRasterdfColumn()      │
│  rasterdf::bitmask_type validity masks                  │
├─────────────────────────────────────────────────────────┤
│             rasterdf Library (linked)                   │
│  context, dispatcher, column, table, device_buffer      │
│  memory_resource, Vulkan compute shaders (.spv)         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 What Gets Reused (Minimal Changes)

These layers need only **namespace/naming renaming** (sirius→rasterdb):

| Layer | Files | Change Type |
|---|---|---|
| Extension entry | `sirius_extension.cpp` | Rename to `rasterdb_extension.cpp`, change function names |
| Planner | `src/planner/*.cpp` | Rename `sirius_` prefix → `rasterdb_` |
| Engine | `sirius_engine.cpp` | Rename, same logic |
| Pipeline | `src/pipeline/*.cpp` | Rename, same logic |
| Sirius operators | `src/op/*.cpp` | Rename `sirius_` prefix → `rasterdb_` |
| Config | `sirius_config.cpp` | Rename to `rasterdb_config.cpp` |
| Interface | `sirius_interface.cpp` | Rename to `rasterdb_interface.cpp` |
| Fallback | `fallback.cpp` | Keep as-is |

### 3.3 What Gets Replaced (Major Changes)

| Sirius Component | RasterDB Replacement | Effort |
|---|---|---|
| `src/cuda/cudf/cudf_join.cu` | `src/gpu/rasterdf/rdf_join.cpp` | **High** — rewrite using rasterdf dispatcher hash join |
| `src/cuda/cudf/cudf_aggregate.cu` | `src/gpu/rasterdf/rdf_aggregate.cpp` | **High** — rewrite using rasterdf reduction dispatches |
| `src/cuda/cudf/cudf_groupby.cu` | `src/gpu/rasterdf/rdf_groupby.cpp` | **High** — rewrite using rasterdf hash aggregate dispatches |
| `src/cuda/cudf/cudf_orderby.cu` | `src/gpu/rasterdf/rdf_orderby.cpp` | **High** — rewrite using rasterdf radix sort dispatches |
| `src/cuda/cudf/cudf_duplicate_elimination.cu` | `src/gpu/rasterdf/rdf_dedup.cpp` | **Medium** — placeholder needed |
| `src/cuda/cudf/cudf_utils.cu` | `src/gpu/rasterdf/rdf_utils.cpp` | **Medium** |
| `src/cuda/operator/*.cu` (14 files) | `src/gpu/operator/*.cpp` (Vulkan dispatches) | **High** — each kernel becomes shader dispatch |
| `src/cuda/allocator.cu` | Removed — rasterdf handles allocation | **Low** |
| `gpu_buffer_manager.cpp` | Rewrite to wrap `rasterdf::memory_manager` | **High** |
| `gpu_columns.cpp` | Replace `convertToCudfColumn()` → `convertToRasterdfColumn()` | **High** |
| `src/cuda/expression_executor/*.cu` | `src/gpu/expression_executor/*.cpp` | **High** — placeholder for unsupported |

---

## 4. Detailed Migration Plan

### Phase 1: Foundation (CMake + Core Types + Memory)

#### 4.1 CMakeLists.txt Changes

**Remove:**
```cmake
project(sirius LANGUAGES CXX CUDA)          # Remove CUDA language
find_package(cudf REQUIRED CONFIG)           # Remove cuDF dependency
# All CUDA_* properties
# All .cu source files
# cuCascade subdirectory (optional — evaluate if needed)
# rmm::rmm, cudf::cudf link targets
# PkgConfig NUMA (evaluate if needed)
```

**Add:**
```cmake
project(rasterdb LANGUAGES CXX C)

# Vulkan
find_package(Vulkan REQUIRED)

# rasterdf library (as subdirectory or pre-built)
# Option A: add_subdirectory for rasterdf
set(RASTERDF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../rasterdf")
add_subdirectory(${RASTERDF_ROOT} "${CMAKE_BINARY_DIR}/rasterdf" EXCLUDE_FROM_ALL)
# Option B: find_package if rasterdf installs a config
# find_package(rasterdf REQUIRED CONFIG)

# Volk (Vulkan meta-loader) — pulled in by rasterdf
# spdlog — keep if still used for logging, or switch to rasterdf::Logger

# Link targets become:
target_link_libraries(${_target} rasterdf spdlog::spdlog)
# Remove: cudf::cudf rmm::rmm cuCascade::cucascade
```

**Source file changes:**
```cmake
# Replace CUDA_SOURCES with:
set(GPU_SOURCES
    src/gpu/rasterdf/rdf_join.cpp
    src/gpu/rasterdf/rdf_aggregate.cpp
    src/gpu/rasterdf/rdf_groupby.cpp
    src/gpu/rasterdf/rdf_orderby.cpp
    src/gpu/rasterdf/rdf_dedup.cpp
    src/gpu/rasterdf/rdf_utils.cpp
    src/gpu/operator/hash_join_inner.cpp
    src/gpu/operator/hash_join_right.cpp
    src/gpu/operator/hash_join_single.cpp
    src/gpu/operator/materialize.cpp
    src/gpu/operator/nested_loop_join.cpp
    src/gpu/operator/comparison_expression.cpp
    src/gpu/operator/arbitrary_expression.cpp
    src/gpu/operator/strings_matching.cpp      # placeholder
    src/gpu/operator/substring.cpp             # placeholder
    src/gpu/expression_executor/gpu_dispatch_materialize.cpp
    src/gpu/expression_executor/gpu_dispatch_select.cpp
    src/gpu/expression_executor/gpu_dispatch_string.cpp  # placeholder
    src/gpu/utils.cpp
    src/gpu/print.cpp)

# Remove CUDA-specific properties:
# No CUDA_STANDARD, CUDA_SEPARABLE_COMPILATION, CUDA_RESOLVE_DEVICE_SYMBOLS
# No set_source_files_properties(... SKIP_PRECOMPILE_HEADERS)

# Add C++20 only:
set_target_properties(${_target} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)
```

#### 4.2 Core Types Bridge (`src/include/rasterdf/rdf_utils.hpp`)

This replaces `src/include/cudf/cudf_utils.hpp`:

```cpp
#pragma once

#include <rasterdf/core/types.hpp>
#include <rasterdf/core/column.hpp>
#include <rasterdf/core/column_view.hpp>
#include <rasterdf/core/table.hpp>
#include <rasterdf/core/table_view.hpp>
#include <rasterdf/core/scalar.hpp>
#include <rasterdf/memory/memory_manager.hpp>
#include <rasterdf/execution/dispatcher.hpp>

#include <duckdb/common/exception.hpp>
#include <duckdb/common/types.hpp>

namespace duckdb {

// Null mask type alias (mirrors cudf::bitmask_type = uint32_t)
using rdf_bitmask_type = uint32_t;

enum class rdf_mask_state { ALL_VALID, ALL_NULL, UNINITIALIZED };

// Type conversion: DuckDB LogicalType → rasterdf data_type
inline rasterdf::data_type GetRasterdfType(const LogicalType& logical_type) {
    switch (logical_type.id()) {
        case LogicalTypeId::TINYINT:     return {rasterdf::type_id::INT8};
        case LogicalTypeId::SMALLINT:    return {rasterdf::type_id::INT16};
        case LogicalTypeId::INTEGER:     return {rasterdf::type_id::INT32};
        case LogicalTypeId::BIGINT:      return {rasterdf::type_id::INT64};
        case LogicalTypeId::HUGEINT:     return {rasterdf::type_id::INT64}; // lossy
        case LogicalTypeId::FLOAT:       return {rasterdf::type_id::FLOAT32};
        case LogicalTypeId::DOUBLE:      return {rasterdf::type_id::FLOAT64};
        case LogicalTypeId::BOOLEAN:     return {rasterdf::type_id::BOOL8};
        case LogicalTypeId::DATE:        return {rasterdf::type_id::TIMESTAMP_DAYS};
        case LogicalTypeId::TIMESTAMP:   return {rasterdf::type_id::TIMESTAMP_MICROSECONDS};
        case LogicalTypeId::VARCHAR:     return {rasterdf::type_id::STRING};
        // DECIMAL, TIMESTAMP_SEC/MS/NS: placeholders
        default:
            throw InvalidInputException(
                "GetRasterdfType: Unsupported type: %d",
                static_cast<int>(logical_type.id()));
    }
}

} // namespace duckdb
```

#### 4.3 GPU Buffer Manager Rewrite

Replace RMM-based `GPUBufferManager` with rasterdf-based version:

**Key changes:**
- `rmm::mr::cuda_memory_resource* cuda_mr` → `rasterdf::memory_manager* mem_mgr`
- `rmm::mr::pool_memory_resource* mr` → `rasterdf::memory_resource* mr` (from `mem_mgr->workspace_resource()`)
- `customCudaMalloc<T>(size, gpu, caching)` → allocate via `rasterdf::device_buffer`
- `customCudaHostAlloc<T>(size)` → host-visible VMA allocation
- `rmm_stored_buffers` → `rdf_stored_buffers` (vector of `rasterdf::device_buffer`)
- `cudaMemcpy*` → `device_buffer::copy_from_host()` / `copy_to_host()`
- `cudaMemset` → Vulkan `vkCmdFillBuffer` or host-mapped memset

#### 4.4 GPU Columns Rewrite

Replace `cudf::column_view convertToCudfColumn()` with:

```cpp
rasterdf::column_view convertToRasterdfColumn() {
    // Build a rasterdf::column_view from GPUColumn's data
    auto rdf_type = /* map GPUColumnTypeId → rasterdf::type_id */;
    return rasterdf::column_view(
        rdf_type,
        static_cast<rasterdf::size_type>(column_length),
        data_wrapper.data_address,   // VkDeviceAddress
        data_wrapper.mask_address,   // VkDeviceAddress (null mask)
        null_count,
        0 /* offset */);
}
```

**Replace `cudf::bitmask_type*` throughout:**
- `cudf::bitmask_type*` → `rdf_bitmask_type*` (uint32_t*, allocated as rasterdf device_buffer)
- `createNullMask(size, state)` → rasterdf equivalent (allocate buffer, fill with 0xFF or 0x00)
- `cudf::mask_state::ALL_VALID` → `rdf_mask_state::ALL_VALID`

---

### Phase 2: Operator Rewrites (cuDF → rasterdf)

#### 4.5 Join (`cudf_join.cu` → `rdf_join.cpp`)

**Sirius uses:** `cudf::hash_join`, `cudf::distinct_hash_join`, `cudf::conditional_inner_join`, `cudf::mixed_inner_join`

**rasterdf has:** `dispatch_hash_join_build_count`, `dispatch_hash_join_build_insert`, `dispatch_hash_join_probe_count`, `dispatch_hash_join_probe_write`

**Migration strategy:**
```
cudf::hash_join(build_table).inner_join(probe_table)
    ↓ becomes ↓
1. dispatch_hash_join_build_count(build_keys, hash_table, hash_counts)
2. prefix_scan on hash_counts → hash_offsets
3. dispatch_hash_join_build_insert(build_keys, hash_table, hash_offsets, right_indices)
4. dispatch_hash_join_probe_count(probe_keys, hash_table, hash_counts, left_match_counts)
5. prefix_scan on left_match_counts → left_match_offsets
6. dispatch_hash_join_probe_write(probe_keys, hash_table, ..., out_left, out_right)
```

**Placeholders needed:**
- `cudf::conditional_inner_join` → **PLACEHOLDER** (rasterdf lacks conditional join)
- `cudf::mixed_inner_join` → **PLACEHOLDER** (rasterdf lacks mixed join)
- Left join, right join → **PLACEHOLDER** (rasterdf only has inner join primitives)
- `cudf::distinct_hash_join` → **PLACEHOLDER** (use regular hash join)
- Decimal type casting in join keys → **PLACEHOLDER**

#### 4.6 Aggregate (`cudf_aggregate.cu` → `rdf_aggregate.cpp`)

**Sirius uses:** `cudf::reduce(column, aggregation, output_type)`

**rasterdf has:** `dispatch_sum`, `dispatch_max`, `dispatch_min`, `dispatch_sum_float`, etc.

**Migration strategy:**
```
cudf::reduce(column, make_sum_aggregation(), type)
    ↓ becomes ↓
1. Allocate output device_buffer (1 element)
2. dispatch_sum({input_addr, output_addr, size, pad})
3. Read back result
```

**Placeholders needed:**
- `MEAN` aggregation → compute SUM + COUNT, divide (rasterdf has no single-dispatch mean reduction)
- `NUNIQUE` (COUNT_DISTINCT) → **PLACEHOLDER**
- `FIRST` → simple gather of index 0 (can implement with `dispatch_gather_indices`)
- Scalar return (`setFromCudfScalar`) → rewrite for rasterdf scalar

#### 4.7 GroupBy (`cudf_groupby.cu` → `rdf_groupby.cpp`)

**Sirius uses:** `cudf::groupby::groupby(keys).aggregate(requests)`

**rasterdf has:** Two paths:
1. Sort-based: `dispatch_groupby_find_boundaries`, `dispatch_groupby_segment_{sum,count,min,max}`, `dispatch_groupby_extract_keys`
2. Hash-based: `dispatch_hash_aggregate_{sum,count,min,max,mean}`, `dispatch_hash_extract_results`

**Migration strategy:**
- For simple single-key groupby: use hash-based path
- For multi-key groupby: sort keys first (radix sort), then sort-based groupby
- `combineColumns` (merge partial aggregates) → Vulkan buffer copy + dispatch

**Placeholders needed:**
- Multi-key hash groupby → **PLACEHOLDER** (rasterdf hash groupby is single-key int32 only)
- String key groupby → **PLACEHOLDER**
- Decimal aggregation → **PLACEHOLDER**
- `cudf::distinct` for deduplication within groupby → **PLACEHOLDER**

#### 4.8 OrderBy (`cudf_orderby.cu` → `rdf_orderby.cpp`)

**Sirius uses:** `cudf::sorted_order`, `cudf::sort`, CUB radix sort

**rasterdf has:** Full radix sort pipeline:
`dispatch_radix_init_indices` → `dispatch_radix_histogram` → `dispatch_radix_scan_reduce` → `dispatch_radix_scan_global` → `dispatch_radix_scatter` → `dispatch_radix_compute_offsets` (+ `dispatch_radix_sort_batched` for all passes)

Also: `dispatch_float_to_sortable`, `dispatch_sortable_to_float`, `dispatch_gather_indices`

**Migration strategy:**
- Single numeric column sort: direct radix sort
- Multi-column sort: radix sort on each key (least significant first)
- Float columns: float_to_sortable → sort → sortable_to_float
- Descending: bitwise complement before sort

**Placeholders needed:**
- String column sort → **PLACEHOLDER**
- Decimal column sort → **PLACEHOLDER**
- TopN optimization (partial sort) → **PLACEHOLDER** (use full sort + take first N)

#### 4.9 Custom Kernels (`src/cuda/operator/` → `src/gpu/operator/`)

| CUDA Kernel | rasterdf Replacement | Status |
|---|---|---|
| `hash_join_inner.cu` | Use rasterdf dispatcher join pipeline | ✅ Available |
| `hash_join_right.cu` | **PLACEHOLDER** | ❌ Not in rasterdf |
| `hash_join_single.cu` | Use rasterdf join with unique build keys | ⚠️ Partial |
| `materialize.cu` | `dispatch_gather_indices` | ✅ Available |
| `nested_loop_join.cu` | **PLACEHOLDER** | ❌ Not in rasterdf |
| `comparison_expression.cu` | `dispatch_compare` / `dispatch_compare_columns` | ✅ Available |
| `arbitrary_expression.cu` | `dispatch_binary_op` / `dispatch_unary_op` | ✅ Available |
| `strings_matching.cu` | **PLACEHOLDER** | ❌ No string ops in rasterdf |
| `substring.cu` | **PLACEHOLDER** | ❌ No string ops in rasterdf |
| `strlen_from_offsets.cu` | **PLACEHOLDER** | ❌ No string ops in rasterdf |
| `empty_str_check.cu` | **PLACEHOLDER** | ❌ No string ops in rasterdf |
| `gpu_dispatch_materialize.cu` | `dispatch_gather_indices` + `dispatch_gather` | ✅ Available |
| `gpu_dispatch_select.cu` | `dispatch_filter_count` + `dispatch_filter_write` | ✅ Available |
| `gpu_dispatch_string.cu` | **PLACEHOLDER** | ❌ No string ops |

---

### Phase 3: Expression Executor

The expression executor (`src/expression_executor/`) translates DuckDB expressions into GPU operations.

**Changes needed:**
- `gpu_execute_comparison.cpp` → use `dispatch_compare` / `dispatch_compare_columns`
- `gpu_execute_cast.cpp` → **PLACEHOLDER** (rasterdf lacks `cast`)
- `gpu_execute_function.cpp` → map to `dispatch_binary_op` / `dispatch_unary_op`
- `gpu_execute_operator.cpp` → map to `dispatch_binary_op`
- `gpu_execute_constant.cpp` → allocate scalar on GPU, use `dispatch_transform` (add 0)
- `gpu_execute_reference.cpp` → pass-through column reference
- `gpu_execute_between.cpp` → two compares + AND
- `gpu_execute_case.cpp` → **PLACEHOLDER** (complex control flow)
- `gpu_execute_conjunction.cpp` → element-wise AND/OR via binary_op

---

## 5. rasterdf Library Reorganization

### 5.1 Current rasterdf Structure
```
rasterdf/
├── include/rasterdf/
│   ├── api/librasterdf_c.h          # C API
│   ├── core/{column,table,context,types,device_buffer,...}.hpp
│   ├── execution/{dispatcher,constants}.hpp
│   ├── memory/{memory_resource,pool_memory_resource,...}.hpp
│   ├── io/binary_reader.hpp
│   └── utils/logger.hpp
├── src/
│   ├── core/      # column, table, context implementations
│   ├── execution/ # dispatcher implementation
│   ├── memory/    # memory resource implementations
│   ├── io/        # binary reader
│   └── utils/     # logger
├── shaders/       # GLSL compute shaders (.comp)
└── cudf/          # Reference cuDF source (read-only)
```

### 5.2 Target rasterdf Structure (Mirror cuDF)

To make rasterdf a drop-in conceptual replacement for cuDF, reorganize headers to match
cuDF's include structure. This enables rasterdb to `#include <rasterdf/...>` in a pattern
similar to `#include <cudf/...>`.

```
rasterdf/
├── include/rasterdf/
│   ├── types.hpp                      # ← core/types.hpp (promoted)
│   │
│   ├── column/
│   │   ├── column.hpp                 # ← core/column.hpp
│   │   ├── column_view.hpp            # ← core/column_view.hpp
│   │   ├── column_device_view.hpp     # ← core/column_device_view.hpp
│   │   └── column_factories.hpp       # NEW: make_empty_column(), etc.
│   │
│   ├── table/
│   │   ├── table.hpp                  # ← core/table.hpp
│   │   ├── table_view.hpp             # ← core/table_view.hpp
│   │   └── table_device_view.hpp      # ← core/table_device_view.hpp
│   │
│   ├── scalar/
│   │   ├── scalar.hpp                 # ← core/scalar.hpp (fix template issues)
│   │   └── scalar_factories.hpp       # NEW: make_*_scalar()
│   │
│   ├── aggregation.hpp                # NEW: aggregation kind enum + factory
│   ├── reduction.hpp                  # NEW: reduce(column_view, agg, type)
│   ├── groupby.hpp                    # NEW: groupby class
│   ├── sorting.hpp                    # NEW: sorted_order(), sort()
│   ├── copying.hpp                    # NEW: gather(), scatter(), empty_like()
│   ├── unary.hpp                      # NEW: unary operations
│   ├── binaryop.hpp                   # NEW: binary operations
│   ├── stream_compaction.hpp          # NEW: distinct(), unique()
│   ├── join/
│   │   ├── hash_join.hpp              # NEW: hash_join class
│   │   ├── join.hpp                   # NEW: inner_join, left_join stubs
│   │   ├── conditional_join.hpp       # NEW: placeholder
│   │   └── mixed_join.hpp             # NEW: placeholder
│   │
│   ├── ast/
│   │   └── expressions.hpp            # NEW: placeholder for AST expressions
│   │
│   ├── detail/                        # Internal implementation details
│   │   └── utilities.hpp
│   │
│   ├── mr/                            # ← memory/ (renamed to match rmm)
│   │   ├── device_memory_resource.hpp # ← memory/memory_resource.hpp
│   │   ├── pool_memory_resource.hpp   # ← memory/pool_memory_resource.hpp
│   │   └── vulkan_memory_resource.hpp # ← memory/vulkan_memory_resource.hpp
│   │
│   ├── utilities/
│   │   ├── error.hpp                  # NEW: RDF_EXPECTS, RDF_CHECK macros
│   │   └── type_dispatcher.hpp        # NEW: type dispatch utility
│   │
│   ├── io/
│   │   └── binary_reader.hpp          # ← io/binary_reader.hpp
│   │
│   ├── context.hpp                    # ← core/context.hpp (promoted)
│   ├── device_buffer.hpp              # ← core/device_buffer.hpp (promoted)
│   ├── memory_manager.hpp             # ← memory/memory_manager.hpp (promoted)
│   │
│   ├── vulkan/                        # Vulkan-specific internals
│   │   ├── VkEngine.hpp               # ← core/VkEngine.hpp
│   │   ├── VulkanDevice.hpp           # ← core/VulkanDevice.hpp
│   │   └── ComputePipelineProperties.hpp
│   │
│   └── execution/
│       ├── dispatcher.hpp             # ← execution/dispatcher.hpp
│       └── constants.hpp              # ← execution/constants.hpp
│
├── src/  (mirrors include/ structure)
│   ├── column/column.cpp, column_view.cpp, column_factories.cpp
│   ├── table/table.cpp, table_view.cpp
│   ├── scalar/scalar.cpp, scalar_factories.cpp
│   ├── reduction/reduction.cpp        # implements reduce()
│   ├── groupby/groupby.cpp            # implements groupby
│   ├── sorting/sorting.cpp            # implements sort, sorted_order
│   ├── join/hash_join.cpp             # implements hash_join class
│   ├── copying/copying.cpp            # implements gather, scatter
│   ├── binaryop/binaryop.cpp          # implements binary operations
│   ├── unary/unary.cpp                # implements unary operations
│   ├── stream_compaction/distinct.cpp  # implements distinct
│   ├── mr/...
│   ├── vulkan/...
│   ├── execution/dispatcher.cpp
│   ├── io/binary_reader.cpp
│   └── utils/logger.cpp
│
├── shaders/  (unchanged)
└── python/   (unchanged)
```

### 5.3 New High-Level APIs to Add to rasterdf

These are the **critical APIs** that rasterdb needs from rasterdf, matching cuDF's interface:

#### 5.3.1 `rasterdf/reduction.hpp`
```cpp
namespace rasterdf {

enum class reduce_op { SUM, MIN, MAX, MEAN, COUNT, NUNIQUE };

// Reduce a column to a scalar
std::unique_ptr<scalar> reduce(
    column_view const& col,
    reduce_op op,
    data_type output_type);

} // namespace rasterdf
```
**Implementation:** Internally dispatches `dispatch_sum`, `dispatch_min`, `dispatch_max`, etc.

#### 5.3.2 `rasterdf/sorting.hpp`
```cpp
namespace rasterdf {

enum class order { ASCENDING, DESCENDING };
enum class null_order { BEFORE, AFTER };

// Returns column of sorted indices
std::unique_ptr<column> sorted_order(
    table_view const& keys,
    std::vector<order> const& column_order);

// Returns sorted table
std::unique_ptr<table> sort(
    table_view const& input,
    std::vector<order> const& column_order);

} // namespace rasterdf
```
**Implementation:** Internally uses `dispatch_radix_sort_batched` + `dispatch_gather_indices`.

#### 5.3.3 `rasterdf/join/hash_join.hpp`
```cpp
namespace rasterdf {

class hash_join {
public:
    hash_join(table_view const& build_keys);

    // Returns (left_indices, right_indices)
    std::pair<std::unique_ptr<column>, std::unique_ptr<column>>
    inner_join(table_view const& probe_keys) const;

    // PLACEHOLDER: left_join, full_join
    std::pair<std::unique_ptr<column>, std::unique_ptr<column>>
    left_join(table_view const& probe_keys) const;  // placeholder

private:
    // Internal hash table state (Vulkan buffers)
};

} // namespace rasterdf
```
**Implementation:** Uses `dispatch_hash_join_build_count/insert` + `dispatch_hash_join_probe_count/write`.

#### 5.3.4 `rasterdf/groupby.hpp`
```cpp
namespace rasterdf {

enum class aggregation_kind { SUM, MIN, MAX, COUNT, MEAN };

struct aggregation_request {
    column_view values;
    std::vector<aggregation_kind> aggregations;
};

class groupby {
public:
    groupby(table_view const& keys);

    struct result {
        std::unique_ptr<table> keys;
        std::vector<std::unique_ptr<column>> values;
    };

    result aggregate(std::vector<aggregation_request> const& requests);

private:
    table_view _keys;
};

} // namespace rasterdf
```
**Implementation:** Uses hash-based groupby dispatches for single int32 keys, sort-based for others.

#### 5.3.5 `rasterdf/copying.hpp`
```cpp
namespace rasterdf {

// Gather rows by index
std::unique_ptr<table> gather(
    table_view const& source,
    column_view const& gather_map);

// Create empty table with same schema
std::unique_ptr<table> empty_like(table_view const& input);

} // namespace rasterdf
```

#### 5.3.6 `rasterdf/stream_compaction.hpp`
```cpp
namespace rasterdf {

// Remove duplicate rows — PLACEHOLDER
std::unique_ptr<table> distinct(
    table_view const& input,
    std::vector<size_type> const& keys);

} // namespace rasterdf
```

#### 5.3.7 Bitmask Infrastructure
```cpp
// In types.hpp or a new bitmask.hpp:
namespace rasterdf {

using bitmask_type = uint32_t;
enum class mask_state { ALL_VALID, ALL_NULL, UNINITIALIZED };

// Allocate a validity bitmask buffer
device_buffer create_null_mask(
    size_type size,
    mask_state state,
    memory_resource* mr);

size_type count_set_bits(bitmask_type const* bitmask, size_type start, size_type stop);

} // namespace rasterdf
```

---

## 6. Feature Gap Analysis

### 6.1 rasterdf Features Available (Can Map Directly)

| cuDF Operation | rasterdf Equivalent | Notes |
|---|---|---|
| `cudf::reduce(SUM)` | `dispatch_sum` / `dispatch_sum_float` | int32 + float32 only |
| `cudf::reduce(MIN)` | `dispatch_min` / `dispatch_min_float` | int32 + float32 only |
| `cudf::reduce(MAX)` | `dispatch_max` / `dispatch_max_float` | int32 + float32 only |
| `cudf::hash_join::inner_join` | hash_join build+probe pipeline | int32 keys only |
| `cudf::sort` | radix sort pipeline | int32 + float32 |
| `cudf::sorted_order` | radix sort with index tracking | int32 + float32 |
| Filter (comparison) | `dispatch_compare` + `dispatch_filter_count/write` | int32 + float32 |
| Materialize (gather) | `dispatch_gather_indices` | ✅ |
| Binary ops | `dispatch_binary_op` | add/sub/mul/div/mod/pow/floordiv |
| Unary ops | `dispatch_unary_op` | negate/abs/sign/square/bitwise_not |
| Groupby (hash) | hash aggregate sum/count/min/max/mean + extract | int32 keys+values |
| Groupby (sort) | sort + find_boundaries + segment ops | int32 |
| Prefix scan | 3-pass scan (local+global+add) | ✅ |

### 6.2 rasterdf Features Missing (Need Placeholders)

| Feature | Priority | Placeholder Strategy |
|---|---|---|
| **INT64/INT16/INT8 support in shaders** | HIGH | Fall back to CPU for non-int32/float32 |
| **FLOAT64 support in shaders** | HIGH | Fall back to CPU |
| **VARCHAR/STRING operations** | HIGH | Fall back to CPU |
| **DECIMAL types** | MEDIUM | Fall back to CPU |
| **NULL mask propagation in shaders** | HIGH | Ignore nulls initially, add mask-aware shaders later |
| **Left/Right/Outer joins** | HIGH | Fall back to CPU |
| **Conditional/Mixed joins** | MEDIUM | Fall back to CPU |
| **Type casting** | MEDIUM | Fall back to CPU |
| **NUNIQUE / COUNT DISTINCT** | LOW | Fall back to CPU |
| **Multi-key hash join** | HIGH | Composite key hashing or fall back |
| **Multi-column sort** | MEDIUM | Sequential radix sort (LSB first) |
| **Distinct (deduplication)** | MEDIUM | Sort + boundary detection |
| **Nested Loop Join** | LOW | Fall back to CPU |

### 6.3 Recommended Type Support Expansion in rasterdf

To cover the most common SQL types, rasterdf shaders should be templated/parameterized for:

1. **INT64** (BIGINT) — most critical gap; many SQL columns are BIGINT
2. **FLOAT64** (DOUBLE) — needed for AVG and financial computations
3. **INT16** (SMALLINT) — used in TPC-H
4. **BOOL8** — needed for filter masks

The Vulkan compute shaders already use push constants with a `type_id` field. Extend the existing
shaders to handle 64-bit types by reading 2×uint32 and operating on uint64/int64.

---

## 7. Directory Structure for rasterdb (After Migration)

```
rasterdb/
├── CMakeLists.txt                     # REWRITTEN: Vulkan + rasterdf, no CUDA
├── ARCHITECTURE.md                    # This document
├── duckdb/                            # DuckDB submodule (unchanged)
├── rasterdf/                          # rasterdf as git submodule
│   └── (rasterdf library source)
├── src/
│   ├── include/
│   │   ├── rasterdf/
│   │   │   └── rdf_utils.hpp          # NEW: replaces cudf/cudf_utils.hpp
│   │   ├── config.hpp                 # Renamed from sirius
│   │   ├── gpu_buffer_manager.hpp     # REWRITTEN: rasterdf memory
│   │   ├── gpu_columns.hpp            # REWRITTEN: rasterdf types
│   │   ├── gpu_context.hpp            # REWRITTEN: rasterdf context
│   │   ├── operator/                  # Renamed sirius→rasterdb
│   │   ├── op/                        # Renamed sirius→rasterdb
│   │   ├── pipeline/                  # Renamed sirius→rasterdb
│   │   ├── planner/                   # Renamed sirius→rasterdb
│   │   ├── expression_executor/       # Modified for rasterdf
│   │   └── ...
│   ├── gpu/                           # NEW: replaces src/cuda/
│   │   ├── rasterdf/                  # replaces src/cuda/cudf/
│   │   │   ├── rdf_join.cpp
│   │   │   ├── rdf_aggregate.cpp
│   │   │   ├── rdf_groupby.cpp
│   │   │   ├── rdf_orderby.cpp
│   │   │   ├── rdf_dedup.cpp
│   │   │   └── rdf_utils.cpp
│   │   ├── operator/                  # replaces src/cuda/operator/
│   │   │   ├── hash_join_inner.cpp
│   │   │   ├── materialize.cpp
│   │   │   ├── comparison_expression.cpp
│   │   │   └── ...
│   │   ├── expression_executor/       # replaces src/cuda/expression_executor/
│   │   │   ├── gpu_dispatch_materialize.cpp
│   │   │   ├── gpu_dispatch_select.cpp
│   │   │   └── gpu_dispatch_string.cpp  # placeholder
│   │   ├── utils.cpp
│   │   └── print.cpp
│   ├── planner/                       # Renamed sirius→rasterdb
│   ├── op/                            # Renamed sirius→rasterdb
│   ├── operator/                      # Modified for rasterdf calls
│   ├── pipeline/                      # Renamed sirius→rasterdb
│   ├── expression_executor/           # Modified for rasterdf
│   ├── rasterdb_engine.cpp            # Renamed from sirius_engine.cpp
│   ├── rasterdb_extension.cpp         # Renamed from sirius_extension.cpp
│   ├── rasterdb_config.cpp            # Renamed from sirius_config.cpp
│   ├── rasterdb_context.cpp           # Renamed from sirius_context.cpp
│   ├── rasterdb_interface.cpp         # Renamed from sirius_interface.cpp
│   ├── gpu_buffer_manager.cpp         # REWRITTEN
│   ├── gpu_columns.cpp                # REWRITTEN
│   ├── gpu_context.cpp                # REWRITTEN
│   └── fallback.cpp                   # Keep
├── test/                              # Adapted tests
└── extension_config.cmake             # Updated for rasterdb
```

---

## 8. Step-by-Step Implementation Roadmap

### Sprint 1: Skeleton (Week 1-2)
1. **Rename** all `sirius_*` files to `rasterdb_*` in rasterdb fork
2. **Rewrite CMakeLists.txt** — remove CUDA, add Vulkan/rasterdf
3. **Add rasterdf as git submodule** in rasterdb
4. **Create `src/include/rasterdf/rdf_utils.hpp`** — type bridge
5. **Stub out `src/gpu/`** directory with placeholder `.cpp` files
6. **Get it compiling** (all GPU calls are no-ops / throw "not implemented")

### Sprint 2: Memory + Columns (Week 2-3)
7. **Rewrite `gpu_buffer_manager.cpp`** to use `rasterdf::memory_manager`
8. **Rewrite `gpu_columns.cpp`** — replace cudf types with rasterdf types
9. **Implement `convertToRasterdfColumn()`**
10. **Implement bitmask infrastructure** in rasterdf

### Sprint 3: Core Operators (Week 3-5)
11. **Implement `rdf_aggregate.cpp`** — SUM, MIN, MAX for int32/float32
12. **Implement `rdf_orderby.cpp`** — radix sort for int32/float32
13. **Implement `rdf_join.cpp`** — hash inner join for int32 keys
14. **Implement `rdf_groupby.cpp`** — hash aggregate for int32
15. **Implement materialize/gather** via `dispatch_gather_indices`
16. **Implement filter** via `dispatch_compare` + `dispatch_filter_count/write`

### Sprint 4: Expression Executor (Week 5-6)
17. **Port expression executor** — comparison, binary ops, unary ops
18. **Add placeholders** for cast, string, case expressions

### Sprint 5: Integration + Testing (Week 6-8)
19. **End-to-end test**: simple SELECT with WHERE, GROUP BY, ORDER BY
20. **TPC-H Q1** (aggregation-heavy) as first target
21. **TPC-H Q6** (filter-heavy) as second target
22. **Fallback testing** — ensure unsupported ops fall back to CPU

### Sprint 6: Type Expansion (Week 8+)
23. **Add INT64 shaders** to rasterdf
24. **Add FLOAT64 shaders** to rasterdf
25. **Expand join to multi-key** 
26. **Add left/outer join** shaders

---

## 9. Key Design Decisions

### 9.1 rasterdf Context Lifetime
- **Sirius**: `GPUBufferManager` is a singleton; RMM pool is created once
- **RasterDB**: `rasterdf::context` + `rasterdf::memory_manager` should be created once in `rasterdb_extension.cpp` load and stored in `RasterDBContext` (equivalent to `SiriusContext`)
- The `rasterdf::execution::dispatcher` should also be long-lived (pipeline creation is expensive)

### 9.2 Vulkan Command Buffer Strategy
- **Sirius**: CUDA launches are fire-and-forget on default stream
- **RasterDB**: rasterdf dispatcher submits commands synchronously (submit + waitIdle)
- For pipeline of operators: batch dispatches into single command buffer where possible
- Use `dispatch_radix_sort_batched` pattern for multi-pass operations

### 9.3 Memory Transfer Strategy
- **Sirius**: `cudaMemcpy` + `cudaMemcpyAsync` with RMM device buffers
- **RasterDB**: `device_buffer::copy_from_host()` / `copy_to_host()` (staging buffer + vkCmdCopyBuffer)
- For ReBAR/resizable BAR GPUs: direct mapped access via VMA
- Host pinned memory → VMA host-visible, host-coherent allocations

### 9.4 Null Mask Handling
- **Sirius**: `cudf::bitmask_type*` (uint32_t*), packed 1-bit-per-row, CUDA kernels propagate masks
- **RasterDB Phase 1**: Ignore null masks (treat all as ALL_VALID). Most TPC-H data has no NULLs.
- **RasterDB Phase 2**: Add null-mask-aware Vulkan shaders

### 9.5 cuCascade Replacement
- **Sirius**: cuCascade provides tiered memory (GPU → CPU → disk) and data repositories
- **RasterDB Phase 1**: Use rasterdf `memory_manager` with data_pool + workspace_pool. No disk tier.
- **RasterDB Phase 2**: Implement tiered memory if needed for large datasets

---

## 10. Placeholder Template

For every unsupported operation, use this pattern:

```cpp
// In rasterdb source files:
void rdf_unsupported_operation(const std::string& op_name) {
    throw duckdb::NotImplementedException(
        "RasterDB: Operation '%s' not yet implemented in rasterdf. "
        "Falling back to DuckDB CPU execution.", op_name.c_str());
}
```

The existing fallback mechanism in `fallback.cpp` will catch this and route to DuckDB CPU execution.

---

## 11. Appendix: File-by-File Rename Map

| Sirius File | RasterDB File | Change Type |
|---|---|---|
| `sirius_extension.cpp` | `rasterdb_extension.cpp` | Rename + modify |
| `sirius_engine.cpp` | `rasterdb_engine.cpp` | Rename |
| `sirius_config.cpp` | `rasterdb_config.cpp` | Rename |
| `sirius_context.cpp` | `rasterdb_context.cpp` | Rename + modify |
| `sirius_interface.cpp` | `rasterdb_interface.cpp` | Rename |
| `src/include/cudf/cudf_utils.hpp` | `src/include/rasterdf/rdf_utils.hpp` | Rewrite |
| `src/cuda/cudf/cudf_join.cu` | `src/gpu/rasterdf/rdf_join.cpp` | Rewrite |
| `src/cuda/cudf/cudf_aggregate.cu` | `src/gpu/rasterdf/rdf_aggregate.cpp` | Rewrite |
| `src/cuda/cudf/cudf_groupby.cu` | `src/gpu/rasterdf/rdf_groupby.cpp` | Rewrite |
| `src/cuda/cudf/cudf_orderby.cu` | `src/gpu/rasterdf/rdf_orderby.cpp` | Rewrite |
| `src/cuda/cudf/cudf_duplicate_elimination.cu` | `src/gpu/rasterdf/rdf_dedup.cpp` | Rewrite |
| `src/cuda/cudf/cudf_utils.cu` | `src/gpu/rasterdf/rdf_utils.cpp` | Rewrite |
| `src/cuda/allocator.cu` | Removed (rasterdf handles) | Delete |
| `src/cuda/communication.cu` | `src/gpu/communication.cpp` | Rewrite |
| `src/cuda/print.cu` | `src/gpu/print.cpp` | Rewrite |
| `src/cuda/utils.cu` | `src/gpu/utils.cpp` | Rewrite |
| `src/cuda/operator/*.cu` (14 files) | `src/gpu/operator/*.cpp` | Rewrite |
| `src/cuda/expression_executor/*.cu` (3) | `src/gpu/expression_executor/*.cpp` | Rewrite |
| `gpu_buffer_manager.cpp` | `gpu_buffer_manager.cpp` | Rewrite in-place |
| `gpu_columns.cpp` | `gpu_columns.cpp` | Rewrite in-place |
| `gpu_context.cpp` | `gpu_context.cpp` | Rewrite in-place |
| All `src/planner/sirius_*` | `src/planner/rasterdb_*` | Rename |
| All `src/op/sirius_*` | `src/op/rasterdb_*` | Rename |
| All `src/pipeline/sirius_*` | `src/pipeline/rasterdb_*` | Rename |
| `src/include/sirius_*.hpp` | `src/include/rasterdb_*.hpp` | Rename |
