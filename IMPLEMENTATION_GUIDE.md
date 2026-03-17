# RasterDB Implementation Guide

## Step-by-Step Instructions for Converting Sirius Fork to RasterDB

This guide provides concrete, file-level instructions for transforming the rasterdb
fork (currently a copy of sirius) into a Vulkan/rasterdf-based GPU SQL engine.

---

## Phase 0: Pre-Flight Checklist

Before starting, verify:
- [ ] rasterdb is a clean fork of sirius (all files identical except test additions)
- [ ] rasterdf builds and tests pass (`cd rasterdf && mkdir build && cd build && cmake .. && make`)
- [ ] Vulkan SDK is installed and `vulkaninfo` works
- [ ] DuckDB submodule in rasterdb is intact

---

## Phase 1: CMake Conversion

### 1.1 Replace CMakeLists.txt

The most critical change. Transform from CUDA+cuDF to CXX+Vulkan+rasterdf.

**Key changes to `rasterdb/CMakeLists.txt`:**

```cmake
# BEFORE (sirius):
project(sirius LANGUAGES CXX CUDA)
find_package(cudf REQUIRED CONFIG)
# CUDA architectures, CUDA properties, .cu files, etc.

# AFTER (rasterdb):
project(rasterdb LANGUAGES CXX C)
find_package(Vulkan REQUIRED)
find_package(spdlog REQUIRED CONFIG)
# Optionally keep: find_package(libconfig++ REQUIRED CONFIG)
# Optionally keep: find_package(absl REQUIRED CONFIG)

# rasterdf as subdirectory (adjust path as needed)
set(RASTERDF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RASTERDF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../rasterdf"
                 "${CMAKE_BINARY_DIR}/rasterdf" EXCLUDE_FROM_ALL)
```

**Remove entirely:**
- `CMAKE_CUDA_ARCHITECTURES`
- `ENABLE_STREAM_CHECK` option
- `cucascade` subdirectory (replace with rasterdf memory management)
- All `CUDA_*` target properties
- `SKIP_PRECOMPILE_HEADERS` for .cu files
- `PkgConfig NUMA` (unless host-side NUMA is still desired)

**Replace link targets:**
```cmake
# BEFORE:
target_link_libraries(${_target} cudf::cudf rmm::rmm spdlog::spdlog cuCascade::cucascade)

# AFTER:
target_link_libraries(${_target} rasterdf spdlog::spdlog Vulkan::Vulkan)
```

**Replace source file lists:**
- All `.cu` files in `CUDA_SOURCES` → `.cpp` files in `GPU_SOURCES`
- See ARCHITECTURE.md Section 4.1 for the full mapping

### 1.2 Add rasterdf as Git Submodule

```bash
cd rasterdb
git submodule add ../../rasterdf rasterdf_lib
# Or if rasterdf is at a known path:
# Just reference it via CMake add_subdirectory with an absolute/relative path
```

Alternative: symlink for development:
```bash
ln -s /home/akashmaji/Device/IMPORTANT/rasterdf rasterdb/rasterdf_lib
```

### 1.3 Update extension_config.cmake

```cmake
# Change extension name from sirius to rasterdb
duckdb_extension_load(rasterdb
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS)
```

---

## Phase 2: Namespace & File Renaming

### 2.1 Automated Rename Script

Create `scripts/rename_sirius_to_rasterdb.sh`:

```bash
#!/bin/bash
# Run from rasterdb root

# Rename files
find src/ -name 'sirius_*' | while read f; do
    new=$(echo "$f" | sed 's/sirius_/rasterdb_/g')
    git mv "$f" "$new"
done

find src/include/ -name 'sirius_*' | while read f; do
    new=$(echo "$f" | sed 's/sirius_/rasterdb_/g')
    git mv "$f" "$new"
done

# Replace in source content
find src/ -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) -exec \
    sed -i 's/sirius_/rasterdb_/g; s/Sirius/RasterDB/g; s/SIRIUS/RASTERDB/g' {} +

# Replace namespace
find src/ -type f \( -name '*.cpp' -o -name '*.hpp' \) -exec \
    sed -i 's/namespace sirius/namespace rasterdb/g' {} +
```

### 2.2 Files to Rename (Complete List)

**Headers (`src/include/`):**
| Old | New |
|---|---|
| `sirius_config.hpp` | `rasterdb_config.hpp` |
| `sirius_context.hpp` | `rasterdb_context.hpp` |
| `sirius_engine.hpp` | `rasterdb_engine.hpp` |
| `sirius_extension.hpp` | `rasterdb_extension.hpp` |
| `sirius_interface.hpp` | `rasterdb_interface.hpp` |
| `sirius_pipeline_hashmap.hpp` | `rasterdb_pipeline_hashmap.hpp` |

**Source files (`src/`):**
| Old | New |
|---|---|
| `sirius_config.cpp` | `rasterdb_config.cpp` |
| `sirius_context.cpp` | `rasterdb_context.cpp` |
| `sirius_engine.cpp` | `rasterdb_engine.cpp` |
| `sirius_extension.cpp` | `rasterdb_extension.cpp` |
| `sirius_interface.cpp` | `rasterdb_interface.cpp` |

**Pipeline files (`src/pipeline/`):**
| Old | New |
|---|---|
| `sirius_meta_pipeline.cpp` | `rasterdb_meta_pipeline.cpp` |
| `sirius_pipeline.cpp` | `rasterdb_pipeline.cpp` |

**Operator files (`src/op/sirius_physical_*`):**
All `sirius_physical_*` → `rasterdb_physical_*` (both `.cpp` and `.hpp`)

**Planner files (`src/planner/sirius_*`):**
All `sirius_plan_*` / `sirius_physical_plan_*` → `rasterdb_plan_*` / `rasterdb_physical_plan_*`

### 2.3 String Replacements in Content

| Pattern | Replacement | Scope |
|---|---|---|
| `sirius_extension` | `rasterdb_extension` | All source files |
| `sirius_loadable_extension` | `rasterdb_loadable_extension` | CMakeLists.txt |
| `sirius_unittest` | `rasterdb_unittest` | CMakeLists.txt, test files |
| `SIRIUS_LOG_DEBUG` | `RASTERDB_LOG_DEBUG` | All source files |
| `SIRIUS_LOG_*` | `RASTERDB_LOG_*` | All source files |
| `SIRIUS_DEFAULT_LOG_DIR` | `RASTERDB_DEFAULT_LOG_DIR` | CMakeLists.txt |
| `SiriusContext` | `RasterDBContext` | Context class references |
| `"sirius_state"` | `"rasterdb_state"` | Extension state key |
| `"sirius"` (extension name) | `"rasterdb"` | Extension registration |
| `gpu_processing` | `gpu_processing` | Keep (or rename to `rasterdb_processing`) |
| `gpu_execution` | `gpu_execution` | Keep (or rename to `rasterdb_execution`) |
| `gpu_buffer_init` | `gpu_buffer_init` | Keep |

---

## Phase 3: Replace CUDA/cuDF Core Files

### 3.1 Create `src/gpu/` Directory Structure

```bash
mkdir -p src/gpu/rasterdf
mkdir -p src/gpu/operator
mkdir -p src/gpu/expression_executor
```

### 3.2 Delete `src/cuda/` Directory

The entire `src/cuda/` directory contains CUDA-specific code. Replace with:

| Delete | Create |
|---|---|
| `src/cuda/cudf/cudf_join.cu` | `src/gpu/rasterdf/rdf_join.cpp` |
| `src/cuda/cudf/cudf_aggregate.cu` | `src/gpu/rasterdf/rdf_aggregate.cpp` |
| `src/cuda/cudf/cudf_groupby.cu` | `src/gpu/rasterdf/rdf_groupby.cpp` |
| `src/cuda/cudf/cudf_orderby.cu` | `src/gpu/rasterdf/rdf_orderby.cpp` |
| `src/cuda/cudf/cudf_duplicate_elimination.cu` | `src/gpu/rasterdf/rdf_dedup.cpp` |
| `src/cuda/cudf/cudf_utils.cu` | `src/gpu/rasterdf/rdf_utils.cpp` |
| `src/cuda/allocator.cu` | (removed — rasterdf handles) |
| `src/cuda/communication.cu` | `src/gpu/communication.cpp` |
| `src/cuda/print.cu` | `src/gpu/print.cpp` |
| `src/cuda/utils.cu` | `src/gpu/utils.cpp` |
| `src/cuda/operator/*.cu` | `src/gpu/operator/*.cpp` |
| `src/cuda/expression_executor/*.cu` | `src/gpu/expression_executor/*.cpp` |

### 3.3 Replace `src/include/cudf/cudf_utils.hpp`

Create `src/include/rasterdf/rdf_utils.hpp` — see ARCHITECTURE.md Section 4.2 for contents.

This is the **type bridge** between DuckDB types and rasterdf types, replacing the
cuDF type bridge in `cudf_utils.hpp`.

### 3.4 Rewrite `gpu_buffer_manager.hpp/cpp`

**Remove all references to:**
- `rmm::mr::cuda_memory_resource`
- `rmm::mr::pool_memory_resource`
- `rmm::device_buffer`
- `cudaMalloc`, `cudaFree`, `cudaMemcpy*`, `cudaMemset`
- `cudaHostAlloc`
- `rmm_stored_buffers`

**Replace with:**
- `rasterdf::memory_manager` — owns data_pool and workspace_pool
- `rasterdf::memory_resource*` — for allocations
- `rasterdf::device_buffer` — for stored buffers
- `device_buffer::copy_from_host()` / `copy_to_host()` — for transfers

**Key method changes:**

```cpp
// BEFORE:
template <typename T>
T* customCudaMalloc(size_t size, int gpu, bool caching);

// AFTER:
rasterdf::device_buffer allocate_buffer(size_t bytes);
// Returns a device_buffer with VkDeviceAddress accessible data
```

```cpp
// BEFORE:
template <typename T>
T* customCudaHostAlloc(size_t size);

// AFTER:
// Use rasterdf memory_manager->host_resource() for host-visible allocations
rasterdf::device_buffer allocate_host_buffer(size_t bytes);
// Uses VMA_MEMORY_USAGE_AUTO with HOST_ACCESS_SEQUENTIAL_WRITE
```

### 3.5 Rewrite `gpu_columns.hpp/cpp`

**Remove all references to:**
- `cudf::column_view`
- `cudf::bitmask_type`
- `cudf::mask_state`
- `cudf::scalar`
- `convertToCudfColumn()`
- `setFromCudfColumn()`, `setFromCudfScalar()`
- `convertSiriusOffsetToCudfOffset()`, etc.

**Replace with:**
- `rasterdf::column_view`
- `rdf_bitmask_type` (alias for uint32_t)
- `rdf_mask_state`
- `rasterdf::scalar`
- `convertToRasterdfColumnView()`
- `setFromRasterdfColumn()`, `setFromRasterdfScalar()`

**DataWrapper changes:**
```cpp
// BEFORE:
cudf::bitmask_type* validity_mask{nullptr};

// AFTER:
rdf_bitmask_type* validity_mask{nullptr};
// Or better: store as VkDeviceAddress for shader access
VkDeviceAddress validity_mask_addr{0};
```

### 3.6 Rewrite `gpu_context.hpp/cpp`

**Remove:**
- CUDA device management (`cudaSetDevice`, `cudaGetDeviceCount`)
- CUDA stream management

**Replace with:**
- `rasterdf::context` — Vulkan instance/device/queue
- `rasterdf::execution::dispatcher` — compute pipeline dispatcher
- `rasterdf::memory_manager` — memory pools

```cpp
class GPUContext {
    rasterdf::context rdf_ctx;
    rasterdf::memory_manager mem_mgr;
    rasterdf::execution::dispatcher dispatcher;
    // ...
};
```

---

## Phase 4: Implement GPU Wrapper Functions

### 4.1 `src/gpu/rasterdf/rdf_join.cpp` (Priority: HIGH)

This replaces `cudf_join.cu`. Use rasterdf's hash join dispatch pipeline.

**Function signatures to implement (same as cudf_join.cu):**
```cpp
void rdf_hash_inner_join(
    vector<shared_ptr<GPUColumn>>& probe_keys,
    vector<shared_ptr<GPUColumn>>& build_keys,
    int num_keys,
    uint64_t*& row_ids_left,
    uint64_t*& row_ids_right,
    uint64_t*& count,
    bool unique_build_keys);

// PLACEHOLDER:
void rdf_mixed_or_conditional_inner_join(...) {
    throw NotImplementedException("Conditional/mixed join not in rasterdf");
}

void rdf_hash_left_join(...) {
    throw NotImplementedException("Left join not yet in rasterdf");
}
```

**Inner join implementation sketch:**
```cpp
void rdf_hash_inner_join(...) {
    auto& ctx = get_rasterdf_context();
    auto& disp = get_dispatcher();
    auto* mr = get_workspace_resource();

    // 1. Build phase
    uint32_t tableSize = next_power_of_2(build_keys[0]->column_length * 2);
    auto hash_keys_buf = rasterdf::device_buffer(mr, tableSize * sizeof(int32_t), ...);
    auto hash_counts_buf = rasterdf::device_buffer(mr, tableSize * sizeof(int32_t), ...);
    // Fill hash_keys with -1, hash_counts with 0

    // 2. Build count
    rasterdf::execution::hash_join_build_count_pc pc1{
        build_keys[0]->getDeviceAddress(),
        hash_keys_buf.data(),
        hash_counts_buf.data(),
        static_cast<uint32_t>(build_keys[0]->column_length),
        tableSize
    };
    disp.dispatch_hash_join_build_count(pc1, numGroups);

    // 3. Prefix scan on hash_counts → hash_offsets
    // ... (use 3-pass scan)

    // 4. Build insert
    // 5. Probe count
    // 6. Prefix scan on probe counts
    // 7. Probe write → out_left, out_right

    count[0] = total_matches;
}
```

### 4.2 `src/gpu/rasterdf/rdf_aggregate.cpp` (Priority: HIGH)

Replaces `cudf_aggregate.cu`. Use rasterdf reduction dispatches.

```cpp
void rdf_aggregate(vector<shared_ptr<GPUColumn>>& column,
                   uint64_t num_aggregates,
                   AggregationType* agg_mode) {
    auto& disp = get_dispatcher();
    auto* mr = get_workspace_resource();

    for (int agg = 0; agg < num_aggregates; agg++) {
        switch (agg_mode[agg]) {
            case AggregationType::SUM: {
                auto output = rasterdf::device_buffer(mr, sizeof(int64_t), ...);
                rasterdf::execution::sum_push_constants pc{
                    column[agg]->getDeviceAddress(),
                    output.data(),
                    static_cast<uint32_t>(column[agg]->column_length),
                    0
                };
                if (column[agg]->isFloat()) disp.dispatch_sum_float(pc);
                else disp.dispatch_sum(pc);
                // Update column[agg] with result
                break;
            }
            case AggregationType::MIN: { /* similar with dispatch_min */ }
            case AggregationType::MAX: { /* similar with dispatch_max */ }
            case AggregationType::COUNT: { /* count non-null */ }
            case AggregationType::COUNT_STAR: { /* return column length */ }
            case AggregationType::AVERAGE: {
                // dispatch_sum + count, then divide
                // PLACEHOLDER for float division on GPU
            }
            case AggregationType::COUNT_DISTINCT: {
                throw NotImplementedException("COUNT DISTINCT not in rasterdf");
            }
            case AggregationType::FIRST: {
                // Gather index 0
            }
        }
    }
}
```

### 4.3 `src/gpu/rasterdf/rdf_orderby.cpp` (Priority: HIGH)

Replaces `cudf_orderby.cu`. Use rasterdf's radix sort pipeline.

```cpp
void rdf_order_by(vector<shared_ptr<GPUColumn>>& columns,
                  vector<OrderType>& orders,
                  uint64_t*& sorted_indices) {
    auto& disp = get_dispatcher();
    auto* mr = get_workspace_resource();
    uint32_t n = columns[0]->column_length;

    // Allocate index array and init to 0..n-1
    auto indices = rasterdf::device_buffer(mr, n * sizeof(uint32_t), ...);
    rasterdf::execution::radix_init_indices_pc init_pc{indices.data(), n};
    disp.dispatch_radix_init_indices(init_pc, (n + 255) / 256);

    // For each sort key (least significant first for multi-key):
    for (int k = columns.size() - 1; k >= 0; k--) {
        auto data_addr = columns[k]->getDeviceAddress();

        // Float keys: convert to sortable uint
        if (columns[k]->isFloat()) {
            rasterdf::execution::radix_init_indices_pc f2s{data_addr, n};
            disp.dispatch_float_to_sortable(f2s, (n + 255) / 256);
        }

        // Radix sort (batched)
        auto buf_b = rasterdf::device_buffer(mr, n * sizeof(uint32_t), ...);
        auto payload_b = rasterdf::device_buffer(mr, n * sizeof(uint32_t), ...);
        // ... allocate histogram, partial, bucket_totals, global_offsets buffers

        uint32_t numGroups = (n + 255) / 256;
        uint32_t numBlocks = (numGroups + 255) / 256;

        disp.dispatch_radix_sort_batched(
            data_addr, buf_b.data(),
            indices.data(), payload_b.data(),
            hist.data(), partial.data(),
            bucket_totals.data(), global_offsets.data(),
            n, numGroups, numBlocks);

        // Float keys: convert back
        if (columns[k]->isFloat()) {
            // dispatch_sortable_to_float on final data
        }

        // Descending: would need bitwise complement before sort
    }

    // sorted_indices now contains the permutation
}
```

### 4.4 `src/gpu/rasterdf/rdf_groupby.cpp` (Priority: HIGH)

Replaces `cudf_groupby.cu`. Use rasterdf hash-based or sort-based groupby.

```cpp
void rdf_groupby(vector<shared_ptr<GPUColumn>>& key_columns,
                 vector<shared_ptr<GPUColumn>>& value_columns,
                 vector<AggregationType>& agg_types,
                 /* output params */) {
    auto& disp = get_dispatcher();
    auto* mr = get_workspace_resource();

    // For single int32 key: use hash-based groupby
    if (key_columns.size() == 1 &&
        key_columns[0]->data_wrapper.type.id() == GPUColumnTypeId::INT32) {

        uint32_t n = key_columns[0]->column_length;
        uint32_t tableSize = next_power_of_2(n * 2);
        uint32_t numGroups = (n + 255) / 256;

        for (int v = 0; v < value_columns.size(); v++) {
            switch (agg_types[v]) {
                case AggregationType::SUM: {
                    // dispatch_hash_aggregate_sum
                    // dispatch_hash_extract_results
                    break;
                }
                case AggregationType::COUNT: {
                    // dispatch_hash_aggregate_count
                    // dispatch_hash_extract_results
                    break;
                }
                // MIN, MAX, MEAN similarly
            }
        }
    } else {
        // Multi-key or non-int32: PLACEHOLDER
        throw NotImplementedException("Multi-key groupby not yet in rasterdf");
    }
}
```

### 4.5 `src/gpu/operator/materialize.cpp` (Priority: HIGH)

Replaces `src/cuda/operator/materialize.cu`. Use `dispatch_gather_indices`.

```cpp
void gpu_materialize(GPUColumn& input, GPUColumn& output,
                     uint32_t* indices, uint32_t count) {
    auto& disp = get_dispatcher();
    rasterdf::execution::gather_indices_pc pc{
        input.getDeviceAddress(),
        /* indices addr */,
        output.getDeviceAddress(),
        count
    };
    disp.dispatch_gather_indices(pc, (count + 255) / 256);
}
```

### 4.6 `src/gpu/operator/comparison_expression.cpp` (Priority: HIGH)

Replaces `src/cuda/operator/comparison_expression.cu`.

```cpp
void gpu_compare(GPUColumn& input, GPUColumn& output,
                 int32_t threshold, CompareOp op) {
    auto& disp = get_dispatcher();
    rasterdf::execution::compare_push_constants pc{
        input.getDeviceAddress(),
        output.getDeviceAddress(),
        static_cast<uint32_t>(input.column_length),
        threshold,
        static_cast<int32_t>(op),  // 0=gt,1=lt,2=ge,3=le,4=eq,5=ne
        input.isFloat() ? 1 : 0
    };
    disp.dispatch_compare(pc);
}
```

---

## Phase 5: Placeholder Files

For every unsupported operation, create a `.cpp` file that throws:

```cpp
// src/gpu/rasterdf/rdf_dedup.cpp
#include "rasterdf/rdf_utils.hpp"
#include <duckdb/common/exception.hpp>

namespace duckdb {

void rdf_duplicate_elimination(/* params */) {
    throw NotImplementedException(
        "RasterDB: duplicate elimination not yet implemented in rasterdf");
}

} // namespace duckdb
```

Create placeholders for:
- `src/gpu/operator/hash_join_right.cpp`
- `src/gpu/operator/hash_join_single.cpp`
- `src/gpu/operator/nested_loop_join.cpp`
- `src/gpu/operator/strings_matching.cpp`
- `src/gpu/operator/substring.cpp`
- `src/gpu/operator/strlen_from_offsets.cpp`
- `src/gpu/operator/empty_str_check.cpp`
- `src/gpu/expression_executor/gpu_dispatch_string.cpp`

---

## Phase 6: Test Strategy

### 6.1 Minimal Smoke Test

Create `test/sql/rasterdb-basic.test`:
```sql
-- Load extension
LOAD 'build/release/extension/rasterdb/rasterdb.duckdb_extension';
CALL gpu_buffer_init('1 GB', '2 GB');

-- Simple filter + projection (int32)
CREATE TABLE t1 AS SELECT i AS a, i*2 AS b FROM range(1000) t(i);
CALL gpu_execution('SELECT a, b FROM t1 WHERE a > 500');

-- Simple aggregation
CALL gpu_execution('SELECT SUM(a), COUNT(*) FROM t1');

-- Simple group by
CALL gpu_execution('SELECT a % 10 AS grp, SUM(b) FROM t1 GROUP BY grp');

-- Simple order by
CALL gpu_execution('SELECT a FROM t1 ORDER BY a DESC LIMIT 10');
```

### 6.2 TPC-H Priority Queries

| Query | Operations | rasterdf Coverage |
|---|---|---|
| Q6 | Filter, SUM | ✅ Full |
| Q1 | Filter, GroupBy, SUM/AVG/COUNT | ⚠️ Partial (need multi-agg groupby) |
| Q3 | Join, Filter, GroupBy, OrderBy | ⚠️ Partial (need join + multi-key) |
| Q5 | Multi-way Join, GroupBy | ❌ Need multi-table join |

---

## Phase 7: Compilation Verification Checklist

After each phase, verify compilation:

```bash
cd rasterdb
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Phase 1 checkpoint:** CMake configures without errors, finds Vulkan and rasterdf
**Phase 2 checkpoint:** All renames compile, no undefined symbols from sirius references
**Phase 3 checkpoint:** Placeholder GPU files compile, link succeeds
**Phase 4 checkpoint:** Basic GPU operations work end-to-end
**Phase 5 checkpoint:** Unsupported ops throw NotImplementedException (caught by fallback)
**Phase 6 checkpoint:** Smoke test SQL queries produce correct results

---

## Appendix A: Global Context Access Pattern

rasterdf requires a `context` and `dispatcher` to be long-lived. Store them in the
extension's registered state:

```cpp
// In rasterdb_context.hpp:
class RasterDBContext {
public:
    rasterdf::context rdf_ctx;
    std::unique_ptr<rasterdf::memory_manager> mem_mgr;
    std::unique_ptr<rasterdf::execution::dispatcher> dispatcher;

    // Initialized during gpu_buffer_init()
    void initialize(size_t gpu_memory_limit) {
        mem_mgr = std::make_unique<rasterdf::memory_manager>(
            rdf_ctx, gpu_memory_limit);
        dispatcher = std::make_unique<rasterdf::execution::dispatcher>(rdf_ctx);
    }
};
```

Accessor pattern (replaces GPUBufferManager singleton):
```cpp
// Global or thread-local accessor
RasterDBContext& get_rasterdb_context();
rasterdf::execution::dispatcher& get_dispatcher();
rasterdf::memory_resource* get_workspace_resource();
rasterdf::memory_resource* get_data_resource();
```

---

## Appendix B: Memory Transfer Patterns

### Host → Device
```cpp
// BEFORE (CUDA):
cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);

// AFTER (rasterdf):
auto buf = rasterdf::device_buffer(mr, size);
buf.copy_from_host(host_ptr, size,
                   ctx.device(), ctx.compute_queue(),
                   dispatcher._command_pool);
```

### Device → Host
```cpp
// BEFORE (CUDA):
cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost);

// AFTER (rasterdf):
buf.copy_to_host(host_ptr, size,
                 ctx.device(), ctx.compute_queue(),
                 dispatcher._command_pool);
```

### Device → Device
```cpp
// BEFORE (CUDA):
cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);

// AFTER (rasterdf):
// Use vkCmdCopyBuffer via a helper function
void rdf_device_copy(VkDeviceAddress src, VkDeviceAddress dst, size_t size) {
    // Record command buffer with vkCmdCopyBuffer
    // Submit and wait
}
```

### Memset
```cpp
// BEFORE (CUDA):
cudaMemset(ptr, value, size);

// AFTER (rasterdf):
// Use vkCmdFillBuffer for uint32_t-aligned fills
// Or use mapped memory + memset for host-visible buffers
```

---

## Appendix C: Null Mask Conversion

### Creating Null Masks
```cpp
// BEFORE (CUDA/cuDF):
cudf::bitmask_type* mask = createNullMask(size, cudf::mask_state::ALL_VALID);

// AFTER (rasterdf):
auto mask_buf = rasterdf::device_buffer(mr, (size + 31) / 32 * sizeof(uint32_t));
// Fill with 0xFFFFFFFF for ALL_VALID
// Fill with 0x00000000 for ALL_NULL
```

### Propagating Null Masks
Phase 1: Skip null mask propagation (assume no NULLs).
Phase 2: Add null-mask-aware shaders or host-side mask merge.
