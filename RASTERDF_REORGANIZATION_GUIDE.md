# RasterDF Library Reorganization Guide

## Goal

Reorganize rasterdf's include/src layout to mirror cuDF's structure, making it a
conceptual drop-in replacement. This enables rasterdb to `#include <rasterdf/...>`
in patterns identical to `#include <cudf/...>`.

---

## 1. Current rasterdf Header Layout

```
include/rasterdf/
├── api/librasterdf_c.h
├── core/
│   ├── column.hpp
│   ├── column_device_view.hpp
│   ├── column_view.hpp
│   ├── context.hpp
│   ├── device_buffer.hpp
│   ├── scalar.hpp
│   ├── table.hpp
│   ├── table_device_view.hpp
│   ├── table_view.hpp
│   ├── types.hpp
│   ├── ComputePipelineProperties.hpp
│   ├── VkEngine.hpp
│   └── VulkanDevice.hpp
├── execution/
│   ├── constants.hpp
│   └── dispatcher.hpp
├── io/
│   ├── binary_reader.hpp
│   └── (csv_reader.hpp — future)
├── memory/
│   ├── memory_manager.hpp
│   ├── memory_resource.hpp
│   ├── pool_memory_resource.hpp
│   └── vulkan_memory_resource.hpp
└── utils/
    ├── logger.hpp
    └── (error.hpp — future)
```

## 2. Target rasterdf Header Layout (Mirroring cuDF)

```
include/rasterdf/
│
├── types.hpp                          # MOVED from core/types.hpp
├── context.hpp                        # MOVED from core/context.hpp
├── device_buffer.hpp                  # MOVED from core/device_buffer.hpp
├── memory_manager.hpp                 # MOVED from memory/memory_manager.hpp
│
├── column/
│   ├── column.hpp                     # MOVED from core/column.hpp
│   ├── column_view.hpp                # MOVED from core/column_view.hpp
│   ├── column_device_view.hpp         # MOVED from core/column_device_view.hpp
│   └── column_factories.hpp           # NEW
│
├── table/
│   ├── table.hpp                      # MOVED from core/table.hpp
│   ├── table_view.hpp                 # MOVED from core/table_view.hpp
│   └── table_device_view.hpp          # MOVED from core/table_device_view.hpp
│
├── scalar/
│   ├── scalar.hpp                     # MOVED from core/scalar.hpp (fix templates)
│   └── scalar_factories.hpp           # NEW
│
├── reduction.hpp                      # NEW — reduce() API
├── aggregation.hpp                    # NEW — aggregation enums/factories
├── sorting.hpp                        # NEW — sort(), sorted_order()
├── copying.hpp                        # NEW — gather(), scatter(), empty_like()
├── unary.hpp                          # NEW — unary_operation()
├── binaryop.hpp                       # NEW — binary_operation()
├── round.hpp                          # NEW — placeholder
├── stream_compaction.hpp              # NEW — distinct(), unique()
│
├── join/
│   ├── hash_join.hpp                  # NEW — hash_join class
│   ├── join.hpp                       # NEW — inner_join(), left_join() free fns
│   ├── conditional_join.hpp           # NEW — placeholder
│   └── mixed_join.hpp                 # NEW — placeholder
│
├── ast/
│   └── expressions.hpp                # NEW — placeholder for AST expr trees
│
├── mr/                                # MOVED+RENAMED from memory/
│   ├── device_memory_resource.hpp     # RENAMED from memory_resource.hpp
│   ├── pool_memory_resource.hpp       # MOVED from memory/pool_memory_resource.hpp
│   └── vulkan_memory_resource.hpp     # MOVED from memory/vulkan_memory_resource.hpp
│
├── vulkan/                            # MOVED Vulkan internals from core/
│   ├── VkEngine.hpp
│   ├── VulkanDevice.hpp
│   └── ComputePipelineProperties.hpp
│
├── execution/                         # KEPT as-is
│   ├── dispatcher.hpp
│   └── constants.hpp
│
├── io/                                # KEPT as-is
│   └── binary_reader.hpp
│
├── utilities/                         # NEW
│   ├── error.hpp                      # RDF_EXPECTS, RDF_CHECK macros
│   └── type_dispatcher.hpp            # Type-based dispatch utility
│
└── api/                               # KEPT as-is
    └── librasterdf_c.h
```

## 3. Migration Steps (In Order)

### Step 1: Add Forwarding Headers (Non-Breaking)

Before moving files, create forwarding headers at the OLD locations that `#include`
the new locations. This way existing code keeps compiling during the transition.

Example — create `include/rasterdf/core/column.hpp` as:
```cpp
#pragma once
// Forwarding header — column.hpp moved to rasterdf/column/column.hpp
#include <rasterdf/column/column.hpp>
```

### Step 2: Move Core Type Files to Top Level

| Old Path | New Path |
|---|---|
| `core/types.hpp` | `types.hpp` |
| `core/context.hpp` | `context.hpp` |
| `core/device_buffer.hpp` | `device_buffer.hpp` |
| `memory/memory_manager.hpp` | `memory_manager.hpp` |

**Why:** cuDF has `cudf/types.hpp` at the top level, not `cudf/core/types.hpp`.

### Step 3: Move Column/Table/Scalar into Dedicated Directories

| Old Path | New Path |
|---|---|
| `core/column.hpp` | `column/column.hpp` |
| `core/column_view.hpp` | `column/column_view.hpp` |
| `core/column_device_view.hpp` | `column/column_device_view.hpp` |
| `core/table.hpp` | `table/table.hpp` |
| `core/table_view.hpp` | `table/table_view.hpp` |
| `core/table_device_view.hpp` | `table/table_device_view.hpp` |
| `core/scalar.hpp` | `scalar/scalar.hpp` |

**Matches cuDF pattern:** `cudf/column/column.hpp`, `cudf/table/table.hpp`, etc.

### Step 4: Move Memory to `mr/` Directory

| Old Path | New Path |
|---|---|
| `memory/memory_resource.hpp` | `mr/device_memory_resource.hpp` |
| `memory/pool_memory_resource.hpp` | `mr/pool_memory_resource.hpp` |
| `memory/vulkan_memory_resource.hpp` | `mr/vulkan_memory_resource.hpp` |

**Matches RMM pattern:** `rmm/mr/device_memory_resource.hpp`

### Step 5: Move Vulkan Internals to `vulkan/`

| Old Path | New Path |
|---|---|
| `core/VkEngine.hpp` | `vulkan/VkEngine.hpp` |
| `core/VulkanDevice.hpp` | `vulkan/VulkanDevice.hpp` |
| `core/ComputePipelineProperties.hpp` | `vulkan/ComputePipelineProperties.hpp` |

**Why:** These are implementation details, not part of the public API.

### Step 6: Create New High-Level API Headers

These are **new files** that expose cuDF-like function signatures. They wrap
the low-level dispatcher calls into clean APIs.

#### `include/rasterdf/reduction.hpp`
```cpp
#pragma once
#include <rasterdf/column/column.hpp>
#include <rasterdf/column/column_view.hpp>
#include <rasterdf/scalar/scalar.hpp>
#include <rasterdf/aggregation.hpp>

namespace rasterdf {

// Mirrors cudf::reduce()
std::unique_ptr<scalar> reduce(
    column_view const& col,
    reduce_aggregation const& agg,
    data_type output_type,
    context& ctx);

} // namespace rasterdf
```

#### `include/rasterdf/aggregation.hpp`
```cpp
#pragma once
#include <memory>

namespace rasterdf {

struct reduce_aggregation {
    enum Kind { SUM, MIN, MAX, MEAN, COUNT, NUNIQUE };
    Kind kind;
};

struct groupby_aggregation {
    enum Kind { SUM, MIN, MAX, COUNT, MEAN, FIRST };
    Kind kind;
};

// Factory functions (mirror cudf::make_*_aggregation)
inline auto make_sum_aggregation() { return reduce_aggregation{reduce_aggregation::SUM}; }
inline auto make_min_aggregation() { return reduce_aggregation{reduce_aggregation::MIN}; }
inline auto make_max_aggregation() { return reduce_aggregation{reduce_aggregation::MAX}; }
inline auto make_mean_aggregation() { return reduce_aggregation{reduce_aggregation::MEAN}; }
inline auto make_count_aggregation() { return reduce_aggregation{reduce_aggregation::COUNT}; }
inline auto make_nunique_aggregation() { return reduce_aggregation{reduce_aggregation::NUNIQUE}; }

} // namespace rasterdf
```

#### `include/rasterdf/sorting.hpp`
```cpp
#pragma once
#include <rasterdf/column/column.hpp>
#include <rasterdf/table/table.hpp>
#include <rasterdf/table/table_view.hpp>
#include <vector>

namespace rasterdf {

enum class order { ASCENDING, DESCENDING };
enum class null_order { BEFORE, AFTER };

// Mirrors cudf::sorted_order()
std::unique_ptr<column> sorted_order(
    table_view const& keys,
    std::vector<order> const& column_order,
    context& ctx);

// Mirrors cudf::sort()
std::unique_ptr<table> sort(
    table_view const& input,
    std::vector<order> const& column_order,
    context& ctx);

} // namespace rasterdf
```

#### `include/rasterdf/join/hash_join.hpp`
```cpp
#pragma once
#include <rasterdf/column/column.hpp>
#include <rasterdf/table/table_view.hpp>
#include <memory>
#include <utility>

namespace rasterdf {

// Mirrors cudf::hash_join
class hash_join {
public:
    explicit hash_join(table_view const& build_keys, context& ctx);
    ~hash_join();

    std::pair<std::unique_ptr<column>, std::unique_ptr<column>>
    inner_join(table_view const& probe_keys) const;

    // Placeholder — falls back
    std::pair<std::unique_ptr<column>, std::unique_ptr<column>>
    left_join(table_view const& probe_keys) const;

private:
    struct impl;
    std::unique_ptr<impl> _impl;
};

} // namespace rasterdf
```

#### `include/rasterdf/groupby.hpp`
```cpp
#pragma once
#include <rasterdf/column/column.hpp>
#include <rasterdf/table/table.hpp>
#include <rasterdf/table/table_view.hpp>
#include <rasterdf/aggregation.hpp>
#include <vector>

namespace rasterdf {

struct groupby_request {
    column_view values;
    std::vector<groupby_aggregation> aggregations;
};

class groupby {
public:
    explicit groupby(table_view const& keys, context& ctx);

    struct result {
        std::unique_ptr<table> keys;
        std::vector<std::vector<std::unique_ptr<column>>> results;
    };

    result aggregate(std::vector<groupby_request> const& requests);

private:
    table_view _keys;
    context& _ctx;
};

} // namespace rasterdf
```

#### `include/rasterdf/copying.hpp`
```cpp
#pragma once
#include <rasterdf/column/column.hpp>
#include <rasterdf/column/column_view.hpp>
#include <rasterdf/table/table.hpp>
#include <rasterdf/table/table_view.hpp>

namespace rasterdf {

std::unique_ptr<table> gather(
    table_view const& source,
    column_view const& gather_map,
    context& ctx);

std::unique_ptr<table> empty_like(table_view const& input);

std::unique_ptr<column> allocate_like(
    column_view const& input,
    mask_state state,
    context& ctx);

} // namespace rasterdf
```

#### `include/rasterdf/stream_compaction.hpp`
```cpp
#pragma once
#include <rasterdf/table/table.hpp>
#include <rasterdf/table/table_view.hpp>
#include <vector>

namespace rasterdf {

// Placeholder — sort + boundary detect
std::unique_ptr<table> distinct(
    table_view const& input,
    std::vector<size_type> const& keys,
    context& ctx);

} // namespace rasterdf
```

### Step 7: Add Bitmask Infrastructure to `types.hpp`

Add to `include/rasterdf/types.hpp`:
```cpp
// Bitmask types (mirrors cudf::bitmask_type)
using bitmask_type = uint32_t;

enum class mask_state {
    ALL_VALID,      // All bits set to 1 (no nulls)
    ALL_NULL,       // All bits set to 0 (all null)
    UNINITIALIZED   // Bits not set
};

enum class null_equality { EQUAL, UNEQUAL };
enum class null_policy { EXCLUDE, INCLUDE };
enum class nan_policy { NAN_IS_NULL, NAN_IS_VALID };
```

Add utility functions:
```cpp
// In a new include/rasterdf/null_mask.hpp:
device_buffer create_null_mask(size_type size, mask_state state,
                               memory_resource* mr, context& ctx);
size_type count_set_bits(bitmask_type const* bitmask,
                         size_type start, size_type stop);
```

### Step 8: Fix scalar.hpp Template Issues

Current `scalar.hpp` has broken template specializations (lines 62-67 use undefined `T...`).
Fix by removing those broken specializations and adding proper ones:

```cpp
// Remove these broken lines:
// template <> inline data_type get_data_type<std::tuple<T...>>() { ... }
// template <> inline data_type get_data_type<std::variant<T...>>() { ... }

// The get_data_type specializations already exist in types.hpp.
// scalar.hpp should not redefine them. Remove duplicates and use types.hpp.
```

### Step 9: Add Missing type_id Entries

Current `type_id` enum needs:
```cpp
enum class type_id {
    EMPTY,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,         // NEW — needed for cuDF compatibility
    UINT16,        // NEW
    UINT32,        // NEW
    UINT64,        // NEW
    FLOAT32,
    FLOAT64,
    BOOL8,
    TIMESTAMP_DAYS,
    TIMESTAMP_SECONDS,
    TIMESTAMP_MILLISECONDS,
    TIMESTAMP_MICROSECONDS,
    TIMESTAMP_NANOSECONDS,
    DECIMAL32,     // NEW — placeholder
    DECIMAL64,     // NEW — placeholder
    DECIMAL128,    // NEW — placeholder
    DICTIONARY32,
    STRING,
    LIST,
    STRUCT,
    NUM_TYPE_IDS
};
```

### Step 10: Update src/ Layout to Match

Mirror the header reorganization in `src/`:
```
src/
├── column/
│   ├── column.cpp           # MOVED from core/column.cpp
│   ├── column_view.cpp      # MOVED from core/column_view.cpp
│   └── column_factories.cpp # NEW
├── table/
│   ├── table.cpp            # MOVED from core/table.cpp
│   └── table_view.cpp       # MOVED from core/table_view.cpp
├── scalar/
│   └── scalar.cpp           # NEW (implement numeric_scalar properly)
├── reduction/
│   └── reduction.cpp        # NEW (wraps dispatcher reductions)
├── sorting/
│   └── sorting.cpp          # NEW (wraps dispatcher radix sort)
├── join/
│   └── hash_join.cpp        # NEW (wraps dispatcher join pipeline)
├── groupby/
│   └── groupby.cpp          # NEW (wraps dispatcher groupby)
├── copying/
│   └── copying.cpp          # NEW (wraps dispatcher gather)
├── binaryop/
│   └── binaryop.cpp         # NEW (wraps dispatcher binary ops)
├── unary/
│   └── unary.cpp            # NEW (wraps dispatcher unary ops)
├── stream_compaction/
│   └── distinct.cpp         # NEW (placeholder)
├── null_mask/
│   └── null_mask.cpp        # NEW (bitmask utilities)
├── mr/
│   ├── vulkan_memory_resource.cpp  # MOVED from memory/
│   ├── pool_memory_resource.cpp    # MOVED from memory/
│   └── memory_manager.cpp         # MOVED from memory/
├── vulkan/
│   ├── VkEngine.cpp         # MOVED from core/
│   ├── VulkanDevice.cpp     # MOVED from core/
│   └── ComputePipelineProperties.cpp
├── execution/
│   └── dispatcher.cpp       # KEPT
├── io/
│   └── binary_reader.cpp    # KEPT
├── api/
│   └── librasterdf_c.cpp   # KEPT
├── utils/
│   └── logger.cpp           # KEPT
└── examples/                # KEPT
```

### Step 11: Update CMakeLists.txt

After moving files, update the source list in `CMakeLists.txt`:
```cmake
add_library(rasterdf SHARED
    # VMA
    vma/vma.cpp
    # Column
    src/column/column.cpp
    src/column/column_view.cpp
    src/column/column_factories.cpp
    # Table
    src/table/table.cpp
    src/table/table_view.cpp
    # Scalar
    src/scalar/scalar.cpp
    # High-level APIs
    src/reduction/reduction.cpp
    src/sorting/sorting.cpp
    src/join/hash_join.cpp
    src/groupby/groupby.cpp
    src/copying/copying.cpp
    src/binaryop/binaryop.cpp
    src/unary/unary.cpp
    src/stream_compaction/distinct.cpp
    src/null_mask/null_mask.cpp
    # Execution
    src/execution/dispatcher.cpp
    # Vulkan internals
    src/vulkan/VkEngine.cpp
    src/vulkan/VulkanDevice.cpp
    src/vulkan/ComputePipelineProperties.cpp
    # Memory
    src/mr/vulkan_memory_resource.cpp
    src/mr/pool_memory_resource.cpp
    src/mr/memory_manager.cpp
    # IO
    src/io/binary_reader.cpp
    # Utils
    src/utils/logger.cpp
    # C API
    src/api/librasterdf_c.cpp
)
```

---

## 4. Include Path Mapping: cuDF → rasterdf

After reorganization, the include mapping for rasterdb becomes:

| cuDF Include (used in Sirius) | rasterdf Include (for RasterDB) |
|---|---|
| `<cudf/types.hpp>` | `<rasterdf/types.hpp>` |
| `<cudf/column/column_view.hpp>` | `<rasterdf/column/column_view.hpp>` |
| `<cudf/column/column_factories.hpp>` | `<rasterdf/column/column_factories.hpp>` |
| `<cudf/table/table.hpp>` | `<rasterdf/table/table.hpp>` |
| `<cudf/table/table_view.hpp>` | `<rasterdf/table/table_view.hpp>` |
| `<cudf/scalar/scalar.hpp>` | `<rasterdf/scalar/scalar.hpp>` |
| `<cudf/scalar/scalar_factories.hpp>` | `<rasterdf/scalar/scalar_factories.hpp>` |
| `<cudf/aggregation.hpp>` | `<rasterdf/aggregation.hpp>` |
| `<cudf/reduction.hpp>` | `<rasterdf/reduction.hpp>` |
| `<cudf/groupby.hpp>` | `<rasterdf/groupby.hpp>` |
| `<cudf/sorting.hpp>` | `<rasterdf/sorting.hpp>` |
| `<cudf/copying.hpp>` | `<rasterdf/copying.hpp>` |
| `<cudf/unary.hpp>` | `<rasterdf/unary.hpp>` |
| `<cudf/round.hpp>` | `<rasterdf/round.hpp>` (placeholder) |
| `<cudf/stream_compaction.hpp>` | `<rasterdf/stream_compaction.hpp>` |
| `<cudf/join.hpp>` / `<cudf/join/hash_join.hpp>` | `<rasterdf/join/hash_join.hpp>` |
| `<cudf/join/join.hpp>` | `<rasterdf/join/join.hpp>` |
| `<cudf/join/conditional_join.hpp>` | `<rasterdf/join/conditional_join.hpp>` (placeholder) |
| `<cudf/join/mixed_join.hpp>` | `<rasterdf/join/mixed_join.hpp>` (placeholder) |
| `<cudf/ast/expressions.hpp>` | `<rasterdf/ast/expressions.hpp>` (placeholder) |
| `<rmm/mr/device_memory_resource.hpp>` | `<rasterdf/mr/device_memory_resource.hpp>` |
| `<rmm/mr/pool_memory_resource.hpp>` | `<rasterdf/mr/pool_memory_resource.hpp>` |
| `<rmm/mr/cuda_memory_resource.hpp>` | `<rasterdf/mr/vulkan_memory_resource.hpp>` |
| `<rmm/device_buffer.hpp>` | `<rasterdf/device_buffer.hpp>` |

---

## 5. Priority Order for Implementation

1. **types.hpp** additions (bitmask_type, mask_state, UINT types, DECIMAL placeholders)
2. **null_mask.hpp/cpp** — create_null_mask, count_set_bits
3. **scalar.hpp fix** — remove broken template specializations
4. **column_factories.hpp** — make_empty_column, allocate_like
5. **reduction.hpp/cpp** — wraps existing dispatcher sum/min/max
6. **sorting.hpp/cpp** — wraps existing dispatcher radix sort
7. **join/hash_join.hpp/cpp** — wraps existing dispatcher join pipeline
8. **groupby.hpp/cpp** — wraps existing dispatcher groupby
9. **copying.hpp/cpp** — gather via dispatcher
10. **Header moves** (column/, table/, scalar/, mr/, vulkan/)
11. **Forwarding headers** at old locations
12. **aggregation.hpp, binaryop.hpp, unary.hpp** — enum definitions + wrappers
13. **Placeholder headers** (ast/, conditional_join, mixed_join, round, stream_compaction)
