/*
 * cudf compatibility shim for RasterDB.
 * Provides cudf:: types backed by basic C++ types so existing code compiles.
 * GPU operations are routed through rasterdf or fall back to CPU.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <utility>

namespace cudf {

using size_type = int32_t;
using bitmask_type = uint32_t;
using char_utf8 = uint32_t;
using thread_index_type = int64_t;

enum class type_id : int32_t {
    EMPTY = 0,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FLOAT32,
    FLOAT64,
    BOOL8,
    TIMESTAMP_DAYS,
    TIMESTAMP_SECONDS,
    TIMESTAMP_MILLISECONDS,
    TIMESTAMP_MICROSECONDS,
    TIMESTAMP_NANOSECONDS,
    DURATION_DAYS,
    DURATION_SECONDS,
    DURATION_MILLISECONDS,
    DURATION_MICROSECONDS,
    DURATION_NANOSECONDS,
    DICTIONARY32,
    STRING,
    LIST,
    DECIMAL32,
    DECIMAL64,
    DECIMAL128,
    STRUCT,
    NUM_TYPE_IDS
};

class data_type {
public:
    data_type() : _id(type_id::EMPTY), _scale(0) {}
    explicit data_type(type_id id) : _id(id), _scale(0) {}
    data_type(type_id id, int32_t scale) : _id(id), _scale(scale) {}
    type_id id() const { return _id; }
    int32_t scale() const { return _scale; }
    bool operator==(data_type const& o) const { return _id == o._id && _scale == o._scale; }
    bool operator!=(data_type const& o) const { return !(*this == o); }
private:
    type_id _id;
    int32_t _scale;
};

inline size_t size_of(data_type t) {
    switch (t.id()) {
        case type_id::INT8: case type_id::UINT8: case type_id::BOOL8: return 1;
        case type_id::INT16: case type_id::UINT16: return 2;
        case type_id::INT32: case type_id::UINT32: case type_id::FLOAT32:
        case type_id::DECIMAL32: case type_id::DICTIONARY32:
        case type_id::TIMESTAMP_DAYS: case type_id::DURATION_DAYS: return 4;
        case type_id::INT64: case type_id::UINT64: case type_id::FLOAT64:
        case type_id::DECIMAL64:
        case type_id::TIMESTAMP_SECONDS: case type_id::TIMESTAMP_MILLISECONDS:
        case type_id::TIMESTAMP_MICROSECONDS: case type_id::TIMESTAMP_NANOSECONDS:
        case type_id::DURATION_SECONDS: case type_id::DURATION_MILLISECONDS:
        case type_id::DURATION_MICROSECONDS: case type_id::DURATION_NANOSECONDS: return 8;
        case type_id::DECIMAL128: return 16;
        default: return 0;
    }
}

enum class order : bool { ASCENDING, DESCENDING };
enum class null_order : bool { BEFORE, AFTER };
enum class null_policy : bool { EXCLUDE, INCLUDE };
enum class nan_policy : bool { NAN_IS_NULL, NAN_IS_VALID };
enum class null_equality : bool { EQUAL, UNEQUAL };
enum class sorted : bool { NO, YES };
enum class mask_state : uint8_t { UNALLOCATED, UNINITIALIZED, ALL_VALID, ALL_NULL };
enum class out_of_bounds_policy : bool { DONT_CHECK, NULLIFY };
enum class duplicate_keep_option : int32_t { KEEP_ANY, KEEP_FIRST, KEEP_LAST, KEEP_NONE };

// Aggregation kind used by cudf
enum class aggregation_kind : int32_t {
    SUM, PRODUCT, MIN, MAX, COUNT_VALID, COUNT_ALL, ANY, ALL,
    SUM_OF_SQUARES, MEAN, VARIANCE, STD, MEDIAN, QUANTILE, ARGMAX, ARGMIN,
    NUNIQUE, NTH_ELEMENT, ROW_NUMBER, RANK, DENSE_RANK, COLLECT_LIST,
    COLLECT_SET, LEAD, LAG, PTX, CUDA, MERGE_LISTS, MERGE_SETS, MERGE_M2,
    COVARIANCE, CORRELATION, TDIGEST, MERGE_TDIGEST, HISTOGRAM, MERGE_HISTOGRAM,
    EWMA, PAIRWISE_FIRST, PAIRWISE_LAST
};

enum class binary_operator : int32_t {
    ADD, SUB, MUL, DIV, TRUE_DIV, FLOOR_DIV, MOD, PYMOD, POW,
    INT_POW, LOG_BASE, ATAN2, SHIFT_LEFT, SHIFT_RIGHT,
    SHIFT_RIGHT_UNSIGNED, BITWISE_AND, BITWISE_OR, BITWISE_XOR,
    LOGICAL_AND, LOGICAL_OR, EQUAL, NOT_EQUAL, LESS, GREATER,
    LESS_EQUAL, GREATER_EQUAL, NULL_EQUALS, NULL_MAX, NULL_MIN,
    NULL_NOT_EQUALS, NULL_LOGICAL_AND, NULL_LOGICAL_OR,
    GENERIC_BINARY, INVALID_BINARY
};

enum class unary_operator : int32_t {
    SIN, COS, TAN, ARCSIN, ARCCOS, ARCTAN, SINH, COSH, TANH,
    ARCSINH, ARCCOSH, ARCTANH, EXP, LOG, SQRT, CBRT, CEIL, FLOOR,
    ABS, RINT, BIT_INVERT, NOT, CAST_TO_INT64, CAST_TO_UINT64,
    CAST_TO_FLOAT64
};

// Timestamp type aliases
using timestamp_D = int32_t;
using timestamp_s = int64_t;
using timestamp_ms = int64_t;
using timestamp_us = int64_t;
using timestamp_ns = int64_t;
using duration_D = int32_t;
using duration_s = int64_t;
using duration_ms = int64_t;
using duration_us = int64_t;
using duration_ns = int64_t;

// Type traits
template <typename T> struct type_to_id { static constexpr type_id value = type_id::EMPTY; };
template <> struct type_to_id<int8_t> { static constexpr type_id value = type_id::INT8; };
template <> struct type_to_id<int16_t> { static constexpr type_id value = type_id::INT16; };
template <> struct type_to_id<int32_t> { static constexpr type_id value = type_id::INT32; };
template <> struct type_to_id<int64_t> { static constexpr type_id value = type_id::INT64; };
template <> struct type_to_id<uint8_t> { static constexpr type_id value = type_id::UINT8; };
template <> struct type_to_id<uint16_t> { static constexpr type_id value = type_id::UINT16; };
template <> struct type_to_id<uint32_t> { static constexpr type_id value = type_id::UINT32; };
template <> struct type_to_id<uint64_t> { static constexpr type_id value = type_id::UINT64; };
template <> struct type_to_id<float> { static constexpr type_id value = type_id::FLOAT32; };
template <> struct type_to_id<double> { static constexpr type_id value = type_id::FLOAT64; };
template <> struct type_to_id<bool> { static constexpr type_id value = type_id::BOOL8; };

template <type_id Id> struct id_to_type {};
template <> struct id_to_type<type_id::INT8> { using type = int8_t; };
template <> struct id_to_type<type_id::INT16> { using type = int16_t; };
template <> struct id_to_type<type_id::INT32> { using type = int32_t; };
template <> struct id_to_type<type_id::INT64> { using type = int64_t; };
template <> struct id_to_type<type_id::UINT8> { using type = uint8_t; };
template <> struct id_to_type<type_id::UINT16> { using type = uint16_t; };
template <> struct id_to_type<type_id::UINT32> { using type = uint32_t; };
template <> struct id_to_type<type_id::UINT64> { using type = uint64_t; };
template <> struct id_to_type<type_id::FLOAT32> { using type = float; };
template <> struct id_to_type<type_id::FLOAT64> { using type = double; };
template <> struct id_to_type<type_id::BOOL8> { using type = bool; };

template <typename T>
using scalar_type_t = T;

// Type name helpers
inline const char* type_to_name(data_type t) {
    switch (t.id()) {
        case type_id::INT8: return "INT8";
        case type_id::INT16: return "INT16";
        case type_id::INT32: return "INT32";
        case type_id::INT64: return "INT64";
        case type_id::UINT8: return "UINT8";
        case type_id::UINT16: return "UINT16";
        case type_id::UINT32: return "UINT32";
        case type_id::UINT64: return "UINT64";
        case type_id::FLOAT32: return "FLOAT32";
        case type_id::FLOAT64: return "FLOAT64";
        case type_id::BOOL8: return "BOOL8";
        case type_id::STRING: return "STRING";
        case type_id::TIMESTAMP_DAYS: return "TIMESTAMP_DAYS";
        case type_id::TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
        case type_id::TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
        case type_id::TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
        case type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
        case type_id::DECIMAL32: return "DECIMAL32";
        case type_id::DECIMAL64: return "DECIMAL64";
        case type_id::DECIMAL128: return "DECIMAL128";
        default: return "UNKNOWN";
    }
}

// Unary operation stub
template <typename... Args>
inline std::unique_ptr<struct column> unary_operation(struct column_view const&, unary_operator, Args&&...) {
    return nullptr;
}

// Binary operation stubs — accept optional stream/mr trailing args via variadic
template <typename... Args>
inline std::unique_ptr<struct column> binary_operation(struct column_view const&, struct column_view const&,
    binary_operator, data_type, Args&&...) {
    return nullptr;
}
template <typename S, typename... Args>
inline std::unique_ptr<struct column> binary_operation(S const&, struct column_view const&,
    binary_operator, data_type, Args&&...) {
    return nullptr;
}

// Cast stub
template <typename... Args>
inline std::unique_ptr<struct column> cast(struct column_view const&, data_type, Args&&...) {
    return nullptr;
}

// Replace nulls stub
template <typename... Args>
inline std::unique_ptr<struct column> replace_nulls(struct column_view const&, struct column_view const&, Args&&...) {
    return nullptr;
}
template <typename S, typename... Args>
inline std::unique_ptr<struct column> replace_nulls(struct column_view const&, S const&, Args&&...) {
    return nullptr;
}

// Type dispatcher - simplified
template <typename Functor, typename... Args>
auto type_dispatcher(data_type dtype, Functor&& f, Args&&... args) {
    return f.template operator()<int32_t>(std::forward<Args>(args)...);
}

} // namespace cudf
