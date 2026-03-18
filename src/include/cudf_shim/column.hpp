/*
 * cudf column/table compatibility shim for RasterDB.
 */
#pragma once

#include "types.hpp"
#include <memory>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cstdlib>

#ifndef CUDF_CUDA_TRY
#define CUDF_CUDA_TRY(expr) (expr)
#endif
#ifndef RMM_CUDA_TRY
#define RMM_CUDA_TRY(expr) (expr)
#endif

namespace rmm {
namespace cuda_stream_view_detail {
struct cuda_stream_view {
    int value() const { return 0; }
    void synchronize() const {}
};
} // namespace cuda_stream_view_detail
using cuda_stream_view = cuda_stream_view_detail::cuda_stream_view;
static const cuda_stream_view cuda_stream_default{};

// rmm::cuda_stream (owning)
class cuda_stream {
public:
    cuda_stream() = default;
    explicit cuda_stream(unsigned int flags) {}
    cuda_stream_view view() const { return {}; }
    operator cuda_stream_view() const { return {}; }
};

struct cuda_device_id {
    int32_t value_{0};
    cuda_device_id() = default;
    cuda_device_id(int32_t v) : value_(v) {}
    int32_t value() const { return value_; }
};

class cuda_set_device_raii {
public:
    cuda_set_device_raii() = default;
    cuda_set_device_raii(cuda_device_id) {}
    ~cuda_set_device_raii() = default;
};

namespace mr {
class device_memory_resource {
public:
    virtual ~device_memory_resource() = default;
    void* allocate(size_t bytes, size_t = 256) { return ::malloc(bytes); }
    void deallocate(void* p, size_t, size_t = 256) { ::free(p); }
};
class cuda_memory_resource : public device_memory_resource {};
inline device_memory_resource* get_current_device_resource() {
    static device_memory_resource default_mr;
    return &default_mr;
}
inline device_memory_resource* set_current_device_resource(device_memory_resource* mr) {
    return mr; // no-op
}
} // namespace mr

using device_async_resource_ref = mr::device_memory_resource*;

// rmm::out_of_memory exception
class out_of_memory : public std::runtime_error {
public:
    out_of_memory(const char* msg = "out of memory") : std::runtime_error(msg) {}
};

static constexpr size_t CUDA_ALLOCATION_ALIGNMENT = 256;
using host_device_async_resource_ref = mr::device_memory_resource*;

class device_buffer {
public:
    device_buffer() : _data(nullptr), _size(0) {}
    device_buffer(size_t size, cuda_stream_view = {}, mr::device_memory_resource* = nullptr)
        : _size(size) { _data = size > 0 ? new uint8_t[size] : nullptr; }
    device_buffer(device_buffer&& o) noexcept : _data(o._data), _size(o._size) {
        o._data = nullptr; o._size = 0;
    }
    device_buffer& operator=(device_buffer&& o) noexcept {
        delete[] static_cast<uint8_t*>(_data);
        _data = o._data; _size = o._size;
        o._data = nullptr; o._size = 0;
        return *this;
    }
    ~device_buffer() { delete[] static_cast<uint8_t*>(_data); }
    void* data() { return _data; }
    const void* data() const { return _data; }
    size_t size() const { return _size; }
    void resize(size_t new_size, cuda_stream_view = {}) {
        auto* newbuf = new uint8_t[new_size];
        if (_data && _size > 0) std::memcpy(newbuf, _data, std::min(_size, new_size));
        delete[] static_cast<uint8_t*>(_data);
        _data = newbuf; _size = new_size;
    }
private:
    void* _data;
    size_t _size;
};

} // namespace rmm

namespace cudf {

// Forward declarations
class column_view;
class mutable_column_view;
class table_view;
class mutable_table_view;

class column_view {
public:
    column_view() : _type{}, _size(0), _data(nullptr), _null_mask(nullptr), _null_count(0), _offset(0) {}
    column_view(data_type type, size_type size, const void* data,
                const bitmask_type* null_mask = nullptr,
                size_type null_count = 0,
                size_type offset = 0,
                std::vector<column_view> children = {})
        : _type(type), _size(size), _data(data), _null_mask(null_mask),
          _null_count(null_count), _offset(offset), _children(std::move(children)) {}

    data_type type() const { return _type; }
    size_type size() const { return _size; }
    size_type null_count() const { return _null_count; }
    bool nullable() const { return _null_mask != nullptr; }
    bool has_nulls() const { return _null_count > 0; }
    size_type offset() const { return _offset; }

    template <typename T> const T* data() const {
        return static_cast<const T*>(_data) + _offset;
    }
    template <typename T> const T* head() const {
        return static_cast<const T*>(_data);
    }
    const bitmask_type* null_mask() const { return _null_mask; }

    size_type num_children() const { return static_cast<size_type>(_children.size()); }
    column_view child(size_type idx) const { return _children[idx]; }
    void set_children(std::vector<column_view> c) { _children = std::move(c); }

private:
    data_type _type;
    size_type _size;
    const void* _data;
    const bitmask_type* _null_mask;
    size_type _null_count;
    size_type _offset;
    std::vector<column_view> _children;
};

class mutable_column_view : public column_view {
public:
    using column_view::column_view;
    template <typename T> T* data() const {
        return const_cast<T*>(column_view::data<T>());
    }
    bitmask_type* null_mask() const {
        return const_cast<bitmask_type*>(column_view::null_mask());
    }
};

class strings_column_view {
public:
    strings_column_view(column_view const& cv) : _cv(cv) {}
    column_view parent() const { return _cv; }
    column_view offsets() const { return _cv.num_children() > 0 ? _cv.child(0) : column_view{}; }
    column_view chars() const { return _cv.num_children() > 1 ? _cv.child(1) : column_view{}; }
    size_type size() const { return _cv.size(); }
private:
    column_view _cv;
};

class table_view {
public:
    table_view() = default;
    table_view(std::vector<column_view> const& cols) : _columns(cols) {}
    column_view const& column(size_type idx) const { return _columns[idx]; }
    size_type num_columns() const { return static_cast<size_type>(_columns.size()); }
    size_type num_rows() const { return _columns.empty() ? 0 : _columns[0].size(); }
    auto begin() const { return _columns.begin(); }
    auto end() const { return _columns.end(); }
    table_view select(std::vector<size_type> const& col_indices) const {
        std::vector<column_view> result;
        for (auto i : col_indices) result.push_back(_columns[i]);
        return table_view(result);
    }
private:
    std::vector<column_view> _columns;
};

class mutable_table_view {
public:
    mutable_table_view() = default;
    mutable_table_view(std::vector<mutable_column_view> const& cols) : _columns(cols) {}
    mutable_column_view& column(size_type idx) { return _columns[idx]; }
    size_type num_columns() const { return static_cast<size_type>(_columns.size()); }
private:
    std::vector<mutable_column_view> _columns;
};

// Owning column
class column {
public:
    column() = default;
    column(data_type type, size_type size, rmm::device_buffer&& data,
           rmm::device_buffer&& null_mask = {}, size_type null_count = 0,
           std::vector<std::unique_ptr<column>>&& children = {})
        : _type(type), _size(size), _data(std::move(data)),
          _null_mask(std::move(null_mask)), _null_count(null_count),
          _children(std::move(children)) {}
    // Copy from column_view
    column(column_view const& cv, rmm::cuda_stream_view = {}, rmm::mr::device_memory_resource* = nullptr)
        : _type(cv.type()), _size(cv.size()), _null_count(cv.null_count()) {
        size_t bytes = cv.size() * size_of(cv.type());
        _data = rmm::device_buffer(bytes);
        if (cv.template data<uint8_t>()) std::memcpy(_data.data(), cv.template head<uint8_t>(), bytes);
    }
    column(column&&) = default;
    column& operator=(column&&) = default;

    column_view view() const {
        return column_view(_type, _size, _data.data(),
                          static_cast<const bitmask_type*>(_null_mask.data()),
                          _null_count);
    }
    operator column_view() const { return view(); }
    mutable_column_view mutable_view() {
        return mutable_column_view(_type, _size, _data.data(),
                                   static_cast<const bitmask_type*>(_null_mask.data()),
                                   _null_count);
    }
    data_type type() const { return _type; }
    size_type size() const { return _size; }
    size_type null_count() const { return _null_count; }
    bool nullable() const { return _null_mask.size() > 0; }
    bool has_nulls() const { return _null_count > 0; }
    void set_null_count(size_type c) { _null_count = c; }
    rmm::device_buffer& mutable_null_mask() { return _null_mask; }

    struct contents {
        std::unique_ptr<column> data;
        rmm::device_buffer null_mask;
        std::vector<std::unique_ptr<column>> children;
    };
    contents release() {
        return contents{nullptr, std::move(_null_mask), {}};
    }

private:
    data_type _type{};
    size_type _size{0};
    rmm::device_buffer _data;
    rmm::device_buffer _null_mask;
    size_type _null_count{0};
    std::vector<std::unique_ptr<column>> _children;
};

// Owning table
class table {
public:
    table() = default;
    table(std::vector<std::unique_ptr<column>>&& cols) : _columns(std::move(cols)) {}
    table(std::vector<std::unique_ptr<column>>&& cols, rmm::cuda_stream_view, rmm::mr::device_memory_resource* = nullptr)
        : _columns(std::move(cols)) {}
    table(table&&) = default;
    table& operator=(table&&) = default;

    table_view view() const {
        std::vector<column_view> views;
        for (auto& c : _columns) views.push_back(c->view());
        return table_view(views);
    }
    operator table_view() const { return view(); }
    size_type num_columns() const { return static_cast<size_type>(_columns.size()); }
    size_type num_rows() const { return _columns.empty() ? 0 : _columns[0]->size(); }
    column& get_column(size_type i) { return *_columns[i]; }
    std::vector<std::unique_ptr<column>>& release() { return _columns; }

private:
    std::vector<std::unique_ptr<column>> _columns;
};

// Scalar types
class scalar {
public:
    virtual ~scalar() = default;
    data_type type() const { return _type; }
    bool is_valid() const { return _is_valid; }
    void set_valid_async(bool v, rmm::cuda_stream_view = {}) { _is_valid = v; }
    void* data() { return nullptr; }
protected:
    scalar(data_type type, bool valid = true) : _type(type), _is_valid(valid) {}
    data_type _type;
    bool _is_valid;
};

template <typename T>
class numeric_scalar : public scalar {
public:
    numeric_scalar() : scalar(data_type{type_id::INT32}), _value{} {}
    numeric_scalar(T val, bool valid = true, rmm::cuda_stream_view = {},
                   rmm::mr::device_memory_resource* = nullptr)
        : scalar(data_type{type_id::INT32}, valid), _value(val) {}
    T value(rmm::cuda_stream_view = {}) const { return _value; }
    void set_value(T v, rmm::cuda_stream_view = {}) { _value = v; }
    T* data() { return &_value; }
    const T* data() const { return &_value; }
private:
    T _value;
};

template <typename T>
class fixed_point_scalar : public scalar {
public:
    fixed_point_scalar(T val = T{}, bool valid = true, rmm::cuda_stream_view = {},
                       rmm::mr::device_memory_resource* = nullptr)
        : scalar(data_type{type_id::DECIMAL64}, valid), _value(val) {}
    T value(rmm::cuda_stream_view = {}) const { return _value; }
private:
    T _value;
};

class string_scalar : public scalar {
public:
    string_scalar(std::string val = "", bool valid = true, rmm::cuda_stream_view = {},
                  rmm::mr::device_memory_resource* = nullptr)
        : scalar(data_type{type_id::STRING}, valid), _value(std::move(val)) {}
    std::string to_string(rmm::cuda_stream_view = {}) const { return _value; }
    size_type size() const { return static_cast<size_type>(_value.size()); }
    const char* data() const { return _value.c_str(); }
private:
    std::string _value;
};

template <typename Rep = int64_t>
class timestamp_scalar : public scalar {
public:
    timestamp_scalar() : scalar(data_type{type_id::TIMESTAMP_MICROSECONDS}), _value{} {}
    timestamp_scalar(Rep val, bool valid = true, rmm::cuda_stream_view = {},
                     rmm::mr::device_memory_resource* = nullptr)
        : scalar(data_type{type_id::TIMESTAMP_MICROSECONDS}, valid), _value(static_cast<int64_t>(val)) {}
    int64_t value(rmm::cuda_stream_view = {}) const { return _value; }
    int64_t* data() { return &_value; }
    const int64_t* data() const { return &_value; }
private:
    int64_t _value;
};

// Factory stubs
inline std::unique_ptr<column> make_empty_column(data_type type) {
    return std::make_unique<column>(type, 0, rmm::device_buffer{});
}

inline std::unique_ptr<column> make_empty_column(type_id id) {
    return make_empty_column(data_type{id});
}

inline std::unique_ptr<table> empty_like(table_view const& tv) {
    std::vector<std::unique_ptr<column>> cols;
    for (size_type i = 0; i < tv.num_columns(); i++) {
        cols.push_back(make_empty_column(tv.column(i).type()));
    }
    return std::make_unique<table>(std::move(cols));
}

inline std::unique_ptr<column> make_column_from_scalar(scalar const& /*s*/,
                                                        size_type /*size*/,
                                                        rmm::cuda_stream_view = {},
                                                        rmm::mr::device_memory_resource* = nullptr) {
    throw std::runtime_error("make_column_from_scalar: not implemented in RasterDB shim");
}

// get_default_stream stub
inline rmm::cuda_stream_view get_default_stream() { return rmm::cuda_stream_view{}; }

// host_span / device_span at cudf namespace level
template <typename T>
struct host_span {
    T* _data{nullptr};
    size_t _size{0};
    host_span() = default;
    host_span(T* d, size_t s) : _data(d), _size(s) {}
    template <typename Container>
    host_span(Container& c) : _data(c.data()), _size(c.size()) {}
    T* data() const { return _data; }
    size_t size() const { return _size; }
    T& operator[](size_t i) const { return _data[i]; }
    T* begin() const { return _data; }
    T* end() const { return _data + _size; }
};

template <typename T>
struct device_span {
    T* _data{nullptr};
    size_t _size{0};
    device_span() = default;
    device_span(T* d, size_t s) : _data(d), _size(s) {}
    T* data() const { return _data; }
    size_t size() const { return _size; }
};

// Concatenate stub
inline std::unique_ptr<table> concatenate(std::vector<table_view> const&) {
    throw std::runtime_error("cudf::concatenate not implemented — use rasterdf");
}
inline std::unique_ptr<column> concatenate(std::vector<column_view> const&) {
    throw std::runtime_error("cudf::concatenate columns not implemented — use rasterdf");
}

// Contiguous split stub
struct packed_columns {
    std::unique_ptr<table> metadata;
    rmm::device_buffer gpu_data;
};
inline std::vector<packed_columns> contiguous_split(table_view const&, std::vector<size_type> const&) {
    throw std::runtime_error("cudf::contiguous_split not implemented — use rasterdf");
}

// Stream compaction stubs
inline std::unique_ptr<table> drop_nulls(table_view const&, std::vector<size_type> const&, size_type = 1,
    rmm::cuda_stream_view = {}, rmm::mr::device_memory_resource* = nullptr) {
    throw std::runtime_error("cudf::drop_nulls not implemented — use rasterdf");
}
inline std::unique_ptr<table> apply_boolean_mask(table_view const&, column_view const&,
    rmm::cuda_stream_view = {}, rmm::mr::device_memory_resource* = nullptr) {
    throw std::runtime_error("cudf::apply_boolean_mask not implemented — use rasterdf");
}
inline std::unique_ptr<table> drop_duplicates(table_view const&, std::vector<size_type> const&,
    duplicate_keep_option keep = duplicate_keep_option::KEEP_FIRST,
    null_equality = null_equality::EQUAL,
    rmm::cuda_stream_view = {}, rmm::mr::device_memory_resource* = nullptr) {
    throw std::runtime_error("cudf::drop_duplicates not implemented — use rasterdf");
}
inline size_type distinct_count(column_view const&, null_policy = null_policy::INCLUDE,
    nan_policy = nan_policy::NAN_IS_VALID) {
    return 0;
}

// null_count helper
inline size_type null_count(bitmask_type const*, size_type, size_type,
    rmm::cuda_stream_view = {}) {
    return 0;
}

// set_current_device_resource
inline rmm::mr::device_memory_resource* set_current_device_resource(rmm::mr::device_memory_resource* mr) {
    return mr;
}

// Round stub
enum class rounding_method : int32_t { HALF_UP, HALF_EVEN };
inline std::unique_ptr<column> round(column_view const&, int32_t = 0,
    rounding_method = rounding_method::HALF_UP,
    rmm::cuda_stream_view = {}, rmm::mr::device_memory_resource* = nullptr) {
    throw std::runtime_error("cudf::round not implemented — use rasterdf");
}

// Concatenate with stream/mr
inline std::unique_ptr<table> concatenate(std::vector<table_view> const&,
    rmm::cuda_stream_view, rmm::mr::device_memory_resource* = nullptr) {
    throw std::runtime_error("cudf::concatenate not implemented — use rasterdf");
}

// copy_if_else with stream/mr
inline std::unique_ptr<column> copy_if_else(column_view const&, column_view const&, column_view const&,
    rmm::cuda_stream_view, rmm::mr::device_memory_resource* = nullptr) {
    throw std::runtime_error("cudf::copy_if_else not implemented — use rasterdf");
}
inline std::unique_ptr<column> copy_if_else(scalar const&, column_view const&, column_view const&,
    rmm::cuda_stream_view = {}, rmm::mr::device_memory_resource* = nullptr) {
    throw std::runtime_error("cudf::copy_if_else not implemented — use rasterdf");
}

// cudf::contains stub
inline bool contains(column_view const&, scalar const&) { return false; }

// cudf::is_null predicate stub
inline std::unique_ptr<column> is_null(column_view const&,
    rmm::cuda_stream_view = {}, rmm::mr::device_memory_resource* = nullptr) {
    return nullptr;
}

// cudf::make_structs_column stub
template <typename... Args>
inline std::unique_ptr<column> make_structs_column(Args&&...) {
    return nullptr;
}

// cudf::io::datasource::create — defined in io/datasource.hpp, just a forward stub here

// cudf::strings namespace stubs
namespace strings {
    template <typename... Args>
    inline std::unique_ptr<column> contains_re(Args&&...) { return nullptr; }
    template <typename... Args>
    inline std::unique_ptr<column> like(Args&&...) { return nullptr; }
} // namespace strings

// cudf::datetime namespace stubs
namespace datetime {
    template <typename... Args>
    inline std::unique_ptr<column> extract_year(Args&&...) { return nullptr; }
    template <typename... Args>
    inline std::unique_ptr<column> extract_month(Args&&...) { return nullptr; }
    template <typename... Args>
    inline std::unique_ptr<column> extract_day(Args&&...) { return nullptr; }
} // namespace datetime

} // namespace cudf

// Namespace alias for cudf::numeric = cudf::numeric_scalar
namespace cudf { template <typename T> using numeric = numeric_scalar<T>; }
