/* RasterDB rmm::device_uvector compatibility shim */
#pragma once
#include "../types.hpp"
#include "../column.hpp"
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace rmm {

template <typename T>
class device_uvector {
public:
    device_uvector() = default;
    explicit device_uvector(std::size_t size, cuda_stream_view = {}, mr::device_memory_resource* = nullptr)
        : _size(size) { _data = size > 0 ? static_cast<T*>(::malloc(size * sizeof(T))) : nullptr; }
    device_uvector(device_uvector&& o) noexcept : _data(o._data), _size(o._size) {
        o._data = nullptr; o._size = 0;
    }
    device_uvector& operator=(device_uvector&& o) noexcept {
        ::free(_data); _data = o._data; _size = o._size;
        o._data = nullptr; o._size = 0; return *this;
    }
    ~device_uvector() { ::free(_data); }

    T* data() { return _data; }
    const T* data() const { return _data; }
    T* begin() { return _data; }
    T* end() { return _data + _size; }
    const T* begin() const { return _data; }
    const T* end() const { return _data + _size; }
    std::size_t size() const { return _size; }
    bool is_empty() const { return _size == 0; }
    T element(std::size_t i, cuda_stream_view = {}) const { return _data[i]; }
    void set_element(std::size_t i, T val, cuda_stream_view = {}) { _data[i] = val; }
    void set_element_async(std::size_t i, T val, cuda_stream_view = {}) { _data[i] = val; }
    void resize(std::size_t new_size, cuda_stream_view = {}) {
        _data = static_cast<T*>(::realloc(_data, new_size * sizeof(T)));
        _size = new_size;
    }
    device_buffer release() {
        device_buffer buf(_size * sizeof(T));
        if (_data) std::memcpy(buf.data(), _data, _size * sizeof(T));
        ::free(_data); _data = nullptr; _size = 0;
        return buf;
    }

private:
    T* _data{nullptr};
    std::size_t _size{0};
};

} // namespace rmm
