/*
 * RasterDB cucascade::data_batch shim.
 */
#pragma once
#include "common.hpp"
#include "../memory/memory_reservation.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <functional>

namespace cucascade {

class data_batch : public std::enable_shared_from_this<data_batch> {
public:
    data_batch() = default;
    data_batch(std::size_t num_rows, std::unique_ptr<idata_representation> repr = nullptr)
        : _num_rows(num_rows) {}
    template <typename T>
    data_batch(std::size_t num_rows, std::unique_ptr<T> repr)
        : _num_rows(num_rows) {}
    ~data_batch() = default;

    batch_state state() const { return _state.load(); }
    void set_state(batch_state s) { _state.store(s); }

    std::size_t num_rows() const { return _num_rows; }
    void set_num_rows(std::size_t n) { _num_rows = n; }

    std::size_t num_columns() const { return _num_cols; }
    void set_num_columns(std::size_t n) { _num_cols = n; }

    std::size_t size_bytes() const { return _size_bytes; }
    void set_size_bytes(std::size_t n) { _size_bytes = n; }

    // Representation access
    template <typename T>
    std::shared_ptr<T> get_representation() const { return nullptr; }

    template <typename T>
    std::shared_ptr<T> get_data() const { return nullptr; }

    template <typename T>
    void set_representation(std::shared_ptr<T>) {}

    template <typename Target>
    void convert_to(memory::memory_space*) {}

    // Processing handle
    data_batch_processing_handle acquire_processing() {
        _state.store(batch_state::processing);
        return data_batch_processing_handle{};
    }

    bool try_set_task_created() {
        auto expected = batch_state::idle;
        return _state.compare_exchange_strong(expected, batch_state::task_created);
    }

    bool try_to_lock_for_in_transit() {
        auto expected = batch_state::idle;
        return _state.compare_exchange_strong(expected, batch_state::in_transit);
    }
    void try_to_release_in_transit() {
        _state.store(batch_state::idle);
    }

    // State
    batch_state get_state() const { return _state.load(); }

    // Memory reservation
    void set_reservation(memory::reservation&& r) { _reservation = std::move(r); }
    memory::reservation& reservation() { return _reservation; }
    const memory::reservation& reservation() const { return _reservation; }

    memory::memory_space_id location() const {
        return memory::memory_space_id{memory::Tier::HOST, 0};
    }
    memory::memory_space* get_memory_space() const { return nullptr; }

    // Index in repository
    int32_t index() const { return _index; }
    void set_index(int32_t i) { _index = i; }
    int32_t get_batch_id() const { return _index; }

private:
    std::atomic<batch_state> _state{batch_state::idle};
    std::size_t _num_rows{0};
    std::size_t _num_cols{0};
    std::size_t _size_bytes{0};
    int32_t _index{-1};
    memory::reservation _reservation;
};

using shared_data_batch = std::shared_ptr<data_batch>;

} // namespace cucascade
