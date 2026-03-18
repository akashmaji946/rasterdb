/*
 * RasterDB cucascade::shared_data_repository shim.
 */
#pragma once
#include "data_batch.hpp"
#include "../memory/memory_reservation.hpp"
#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <functional>

namespace cucascade {

class shared_data_repository {
public:
    shared_data_repository() = default;
    ~shared_data_repository() = default;

    std::size_t num_batches() const { return _batches.size(); }
    std::size_t total_rows() const {
        std::size_t total = 0;
        for (auto& b : _batches) total += b->num_rows();
        return total;
    }

    shared_data_batch get_batch(std::size_t idx) const {
        return (idx < _batches.size()) ? _batches[idx] : nullptr;
    }

    void add_batch(shared_data_batch batch) { _batches.push_back(std::move(batch)); }

    const std::vector<shared_data_batch>& batches() const { return _batches; }
    std::vector<shared_data_batch>& batches() { return _batches; }

    // Iteration
    auto begin() { return _batches.begin(); }
    auto end() { return _batches.end(); }
    auto begin() const { return _batches.begin(); }
    auto end() const { return _batches.end(); }

private:
    std::vector<shared_data_batch> _batches;
};

} // namespace cucascade
