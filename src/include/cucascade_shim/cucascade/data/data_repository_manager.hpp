/*
 * RasterDB cucascade::shared_data_repository_manager shim.
 */
#pragma once
#include "data_repository.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>

namespace cucascade {

class shared_data_repository_manager {
public:
    shared_data_repository_manager() = default;
    ~shared_data_repository_manager() = default;

    std::shared_ptr<shared_data_repository> get_or_create(const std::string& name) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _repos.find(name);
        if (it != _repos.end()) return it->second;
        auto repo = std::make_shared<shared_data_repository>();
        _repos[name] = repo;
        return repo;
    }

    std::shared_ptr<shared_data_repository> get(const std::string& name) const {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _repos.find(name);
        return (it != _repos.end()) ? it->second : nullptr;
    }

    void remove(const std::string& name) {
        std::lock_guard<std::mutex> lock(_mutex);
        _repos.erase(name);
    }

private:
    mutable std::mutex _mutex;
    std::unordered_map<std::string, std::shared_ptr<shared_data_repository>> _repos;
};

} // namespace cucascade
