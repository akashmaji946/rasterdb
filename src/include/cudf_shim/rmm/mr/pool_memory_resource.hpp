/* RasterDB rmm::mr::pool_memory_resource compatibility shim */
#pragma once
#include "../../types.hpp"
#include "../../column.hpp"
#include <cstddef>

namespace rmm {
namespace mr {

template <typename Upstream = device_memory_resource>
class pool_memory_resource : public device_memory_resource {
public:
    pool_memory_resource() = default;
    explicit pool_memory_resource(Upstream* upstream,
        std::size_t initial_pool_size = 0, std::size_t max_pool_size = 0)
        : _upstream(upstream) {}
    ~pool_memory_resource() override = default;
    Upstream* get_upstream() const { return _upstream; }
private:
    Upstream* _upstream{nullptr};
};

} // namespace mr
} // namespace rmm
