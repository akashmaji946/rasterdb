/*
 * RasterDB cucascade::representation_converter_registry shim.
 */
#pragma once
#include "common.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace cucascade {

class representation_converter_registry {
public:
    representation_converter_registry() = default;
    ~representation_converter_registry() = default;
};

} // namespace cucascade
