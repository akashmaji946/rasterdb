/*
 * RasterDB GPU stubs header.
 * All GPU operations that are not yet implemented in rasterdf
 * throw NotImplementedException, which triggers DuckDB CPU fallback.
 */
#pragma once

#include <stdexcept>
#include <string>

namespace duckdb {

inline void rdf_not_implemented(const char* op_name) {
    throw std::runtime_error(
        std::string("RasterDB: GPU operation '") + op_name +
        "' not yet implemented via rasterdf. Falling back to CPU.");
}

} // namespace duckdb
