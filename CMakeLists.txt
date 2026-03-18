# =============================================================================
# Copyright 2025, RasterDB Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(rasterdb LANGUAGES CXX C)

# Dependencies
# Hint Vulkan location for older CMake versions
if(NOT DEFINED ENV{VULKAN_SDK})
  if(EXISTS "/usr/include/vulkan/vulkan.h")
    set(Vulkan_INCLUDE_DIR "/usr/include" CACHE PATH "Vulkan include directory")
    set(Vulkan_LIBRARY "/usr/lib/x86_64-linux-gnu/libvulkan.so" CACHE FILEPATH "Vulkan library")
  endif()
endif()
find_package(Vulkan REQUIRED)
find_package(spdlog REQUIRED CONFIG)

# rasterdf - Vulkan GPU DataFrame Library (use pre-built shared library)
set(RASTERDF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../rasterdf")
# Look for pre-built librasterdf.so
find_library(RASTERDF_LIBRARY
  NAMES rasterdf
  PATHS "${RASTERDF_ROOT}/build_release" "${RASTERDF_ROOT}/build"
  NO_DEFAULT_PATH)

if(RASTERDF_LIBRARY)
  message(STATUS "Found librasterdf: ${RASTERDF_LIBRARY}")
else()
  message(WARNING "librasterdf.so not found — GPU operations will be stubs only")
  set(RASTERDF_LIBRARY "")
endif()

# Create imported target so we can link without export-set issues
if(RASTERDF_LIBRARY)
  add_library(rasterdf_imported SHARED IMPORTED)
  set_target_properties(rasterdf_imported PROPERTIES
    IMPORTED_LOCATION "${RASTERDF_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${RASTERDF_ROOT}/include;${RASTERDF_ROOT}")
endif()

# ==========================================================================
# MINIMAL SOURCE LIST — bare-minimum DuckDB extension skeleton.
# All cudf/rmm/cucascade/CUDA dependencies have been removed.
# GPU operators will be added back with Vulkan/rasterdf implementations.
# ==========================================================================

# cmake-format: off
set(EXTENSION_SOURCES
    # Core extension entry point
    src/rasterdb_extension.cpp
    # Config (no external deps)
    src/config.cpp
    # Utilities
    src/util/segfault_backtrace_handler.cpp)

# GPU_SOURCES — Vulkan/rasterdf GPU execution layer
set(GPU_SOURCES
    src/gpu/gpu_context.cpp
    src/gpu/gpu_table.cpp
    src/gpu/gpu_executor.cpp)
# cmake-format: on

build_static_extension(rasterdb ${EXTENSION_SOURCES} ${GPU_SOURCES})
build_loadable_extension(rasterdb CPP ${EXTENSION_SOURCES} ${GPU_SOURCES})

# Shared configuration for both extension targets
foreach(_target rasterdb_extension rasterdb_loadable_extension)
  set_target_properties(
    ${_target}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON)

  target_include_directories(
    ${_target}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
      $<BUILD_INTERFACE:${RASTERDF_ROOT}/include>
      $<BUILD_INTERFACE:${RASTERDF_ROOT}>
      $<BUILD_INTERFACE:${RASTERDF_ROOT}/build/_deps/volk-src>)

  target_compile_definitions(
    ${_target}
    PRIVATE
      $<BUILD_INTERFACE:RASTERDB_DEFAULT_LOG_DIR="${CMAKE_BINARY_DIR}/log">
      VK_NO_PROTOTYPES)

  if(RASTERDF_LIBRARY)
    target_link_libraries(${_target} rasterdf_imported spdlog::spdlog Vulkan::Vulkan)
  else()
    target_link_libraries(${_target} spdlog::spdlog Vulkan::Vulkan)
  endif()

  add_dependencies(${_target} duckdb_static)
endforeach()

# Precompiled headers for extension targets
target_precompile_headers(
  rasterdb_extension
  PRIVATE
  <algorithm>
  <memory>
  <optional>
  <string>
  <string_view>
  <vector>
  "src/include/log/logging.hpp")
target_precompile_headers(rasterdb_loadable_extension REUSE_FROM rasterdb_extension)

# Install rasterdb extension
install(
  TARGETS rasterdb_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")

# ==========================================================================
# Tests — disabled until GPU operators are reimplemented with Vulkan/rasterdf
# ==========================================================================
# TODO: Re-enable tests after Vulkan/rasterdf operators are implemented
