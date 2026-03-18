#!/bin/bash
# build.sh — Build the RasterDB DuckDB extension
#
# Prerequisites:
#   - rasterdf must be built first (../rasterdf/build.sh --release)
#   - spdlog must be installed (apt install libspdlog-dev or via conda)
#   - Vulkan SDK / headers must be available
#
# Usage: ./build.sh [--release|--debug|--clean]
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

BUILD_PRESET="release"
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --release) BUILD_PRESET="release" ;;
        --debug)   BUILD_PRESET="debug" ;;
        --clean)   CLEAN=1 ;;
    esac
done

echo "============================================"
echo "  RasterDB — DuckDB Extension Build"
echo "  Preset: ${BUILD_PRESET}"
echo "============================================"

# --- 0. Clean if requested ---
if [ "$CLEAN" -eq 1 ]; then
    echo ""
    echo "[0] Cleaning build directory..."
    rm -rf build
fi

# --- 1. Check rasterdf is built ---
echo ""
echo "[1/4] Checking rasterdf library..."
RASTERDF_ROOT="${PROJECT_DIR}/../rasterdf"
SO_FILE=""
for candidate in "${RASTERDF_ROOT}/build_release/librasterdf.so" \
                 "${RASTERDF_ROOT}/build/librasterdf.so" \
                 "/usr/local/lib/librasterdf.so"; do
    if [ -f "$candidate" ]; then
        SO_FILE="$candidate"
        break
    fi
done

if [ -z "$SO_FILE" ]; then
    echo "  ERROR: librasterdf.so not found!"
    echo "  Build it first:  cd ../rasterdf && ./build.sh --release"
    exit 1
fi
echo "  Found: ${SO_FILE}"

# --- 2. Check spdlog ---
echo ""
echo "[2/4] Checking spdlog..."
# Try conda env first
SPDLOG_PREFIX=""
if [ -d "${PROJECT_DIR}/../.pixi/envs/default" ]; then
    SPDLOG_PREFIX="${PROJECT_DIR}/../.pixi/envs/default"
elif [ -n "$CONDA_PREFIX" ]; then
    SPDLOG_PREFIX="$CONDA_PREFIX"
fi

if [ -n "$SPDLOG_PREFIX" ] && [ -f "${SPDLOG_PREFIX}/lib/cmake/spdlog/spdlogConfig.cmake" ]; then
    echo "  Found spdlog in: ${SPDLOG_PREFIX}"
elif pkg-config --exists spdlog 2>/dev/null; then
    echo "  Found spdlog via pkg-config"
    SPDLOG_PREFIX=""
else
    echo "  WARNING: spdlog not found. Trying to build anyway..."
    SPDLOG_PREFIX=""
fi

# --- 3. Symlink CMakePresets.json ---
echo ""
echo "[3/4] Setting up CMake presets..."
PRESETS_LINK="${PROJECT_DIR}/duckdb/CMakePresets.json"
if [ ! -L "$PRESETS_LINK" ] || [ "$(readlink -f "$PRESETS_LINK")" != "$(readlink -f "${PROJECT_DIR}/cmake/CMakePresets.json")" ]; then
    rm -f "$PRESETS_LINK"
    ln -sf ../cmake/CMakePresets.json "$PRESETS_LINK"
    echo "  Symlinked CMakePresets.json"
else
    echo "  CMakePresets.json already linked"
fi

# --- 4. Build ---
echo ""
echo "[4/4] Building rasterdb extension (${BUILD_PRESET})..."

# Add spdlog prefix to CMAKE_PREFIX_PATH if found
CMAKE_EXTRA=""
if [ -n "$SPDLOG_PREFIX" ]; then
    CMAKE_EXTRA="-DCMAKE_PREFIX_PATH=${SPDLOG_PREFIX}"
fi

cd "${PROJECT_DIR}/duckdb"
cmake --preset "${BUILD_PRESET}" ${CMAKE_EXTRA}
cmake --build --preset "${BUILD_PRESET}" -j$(nproc)

# --- 5. Verify ---
echo ""
echo "============================================"
echo "  Build complete!"
echo "============================================"

# Find the extension .duckdb_extension file
EXT_FILE=""
for candidate in \
    "${PROJECT_DIR}/build/${BUILD_PRESET}/extension/rasterdb/rasterdb.duckdb_extension" \
    "${PROJECT_DIR}/build/${BUILD_PRESET}/rasterdb.duckdb_extension"; do
    if [ -f "$candidate" ]; then
        EXT_FILE="$candidate"
        break
    fi
done

if [ -n "$EXT_FILE" ]; then
    echo "  Extension: ${EXT_FILE} ($(du -h "$EXT_FILE" | cut -f1))"
else
    echo "  Extension built (check build/${BUILD_PRESET}/ for output)"
fi

echo ""
echo "  Usage:"
echo "    export RASTERDF_SHADER_DIR=/usr/local/share/rasterdf/shaders"
echo "    duckdb -unsigned"
echo "    LOAD '${EXT_FILE:-build/${BUILD_PRESET}/.../rasterdb.duckdb_extension}';"
echo "    SELECT * FROM gpu_execution('SELECT sum(a) FROM t');"
