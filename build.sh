#!/bin/bash
# build.sh — Build the RasterDB DuckDB extension
#
# Prerequisites:
#   - rasterdf must be built first (../rasterdf/build.sh --release)
#   - spdlog must be installed (apt install libspdlog-dev or via conda)
#   - Vulkan SDK / headers must be available
#
# Usage: ./build.sh [--release|--debug|--clean] [--log-level=debug|info|warn|error|none]
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

BUILD_PRESET="release"
CLEAN=0
LOG_LEVEL="${RASTERDB_LOG_LEVEL:-info}"
for arg in "$@"; do
    case "$arg" in
        --release) BUILD_PRESET="release" ;;
        --debug)   BUILD_PRESET="debug" ;;
        --clean)   CLEAN=1 ;;
        --log-level=*) LOG_LEVEL="${arg#*=}" ;;
        --log-level)
            echo "ERROR: --log-level requires a value, e.g. --log-level=debug"
            exit 1
            ;;
        --trace|--debug-log|--info|--warn|--error|--none)
            LOG_LEVEL="${arg#--}"
            [ "$LOG_LEVEL" = "debug-log" ] && LOG_LEVEL="debug"
            ;;
    esac
done

LOG_LEVEL="${LOG_LEVEL,,}"
case "$LOG_LEVEL" in
    trace|debug|info|warn|warning|error|err|critical|fatal|none|off) ;;
    *)
        echo "ERROR: invalid log level '${LOG_LEVEL}'"
        echo "       expected trace, debug, info, warn, error, critical, or none"
        exit 1
        ;;
esac

echo "============================================"
echo "  RasterDB — DuckDB Extension Build"
echo "  Preset: ${BUILD_PRESET}"
echo "  Default log level: ${LOG_LEVEL}"
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
for candidate in "${RASTERDF_ROOT}/build/librasterdf.so" \
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

# Add spdlog prefix to CMAKE_PREFIX_PATH if found.
# Keep -D arguments before --preset; this CMake version otherwise lets the
# preset/default cache value win during DuckDB's configure step.
CMAKE_ARGS=("-DRASTERDB_LOG_LEVEL=${LOG_LEVEL}")
if [ -n "$SPDLOG_PREFIX" ]; then
    CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${SPDLOG_PREFIX}")
fi

cd "${PROJECT_DIR}/duckdb"
cmake "${CMAKE_ARGS[@]}" --preset "${BUILD_PRESET}"
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

echo "============================================"
echo "BUILD SUCCESS"
echo "============================================"
