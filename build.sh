#!/usr/bin/env bash
# build.sh — Build the Sirius DuckDB extension using pixi
# Usage:
#   ./build.sh              # incremental release build (only sirius extension)
#   ./build.sh release      # same as above
#   ./build.sh full         # full release build (all targets)
#   ./build.sh configure    # reconfigure CMake (release preset)
#   ./build.sh clean        # remove build directory and rebuild
#   ./build.sh debug        # incremental debug build
#   ./build.sh --use-compiler-launcher release
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIXI="${HOME}/.pixi/bin/pixi"
USE_COMPILER_LAUNCHER=0

if [[ "${1:-}" == "--use-compiler-launcher" ]]; then
    USE_COMPILER_LAUNCHER=1
    shift
fi

BUILD_TYPE="${1:-release}"

if [[ ! -x "$PIXI" ]]; then
    echo "ERROR: pixi not found at $PIXI"
    echo "Install pixi: curl -fsSL https://pixi.sh/install.sh | bash"
    exit 1
fi

# Number of parallel jobs
JOBS="$(nproc)"
TMP_ROOT="$SCRIPT_DIR/build/tmp"

mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
export TMP="$TMP_ROOT"
export TEMP="$TMP_ROOT"

CONFIGURE_ARGS=()
if [[ "$USE_COMPILER_LAUNCHER" -eq 0 ]]; then
    CONFIGURE_ARGS+=(
        -DCMAKE_C_COMPILER_LAUNCHER=
        -DCMAKE_CXX_COMPILER_LAUNCHER=
        -DCMAKE_CUDA_COMPILER_LAUNCHER=
    )
fi

case "$BUILD_TYPE" in
    configure)
        echo "=== Configuring CMake (release preset) ==="
        cd "$SCRIPT_DIR/duckdb"
        $PIXI run -e default cmake --preset release "${CONFIGURE_ARGS[@]}"
        ;;
    clean)
        echo "=== Cleaning build directory ==="
        rm -rf "$SCRIPT_DIR/build"
        mkdir -p "$TMP_ROOT"
        echo "=== Configuring CMake (release preset) ==="
        cd "$SCRIPT_DIR/duckdb"
        $PIXI run -e default cmake --preset release "${CONFIGURE_ARGS[@]}"
        echo "=== Building sirius extension ==="
        cd "$SCRIPT_DIR/duckdb"
        $PIXI run -e default cmake --build --preset release --target sirius_loadable_extension -j"$JOBS"
        ;;
    full)
        echo "=== Full release build (all targets) ==="
        # Ensure configured
        if [[ ! -f "$SCRIPT_DIR/build/release/build.ninja" ]]; then
            echo "--- Configuring CMake (release preset) ---"
            cd "$SCRIPT_DIR/duckdb"
            $PIXI run -e default cmake --preset release "${CONFIGURE_ARGS[@]}"
        fi
        cd "$SCRIPT_DIR/duckdb"
        $PIXI run -e default cmake --build --preset release -j"$JOBS"
        ;;
    debug)
        echo "=== Debug build (sirius extension) ==="
        if [[ ! -f "$SCRIPT_DIR/build/debug/build.ninja" ]]; then
            echo "--- Configuring CMake (debug preset) ---"
            cd "$SCRIPT_DIR/duckdb"
            $PIXI run -e default cmake --preset debug "${CONFIGURE_ARGS[@]}"
        fi
        cd "$SCRIPT_DIR/duckdb"
        $PIXI run -e default cmake --build --preset debug --target sirius_loadable_extension -j"$JOBS"
        echo ""
        echo "=== Debug build complete ==="
        echo "Extension: $SCRIPT_DIR/build/debug/extension/sirius/sirius.duckdb_extension"
        ;;
    release|"")
        echo "=== Incremental release build (sirius extension) ==="
        # Ensure configured
        if [[ ! -f "$SCRIPT_DIR/build/release/build.ninja" ]]; then
            echo "--- Configuring CMake (release preset) ---"
            cd "$SCRIPT_DIR/duckdb"
            $PIXI run -e default cmake --preset release "${CONFIGURE_ARGS[@]}"
        fi
        cd "$SCRIPT_DIR/duckdb"
        $PIXI run -e default cmake --build --preset release --target sirius_loadable_extension -j"$JOBS"
        echo ""
        echo "=== Build complete ==="
        echo "Extension: $SCRIPT_DIR/build/release/extension/sirius/sirius.duckdb_extension"
        ;;
    *)
        echo "Unknown build type: $BUILD_TYPE"
        echo "Usage: $0 [--use-compiler-launcher] [release|full|debug|configure|clean]"
        exit 1
        ;;
esac
