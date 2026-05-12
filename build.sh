#!/usr/bin/env bash
# build.sh — Build the Sirius DuckDB extension using pixi
# Usage:
#   ./build.sh              # incremental release build (only sirius extension)
#   ./build.sh release      # same as above
#   ./build.sh full         # full release build (all targets)
#   ./build.sh configure    # reconfigure CMake (release preset)
#   ./build.sh clean        # remove build directory and rebuild
#   ./build.sh debug        # incremental debug build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIXI="${HOME}/.pixi/bin/pixi"
BUILD_TYPE="${1:-release}"

if [[ ! -x "$PIXI" ]]; then
    echo "ERROR: pixi not found at $PIXI"
    echo "Install pixi: curl -fsSL https://pixi.sh/install.sh | bash"
    exit 1
fi

# Number of parallel jobs
JOBS="$(nproc)"

case "$BUILD_TYPE" in
    configure)
        echo "=== Configuring CMake (release preset) ==="
        cd "$SCRIPT_DIR"
        $PIXI run -e default cmake --preset release
        ;;
    clean)
        echo "=== Cleaning build directory ==="
        rm -rf "$SCRIPT_DIR/build"
        echo "=== Configuring CMake (release preset) ==="
        cd "$SCRIPT_DIR/duckdb"
        $PIXI run -e default cmake --preset release
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
            $PIXI run -e default cmake --preset release
        fi
        cd "$SCRIPT_DIR/duckdb"
        $PIXI run -e default cmake --build --preset release -j"$JOBS"
        ;;
    debug)
        echo "=== Debug build (sirius extension) ==="
        if [[ ! -f "$SCRIPT_DIR/build/debug/build.ninja" ]]; then
            echo "--- Configuring CMake (debug preset) ---"
            cd "$SCRIPT_DIR/duckdb"
            $PIXI run -e default cmake --preset debug
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
            $PIXI run -e default cmake --preset release
        fi
        cd "$SCRIPT_DIR/duckdb"
        $PIXI run -e default cmake --build --preset release --target sirius_loadable_extension -j"$JOBS"
        echo ""
        echo "=== Build complete ==="
        echo "Extension: $SCRIPT_DIR/build/release/extension/sirius/sirius.duckdb_extension"
        ;;
    *)
        echo "Unknown build type: $BUILD_TYPE"
        echo "Usage: $0 [release|full|debug|configure|clean]"
        exit 1
        ;;
esac
