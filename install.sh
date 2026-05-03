#!/bin/bash
# install.sh — Install RasterDB DuckDB extension and its dependencies
#
# This script:
#   1. Checks and installs system dependencies (Vulkan, spdlog, etc.)
#   2. Verifies rasterdf is built
#   3. Initializes DuckDB submodule
#   4. Builds DuckDB
#   5. Builds the RasterDB extension
#   6. Installs the extension to a standard location
#
# Usage: ./install.sh [--release|--debug] [--log-level=LEVEL]

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

BUILD_PRESET="release"
LOG_LEVEL="${RASTERDB_LOG_LEVEL:-info}"
SKIP_DEPS=0

for arg in "$@"; do
    case "$arg" in
        --release) BUILD_PRESET="release" ;;
        --debug)   BUILD_PRESET="debug" ;;
        --log-level=*) LOG_LEVEL="${arg#*=}" ;;
        --skip-deps) SKIP_DEPS=1 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --release          Build in release mode (optimized) [default]"
            echo "  --debug            Build in debug mode (with symbols)"
            echo "  --log-level=LEVEL  Set default log level (debug|info|warn|error|none)"
            echo "  --skip-deps        Skip system dependency installation"
            echo "  --help, -h         Show this help message"
            exit 0
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
echo "  RasterDB Installation Script"
echo "  Preset: ${BUILD_PRESET}"
echo "  Log level: ${LOG_LEVEL}"
echo "============================================"

# --- 1. Check for sudo access ---
if [ "$SKIP_DEPS" -eq 0 ]; then
    if ! sudo -n true 2>/dev/null; then
        echo ""
        echo "This script requires sudo access to install system dependencies."
        echo "Please enter your password when prompted."
    fi
fi

# --- 2. Install system dependencies ---
if [ "$SKIP_DEPS" -eq 0 ]; then
    echo ""
    echo "[1/6] Installing system dependencies..."
    
    # Update package list
    sudo apt update
    
    # Install Vulkan SDK
    echo "  Installing Vulkan SDK..."
    sudo apt install -y vulkan-sdk libvulkan-dev || {
        echo "  WARNING: Failed to install Vulkan SDK via apt."
        echo "  Please install manually from https://vulkan.lunarg.com/sdk/home"
    }
    
    # Install spdlog
    echo "  Installing spdlog..."
    sudo apt install -y libspdlog-dev || {
        echo "  WARNING: Failed to install spdlog via apt."
        echo "  You can install via pixi: pixi install spdlog"
    }
    
    # Install build tools
    echo "  Installing build tools..."
    sudo apt install -y cmake build-essential git
    
    echo "  System dependencies installed."
else
    echo ""
    echo "[1/6] Skipping system dependency installation (--skip-deps)."
fi

# --- 3. Check rasterdf is built ---
echo ""
echo "[2/6] Checking rasterdf library..."
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
    echo "  Please build rasterdf first:"
    echo "    cd ${RASTERDF_ROOT}"
    echo "    ./build.sh --release"
    exit 1
fi
echo "  Found: ${SO_FILE}"

# --- 4. Initialize DuckDB submodule ---
echo ""
echo "[3/6] Initializing DuckDB submodule..."
if [ ! -d "${PROJECT_DIR}/duckdb/.git" ]; then
    git submodule update --init --recursive
    echo "  DuckDB submodule initialized."
else
    echo "  DuckDB submodule already initialized."
fi

# --- 5. Build RasterDB extension ---
echo ""
echo "[4/6] Building RasterDB extension..."
./build.sh --${BUILD_PRESET} --log-level="${LOG_LEVEL}"

# --- 6. Install extension ---
echo ""
echo "[5/6] Installing RasterDB extension..."

# Find the extension file
EXT_FILE=""
for candidate in \
    "${PROJECT_DIR}/build/${BUILD_PRESET}/extension/rasterdb/rasterdb.duckdb_extension" \
    "${PROJECT_DIR}/build/${BUILD_PRESET}/rasterdb.duckdb_extension"; do
    if [ -f "$candidate" ]; then
        EXT_FILE="$candidate"
        break
    fi
done

if [ -z "$EXT_FILE" ]; then
    echo "  ERROR: Extension file not found after build."
    echo "  Check build/${BUILD_PRESET}/ for output."
    exit 1
fi

# Install to standard location
INSTALL_DIR="${HOME}/.duckdb/extensions"
mkdir -p "${INSTALL_DIR}"
cp "${EXT_FILE}" "${INSTALL_DIR}/"
echo "  Extension installed to: ${INSTALL_DIR}/$(basename ${EXT_FILE})"

# --- 7. Set up environment ---
echo ""
echo "[6/6] Setting up environment..."

# Create a helper script for setting up the environment
ENV_SCRIPT="${PROJECT_DIR}/setup_env.sh"
cat > "${ENV_SCRIPT}" << 'EOF'
#!/bin/bash
# Source this file to set up the RasterDB environment
export RASTERDF_SHADER_DIR=/usr/local/share/rasterdf/shaders
export RASTERDB_LOG_LEVEL=info
export SIRIUS_LOG_LEVEL=info
echo "RasterDB environment variables set:"
echo "  RASTERDF_SHADER_DIR=${RASTERDF_SHADER_DIR}"
echo "  RASTERDB_LOG_LEVEL=${RASTERDB_LOG_LEVEL}"
echo "  SIRIUS_LOG_LEVEL=${SIRIUS_LOG_LEVEL}"
EOF
chmod +x "${ENV_SCRIPT}"
echo "  Environment setup script created: ${ENV_SCRIPT}"

# --- Summary ---
echo ""
echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo ""
echo "Extension installed: ${INSTALL_DIR}/$(basename ${EXT_FILE})"
echo ""
echo "To use RasterDB:"
echo "  1. Source the environment script:"
echo "     source ${ENV_SCRIPT}"
echo ""
echo "  2. Start DuckDB:"
echo "     duckdb -unsigned"
echo ""
echo "  3. Load the extension:"
echo "     LOAD '${INSTALL_DIR}/$(basename ${EXT_FILE)}';"
echo ""
echo "  4. Run GPU queries:"
echo "     SELECT * FROM gpu_execution('SELECT sum(a) FROM t');"
echo ""
echo "============================================"
