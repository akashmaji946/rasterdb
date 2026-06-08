#!/bin/bash
# install.sh - Build and install RasterDB.
#
# This installer is intentionally narrow in scope:
#   1. Verifies RasterDF is already installed and available.
#   2. Builds DuckDB plus the RasterDB extension.
#   3. Installs the DuckDB shell as `rduckdb`.
#   4. Installs the RasterDB extension so `rduckdb` auto-loads it.
#
# RasterDF installation is handled separately by ../rasterdf/install.sh.
#
# Typical usage:
#   ./install.sh
#   sudo ./install.sh
#   sudo ./install.sh --skip-build
#   sudo ./install.sh --debug
#
# After install:
#   which rduckdb
#   rduckdb --version
#   rduckdb my.db

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"

BUILD_PRESET="release"
SKIP_BUILD=0
SKIP_DEPS=0
PREFIX="/usr/local"

RDUCKDB_VERSION="1.0.0"
RDUCKDB_AUTHOR="Akash Maji"
RDUCKDB_CONTACT="akashmaji@iisc.ac.in"
RDUCKDB_LICENSE="MIT"

INSTALL_BIN_DIR="${PREFIX}/bin"
INSTALL_LIB_DIR="${PREFIX}/lib/rasterdb"
PROFILE_SCRIPT="/etc/profile.d/rasterdb.sh"

usage() {
    cat <<EOF
Usage:
  ./install.sh [--debug] [--skip-build] [--skip-deps] [--prefix PATH]

Common workflows:
  ./install.sh
      Build RasterDB in release mode and install it under ${PREFIX}.

  sudo ./install.sh
      Same as above, with permission to write system files.

  sudo ./install.sh --skip-build
      Reuse the existing build outputs and reinstall the system command.

Options:
  --debug         Build/install from the debug preset instead of release.
  --skip-build    Reuse existing build outputs.
  --skip-deps     Skip system dependency checks.
  --prefix PATH   Install under a different prefix. Default: ${PREFIX}
  --help, -h      Show this help message.

Notes:
  - RasterDF must already be installed separately.
  - This script installs the command as 'rduckdb'.
  - 'rduckdb' auto-loads the RasterDB extension on startup.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --debug)
            BUILD_PRESET="debug"
            shift
            ;;
        --release)
            BUILD_PRESET="release"
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=1
            shift
            ;;
        --prefix)
            if [ $# -lt 2 ]; then
                echo "ERROR: --prefix requires a value"
                exit 1
            fi
            PREFIX="$2"
            INSTALL_BIN_DIR="${PREFIX}/bin"
            INSTALL_LIB_DIR="${PREFIX}/lib/rasterdb"
            shift 2
            ;;
        --prefix=*)
            PREFIX="${1#*=}"
            INSTALL_BIN_DIR="${PREFIX}/bin"
            INSTALL_LIB_DIR="${PREFIX}/lib/rasterdb"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  RasterDB Installer"
echo "============================================"
echo ""
echo "Project root : ${PROJECT_DIR}"
echo "Build preset : ${BUILD_PRESET}"
echo "Install root : ${PREFIX}"

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        echo "  OK: $1 ($(command -v "$1"))"
        return 0
    fi
    echo "  MISSING: $1"
    return 1
}

check_apt_package() {
    local package="$1"
    if command -v dpkg >/dev/null 2>&1 && dpkg -s "$package" >/dev/null 2>&1; then
        echo "  OK: ${package} already installed"
        return 0
    fi
    echo "  MISSING: ${package}"
    return 1
}

if [ "$SKIP_DEPS" -eq 0 ]; then
    echo ""
    echo "[1/6] Checking system dependencies..."
    DEPS_OK=1
    check_apt_package "build-essential" || DEPS_OK=0
    check_command "cmake" || DEPS_OK=0
    check_command "git" || DEPS_OK=0
    check_apt_package "libspdlog-dev" || DEPS_OK=0
    check_apt_package "libvulkan-dev" || DEPS_OK=0
    check_command "glslc" || DEPS_OK=0
    if [ "$DEPS_OK" -eq 1 ]; then
        echo "  All required dependencies are available."
    else
        echo ""
        echo "  Some build dependencies are missing."
        echo "  Install them manually, for example on Ubuntu/Debian:"
        echo "    sudo apt install build-essential cmake git libspdlog-dev libvulkan-dev glslang-tools"
        exit 1
    fi
else
    echo ""
    echo "[1/6] Skipping dependency checks (--skip-deps)."
fi

echo ""
echo "[2/6] Checking RasterDF prerequisite..."
RASTERDF_LIB="/usr/local/lib/librasterdf.so"
RASTERDF_SHADERS="/usr/local/share/rasterdf/shaders"
if [ ! -f "${RASTERDF_LIB}" ]; then
    echo "ERROR: ${RASTERDF_LIB} not found."
    echo "Please install RasterDF first:"
    echo "  cd ../rasterdf"
    echo "  sudo ./install.sh --system"
    exit 1
fi
if [ ! -d "${RASTERDF_SHADERS}" ]; then
    echo "ERROR: ${RASTERDF_SHADERS} not found."
    echo "Please install RasterDF shaders first:"
    echo "  cd ../rasterdf"
    echo "  sudo ./install.sh --system"
    exit 1
fi
echo "  Found RasterDF library : ${RASTERDF_LIB}"
echo "  Found RasterDF shaders : ${RASTERDF_SHADERS}"

echo ""
echo "[3/6] Initializing DuckDB submodule..."
if [ ! -d "${PROJECT_DIR}/duckdb/.git" ]; then
    git submodule update --init --recursive
    echo "  DuckDB submodule initialized."
else
    echo "  DuckDB submodule already initialized."
fi

if [ "$SKIP_BUILD" -eq 0 ]; then
    echo ""
    echo "[4/6] Building RasterDB..."
    ./build.sh "--${BUILD_PRESET}"
else
    echo ""
    echo "[4/6] Skipping build (--skip-build)."
fi

DUCKDB_BIN="${PROJECT_DIR}/build/${BUILD_PRESET}/duckdb"
EXT_FILE="${PROJECT_DIR}/build/${BUILD_PRESET}/extension/rasterdb/rasterdb.duckdb_extension"

if [ ! -x "${DUCKDB_BIN}" ]; then
    echo "ERROR: built DuckDB binary not found at ${DUCKDB_BIN}"
    exit 1
fi
if [ ! -f "${EXT_FILE}" ]; then
    echo "ERROR: built RasterDB extension not found at ${EXT_FILE}"
    exit 1
fi

echo ""
echo "[5/6] Installing system files..."
sudo mkdir -p "${INSTALL_BIN_DIR}" "${INSTALL_LIB_DIR}"
sudo cp -f "${DUCKDB_BIN}" "${INSTALL_LIB_DIR}/duckdb"
sudo cp -f "${EXT_FILE}" "${INSTALL_LIB_DIR}/rasterdb.duckdb_extension"

TMP_WRAPPER="$(mktemp)"
cat > "${TMP_WRAPPER}" <<EOF
#!/bin/bash
set -euo pipefail

RDUCKDB_VERSION="${RDUCKDB_VERSION}"
RDUCKDB_AUTHOR="${RDUCKDB_AUTHOR}"
RDUCKDB_CONTACT="${RDUCKDB_CONTACT}"
RDUCKDB_LICENSE="${RDUCKDB_LICENSE}"
RDUCKDB_BIN="${INSTALL_LIB_DIR}/duckdb"
RDUCKDB_EXT="${INSTALL_LIB_DIR}/rasterdb.duckdb_extension"

case "\${1:-}" in
    --version)
        printf '%s\n' "\${RDUCKDB_VERSION}"
        exit 0
        ;;
    --author)
        printf '%s\n' "\${RDUCKDB_AUTHOR}"
        exit 0
        ;;
    --contact)
        printf '%s\n' "\${RDUCKDB_CONTACT}"
        exit 0
        ;;
    --license)
        printf '%s\n' "\${RDUCKDB_LICENSE}"
        exit 0
        ;;
esac

export RASTERDF_SHADER_DIR="\${RASTERDF_SHADER_DIR:-${RASTERDF_SHADERS}}"
export RASTERDB_LOG_LEVEL="\${RASTERDB_LOG_LEVEL:-info}"

exec "\${RDUCKDB_BIN}" -unsigned -cmd "LOAD '\${RDUCKDB_EXT}';" "\$@"
EOF
sudo install -m 0755 "${TMP_WRAPPER}" "${INSTALL_BIN_DIR}/rduckdb"
rm -f "${TMP_WRAPPER}"

TMP_PROFILE="$(mktemp)"
cat > "${TMP_PROFILE}" <<EOF
# RasterDB runtime defaults
export RASTERDF_SHADER_DIR="${RASTERDF_SHADERS}"
export RASTERDB_LOG_LEVEL="\${RASTERDB_LOG_LEVEL:-info}"
EOF
sudo install -m 0644 "${TMP_PROFILE}" "${PROFILE_SCRIPT}"
rm -f "${TMP_PROFILE}"

echo "  Installed shell    : ${INSTALL_BIN_DIR}/rduckdb"
echo "  Installed DuckDB   : ${INSTALL_LIB_DIR}/duckdb"
echo "  Installed extension: ${INSTALL_LIB_DIR}/rasterdb.duckdb_extension"
echo "  Runtime profile    : ${PROFILE_SCRIPT}"

echo ""
echo "[6/6] Verifying installation..."
if command -v "${INSTALL_BIN_DIR}/rduckdb" >/dev/null 2>&1; then
    echo "  which rduckdb      : ${INSTALL_BIN_DIR}/rduckdb"
fi
echo "  rduckdb --version  : $("${INSTALL_BIN_DIR}/rduckdb" --version)"
echo "  rduckdb --author   : $("${INSTALL_BIN_DIR}/rduckdb" --author)"
echo "  rduckdb --contact  : $("${INSTALL_BIN_DIR}/rduckdb" --contact)"
echo "  rduckdb --license  : $("${INSTALL_BIN_DIR}/rduckdb" --license)"

echo ""
echo "============================================"
echo "  Install complete"
echo "============================================"
echo ""
echo "Try:"
echo "  which rduckdb"
echo "  rduckdb --version"
echo "  rduckdb"
echo "  rduckdb my.db"
echo ""
echo "Note:"
echo "  RasterDF remains a separate install managed by ../rasterdf/install.sh"
