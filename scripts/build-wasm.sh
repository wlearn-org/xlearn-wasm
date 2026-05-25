#!/bin/bash
set -euo pipefail

# Build xLearn as WASM via Emscripten
# Prerequisites: emsdk activated (emcc in PATH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPSTREAM_DIR="${PROJECT_DIR}/upstream/xlearn"
OUTPUT_DIR="${PROJECT_DIR}/wasm"
PATCH_DIR="${PROJECT_DIR}/build/xlearn-patched"

# Verify prerequisites
if ! command -v em++ &> /dev/null; then
  echo "ERROR: em++ not found. Activate emsdk first:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

if [ ! -f "$UPSTREAM_DIR/src/c_api/c_api.h" ]; then
  echo "ERROR: xLearn upstream not found at ${UPSTREAM_DIR}"
  echo "  git submodule update --init"
  exit 1
fi

echo "=== Applying patches ==="

# Patch copies under build/ so the upstream submodule stays clean.
mkdir -p "${PATCH_DIR}/src/solver" "${PATCH_DIR}/src/base"
cp "${UPSTREAM_DIR}/src/solver/solver.cc" "${PATCH_DIR}/src/solver/solver.cc"
cp "${UPSTREAM_DIR}/src/solver/checker.cc" "${PATCH_DIR}/src/solver/checker.cc"
cp "${UPSTREAM_DIR}/src/base/file_util.h" "${PATCH_DIR}/src/base/file_util.h"

# Patch 1: Replace exit() with throw in solver.cc and checker.cc
# (exit() would kill the WASM process; throw gets caught by API_BEGIN/API_END)
if grep -q 'exit(0)' "${PATCH_DIR}/src/solver/solver.cc" 2>/dev/null; then
  echo "  Patching exit() -> throw in solver.cc"
  sed -i 's/exit(0)/throw std::runtime_error("xLearn parameter check failed")/g' "${PATCH_DIR}/src/solver/solver.cc"
  sed -i 's/exit(1)/throw std::runtime_error("xLearn argument error")/g' "${PATCH_DIR}/src/solver/solver.cc"
fi
if grep -q 'exit(0)' "${PATCH_DIR}/src/solver/checker.cc" 2>/dev/null; then
  echo "  Patching exit() -> throw in checker.cc"
  sed -i 's/exit(0)/throw std::runtime_error("xLearn checker failed")/g' "${PATCH_DIR}/src/solver/checker.cc"
fi

# Patch 2: Fix std::min type mismatch in file_util.h (uint32 vs long)
if grep -q 'pos + kChunkSize, end' "${PATCH_DIR}/src/base/file_util.h" 2>/dev/null; then
  echo "  Patching std::min type mismatch in file_util.h"
  sed -i 's/std::min(pos + kChunkSize, end)/std::min(pos + (long)kChunkSize, end)/g' "${PATCH_DIR}/src/base/file_util.h"
fi

echo "=== Compiling WASM ==="
mkdir -p "$OUTPUT_DIR"

# Source files (exclude main() files and distributed code)
SOURCES=(
  "${PROJECT_DIR}/csrc/wl_api.cpp"
  "${UPSTREAM_DIR}/src/base/format_print.cc"
  "${UPSTREAM_DIR}/src/base/levenshtein_distance.cc"
  "${UPSTREAM_DIR}/src/base/logging.cc"
  "${UPSTREAM_DIR}/src/base/split_string.cc"
  "${UPSTREAM_DIR}/src/base/stringprintf.cc"
  "${UPSTREAM_DIR}/src/base/timer.cc"
  "${UPSTREAM_DIR}/src/c_api/c_api.cc"
  "${UPSTREAM_DIR}/src/c_api/c_api_error.cc"
  "${UPSTREAM_DIR}/src/data/model_parameters.cc"
  "${UPSTREAM_DIR}/src/loss/cross_entropy_loss.cc"
  "${UPSTREAM_DIR}/src/loss/loss.cc"
  "${UPSTREAM_DIR}/src/loss/metric.cc"
  "${UPSTREAM_DIR}/src/loss/squared_loss.cc"
  "${UPSTREAM_DIR}/src/reader/file_splitor.cc"
  "${UPSTREAM_DIR}/src/reader/parser.cc"
  "${UPSTREAM_DIR}/src/reader/reader.cc"
  "${UPSTREAM_DIR}/src/score/ffm_score.cc"
  "${UPSTREAM_DIR}/src/score/fm_score.cc"
  "${UPSTREAM_DIR}/src/score/linear_score.cc"
  "${UPSTREAM_DIR}/src/score/score_function.cc"
  "${PATCH_DIR}/src/solver/checker.cc"
  "${UPSTREAM_DIR}/src/solver/inference.cc"
  "${PATCH_DIR}/src/solver/solver.cc"
  "${UPSTREAM_DIR}/src/solver/trainer.cc"
)

EXPORTED_FUNCTIONS='["_wl_xl_get_last_error","_wl_xl_create","_wl_xl_free_handle","_wl_xl_set_str","_wl_xl_set_int","_wl_xl_set_float","_wl_xl_set_bool","_wl_xl_create_dmatrix_dense","_wl_xl_create_dmatrix_csr","_wl_xl_free_dmatrix","_wl_xl_fit","_wl_xl_predict","_wl_xl_free_buffer","_malloc","_free"]'

EXPORTED_RUNTIME_METHODS='["ccall","getValue","setValue","HEAPF32","HEAPU8"]'

em++ \
  "${SOURCES[@]}" \
  -I "${PROJECT_DIR}/csrc" \
  -I "${PATCH_DIR}" \
  -I "${PATCH_DIR}/src" \
  -I "${UPSTREAM_DIR}" \
  -I "${UPSTREAM_DIR}/src" \
  -include "${PROJECT_DIR}/csrc/thread_pool_wasm.h" \
  -o "${OUTPUT_DIR}/xlearn.js" \
  -std=c++11 \
  -msimd128 -msse3 \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s SINGLE_FILE_BINARY_ENCODE=0 \
  -s EXPORT_NAME=createXLearn \
  -s FORCE_FILESYSTEM=1 \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 \
  -s ENVIRONMENT='web,node' \
  -Wno-deprecated-register \
  -Wno-sign-compare \
  -Wno-unused-variable \
  -Wno-unused-but-set-variable \
  -O2

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: xlearn v0.44
upstream_commit: $(cd "$UPSTREAM_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(em++ --version | head -1)
build_flags: -O2 SINGLE_FILE=1 sequential-threadpool
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/xlearn.js"
cat "${OUTPUT_DIR}/BUILD_INFO"
