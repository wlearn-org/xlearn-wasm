#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WASM_FILE="${PROJECT_DIR}/wasm/xlearn.cjs"

if [ ! -f "$WASM_FILE" ]; then
  echo "ERROR: ${WASM_FILE} not found. Run build-wasm.sh first."
  exit 1
fi

EXPECTED_EXPORTS=(
  wl_xl_get_last_error
  wl_xl_create
  wl_xl_free_handle
  wl_xl_set_str
  wl_xl_set_int
  wl_xl_set_float
  wl_xl_set_bool
  wl_xl_create_dmatrix_dense
  wl_xl_create_dmatrix_csr
  wl_xl_free_dmatrix
  wl_xl_fit
  wl_xl_predict
  wl_xl_free_buffer
)

MISSING=0
for fn in "${EXPECTED_EXPORTS[@]}"; do
  if ! grep -q "\"_${fn}\"" "$WASM_FILE"; then
    echo "MISSING: _${fn}"
    MISSING=$((MISSING + 1))
  fi
done

if [ "$MISSING" -gt 0 ]; then
  echo "ERROR: ${MISSING} export(s) missing from WASM build"
  exit 1
fi

echo "All ${#EXPECTED_EXPORTS[@]} exports verified OK"
