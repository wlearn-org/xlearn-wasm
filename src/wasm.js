// WASM loader -- loads the xLearn WASM module (singleton, lazy init)

import { createRequire } from 'module'

let wasmModule = null
let loading = null

export async function loadXLearn(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const require = createRequire(import.meta.url)
    const createXLearn = require('../wasm/xlearn.cjs')
    wasmModule = await createXLearn(options)
    return wasmModule
  })()

  return loading
}

export function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadXLearn() first')
  return wasmModule
}
