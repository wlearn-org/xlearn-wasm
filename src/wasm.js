// WASM loader -- loads the xLearn WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadXLearn(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const createXLearn = require('../wasm/xlearn.js')
    wasmModule = await createXLearn(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadXLearn() first')
  return wasmModule
}

module.exports = { loadXLearn, getWasm }
