import { getWasm, loadXLearn } from './wasm.js'
import {
  normalizeX, normalizeY,
  encodeBundle, decodeBundle,
  register,
  DisposedError, NotFittedError
} from '@wlearn/core'

// FinalizationRegistry safety net
const leakRegistry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ref, freeFn }) => {
    if (ref[0]) {
      console.warn('@wlearn/xlearn: Model was not disposed -- calling free() automatically. This is a bug in your code.')
      freeFn(ref[0])
    }
  })
  : null

// Internal sentinel for load path
const LOAD_SENTINEL = Symbol('load')

// Helper: C string allocation
function withCString(wasm, str, fn) {
  const bytes = new TextEncoder().encode(str + '\0')
  const ptr = wasm._malloc(bytes.length)
  wasm.HEAPU8.set(bytes, ptr)
  try {
    return fn(ptr)
  } finally {
    wasm._free(ptr)
  }
}

function getLastError() {
  return getWasm().ccall('wl_xl_get_last_error', 'string', [], [])
}

// Detect CSR matrix: has indices + indptr arrays
function isCSR(X) {
  return X && typeof X === 'object' && !Array.isArray(X)
    && X.indices instanceof Int32Array
    && X.indptr instanceof Int32Array
}

// Sigmoid function
function sigmoid(x) {
  if (x >= 0) {
    const e = Math.exp(-x)
    return 1 / (1 + e)
  }
  const e = Math.exp(x)
  return e / (1 + e)
}

// --- XLearnBase ---

export class XLearnBase {
  #handle = null
  #handleRef = null
  #modelBytes = null
  #params = {}
  #algo = ''
  #task = ''
  #nFeatures = 0
  #nClasses = 0
  #classes = null
  #featureFields = null
  #fitted = false
  #freed = false

  constructor(sentinel, algo, task, params) {
    if (sentinel === LOAD_SENTINEL) {
      // Load path: algo/task/params set by _fromBundle
      this.#algo = algo
      this.#task = task
      this.#params = params || {}
      this.#fitted = false // set to true after model data is assigned
    } else {
      // Normal create path: sentinel=algo, algo=task, task=params
      this.#algo = sentinel
      this.#task = algo
      this.#params = task || {}
    }
    this.#freed = false
  }

  static async _create(algo, task, params, TypeClass) {
    await loadXLearn()
    return new TypeClass(algo, task, params)
  }

  // --- Estimator interface ---

  fit(X, y) {
    this.#ensureNotDisposed()
    const wasm = getWasm()

    // Dispose previous handle if refitting
    if (this.#handle) {
      wasm._wl_xl_free_handle(this.#handle)
      this.#handle = null
      if (this.#handleRef) this.#handleRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }
    this.#modelBytes = null
    this.#fitted = false

    // Normalize labels
    const yNorm = normalizeY(y)
    const yF64 = yNorm instanceof Float64Array ? yNorm : new Float64Array(yNorm)

    // Build DMatrix (CSR or dense)
    let dmatrix, rows, cols
    if (isCSR(X)) {
      ({ dmatrix, rows, cols } = this.#buildCSRDMatrix(wasm, X, yF64))
    } else {
      ({ dmatrix, rows, cols } = this.#buildDenseDMatrix(wasm, X, yF64))
    }

    if (yF64.length !== rows) {
      wasm._wl_xl_free_dmatrix(dmatrix)
      throw new Error(`y length (${yF64.length}) does not match X rows (${rows})`)
    }

    this.#nFeatures = cols

    // Detect classes for classifier
    if (this.#task === 'binary') {
      const classSet = new Set()
      for (let i = 0; i < yF64.length; i++) classSet.add(yF64[i])
      const sorted = [...classSet].sort((a, b) => a - b)
      this.#nClasses = sorted.length
      this.#classes = new Int32Array(sorted)
    }

    // Create xLearn handle
    const handlePtr = wasm._malloc(4)
    const algo = this.#algo
    const ret = withCString(wasm, algo, (algoCStr) => {
      return wasm._wl_xl_create(algoCStr, handlePtr)
    })

    if (ret !== 0) {
      wasm._free(handlePtr)
      wasm._wl_xl_free_dmatrix(dmatrix)
      throw new Error(`Create failed: ${getLastError()}`)
    }

    const handle = wasm.getValue(handlePtr, 'i32')
    wasm._free(handlePtr)

    // Set task
    const taskStr = this.#task === 'binary' ? 'binary' : 'reg'
    withCString(wasm, 'task', (kPtr) => {
      withCString(wasm, taskStr, (vPtr) => {
        wasm._wl_xl_set_str(handle, kPtr, vPtr)
      })
    })

    // Set parameters
    this.#applyParams(wasm, handle)

    // Train
    const modelBufPtr = wasm._malloc(4)
    const modelLenPtr = wasm._malloc(4)

    const fitRet = wasm._wl_xl_fit(handle, dmatrix, 0, modelBufPtr, modelLenPtr)

    wasm._wl_xl_free_dmatrix(dmatrix)

    if (fitRet !== 0) {
      wasm._free(modelBufPtr)
      wasm._free(modelLenPtr)
      wasm._wl_xl_free_handle(handle)
      throw new Error(`Fit failed: ${getLastError()}`)
    }

    const modelBuf = wasm.getValue(modelBufPtr, 'i32')
    const modelLen = wasm.getValue(modelLenPtr, 'i32')
    wasm._free(modelBufPtr)
    wasm._free(modelLenPtr)

    // Copy model bytes to JS
    this.#modelBytes = new Uint8Array(modelLen)
    this.#modelBytes.set(wasm.HEAPU8.subarray(modelBuf, modelBuf + modelLen))
    wasm._wl_xl_free_buffer(modelBuf)

    // Keep handle for prediction
    this.#handle = handle
    this.#fitted = true

    this.#handleRef = [this.#handle]
    if (leakRegistry) {
      leakRegistry.register(this, {
        ref: this.#handleRef,
        freeFn: (h) => { try { getWasm()._wl_xl_free_handle(h) } catch {} }
      }, this)
    }

    return this
  }

  predict(X) {
    this.#ensureFitted()
    return this.#rawPredict(X)
  }

  predictProba(X) {
    this.#ensureFitted()
    if (this.#task !== 'binary') {
      throw new Error('predictProba is only available for classifiers')
    }

    const margins = this.#rawPredict(X)
    const n = margins.length
    const proba = new Float64Array(n * 2)
    for (let i = 0; i < n; i++) {
      const p1 = sigmoid(margins[i])
      proba[i * 2] = 1 - p1
      proba[i * 2 + 1] = p1
    }
    return proba
  }

  decisionFunction(X) {
    this.#ensureFitted()
    return this.#rawPredict(X)
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = normalizeY(y)

    if (this.#task === 'binary') {
      // Accuracy (apply threshold to raw margins for classifier)
      let correct = 0
      for (let i = 0; i < preds.length; i++) {
        const predClass = preds[i] > 0 ? 1 : 0
        const trueClass = yArr[i] > 0 ? 1 : 0
        if (predClass === trueClass) correct++
      }
      return correct / preds.length
    } else {
      // R-squared
      let ssRes = 0, ssTot = 0, yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    }
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()

    const artifacts = [
      { id: 'model', data: this.#modelBytes }
    ]

    // FFM field map
    if (this.#algo === 'ffm' && this.#featureFields) {
      const fieldBytes = new Uint8Array(this.#featureFields.buffer,
        this.#featureFields.byteOffset, this.#featureFields.byteLength)
      artifacts.push({ id: 'field_map', data: fieldBytes })
    }

    const metadata = {
      algo: this.#algo,
      task: this.#task,
      nFeatures: this.#nFeatures,
      nClasses: this.#nClasses,
      classes: this.#classes ? Array.from(this.#classes) : null
    }

    return encodeBundle(
      { typeId: this._typeId, params: this.getParams(), metadata },
      artifacts
    )
  }

  static async _load(bytes, TypeClass) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return TypeClass._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs, TypeClass) {
    await loadXLearn()

    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const modelData = new Uint8Array(entry.length)
    modelData.set(blobs.subarray(entry.offset, entry.offset + entry.length))

    const meta = manifest.metadata || {}
    const params = manifest.params || {}

    const instance = new TypeClass(LOAD_SENTINEL, meta.algo, meta.task, params)
    instance.#modelBytes = modelData
    instance.#nFeatures = meta.nFeatures || 0
    instance.#nClasses = meta.nClasses || 0
    instance.#classes = meta.classes ? new Int32Array(meta.classes) : null

    // Load field_map if present
    const fieldEntry = toc.find(e => e.id === 'field_map')
    if (fieldEntry) {
      const raw = blobs.subarray(fieldEntry.offset, fieldEntry.offset + fieldEntry.length)
      instance.#featureFields = new Int32Array(raw.buffer.slice(
        raw.byteOffset, raw.byteOffset + raw.byteLength
      ))
      if (params.featureFields === undefined) {
        params.featureFields = instance.#featureFields
      }
    }

    // Create handle for prediction
    const wasm = getWasm()
    const handlePtr = wasm._malloc(4)
    const ret = withCString(wasm, meta.algo || 'fm', (algoCStr) => {
      return wasm._wl_xl_create(algoCStr, handlePtr)
    })

    if (ret !== 0) {
      wasm._free(handlePtr)
      throw new Error(`Create failed: ${getLastError()}`)
    }

    instance.#handle = wasm.getValue(handlePtr, 'i32')
    wasm._free(handlePtr)

    // Set task
    const taskStr = meta.task === 'binary' ? 'binary' : 'reg'
    withCString(wasm, 'task', (kPtr) => {
      withCString(wasm, taskStr, (vPtr) => {
        wasm._wl_xl_set_str(instance.#handle, kPtr, vPtr)
      })
    })

    instance.#fitted = true

    instance.#handleRef = [instance.#handle]
    if (leakRegistry) {
      leakRegistry.register(instance, {
        ref: instance.#handleRef,
        freeFn: (h) => { try { getWasm()._wl_xl_free_handle(h) } catch {} }
      }, instance)
    }

    return instance
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#handle) {
      const wasm = getWasm()
      wasm._wl_xl_free_handle(this.#handle)
    }

    if (this.#handleRef) this.#handleRef[0] = null
    if (leakRegistry) leakRegistry.unregister(this)

    this.#handle = null
    this.#modelBytes = null
    this.#fitted = false
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    if (p.featureFields !== undefined) {
      this.#featureFields = p.featureFields
    }
    return this
  }

  // --- Inspection ---

  get isFitted() {
    return this.#fitted && !this.#freed
  }

  get nFeatures() {
    return this.#nFeatures
  }

  get nClasses() {
    return this.#nClasses
  }

  get classes() {
    return this.#classes ? new Int32Array(this.#classes) : null
  }

  get _typeId() {
    throw new Error('Subclass must implement _typeId')
  }

  // --- Private helpers ---

  #rawPredict(X) {
    const wasm = getWasm()

    // Build DMatrix for query
    let dmatrix, rows
    if (isCSR(X)) {
      ({ dmatrix, rows } = this.#buildCSRDMatrix(wasm, X, null))
    } else {
      ({ dmatrix, rows } = this.#buildDenseDMatrix(wasm, X, null))
    }

    // Write model bytes to WASM heap
    const modelPtr = wasm._malloc(this.#modelBytes.length)
    wasm.HEAPU8.set(this.#modelBytes, modelPtr)

    const outPredsPtr = wasm._malloc(4)
    const outLenPtr = wasm._malloc(4)

    const ret = wasm._wl_xl_predict(
      this.#handle, modelPtr, this.#modelBytes.length,
      dmatrix, outPredsPtr, outLenPtr
    )

    wasm._free(modelPtr)
    wasm._wl_xl_free_dmatrix(dmatrix)

    if (ret !== 0) {
      wasm._free(outPredsPtr)
      wasm._free(outLenPtr)
      throw new Error(`Predict failed: ${getLastError()}`)
    }

    const predsPtr = wasm.getValue(outPredsPtr, 'i32')
    const predsLen = wasm.getValue(outLenPtr, 'i32')
    wasm._free(outPredsPtr)
    wasm._free(outLenPtr)

    // Copy float predictions to Float64Array
    const result = new Float64Array(predsLen)
    for (let i = 0; i < predsLen; i++) {
      result[i] = wasm.HEAPF32[predsPtr / 4 + i]
    }
    wasm._wl_xl_free_buffer(predsPtr)

    return result
  }

  #buildDenseDMatrix(wasm, X, y) {
    const { data: xData, rows, cols } = normalizeX(X)

    // xLearn uses float32 internally
    const xF32 = new Float32Array(xData.length)
    for (let i = 0; i < xData.length; i++) xF32[i] = xData[i]

    const xPtr = wasm._malloc(xF32.length * 4)
    wasm.HEAPF32.set(xF32, xPtr / 4)

    // Labels: remap {0,1} -> {-1,+1} for binary classification
    let yPtr = 0
    if (y) {
      const yF32 = new Float32Array(y.length)
      if (this.#task === 'binary') {
        for (let i = 0; i < y.length; i++) yF32[i] = y[i] > 0 ? 1 : -1
      } else {
        for (let i = 0; i < y.length; i++) yF32[i] = y[i]
      }
      yPtr = wasm._malloc(yF32.length * 4)
      wasm.HEAPF32.set(yF32, yPtr / 4)
    }

    // Field map for FFM
    let fieldPtr = 0
    const featureFields = this.#params.featureFields || this.#featureFields
    if (featureFields) {
      this.#featureFields = featureFields
      fieldPtr = wasm._malloc(featureFields.length * 4)
      for (let i = 0; i < featureFields.length; i++) {
        wasm.setValue(fieldPtr + i * 4, featureFields[i], 'i32')
      }
    }

    const outPtr = wasm._malloc(4)
    const ret = wasm._wl_xl_create_dmatrix_dense(
      xPtr, rows, cols, yPtr, fieldPtr, outPtr
    )

    wasm._free(xPtr)
    if (yPtr) wasm._free(yPtr)
    if (fieldPtr) wasm._free(fieldPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`DMatrix creation failed: ${getLastError()}`)
    }

    const dmatrix = wasm.getValue(outPtr, 'i32')
    wasm._free(outPtr)

    return { dmatrix, rows, cols }
  }

  #buildCSRDMatrix(wasm, X, y) {
    const { rows, cols, data, indices, indptr } = X

    // Convert data to Float32
    const valF32 = new Float32Array(data.length)
    for (let i = 0; i < data.length; i++) valF32[i] = data[i]

    const valPtr = wasm._malloc(valF32.length * 4)
    wasm.HEAPF32.set(valF32, valPtr / 4)

    const idxPtr = wasm._malloc(indices.length * 4)
    for (let i = 0; i < indices.length; i++) {
      wasm.setValue(idxPtr + i * 4, indices[i], 'i32')
    }

    const indptrPtr = wasm._malloc(indptr.length * 4)
    for (let i = 0; i < indptr.length; i++) {
      wasm.setValue(indptrPtr + i * 4, indptr[i], 'i32')
    }

    // Labels
    let yPtr = 0
    if (y) {
      const yF32 = new Float32Array(y.length)
      if (this.#task === 'binary') {
        for (let i = 0; i < y.length; i++) yF32[i] = y[i] > 0 ? 1 : -1
      } else {
        for (let i = 0; i < y.length; i++) yF32[i] = y[i]
      }
      yPtr = wasm._malloc(yF32.length * 4)
      wasm.HEAPF32.set(yF32, yPtr / 4)
    }

    // Field map
    let fieldPtr = 0
    const featureFields = this.#params.featureFields || this.#featureFields
    if (featureFields) {
      this.#featureFields = featureFields
      fieldPtr = wasm._malloc(featureFields.length * 4)
      for (let i = 0; i < featureFields.length; i++) {
        wasm.setValue(fieldPtr + i * 4, featureFields[i], 'i32')
      }
    }

    const outPtr = wasm._malloc(4)
    const ret = wasm._wl_xl_create_dmatrix_csr(
      valPtr, valF32.length,
      idxPtr, indptrPtr, rows, cols,
      yPtr, fieldPtr, outPtr
    )

    wasm._free(valPtr)
    wasm._free(idxPtr)
    wasm._free(indptrPtr)
    if (yPtr) wasm._free(yPtr)
    if (fieldPtr) wasm._free(fieldPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`CSR DMatrix creation failed: ${getLastError()}`)
    }

    const dmatrix = wasm.getValue(outPtr, 'i32')
    wasm._free(outPtr)

    return { dmatrix, rows, cols }
  }

  #applyParams(wasm, handle) {
    const p = this.#params

    const setStr = (key, val) => {
      withCString(wasm, key, (kPtr) => {
        withCString(wasm, val, (vPtr) => {
          wasm._wl_xl_set_str(handle, kPtr, vPtr)
        })
      })
    }

    const setInt = (key, val) => {
      withCString(wasm, key, (kPtr) => {
        wasm._wl_xl_set_int(handle, kPtr, val)
      })
    }

    const setFloat = (key, val) => {
      withCString(wasm, key, (kPtr) => {
        wasm._wl_xl_set_float(handle, kPtr, val)
      })
    }

    const setBool = (key, val) => {
      withCString(wasm, key, (kPtr) => {
        wasm._wl_xl_set_bool(handle, kPtr, val ? 1 : 0)
      })
    }

    if (p.lr !== undefined) setFloat('lr', p.lr)
    if (p.lambda !== undefined) setFloat('lambda', p.lambda)
    if (p.k !== undefined) setInt('k', p.k)
    if (p.epoch !== undefined) setInt('epoch', p.epoch)
    if (p.opt !== undefined) setStr('opt', p.opt)
    if (p.alpha !== undefined) setFloat('alpha', p.alpha)
    if (p.beta !== undefined) setFloat('beta', p.beta)
    if (p.lambda_1 !== undefined) setFloat('lambda_1', p.lambda_1)
    if (p.lambda_2 !== undefined) setFloat('lambda_2', p.lambda_2)
    if (p.normalize !== undefined) setBool('norm', p.normalize)
  }

  #ensureNotDisposed() {
    if (this.#freed) throw new DisposedError('XLearn model has been disposed.')
  }

  #ensureFitted() {
    this.#ensureNotDisposed()
    if (!this.#fitted) throw new NotFittedError('XLearn model is not fitted. Call fit() first.')
  }
}

export { LOAD_SENTINEL }
