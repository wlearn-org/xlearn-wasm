const { getWasm, loadXLearn } = require('./wasm.js')
const {
  normalizeX, normalizeY,
  encodeBundle, validateBundle,
  register,
  DisposedError, NotFittedError
} = require('@wlearn/core')

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

function parseModelHeader(bytes) {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
  const decoder = new TextDecoder('ascii', { fatal: true })
  let offset = 0
  const readU32 = name => {
    if (offset > data.byteLength - 4) throw new Error(`xLearn model header is truncated before ${name}`)
    const value = view.getUint32(offset, true)
    offset += 4
    return value
  }
  const readString = name => {
    const length = readU32(`${name} length`)
    if (length < 1 || length > 64 || offset > data.byteLength - length) {
      throw new Error(`xLearn model header has an invalid ${name}`)
    }
    const value = decoder.decode(data.subarray(offset, offset + length))
    offset += length
    return value
  }
  const algo = readString('score function')
  const loss = readString('loss function')
  const nFeatures = readU32('feature count')
  const nFields = readU32('field count')
  const nFactors = readU32('factor count')
  const auxSize = readU32('auxiliary parameter count')
  const nWeights = readU32('linear weight count')
  const nFactorsStored = algo === 'linear' ? 0 : readU32('latent factor count')
  if (!['linear', 'fm', 'ffm'].includes(algo) ||
      !['cross-entropy', 'squared'].includes(loss) || nFeatures < 1) {
    throw new Error('xLearn model header has unsupported identity fields')
  }
  if (auxSize < 1 ||
      (algo === 'linear' && nFields !== 0) ||
      (algo === 'fm' && nFields !== 0) ||
      (algo !== 'linear' && nFactors < 1) ||
      (algo === 'ffm' && nFields < 1)) {
    throw new Error('xLearn model header has invalid dimensions')
  }

  const checkedProduct = (name, ...values) => {
    let product = 1
    for (const value of values) {
      if (value < 1 || product > Math.floor(0xffffffff / value)) {
        throw new Error(`xLearn model ${name} exceeds uint32 limits`)
      }
      product *= value
    }
    return product
  }
  const expectedWeights = checkedProduct('weight count', nFeatures, auxSize)
  if (nWeights !== expectedWeights) {
    throw new Error('xLearn model linear weight count is inconsistent')
  }

  let expectedFactors = 0
  if (algo !== 'linear') {
    const alignedFactors = Math.ceil(nFactors / 4) * 4
    expectedFactors = algo === 'fm'
      ? checkedProduct('latent factor count', nFeatures, alignedFactors, auxSize)
      : checkedProduct('latent factor count', nFeatures, nFields, alignedFactors, auxSize)
    if (nFactorsStored !== expectedFactors) {
      throw new Error('xLearn model latent factor count is inconsistent')
    }
  }

  const floatCount = expectedWeights + auxSize + expectedFactors
  if (!Number.isSafeInteger(floatCount) ||
      offset + floatCount * Float32Array.BYTES_PER_ELEMENT !== data.byteLength) {
    throw new Error('xLearn model byte length is inconsistent with its parameter counts')
  }
  return {
    algo, loss, nFeatures, nFields, nFactors, auxSize,
    nWeights, nFactorsStored
  }
}

// --- XLearnBase ---

class XLearnBase {
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
  #fittedSeed = 1
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

    // Validate the complete JS-side boundary before releasing a fitted model or
    // passing pointers to C. The C DMatrix constructors read one label per row
    // and one field ID per feature.
    const input = this.#prepareMatrixInput(X)
    const yNorm = normalizeY(y)
    const yF64 = yNorm instanceof Float64Array ? yNorm : new Float64Array(yNorm)
    if (yF64.length !== input.rows) {
      throw new Error(`y length (${yF64.length}) does not match X rows (${input.rows})`)
    }

    // xLearn's binary loss requires {-1,+1}. Keep sorted public int32 labels
    // separately and map them only at the DMatrix boundary.
    let classes = null
    if (this.#task === 'binary') {
      const classSet = new Set()
      for (let i = 0; i < yF64.length; i++) {
        const value = yF64[i]
        if (!Number.isInteger(value) || value < -2147483648 || value > 2147483647) {
          throw new Error(`Classifier labels must be int32 values, got ${value} at index ${i}`)
        }
        classSet.add(value)
      }
      if (classSet.size !== 2) {
        throw new Error(`Binary classification requires exactly 2 classes, got ${classSet.size}`)
      }
      const sorted = [...classSet].sort((a, b) => a - b)
      classes = new Int32Array(sorted)
    } else {
      for (let i = 0; i < yF64.length; i++) {
        if (!Number.isFinite(yF64[i])) {
          throw new Error(`Regression labels must be finite, got ${yF64[i]} at index ${i}`)
        }
      }
    }

    const featureFields = this.#prepareFeatureFields(input.cols)
    const seed = this.#resolvedSeed()

    // Validation succeeded. A refit may now release the prior native model.
    if (this.#handle) {
      wasm._wl_xl_free_handle(this.#handle)
      this.#handle = null
      if (this.#handleRef) this.#handleRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }
    this.#modelBytes = null
    this.#fitted = false
    this.#nClasses = classes ? 2 : 0
    this.#classes = classes
    this.#featureFields = featureFields
    this.#fittedSeed = seed

    // Build DMatrix (CSR or dense) from the already validated input.
    let dmatrix
    if (input.csr) {
      ({ dmatrix } = this.#buildCSRDMatrix(wasm, input.matrix, yF64))
    } else {
      ({ dmatrix } = this.#buildDenseDMatrix(wasm, input.matrix, yF64))
    }

    this.#nFeatures = input.cols

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
    this.#applyParams(wasm, handle, seed)

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
    const margins = this.#rawPredict(X)
    if (this.#task !== 'binary') return margins

    const labels = new Int32Array(margins.length)
    for (let i = 0; i < margins.length; i++) {
      labels[i] = this.#classes[margins[i] > 0 ? 1 : 0]
    }
    return labels
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

    if (yArr.length !== preds.length) {
      throw new Error(`y length (${yArr.length}) does not match prediction rows (${preds.length})`)
    }

    if (this.#task === 'binary') {
      let correct = 0
      for (let i = 0; i < preds.length; i++) {
        if (preds[i] === yArr[i]) correct++
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
    if (this.#algo === 'ffm') {
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

    if (this.#task === 'binary') {
      metadata.labelEncoding = 'sorted-int32-sign-v1'
    }

    const bundleParams = this.getParams()
    // field_map is the sole serialized source of truth. Typed arrays are not
    // valid manifest JSON and duplicating the map risks divergent state.
    delete bundleParams.featureFields
    bundleParams.seed = this.#fittedSeed

    return encodeBundle(
      {
        typeId: this._typeId,
        params: bundleParams,
        seed: this.#fittedSeed,
        metadata
      },
      artifacts
    )
  }

  static async _load(bytes, TypeClass) {
    const { manifest, toc, blobs } = validateBundle(bytes)
    return TypeClass._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs, TypeClass) {
    await loadXLearn()

    const spec = TypeClass.bundleSpec
    if (!spec || !spec.typeIds.includes(manifest.typeId)) {
      throw new Error(
        `${TypeClass.name} cannot load bundle type ${JSON.stringify(manifest.typeId)}`
      )
    }

    const meta = manifest.metadata || {}
    if (meta.algo !== spec.algo || meta.task !== spec.task) {
      throw new Error(
        `${TypeClass.name} bundle metadata does not match ${spec.algo}/${spec.task}`
      )
    }
    if (!Number.isInteger(meta.nFeatures) || meta.nFeatures < 1) {
      throw new Error(`${TypeClass.name} bundle has invalid nFeatures`)
    }
    if (spec.task === 'binary') {
      if (meta.nClasses !== 2 || !Array.isArray(meta.classes) || meta.classes.length !== 2 ||
          !meta.classes.every(value => Number.isInteger(value) &&
            value >= -2147483648 && value <= 2147483647) ||
          meta.classes[0] >= meta.classes[1]) {
        throw new Error(`${TypeClass.name} bundle has invalid binary class metadata`)
      }
      if (manifest.typeId.endsWith('@2') &&
          meta.labelEncoding !== 'sorted-int32-sign-v1') {
        throw new Error(`${TypeClass.name} @2 bundle has invalid label encoding`)
      }
      if (manifest.typeId.endsWith('@1') &&
          !(meta.classes[0] <= 0 && meta.classes[1] > 0)) {
        throw new Error(
          `${TypeClass.name} legacy @1 bundle used an ambiguous same-sign label mapping; retrain the model`
        )
      }
    } else if (meta.nClasses !== 0 || meta.classes != null) {
      throw new Error(`${TypeClass.name} regressor bundle has classifier metadata`)
    }

    const modelEntries = toc.filter(e => e.id === 'model')
    const fieldEntries = toc.filter(e => e.id === 'field_map')
    const ffmWithMap = spec.algo === 'ffm' &&
      fieldEntries.length === 1 && toc.length === 2
    if (modelEntries.length !== 1 ||
        !(toc.length === 1 || ffmWithMap) ||
        toc.some(entry => entry.mediaType !== 'application/octet-stream') ||
        toc.some(entry => entry.id !== 'model' && entry.id !== 'field_map') ||
        (spec.algo !== 'ffm' && fieldEntries.length !== 0)) {
      throw new Error(`${TypeClass.name} bundle has an invalid artifact set`)
    }
    const entry = modelEntries[0]
    const modelData = new Uint8Array(entry.length)
    modelData.set(blobs.subarray(entry.offset, entry.offset + entry.length))
    const modelIdentity = parseModelHeader(modelData)
    const expectedLoss = spec.task === 'binary' ? 'cross-entropy' : 'squared'
    if (modelIdentity.algo !== spec.algo || modelIdentity.loss !== expectedLoss ||
        modelIdentity.nFeatures !== meta.nFeatures) {
      throw new Error(`${TypeClass.name} model blob identity does not match its bundle`)
    }

    const params = { ...(manifest.params || {}) }
    // Legacy writers persisted a requested params.seed but never forwarded it;
    // without the top-level seed contract the effective upstream seed was 1.
    const resolvedSeed = manifest.seed === undefined ? 1 : params.seed
    if (!Number.isInteger(resolvedSeed) || resolvedSeed < 1 ||
        resolvedSeed > 2147483647) {
      throw new Error(`${TypeClass.name} bundle has invalid seed`)
    }
    if (manifest.seed !== undefined && manifest.seed !== resolvedSeed) {
      throw new Error(`${TypeClass.name} bundle seed does not match params.seed`)
    }
    if (manifest.typeId.endsWith('@2') && manifest.seed === undefined) {
      throw new Error(`${TypeClass.name} @2 bundle is missing its training seed`)
    }
    params.seed = resolvedSeed

    const instance = new TypeClass(LOAD_SENTINEL, meta.algo, meta.task, params)
    instance.#modelBytes = modelData
    instance.#nFeatures = meta.nFeatures || 0
    instance.#nClasses = meta.nClasses || 0
    instance.#classes = meta.classes ? new Int32Array(meta.classes) : null
    instance.#fittedSeed = resolvedSeed

    // Load field_map if present
    const fieldEntry = fieldEntries[0]
    if (fieldEntry) {
      if (spec.algo !== 'ffm') {
        throw new Error('Non-FFM bundle must not contain a field_map artifact')
      }
      if (fieldEntry.length % Int32Array.BYTES_PER_ELEMENT !== 0 ||
          fieldEntry.length / Int32Array.BYTES_PER_ELEMENT !== meta.nFeatures) {
        throw new Error('field_map artifact length must match nFeatures')
      }
      const raw = blobs.subarray(fieldEntry.offset, fieldEntry.offset + fieldEntry.length)
      instance.#featureFields = new Int32Array(raw.buffer.slice(
        raw.byteOffset, raw.byteOffset + raw.byteLength
      ))
      for (let i = 0; i < instance.#featureFields.length; i++) {
        if (instance.#featureFields[i] < 0 ||
            instance.#featureFields[i] >= modelIdentity.nFields) {
          throw new Error(`field_map contains an invalid field ID at index ${i}`)
        }
      }
      params.featureFields = instance.#featureFields
    } else if (spec.algo === 'ffm') {
      if (manifest.typeId.endsWith('@2')) {
        throw new Error('FFM @2 bundle missing "field_map" artifact')
      }
      instance.#featureFields = new Int32Array(meta.nFeatures)
      params.featureFields = instance.#featureFields
    } else {
      delete params.featureFields
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
    const input = this.#prepareMatrixInput(X)
    if (input.cols !== this.#nFeatures) {
      throw new Error(`X has ${input.cols} features; model expects ${this.#nFeatures}`)
    }

    // Build DMatrix for query
    let dmatrix, rows
    if (input.csr) {
      ({ dmatrix, rows } = this.#buildCSRDMatrix(wasm, input.matrix, null))
    } else {
      ({ dmatrix, rows } = this.#buildDenseDMatrix(wasm, input.matrix, null))
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
    const { data: xData, rows, cols } = X

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
        for (let i = 0; i < y.length; i++) {
          yF32[i] = y[i] === this.#classes[1] ? 1 : -1
        }
      } else {
        for (let i = 0; i < y.length; i++) yF32[i] = y[i]
      }
      yPtr = wasm._malloc(yF32.length * 4)
      wasm.HEAPF32.set(yF32, yPtr / 4)
    }

    // Field map for FFM
    let fieldPtr = 0
    const featureFields = this.#algo === 'ffm' ? this.#featureFields : null
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
        for (let i = 0; i < y.length; i++) {
          yF32[i] = y[i] === this.#classes[1] ? 1 : -1
        }
      } else {
        for (let i = 0; i < y.length; i++) yF32[i] = y[i]
      }
      yPtr = wasm._malloc(yF32.length * 4)
      wasm.HEAPF32.set(yF32, yPtr / 4)
    }

    // Field map
    let fieldPtr = 0
    const featureFields = this.#algo === 'ffm' ? this.#featureFields : null
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

  #applyParams(wasm, handle, seed) {
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
    setInt('seed', seed)
  }

  #prepareMatrixInput(X) {
    const csrShaped = X && typeof X === 'object' && !Array.isArray(X) &&
      ('indices' in X || 'indptr' in X)
    if (csrShaped) {
      if (!isCSR(X)) {
        throw new Error('CSR indices and indptr must be Int32Array values')
      }
      const { rows, cols, data, indices, indptr } = X
      if (!Number.isInteger(rows) || rows < 1 ||
          !Number.isInteger(cols) || cols < 1) {
        throw new Error(`Invalid CSR dimensions: rows=${rows}, cols=${cols}`)
      }
      if (!(data instanceof Float32Array) && !(data instanceof Float64Array)) {
        throw new Error('CSR data must be Float32Array or Float64Array')
      }
      if (data.length !== indices.length) {
        throw new Error(`CSR data length (${data.length}) does not match indices length (${indices.length})`)
      }
      if (indptr.length !== rows + 1 || indptr[0] !== 0 ||
          indptr[indptr.length - 1] !== data.length) {
        throw new Error('CSR indptr must start at 0, end at nnz, and have rows + 1 entries')
      }
      for (let i = 0; i < data.length; i++) {
        if (!Number.isFinite(data[i])) {
          throw new Error(`CSR data must be finite, got ${data[i]} at index ${i}`)
        }
        if (indices[i] < 0 || indices[i] >= cols) {
          throw new Error(`CSR column index ${indices[i]} is out of bounds at index ${i}`)
        }
      }
      for (let i = 1; i < indptr.length; i++) {
        if (indptr[i] < indptr[i - 1]) {
          throw new Error(`CSR indptr must be nondecreasing at index ${i}`)
        }
      }
      return { csr: true, matrix: X, rows, cols }
    }

    if (Array.isArray(X)) {
      if (X.length === 0 || !Array.isArray(X[0]) || X[0].length === 0) {
        throw new Error('Dense matrix must contain at least one row and one column')
      }
      const expectedCols = X[0].length
      for (let i = 0; i < X.length; i++) {
        if (!Array.isArray(X[i]) || X[i].length !== expectedCols) {
          throw new Error(`Dense matrix row ${i} has inconsistent length`)
        }
      }
    }
    const matrix = normalizeX(X)
    const { data, rows, cols } = matrix
    if (!Number.isInteger(rows) || rows < 1 ||
        !Number.isInteger(cols) || cols < 1 || data.length !== rows * cols) {
      throw new Error(`Invalid dense matrix shape: rows=${rows}, cols=${cols}, data.length=${data.length}`)
    }
    for (let i = 0; i < data.length; i++) {
      if (!Number.isFinite(data[i])) {
        throw new Error(`Dense matrix data must be finite, got ${data[i]} at index ${i}`)
      }
    }
    return { csr: false, matrix, rows, cols }
  }

  #prepareFeatureFields(cols) {
    const provided = this.#params.featureFields
    if (this.#algo !== 'ffm') {
      if (provided !== undefined) {
        throw new Error('featureFields is supported only by FFM models')
      }
      return null
    }

    if (provided === undefined || provided === null) {
      return new Int32Array(cols)
    }
    if (!(provided instanceof Int32Array)) {
      throw new Error('featureFields must be an Int32Array')
    }
    if (provided.length !== cols) {
      throw new Error(`featureFields length (${provided.length}) does not match X columns (${cols})`)
    }
    const fields = new Int32Array(provided.length)
    const compactIds = new Map()
    for (let i = 0; i < fields.length; i++) {
      const field = provided[i]
      if (field < 0) {
        throw new Error(`featureFields contains a negative field ID at index ${i}`)
      }
      if (!compactIds.has(field)) compactIds.set(field, compactIds.size)
      fields[i] = compactIds.get(field)
    }
    return fields
  }

  #resolvedSeed() {
    const seed = this.#params.seed === undefined ? 1 : this.#params.seed
    if (!Number.isInteger(seed) || seed < 1 || seed > 2147483647) {
      throw new Error(`seed must be an integer in [1, 2147483647], got ${seed}`)
    }
    return seed
  }

  #ensureNotDisposed() {
    if (this.#freed) throw new DisposedError('XLearn model has been disposed.')
  }

  #ensureFitted() {
    this.#ensureNotDisposed()
    if (!this.#fitted) throw new NotFittedError('XLearn model is not fitted. Call fit() first.')
  }
}

module.exports = { XLearnBase, LOAD_SENTINEL }
