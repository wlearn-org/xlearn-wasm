const { decodeBundle, encodeBundle, load: coreLoad } = require('@wlearn/core')

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    if (err.stack) {
      const lines = err.stack.split('\n').slice(1, 3)
      for (const line of lines) console.log(`        ${line.trim()}`)
    }
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

async function expectReject(promise, pattern) {
  let error = null
  try {
    await promise
  } catch (caught) {
    error = caught
  }
  assert(error, 'expected operation to reject')
  if (pattern) {
    assert(pattern.test(error.message), `unexpected error: ${error.message}`)
  }
}

function rewriteBundle(bytes, mutateManifest, mutateArtifacts = artifacts => artifacts) {
  const { manifest, toc, blobs } = decodeBundle(bytes)
  const nextManifest = JSON.parse(JSON.stringify(manifest))
  mutateManifest(nextManifest)
  const artifacts = toc.map(entry => ({
    id: entry.id,
    data: new Uint8Array(blobs.slice(entry.offset, entry.offset + entry.length)),
    mediaType: entry.mediaType
  }))
  return encodeBundle(nextManifest, mutateArtifacts(artifacts))
}

function bundleArtifact(bytes, id) {
  const { toc, blobs } = decodeBundle(bytes)
  const entry = toc.find(item => item.id === id)
  assert(entry, `missing ${id} artifact`)
  return new Uint8Array(blobs.slice(entry.offset, entry.offset + entry.length))
}

function modelScalarOffsets(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let offset = 0
  for (let i = 0; i < 2; i++) {
    const length = view.getUint32(offset, true)
    offset += 4 + length
  }
  const nFeatures = offset
  return {
    nFeatures,
    nWeights: nFeatures + 4 * 4
  }
}

// Deterministic data generation
function makeLinearData(n, seed = 7) {
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const t = ((i * seed + 3) % n) / n
    const s = ((i * (seed + 6) + 7) % n) / n
    const x1 = t * 2 - 1
    const x2 = s * 2 - 1
    X.push([x1, x2])
    y.push(x1 + x2 > 0 ? 1 : 0)
  }
  return { X, y }
}

function makeRegressionData(n) {
  const X = []
  const y = []
  for (let i = 0; i < n; i++) {
    const x1 = ((i * 7 + 3) % n) / (n / 2) - 1
    const x2 = ((i * 13 + 7) % n) / (n / 2) - 1
    const noise = ((i * 31 + 11) % n) / (n * 5) - 0.1
    X.push([x1, x2])
    y.push(2 * x1 + 3 * x2 + noise)
  }
  return { X, y }
}

function toCSR(X) {
  const data = []
  const indices = []
  const indptr = [0]
  const rows = X.length
  const cols = X[0].length
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (X[i][j] !== 0) {
        data.push(X[i][j])
        indices.push(j)
      }
    }
    indptr.push(data.length)
  }
  return {
    rows, cols,
    data: new Float64Array(data),
    indices: new Int32Array(indices),
    indptr: new Int32Array(indptr)
  }
}

async function main() {

const {
  loadXLearn, getWasm,
  XLearnLR, XLearnFM, XLearnFFM,
  XLearnLRClassifier, XLearnLRRegressor,
  XLearnFMClassifier, XLearnFMRegressor,
  XLearnFFMClassifier, XLearnFFMRegressor
} = require('../src/index.js')

// ============================================================
// WASM loading
// ============================================================
console.log('\n=== WASM Loading ===')

await test('WASM module loads', async () => {
  const wasm = await loadXLearn()
  assert(wasm, 'wasm module is null')
  assert(typeof wasm.ccall === 'function', 'ccall not available')
})

await test('get_last_error returns string', async () => {
  const { getWasm } = require('../src/wasm.js')
  const wasm = getWasm()
  const err = wasm.ccall('wl_xl_get_last_error', 'string', [], [])
  assert(typeof err === 'string', `expected string, got ${typeof err}`)
})

// ============================================================
// LR Classifier
// ============================================================
console.log('\n=== LR Classifier ===')

await test('LR classifier: create, fit, predict', async () => {
  const m = await XLearnLRClassifier.create({ epoch: 10 })
  assert(!m.isFitted, 'should not be fitted')

  const { X, y } = makeLinearData(80)
  m.fit(X, y)
  assert(m.isFitted, 'should be fitted')
  assert(m.nFeatures === 2, `nFeatures=${m.nFeatures}`)
  assert(m.nClasses === 2, `nClasses=${m.nClasses}`)

  const preds = m.predict(X)
  assert(preds instanceof Int32Array, 'classifier predict should return Int32Array labels')
  assert(preds.length === 80, `expected 80, got ${preds.length}`)
  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] === 0 || preds[i] === 1, `unexpected class label ${preds[i]}`)
  }

  const acc = m.score(X, y)
  assert(acc > 0.6, `accuracy ${acc.toFixed(3)} too low`)

  m.dispose()
})

await test('LR classifier: predictProba', async () => {
  const m = await XLearnLRClassifier.create({ epoch: 10 })
  const { X, y } = makeLinearData(60)
  m.fit(X, y)

  const proba = m.predictProba(X)
  assert(proba.length === 120, `expected 120, got ${proba.length}`)

  for (let r = 0; r < 60; r++) {
    const p0 = proba[r * 2]
    const p1 = proba[r * 2 + 1]
    assert(p0 >= 0 && p0 <= 1, `P(0) out of [0,1]: ${p0}`)
    assert(p1 >= 0 && p1 <= 1, `P(1) out of [0,1]: ${p1}`)
    assertClose(p0 + p1, 1.0, 1e-6, `row ${r} proba sum=${p0 + p1}`)
  }

  m.dispose()
})

await test('LR classifier: save/load round-trip', async () => {
  const m = await XLearnLRClassifier.create({ epoch: 10 })
  const { X, y } = makeLinearData(60)
  m.fit(X, y)

  const p1 = m.predict(X)
  const bytes = m.save()
  m.dispose()

  const m2 = await XLearnLRClassifier.load(bytes)
  assert(m2.isFitted, 'loaded should be fitted')
  const p2 = m2.predict(X)

  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }

  m2.dispose()
})

await test('LR classifier: capabilities', async () => {
  const m = await XLearnLRClassifier.create()
  const c = m.capabilities
  assert(c.classifier === true, 'should be classifier')
  assert(c.regressor === false, 'should not be regressor')
  assert(c.predictProba === true, 'should support predictProba')
  assert(c.csr === true, 'should support csr')
  m.dispose()
})

await test('LR classifier: defaultSearchSpace', async () => {
  const space = XLearnLRClassifier.defaultSearchSpace()
  assert(space.lr, 'missing lr')
  assert(space.lambda, 'missing lambda')
  assert(space.opt, 'missing opt')
  assert(space.epoch, 'missing epoch')
})

// ============================================================
// LR Regressor
// ============================================================
console.log('\n=== LR Regressor ===')

await test('LR regressor: fit, predict, score', async () => {
  const m = await XLearnLRRegressor.create({ epoch: 20 })
  const { X, y } = makeRegressionData(80)
  m.fit(X, y)

  const preds = m.predict(X)
  assert(preds.length === 80, `expected 80, got ${preds.length}`)
  for (let i = 0; i < preds.length; i++) {
    assert(!isNaN(preds[i]), `prediction ${i} is NaN`)
  }

  const r2 = m.score(X, y)
  assert(r2 > 0.2, `R-squared ${r2.toFixed(3)} too low`)

  m.dispose()
})

await test('LR regressor: predictProba throws', async () => {
  const m = await XLearnLRRegressor.create({ epoch: 10 })
  const { X, y } = makeRegressionData(40)
  m.fit(X, y)

  let threw = false
  try { m.predictProba(X) } catch { threw = true }
  assert(threw, 'predictProba should throw for regressor')

  m.dispose()
})

await test('LR regressor: save/load round-trip', async () => {
  const m = await XLearnLRRegressor.create({ epoch: 10 })
  const { X, y } = makeRegressionData(60)
  m.fit(X, y)

  const p1 = m.predict(X)
  const bytes = m.save()
  m.dispose()

  const m2 = await XLearnLRRegressor.load(bytes)
  const p2 = m2.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }

  m2.dispose()
})

// ============================================================
// FM Classifier
// ============================================================
console.log('\n=== FM Classifier ===')

await test('FM classifier: fit, predict, score', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 15, k: 4 })
  const { X, y } = makeLinearData(80)
  m.fit(X, y)

  const preds = m.predict(X)
  assert(preds.length === 80, `expected 80, got ${preds.length}`)
  for (let i = 0; i < preds.length; i++) {
    assert(!isNaN(preds[i]), `prediction ${i} is NaN`)
  }

  const acc = m.score(X, y)
  assert(acc > 0.6, `accuracy ${acc.toFixed(3)} too low`)

  m.dispose()
})

await test('FM classifier: predictProba', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 15, k: 4 })
  const { X, y } = makeLinearData(60)
  m.fit(X, y)

  const proba = m.predictProba(X)
  assert(proba.length === 120, `expected 120, got ${proba.length}`)

  for (let r = 0; r < 60; r++) {
    const p0 = proba[r * 2]
    const p1 = proba[r * 2 + 1]
    assert(p0 >= 0 && p0 <= 1, `P(0) out of [0,1]: ${p0}`)
    assert(p1 >= 0 && p1 <= 1, `P(1) out of [0,1]: ${p1}`)
    assertClose(p0 + p1, 1.0, 1e-6, `row ${r} proba sum=${p0 + p1}`)
  }

  m.dispose()
})

await test('FM classifier: save/load round-trip', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 10, k: 4 })
  const { X, y } = makeLinearData(60)
  m.fit(X, y)

  const p1 = m.predict(X)
  const bytes = m.save()

  const { manifest, toc } = decodeBundle(bytes)
  assert(manifest.typeId === 'wlearn.xlearn.fm.classifier@2', `typeId=${manifest.typeId}`)
  assert(toc.length === 1, `expected 1 TOC entry, got ${toc.length}`)
  assert(toc[0].id === 'model', `expected TOC entry "model", got ${toc[0].id}`)

  m.dispose()

  const m2 = await XLearnFMClassifier.load(bytes)
  const p2 = m2.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }
  m2.dispose()
})

// ============================================================
// FM Regressor
// ============================================================
console.log('\n=== FM Regressor ===')

await test('FM regressor: fit, predict, score', async () => {
  const m = await XLearnFMRegressor.create({ epoch: 20, k: 4 })
  const { X, y } = makeRegressionData(80)
  m.fit(X, y)

  const preds = m.predict(X)
  assert(preds.length === 80, `expected 80, got ${preds.length}`)

  const r2 = m.score(X, y)
  assert(r2 > 0.2, `R-squared ${r2.toFixed(3)} too low`)

  m.dispose()
})

await test('FM regressor: save/load round-trip', async () => {
  const m = await XLearnFMRegressor.create({ epoch: 10, k: 4 })
  const { X, y } = makeRegressionData(60)
  m.fit(X, y)

  const p1 = m.predict(X)
  const bytes = m.save()

  const { manifest } = decodeBundle(bytes)
  assert(manifest.typeId === 'wlearn.xlearn.fm.regressor@1', `typeId=${manifest.typeId}`)

  m.dispose()

  const m2 = await XLearnFMRegressor.load(bytes)
  const p2 = m2.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }
  m2.dispose()
})

// ============================================================
// FFM Classifier
// ============================================================
console.log('\n=== FFM Classifier ===')

await test('FFM classifier: fit with featureFields', async () => {
  const featureFields = new Int32Array([0, 1])
  const m = await XLearnFFMClassifier.create({ epoch: 15, k: 4, featureFields })
  const { X, y } = makeLinearData(80)
  m.fit(X, y)

  const preds = m.predict(X)
  assert(preds.length === 80, `expected 80, got ${preds.length}`)
  for (let i = 0; i < preds.length; i++) {
    assert(!isNaN(preds[i]), `prediction ${i} is NaN`)
  }

  const acc = m.score(X, y)
  assert(acc > 0.6, `accuracy ${acc.toFixed(3)} too low`)

  m.dispose()
})

await test('FFM classifier: predictProba', async () => {
  const featureFields = new Int32Array([0, 1])
  const m = await XLearnFFMClassifier.create({ epoch: 15, k: 4, featureFields })
  const { X, y } = makeLinearData(60)
  m.fit(X, y)

  const proba = m.predictProba(X)
  for (let r = 0; r < 60; r++) {
    const p0 = proba[r * 2]
    const p1 = proba[r * 2 + 1]
    assert(p0 >= 0 && p0 <= 1, `P(0) out of [0,1]: ${p0}`)
    assert(p1 >= 0 && p1 <= 1, `P(1) out of [0,1]: ${p1}`)
    assertClose(p0 + p1, 1.0, 1e-6, `row ${r} proba sum=${p0 + p1}`)
  }

  m.dispose()
})

await test('FFM classifier: save/load preserves field_map', async () => {
  const featureFields = new Int32Array([0, 1])
  const m = await XLearnFFMClassifier.create({ epoch: 10, k: 4, featureFields })
  const { X, y } = makeLinearData(60)
  m.fit(X, y)

  const p1 = m.predict(X)
  const bytes = m.save()

  const { manifest, toc } = decodeBundle(bytes)
  assert(manifest.typeId === 'wlearn.xlearn.ffm.classifier@2', `typeId=${manifest.typeId}`)
  assert(manifest.params.featureFields === undefined, 'featureFields must not be duplicated in manifest params')
  // FFM should have both model and field_map artifacts
  assert(toc.length === 2, `expected 2 TOC entries, got ${toc.length}`)
  const fieldEntry = toc.find(e => e.id === 'field_map')
  assert(fieldEntry, 'missing field_map artifact')

  m.dispose()

  const m2 = await XLearnFFMClassifier.load(bytes)
  const p2 = m2.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 0.1, `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }
  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `label mismatch at ${i}: ${p1[i]} vs ${p2[i]}`)
  }
  assert(m2.getParams().featureFields instanceof Int32Array, 'loaded field map should be Int32Array')
  m2.dispose()
})

// ============================================================
// FFM Regressor
// ============================================================
console.log('\n=== FFM Regressor ===')

await test('FFM regressor: fit, predict, score', async () => {
  const featureFields = new Int32Array([0, 1])
  const m = await XLearnFFMRegressor.create({ epoch: 20, k: 4, featureFields })
  const { X, y } = makeRegressionData(80)
  m.fit(X, y)

  const preds = m.predict(X)
  assert(preds.length === 80, `expected 80, got ${preds.length}`)

  const r2 = m.score(X, y)
  assert(r2 > 0.1, `R-squared ${r2.toFixed(3)} too low`)

  m.dispose()
})

await test('FFM regressor: save/load round-trip', async () => {
  const featureFields = new Int32Array([0, 1])
  const m = await XLearnFFMRegressor.create({ epoch: 10, k: 4, featureFields })
  const { X, y } = makeRegressionData(60)
  m.fit(X, y)

  const p1 = m.predict(X)
  const bytes = m.save()

  const { manifest } = decodeBundle(bytes)
  assert(manifest.typeId === 'wlearn.xlearn.ffm.regressor@1', `typeId=${manifest.typeId}`)

  m.dispose()

  const m2 = await XLearnFFMRegressor.load(bytes)
  const p2 = m2.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 0.1, `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }
  m2.dispose()
})

// ============================================================
// CSR sparse input
// ============================================================
console.log('\n=== CSR Sparse Input ===')

await test('CSR input produces same predictions as dense (LR)', async () => {
  const m1 = await XLearnLRClassifier.create({ epoch: 10 })
  const { X, y } = makeLinearData(60)
  m1.fit(X, y)
  const p1 = m1.predict(X)
  const bytes = m1.save()
  m1.dispose()

  const m2 = await XLearnLRClassifier.load(bytes)
  const csr = toCSR(X)
  const p2 = m2.predict(csr)
  assert(p1.length === p2.length, 'length mismatch')
  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 1e-5, `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }
  m2.dispose()
})

await test('CSR input: FM classifier fit and predict', async () => {
  const { X, y } = makeLinearData(60)
  const csr = toCSR(X)

  const m = await XLearnFMClassifier.create({ epoch: 10, k: 4 })
  m.fit(csr, y)

  const preds = m.predict(csr)
  assert(preds.length === 60, `expected 60, got ${preds.length}`)
  for (let i = 0; i < preds.length; i++) {
    assert(!isNaN(preds[i]), `prediction ${i} is NaN`)
  }

  const acc = m.score(csr, y)
  assert(acc > 0.5, `accuracy ${acc.toFixed(3)} too low`)

  m.dispose()
})

await test('CSR input: FFM with featureFields', async () => {
  const featureFields = new Int32Array([0, 1])
  const { X, y } = makeLinearData(60)
  const csr = toCSR(X)

  const m = await XLearnFFMClassifier.create({ epoch: 10, k: 4, featureFields })
  m.fit(csr, y)

  const preds = m.predict(csr)
  assert(preds.length === 60, `expected 60, got ${preds.length}`)
  for (let i = 0; i < preds.length; i++) {
    assert(!isNaN(preds[i]), `prediction ${i} is NaN`)
  }

  m.dispose()
})

// ============================================================
// Registry dispatch
// ============================================================
console.log('\n=== Registry Dispatch ===')

await test('core.load() dispatches to FM classifier', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 10, k: 4 })
  const { X, y } = makeLinearData(60)
  m.fit(X, y)

  const p1 = m.predict(X)
  const bytes = m.save()
  m.dispose()

  const m2 = await coreLoad(bytes)
  assert(m2.isFitted, 'loaded should be fitted')
  const p2 = m2.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }

  m2.dispose()
})

await test('core.load() dispatches to LR regressor', async () => {
  const m = await XLearnLRRegressor.create({ epoch: 10 })
  const { X, y } = makeRegressionData(60)
  m.fit(X, y)

  const p1 = m.predict(X)
  const bytes = m.save()
  m.dispose()

  const m2 = await coreLoad(bytes)
  const p2 = m2.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }

  m2.dispose()
})

await test('core.load() dispatches to FFM classifier', async () => {
  const featureFields = new Int32Array([0, 1])
  const m = await XLearnFFMClassifier.create({ epoch: 10, k: 4, featureFields })
  const { X, y } = makeLinearData(60)
  m.fit(X, y)

  const p1 = m.predict(X)
  const bytes = m.save()
  m.dispose()

  const m2 = await coreLoad(bytes)
  const p2 = m2.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 0.1, `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }

  m2.dispose()
})

// ============================================================
// Params
// ============================================================
console.log('\n=== Params ===')

await test('getParams / setParams', async () => {
  const m = await XLearnFMClassifier.create({ lr: 0.1, epoch: 20, k: 8 })

  const params = m.getParams()
  assert(params.lr === 0.1, `expected lr=0.1, got ${params.lr}`)
  assert(params.epoch === 20, `expected epoch=20, got ${params.epoch}`)
  assert(params.k === 8, `expected k=8, got ${params.k}`)

  m.setParams({ epoch: 50 })
  assert(m.getParams().epoch === 50, 'setParams should update epoch')

  m.dispose()
})

await test('save/load preserves params', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 15, k: 8 })
  const { X, y } = makeLinearData(60)
  m.fit(X, y)

  const bytes = m.save()
  m.dispose()

  const m2 = await XLearnFMClassifier.load(bytes)
  const params = m2.getParams()
  assert(params.epoch === 15, `expected epoch=15, got ${params.epoch}`)
  assert(params.k === 8, `expected k=8, got ${params.k}`)

  m2.dispose()
})

// ============================================================
// Bundle format
// ============================================================
console.log('\n=== Bundle Format ===')

await test('bundle has correct WLRN magic', async () => {
  const m = await XLearnLRClassifier.create({ epoch: 5 })
  const { X, y } = makeLinearData(40)
  m.fit(X, y)

  const buf = m.save()
  assert(buf[0] === 0x57, 'bad magic[0]')
  assert(buf[1] === 0x4c, 'bad magic[1]')
  assert(buf[2] === 0x52, 'bad magic[2]')
  assert(buf[3] === 0x4e, 'bad magic[3]')

  m.dispose()
})

await test('bundle manifest has required fields', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 5, k: 4 })
  const { X, y } = makeLinearData(40)
  m.fit(X, y)

  const bytes = m.save()
  const { manifest } = decodeBundle(bytes)
  assert(manifest.typeId === 'wlearn.xlearn.fm.classifier@2', `typeId=${manifest.typeId}`)
  assert(manifest.metadata, 'missing metadata')
  assert(manifest.metadata.algo === 'fm', `algo=${manifest.metadata.algo}`)
  assert(manifest.metadata.task === 'binary', `task=${manifest.metadata.task}`)
  assert(manifest.metadata.nFeatures === 2, `nFeatures=${manifest.metadata.nFeatures}`)
  assert(manifest.metadata.nClasses === 2, `nClasses=${manifest.metadata.nClasses}`)
  assert(manifest.metadata.labelEncoding === 'sorted-int32-sign-v1', 'missing label encoding')
  assert(manifest.seed === 1, `seed=${manifest.seed}`)

  m.dispose()
})

await test('bundle TOC has SHA-256 hashes', async () => {
  const m = await XLearnLRClassifier.create({ epoch: 5 })
  const { X, y } = makeLinearData(40)
  m.fit(X, y)

  const bytes = m.save()
  const { toc } = decodeBundle(bytes)
  assert(toc.length >= 1, 'expected at least 1 TOC entry')
  assert(typeof toc[0].sha256 === 'string', 'TOC entry missing sha256')
  assert(toc[0].sha256.length === 64, `sha256 length=${toc[0].sha256.length}`)

  m.dispose()
})

// ============================================================
// Resource management
// ============================================================
console.log('\n=== Resource Management ===')

await test('dispose is idempotent', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 5 })
  const { X, y } = makeLinearData(40)
  m.fit(X, y)
  m.dispose()
  m.dispose() // should not throw
})

await test('throws after dispose', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 5 })
  const { X, y } = makeLinearData(40)
  m.fit(X, y)
  m.dispose()

  let threw = false
  try { m.predict(X) } catch { threw = true }
  assert(threw, 'predict after dispose should throw')
})

await test('throws before fit', async () => {
  const m = await XLearnFMClassifier.create()

  let threw = false
  try { m.predict([[1, 2]]) } catch { threw = true }
  assert(threw, 'predict before fit should throw')

  m.dispose()
})

await test('refit does not leak', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 5, k: 4 })
  const { X, y } = makeLinearData(40)
  m.fit(X, y)
  m.fit(X, y) // refit

  const preds = m.predict(X)
  assert(preds.length === 40, 'should predict after refit')

  m.dispose()
})

// ============================================================
// Typed matrix input
// ============================================================
console.log('\n=== Typed Matrix Input ===')

await test('typed matrix fast path', async () => {
  const m = await XLearnLRClassifier.create({ epoch: 10 })
  const { X, y } = makeLinearData(40)

  const data = new Float64Array(40 * 2)
  for (let i = 0; i < 40; i++) {
    data[i * 2] = X[i][0]
    data[i * 2 + 1] = X[i][1]
  }

  m.fit({ data, rows: 40, cols: 2 }, y)
  const preds = m.predict({ data, rows: 40, cols: 2 })
  assert(preds.length === 40, `expected 40, got ${preds.length}`)
  for (let i = 0; i < preds.length; i++) {
    assert(!isNaN(preds[i]), `prediction ${i} is NaN`)
  }

  m.dispose()
})

// ============================================================
// decisionFunction
// ============================================================
console.log('\n=== decisionFunction ===')

await test('decisionFunction returns raw margins', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 10, k: 4 })
  const { X, y } = makeLinearData(40)
  m.fit(X, y)

  const df = m.decisionFunction(X)
  const preds = m.predict(X)

  assert(df.length === preds.length, 'length mismatch')
  assert(df instanceof Float64Array, 'decisionFunction should return Float64Array margins')
  for (let i = 0; i < df.length; i++) {
    const expected = m.classes[df[i] > 0 ? 1 : 0]
    assert(preds[i] === expected, `margin/label mismatch at ${i}`)
  }

  m.dispose()
})

// ============================================================
// Determinism
// ============================================================
console.log('\n=== Determinism ===')

await test('same model predicts consistently', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 10, k: 4 })
  const { X, y } = makeLinearData(60)
  m.fit(X, y)

  const p1 = m.predict(X)
  const p2 = m.predict(X)

  for (let i = 0; i < p1.length; i++) {
    assert(p1[i] === p2[i], `pred ${i}: ${p1[i]} !== ${p2[i]}`)
  }

  m.dispose()
})

await test('seed makes LR/FM/FFM training reproducible within one WASM instance', async () => {
  const { X, y } = makeLinearData(60)
  for (const Cls of [XLearnLRClassifier, XLearnFMClassifier, XLearnFFMClassifier]) {
    const runs = []
    for (const seed of [42, 42, 43, 42]) {
      const m = await Cls.create({ epoch: 10, k: 4, seed })
      m.fit(X, y)
      runs.push({ margins: m.decisionFunction(X), bundle: m.save() })
      m.dispose()
    }

    for (const sameSeedRun of [runs[1], runs[3]]) {
      assert(Buffer.from(runs[0].bundle).equals(Buffer.from(sameSeedRun.bundle)),
        `${Cls.name} bundle differs for the same seed`)
      for (let i = 0; i < runs[0].margins.length; i++) {
        assert(runs[0].margins[i] === sameSeedRun.margins[i],
          `${Cls.name} margin differs at ${i}`)
      }
    }

    let different = false
    for (let i = 0; i < runs[0].margins.length; i++) {
      if (runs[0].margins[i] !== runs[2].margins[i]) different = true
    }
    assert(different, `${Cls.name} different seed did not change a nondegenerate fit`)
  }
})

// ============================================================
// Score
// ============================================================
console.log('\n=== Score ===')

await test('score returns accuracy for classifier', async () => {
  const m = await XLearnFMClassifier.create({ epoch: 15, k: 4 })
  const { X, y } = makeLinearData(80)
  m.fit(X, y)

  const acc = m.score(X, y)
  assert(typeof acc === 'number', 'score should be a number')
  assert(acc > 0.5, `accuracy ${acc} too low`)
  assert(acc <= 1.0, `accuracy ${acc} > 1`)

  m.dispose()
})

await test('score returns R-squared for regressor', async () => {
  const m = await XLearnFMRegressor.create({ epoch: 20, k: 4 })
  const { X, y } = makeRegressionData(80)
  m.fit(X, y)

  const r2 = m.score(X, y)
  assert(typeof r2 === 'number', 'score should be a number')
  assert(r2 > 0.2, `R-squared ${r2} too low`)

  m.dispose()
})

// ============================================================
// Contract and bundle boundaries
// ============================================================
console.log('\n=== Contract and Bundle Boundaries ===')

await test('classifier maps arbitrary int32 labels for dense and CSR input', async () => {
  const { X, y } = makeLinearData(80)
  const publicY = new Int32Array(y.map(value => value ? 20 : 10))
  const m = await XLearnLRClassifier.create({ epoch: 15, seed: 42 })

  m.fit(X, publicY)
  assert(Array.from(m.classes).join(',') === '10,20', `classes=${Array.from(m.classes)}`)
  const densePreds = m.predict(X)
  const margins = m.decisionFunction(X)
  for (let i = 0; i < densePreds.length; i++) {
    assert(densePreds[i] === (margins[i] > 0 ? 20 : 10), `dense label ${i}`)
  }

  m.fit(toCSR(X), publicY)
  const sparsePreds = m.predict(toCSR(X))
  for (const value of sparsePreds) {
    assert(value === 10 || value === 20, `unexpected sparse label ${value}`)
  }
  m.dispose()
})

await test('classifier label validation is transactional on refit', async () => {
  const { X, y } = makeLinearData(40)
  const m = await XLearnFMClassifier.create({ epoch: 10, k: 4, seed: 7 })
  m.fit(X, y)
  const before = m.decisionFunction(X)

  let threw = false
  try { m.fit(X, y.slice(0, -1)) } catch (error) {
    threw = /y length/.test(error.message)
  }
  assert(threw, 'short y must be rejected before C')
  const after = m.decisionFunction(X)
  for (let i = 0; i < before.length; i++) {
    assert(before[i] === after[i], `fitted model changed at ${i}`)
  }
  m.dispose()
})

await test('classifier rejects non-int32 and non-binary labels', async () => {
  const { X } = makeLinearData(4)
  const m = await XLearnLRClassifier.create()
  let badFloat = false
  let oneClass = false
  try { m.fit(X, [0, 1, 0.5, 1]) } catch (error) { badFloat = /int32/.test(error.message) }
  try { m.fit(X, [3, 3, 3, 3]) } catch (error) { oneClass = /exactly 2/.test(error.message) }
  assert(badFloat, 'fractional class must be rejected')
  assert(oneClass, 'one-class fit must be rejected')
  m.dispose()
})

await test('FFM canonicalizes absent fields and validates provided fields', async () => {
  const { X, y } = makeLinearData(40)
  const m = await XLearnFFMClassifier.create({ epoch: 8, k: 4 })
  m.fit(X, y)
  const bytes = m.save()
  const { toc, blobs } = decodeBundle(bytes)
  const fieldEntry = toc.find(entry => entry.id === 'field_map')
  assert(fieldEntry && fieldEntry.length === 8, 'canonical two-feature field map missing')
  const fields = new Int32Array(blobs.buffer.slice(
    blobs.byteOffset + fieldEntry.offset,
    blobs.byteOffset + fieldEntry.offset + fieldEntry.length
  ))
  assert(fields[0] === 0 && fields[1] === 0, `implicit fields=${Array.from(fields)}`)
  m.dispose()

  const loaded = await XLearnFFMClassifier.load(bytes)
  assert(Array.from(loaded.getParams().featureFields).join(',') === '0,0', 'loaded implicit fields differ')
  loaded.dispose()

  for (const featureFields of [new Int32Array([0]), new Int32Array([0, -1]), [0, 1]]) {
    const invalid = await XLearnFFMClassifier.create({ featureFields })
    let rejected = false
    try { invalid.fit(X, y) } catch (error) { rejected = /featureFields/.test(error.message) }
    assert(rejected, `invalid featureFields accepted: ${featureFields}`)
    invalid.dispose()
  }

  const compact = await XLearnFFMClassifier.create({
    epoch: 3,
    k: 4,
    featureFields: new Int32Array([7, 2147483647])
  })
  compact.fit(X, y)
  const compactBytes = compact.save()
  compact.dispose()
  const compactFields = new Int32Array(bundleArtifact(compactBytes, 'field_map').buffer)
  assert(Array.from(compactFields).join(',') === '0,1',
    `gapped fields were not compacted: ${Array.from(compactFields)}`)
  const compactLoaded = await XLearnFFMClassifier.load(compactBytes)
  assert(Array.from(compactLoaded.getParams().featureFields).join(',') === '0,1',
    'compacted fields changed on load')
  compactLoaded.dispose()

  const invalidFieldMap = rewriteBundle(
    compactBytes,
    () => {},
    artifacts => artifacts.map(artifact => artifact.id === 'field_map'
      ? { ...artifact, data: new Uint8Array(new Int32Array([0, 2]).buffer) }
      : artifact)
  )
  await expectReject(
    XLearnFFMClassifier.load(invalidFieldMap),
    /invalid field ID/
  )
})

await test('non-FFM models reject featureFields', async () => {
  const { X, y } = makeLinearData(20)
  const m = await XLearnLRClassifier.create({ featureFields: new Int32Array([0, 1]) })
  let rejected = false
  try { m.fit(X, y) } catch (error) { rejected = /only by FFM/.test(error.message) }
  assert(rejected, 'LR silently accepted featureFields')
  m.dispose()
})

await test('seed is validated, forwarded to WASM, and persisted', async () => {
  const { X, y } = makeLinearData(80)
  const wasm = getWasm()
  const originalSetInt = wasm._wl_xl_set_int
  const observed = []
  wasm._wl_xl_set_int = (handle, keyPtr, value) => {
    let end = keyPtr
    while (wasm.HEAPU8[end] !== 0) end++
    const key = new TextDecoder().decode(wasm.HEAPU8.subarray(keyPtr, end))
    observed.push([key, value])
    return originalSetInt(handle, keyPtr, value)
  }

  let bytes
  try {
    const m = await XLearnFMClassifier.create({ epoch: 12, k: 4, seed: 42 })
    m.fit(X, y)
    bytes = m.save()
    m.dispose()
  } finally {
    wasm._wl_xl_set_int = originalSetInt
  }
  assert(observed.some(([key, value]) => key === 'seed' && value === 42), 'seed did not cross the WASM setter boundary')
  const { manifest } = decodeBundle(bytes)
  assert(manifest.seed === 42 && manifest.params.seed === 42, 'training seed not persisted')

  const invalid = await XLearnFMClassifier.create({ seed: 0 })
  let rejected = false
  try { invalid.fit(X, y) } catch (error) { rejected = /seed must/.test(error.message) }
  assert(rejected, 'seed=0 must be rejected')
  invalid.dispose()
})

await test('direct load verifies artifact hashes', async () => {
  const { X, y } = makeLinearData(30)
  const m = await XLearnLRClassifier.create({ epoch: 5 })
  m.fit(X, y)
  const tampered = new Uint8Array(m.save())
  tampered[tampered.length - 1] ^= 1
  m.dispose()
  await expectReject(XLearnLRClassifier.load(tampered), /SHA-256 mismatch/)
})

await test('split and unified loaders preserve task identity', async () => {
  const clsData = makeLinearData(40)
  const regData = makeRegressionData(40)
  const cls = await XLearnFMClassifier.create({ epoch: 5, k: 4 })
  const reg = await XLearnFMRegressor.create({ epoch: 5, k: 4 })
  cls.fit(clsData.X, clsData.y)
  reg.fit(regData.X, regData.y)
  const clsBytes = cls.save()
  const regBytes = reg.save()
  cls.dispose()
  reg.dispose()

  await expectReject(XLearnFMClassifier.load(regBytes), /cannot load bundle type/)
  await expectReject(XLearnFMRegressor.load(clsBytes), /cannot load bundle type/)

  const unifiedReg = await XLearnFM.load(regBytes)
  assert(unifiedReg.task === 'regression', `unified task=${unifiedReg.task}`)
  assert(unifiedReg.predict(regData.X) instanceof Float64Array, 'unified regressor prediction type')
  unifiedReg.dispose()
})

await test('legacy classifier @1 loads and resaves as corrected @2', async () => {
  const { X, y } = makeLinearData(40)
  const m = await XLearnLRClassifier.create({ epoch: 6, seed: 42 })
  m.fit(X, y)
  const legacy = rewriteBundle(m.save(), manifest => {
    manifest.typeId = 'wlearn.xlearn.lr.classifier@1'
    delete manifest.metadata.labelEncoding
    delete manifest.seed
  })
  m.dispose()

  const loaded = await XLearnLRClassifier.load(legacy)
  for (const label of loaded.predict(X)) assert(label === 0 || label === 1, `legacy label=${label}`)
  const upgraded = decodeBundle(loaded.save()).manifest
  assert(upgraded.typeId === 'wlearn.xlearn.lr.classifier@2', 'legacy resave not upgraded')
  assert(upgraded.seed === 1 && upgraded.params.seed === 1, 'legacy no-op requested seed was not normalized')
  loaded.dispose()
})

await test('legacy classifier rejects ambiguous same-sign class metadata', async () => {
  const { X, y } = makeLinearData(30)
  const m = await XLearnLRClassifier.create({ epoch: 5 })
  m.fit(X, y)
  const ambiguous = rewriteBundle(m.save(), manifest => {
    manifest.typeId = 'wlearn.xlearn.lr.classifier@1'
    manifest.metadata.classes = [10, 20]
    delete manifest.metadata.labelEncoding
    delete manifest.seed
    delete manifest.params.seed
  })
  m.dispose()
  await expectReject(XLearnLRClassifier.load(ambiguous), /ambiguous same-sign label mapping/)
})

await test('FFM legacy missing field map defaults to field zero; @2 rejects it', async () => {
  const { X, y } = makeLinearData(30)
  const m = await XLearnFFMClassifier.create({ epoch: 5, k: 4 })
  m.fit(X, y)
  const current = m.save()
  m.dispose()

  const withoutFields = (typeId) => rewriteBundle(current, manifest => {
    manifest.typeId = typeId
    if (typeId.endsWith('@1')) {
      delete manifest.metadata.labelEncoding
      delete manifest.seed
      delete manifest.params.seed
    }
  }, artifacts => artifacts.filter(artifact => artifact.id !== 'field_map'))

  const legacy = await XLearnFFMClassifier.load(withoutFields('wlearn.xlearn.ffm.classifier@1'))
  assert(Array.from(legacy.getParams().featureFields).join(',') === '0,0', 'legacy fields not synthesized')
  legacy.dispose()
  await expectReject(
    XLearnFFMClassifier.load(withoutFields('wlearn.xlearn.ffm.classifier@2')),
    /missing "field_map"/
  )
})

await test('bundle rejects inconsistent seed metadata', async () => {
  const { X, y } = makeLinearData(30)
  const m = await XLearnLRClassifier.create({ epoch: 5, seed: 3 })
  m.fit(X, y)
  const inconsistent = rewriteBundle(m.save(), manifest => { manifest.seed = 4 })
  m.dispose()
  await expectReject(XLearnLRClassifier.load(inconsistent), /seed does not match/)
})

await test('bundle rejects model blobs whose algorithm or loss disagrees with the manifest', async () => {
  const { X, y } = makeLinearData(30)
  const regression = makeRegressionData(30)
  const families = [
    [XLearnLRClassifier, XLearnLRRegressor],
    [XLearnFMClassifier, XLearnFMRegressor]
  ]
  const classifierBundles = []
  for (const [Classifier, Regressor] of families) {
    const classifier = await Classifier.create({ epoch: 3, seed: 7 })
    const regressor = await Regressor.create({ epoch: 3, seed: 7 })
    classifier.fit(X, y)
    regressor.fit(regression.X, regression.y)
    const classifierBundle = classifier.save()
    classifierBundles.push(classifierBundle)
    const regressorModel = bundleArtifact(regressor.save(), 'model')
    const wrongLoss = rewriteBundle(
      classifierBundle,
      () => {},
      artifacts => artifacts.map(artifact => artifact.id === 'model'
        ? { ...artifact, data: regressorModel }
        : artifact)
    )
    await expectReject(Classifier.load(wrongLoss), /blob identity/)
    classifier.dispose()
    regressor.dispose()
  }

  const fmModel = bundleArtifact(classifierBundles[1], 'model')
  const wrongAlgorithm = rewriteBundle(
    classifierBundles[0],
    () => {},
    artifacts => artifacts.map(artifact => artifact.id === 'model'
      ? { ...artifact, data: fmModel }
      : artifact)
  )
  await expectReject(XLearnLRClassifier.load(wrongAlgorithm), /blob identity/)
})

await test('bundle rejects truncated and count-inconsistent model blobs before native use', async () => {
  const { X, y } = makeLinearData(30)
  const model = await XLearnLRClassifier.create({ epoch: 3, seed: 7 })
  model.fit(X, y)
  const bundle = model.save()
  model.dispose()
  const raw = bundleArtifact(bundle, 'model')
  const offsets = modelScalarOffsets(raw)

  const cases = [
    raw.subarray(0, offsets.nFeatures + 4),
    (() => {
      const changed = Uint8Array.from(raw)
      const view = new DataView(changed.buffer)
      view.setUint32(offsets.nWeights, view.getUint32(offsets.nWeights, true) + 1, true)
      return changed
    })(),
    (() => {
      const changed = new Uint8Array(raw.length + 1)
      changed.set(raw)
      return changed
    })()
  ]

  for (const modelBytes of cases) {
    const malformed = rewriteBundle(
      bundle,
      () => {},
      artifacts => artifacts.map(artifact => artifact.id === 'model'
        ? { ...artifact, data: modelBytes }
        : artifact)
    )
    await expectReject(
      XLearnLRClassifier.load(malformed),
      /truncated|inconsistent/
    )
  }
})

// ============================================================
// Error handling
// ============================================================
console.log('\n=== Error Handling ===')

await test('save throws before fit', async () => {
  const m = await XLearnFMClassifier.create()
  let threw = false
  try { m.save() } catch { threw = true }
  assert(threw, 'save before fit should throw')
  m.dispose()
})

await test('score throws before fit', async () => {
  const m = await XLearnFMClassifier.create()
  let threw = false
  try { m.score([[1, 2]], [1]) } catch { threw = true }
  assert(threw, 'score before fit should throw')
  m.dispose()
})

await test('fit throws after dispose', async () => {
  const m = await XLearnFMClassifier.create()
  m.dispose()
  let threw = false
  try { m.fit([[1, 0], [0, 1]], [1, 0]) } catch { threw = true }
  assert(threw, 'fit after dispose should throw')
})

// ============================================================
// Summary
// ============================================================
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`)
process.exit(failed > 0 ? 1 : 0)

} // end main

main()
