import { decodeBundle, load as coreLoad } from '@wlearn/core'

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

const {
  loadXLearn,
  XLearnLRClassifier, XLearnLRRegressor,
  XLearnFMClassifier, XLearnFMRegressor,
  XLearnFFMClassifier, XLearnFFMRegressor
} = await import('../src/index.js')

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
  const { getWasm } = await import('../src/wasm.js')
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
  assert(preds instanceof Float64Array, 'should be Float64Array')
  assert(preds.length === 80, `expected 80, got ${preds.length}`)
  // Raw margins -- should be numbers, not NaN
  for (let i = 0; i < preds.length; i++) {
    assert(!isNaN(preds[i]), `prediction ${i} is NaN`)
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
  assert(manifest.typeId === 'wlearn.xlearn.fm.classifier@1', `typeId=${manifest.typeId}`)
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
  assert(manifest.typeId === 'wlearn.xlearn.ffm.classifier@1', `typeId=${manifest.typeId}`)
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
  // Verify sign agreement (classification direction preserved)
  for (let i = 0; i < p1.length; i++) {
    assert(Math.sign(p1[i]) === Math.sign(p2[i]) || Math.abs(p1[i]) < 0.1,
      `sign mismatch at ${i}: ${p1[i]} vs ${p2[i]}`)
  }
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
  assert(manifest.typeId === 'wlearn.xlearn.fm.classifier@1', `typeId=${manifest.typeId}`)
  assert(manifest.metadata, 'missing metadata')
  assert(manifest.metadata.algo === 'fm', `algo=${manifest.metadata.algo}`)
  assert(manifest.metadata.task === 'binary', `task=${manifest.metadata.task}`)
  assert(manifest.metadata.nFeatures === 2, `nFeatures=${manifest.metadata.nFeatures}`)
  assert(manifest.metadata.nClasses === 2, `nClasses=${manifest.metadata.nClasses}`)

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
  for (let i = 0; i < df.length; i++) {
    assert(df[i] === preds[i], `df[${i}]=${df[i]} !== predict[${i}]=${preds[i]}`)
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

await test('separate training runs produce similar predictions', async () => {
  const { X, y } = makeLinearData(60)

  const m1 = await XLearnLRClassifier.create({ epoch: 10 })
  m1.fit(X, y)
  const p1 = m1.predict(X)
  const s1 = m1.score(X, y)
  m1.dispose()

  const m2 = await XLearnLRClassifier.create({ epoch: 10 })
  m2.fit(X, y)
  const p2 = m2.predict(X)
  const s2 = m2.score(X, y)
  m2.dispose()

  // xLearn has minor non-determinism from WASM heap layout, but
  // predictions should be close and classification direction should agree
  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 0.5, `pred ${i}: ${p1[i]} vs ${p2[i]} too different`)
  }
  assert(s1 === s2, `scores should match: ${s1} !== ${s2}`)
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
