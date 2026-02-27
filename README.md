# @wlearn/xlearn

xLearn v0.44 compiled to WebAssembly. Logistic regression, factorization machines (FM), and field-aware factorization machines (FFM) in browsers and Node.js.

Based on [xLearn v0.44](https://github.com/aksnzhy/xlearn) (Apache-2.0). Zero dependencies beyond `@wlearn/core`. ESM.

## Install

```bash
npm install @wlearn/xlearn
```

## Quick start

```js
import { XLearnFMClassifier } from '@wlearn/xlearn'

const model = await XLearnFMClassifier.create({
  epoch: 10,
  k: 4,
  lr: 0.2
})

// Train -- accepts number[][], { data: Float64Array, rows, cols }, or CSR
model.fit(
  [[1, 0], [0, 1], [1, 1], [0, 0]],
  [1, 0, 1, 0]
)

// Predict
const preds = model.predict([[1, 0], [0, 1]])         // Float64Array (raw margins)
const probs = model.predictProba([[1, 0], [0, 1]])     // Float64Array (nrow * 2)
const accuracy = model.score([[1, 0], [0, 1]], [1, 0])

// Save / load
const buf = model.save()  // Uint8Array (WLRN bundle)
const model2 = await XLearnFMClassifier.load(buf)

// Clean up -- required, WASM memory is not garbage collected
model.dispose()
model2.dispose()
```

## Model types

Six model classes covering three algorithms and two tasks:

| Class | Algorithm | Task |
|-------|-----------|------|
| `XLearnLRClassifier` | Logistic regression | Binary classification |
| `XLearnLRRegressor` | Linear regression | Regression |
| `XLearnFMClassifier` | Factorization machine | Binary classification |
| `XLearnFMRegressor` | Factorization machine | Regression |
| `XLearnFFMClassifier` | Field-aware FM | Binary classification |
| `XLearnFFMRegressor` | Field-aware FM | Regression |

All classes share the same API. Binary classification only (no multiclass).

## FFM with field mapping

FFM requires a feature-to-field mapping. Pass `featureFields` as an `Int32Array` where each entry maps a feature index to its field ID:

```js
import { XLearnFFMClassifier } from '@wlearn/xlearn'

// Features 0-2 belong to field 0, features 3-5 belong to field 1
const featureFields = new Int32Array([0, 0, 0, 1, 1, 1])

const model = await XLearnFFMClassifier.create({
  epoch: 10,
  k: 4,
  featureFields
})

model.fit(X, y)
```

The field map is preserved in save/load bundles as a separate `field_map` artifact.

## Sparse input (CSR)

For sparse data (common in CTR/recommender systems), pass a CSR matrix directly:

```js
const csr = {
  rows: 4,
  cols: 3,
  data: new Float64Array([1.0, 2.0, 3.0, 4.0]),
  indices: new Int32Array([0, 2, 1, 0]),
  indptr: new Int32Array([0, 1, 2, 3, 4])
}

model.fit(csr, y)
model.predict(csr)
```

CSR avoids materializing a dense matrix and is passed directly to the WASM layer.

## API

### `Model.create(params?)` -> `Promise<Model>`

Async factory. Loads WASM module on first call, returns a ready-to-use model.

### `model.fit(X, y)` -> `this`

Train on data. Returns `this`.
- `X` -- `number[][]`, `{ data: Float64Array, rows, cols }`, or CSR matrix
- `y` -- `number[]` or `Float64Array`

### `model.predict(X)` -> `Float64Array`

Returns raw margins (classifier) or values (regressor).

### `model.predictProba(X)` -> `Float64Array`

Returns flat array of shape `nrow * 2` (columns: P(class 0), P(class 1)). Classifiers only.

### `model.decisionFunction(X)` -> `Float64Array`

Returns raw decision values. Same as `predict()`.

### `model.score(X, y)` -> `number`

Accuracy (classification) or R-squared (regression).

### `model.save()` / `Model.load(buffer)`

Save to / load from `Uint8Array` (WLRN bundle with xLearn binary model blob).

### `model.dispose()`

Free WASM memory. Required. Idempotent.

### `model.getParams()` / `model.setParams(p)`

Get/set hyperparameters. Enables AutoML grid search and cloning.

### `Model.defaultSearchSpace()`

Returns default hyperparameter search space for AutoML.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 0.2 | Learning rate |
| `lambda` | float | 0.00002 | L2 regularization |
| `k` | int | 4 | Latent factor dimension (FM/FFM only, ignored for LR) |
| `epoch` | int | 10 | Number of training epochs |
| `opt` | string | `'adagrad'` | Optimizer: `'sgd'`, `'adagrad'`, `'ftrl'` |
| `alpha` | float | 0.01 | FTRL alpha |
| `beta` | float | 1.0 | FTRL beta |
| `lambda_1` | float | 0.0 | FTRL L1 penalty |
| `lambda_2` | float | 0.0 | FTRL L2 penalty |
| `normalize` | bool | true | Instance-wise L2 normalization |
| `featureFields` | Int32Array | null | Feature-to-field map (FFM only) |

## Capabilities

| Feature | LR | FM | FFM |
|---------|----|----|-----|
| classifier | yes | yes | yes |
| regressor | yes | yes | yes |
| predictProba | yes | yes | yes |
| decisionFunction | yes | yes | yes |
| csr | yes | yes | yes |
| sampleWeight | no | no | no |
| earlyStopping | no | no | no |

## Resource management

WASM heap memory is not garbage collected. Call `.dispose()` on every model when done. A `FinalizationRegistry` safety net warns if you forget, but do not rely on it.

## Build from source

Requires [Emscripten](https://emscripten.org/) (emsdk) activated.

```bash
git clone --recurse-submodules https://github.com/wlearn-org/xlearn-wasm
cd xlearn-wasm
npm install
npm run build
npm test
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

## Upstream

Based on [xLearn v0.44](https://github.com/aksnzhy/xlearn) (Apache-2.0).

Modifications for WASM:

- **exit() to throw**: xLearn calls `exit(0)` and `exit(1)` in `solver.cc` and `checker.cc` for parameter validation failures. In WASM, `exit()` kills the entire runtime. Replaced with `throw std::runtime_error(...)` so errors are caught by the C API's `API_BEGIN`/`API_END` macros and reported via `XLearnGetLastError()`.

- **std::min type mismatch**: `file_util.h` calls `std::min(pos + kChunkSize, end)` where `kChunkSize` is `uint32` and `end` is `long`. Emscripten's strict type checking rejects this. Fixed by casting `kChunkSize` to `long`.

- **Sequential thread pool**: xLearn's `ThreadPool` uses `std::thread` (not available in WASM without pthreads). Replaced with a drop-in sequential implementation via force-include header that executes tasks inline on the calling thread.

- **Dense DMatrix zero handling**: xLearn's upstream `XlearnCreateDataFromMat` includes zero-valued features when building the internal sparse DMatrix. This causes `norm = 1.0 / 0.0 = inf` for all-zero rows, which propagates NaN through FM/FFM gradient updates (`inf * 0 = NaN`). The custom `wl_xl_create_dmatrix_dense` skips zero values (matching the file-reader behavior) and defaults `norm = 1.0` for all-zero rows.

- **stdout suppression**: xLearn prints verbose banners and progress to stdout even with `quiet=true`. The C adapter redirects fd 1 to `/dev/null` via `dup2` during fit/predict and restores it afterward.

- **SSE to WASM SIMD**: xLearn's FM/FFM scoring uses SSE3 intrinsics for vectorized dot products. Built with `-msimd128 -msse3` for Emscripten's SSE-to-WASM SIMD translation layer.

## License

Apache-2.0 (same as upstream xLearn)
