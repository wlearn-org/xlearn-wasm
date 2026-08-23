# @wlearn/xlearn

xLearn v0.44 compiled to WebAssembly. Logistic regression, factorization machines (FM), and field-aware factorization machines (FFM) in browsers and Node.js.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)). Based on [xLearn v0.44](https://github.com/aksnzhy/xlearn) (Apache-2.0). Zero dependencies beyond `@wlearn/core`. CommonJS.

## Install

```bash
npm install @wlearn/xlearn
```

## Quick start

```js
const { readFileSync, writeFileSync } = require('fs')
const { XLearnFM } = require('@wlearn/xlearn')

const model = await XLearnFM.create({
  task: 'classification',  // or 'regression'; auto-detected from labels if omitted
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
const preds = model.predict([[1, 0], [0, 1]])          // Int32Array class labels
const margins = model.decisionFunction([[1, 0], [0, 1]]) // Float64Array
const probs = model.predictProba([[1, 0], [0, 1]])     // Float64Array (nrow * 2)
const accuracy = model.score([[1, 0], [0, 1]], [1, 0])

// Save / load
writeFileSync('xlearn-fm.wlrn', model.save())
const model2 = await XLearnFM.load(readFileSync('xlearn-fm.wlrn'))
model2.predict([[1, 0]])
```

## Model types

Three unified model classes (recommended), plus six task-specific classes:

| Unified Class | Algorithm | Split Classes |
|---------------|-----------|---------------|
| `XLearnLR` | Logistic / linear regression | `XLearnLRClassifier`, `XLearnLRRegressor` |
| `XLearnFM` | Factorization machine | `XLearnFMClassifier`, `XLearnFMRegressor` |
| `XLearnFFM` | Field-aware FM | `XLearnFFMClassifier`, `XLearnFFMRegressor` |

Unified classes accept `task: 'classification'` or `task: 'regression'` and auto-detect from labels if omitted. Binary classification only (no multiclass).

## FFM with field mapping

For meaningful field-aware interactions, pass `featureFields` as a nonnegative `Int32Array` where each entry maps one feature index to its field ID:

```js
const { XLearnFFMClassifier } = require('@wlearn/xlearn')

// Features 0-2 belong to field 0, features 3-5 belong to field 1
const featureFields = new Int32Array([0, 0, 0, 1, 1, 1])

const model = await XLearnFFMClassifier.create({
  epoch: 10,
  k: 4,
  featureFields
})

model.fit(X, y)
```

The map length must equal the number of input features. If omitted, every feature
belongs to field 0. The resolved map is preserved in save/load bundles as a
separate `field_map` artifact.

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
- classifier `y` -- exactly two arbitrary int32-valued classes; class order is sorted
- regressor `y` -- finite numeric targets

### `model.predict(X)` -> `Int32Array | Float64Array`

Returns public class labels as `Int32Array` for classifiers and numeric values as
`Float64Array` for regressors.

### `model.predictProba(X)` -> `Float64Array`

Returns a flat array of shape `nrow * 2`. Its columns follow the sorted order in
`model.classes`. Classifiers only.

### `model.decisionFunction(X)` -> `Float64Array`

Returns raw decision margins. For a classifier, a margin greater than zero maps
to `model.classes[1]`; other margins map to `model.classes[0]`.

### `model.score(X, y)` -> `number`

Accuracy (classification) or R-squared (regression).

### `model.save()` / `Model.load(buffer)`

Save to / load from `Uint8Array` (WLRN bundle with xLearn binary model blob).

### `model.dispose()`

Release WASM memory immediately. Use in long-running apps, workers, cross-validation, and AutoML loops. Idempotent.

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
| `seed` | int >= 1 | 1 | Data-shuffle seed; saved in the model bundle |
| `featureFields` | Int32Array | all zeros | Nonnegative feature-to-field map (FFM only) |

## Classifier migration from 0.2

The 0.2 package returned raw margins from classifier `predict()`. Current
classifier bundles use `wlearn.xlearn.*.classifier@2`, and `predict()` follows
the common wlearn estimator contract by returning public labels. Code that needs
margins should call `decisionFunction()`.

Legacy `@1` classifier bundles remain loadable when they were trained with
contract-valid integer labels whose lower class is nonpositive and upper class
is positive, such as `[0, 1]` or `[-1, 1]`. The old writer could not distinguish
two same-sign public classes; such bundles are rejected with a retraining error
instead of silently producing incorrect labels.

## Regression target scale

xLearn's squared-loss optimizer operates on the target values supplied to it and
can be sensitive to their scale. This wrapper does not silently transform `y`.
For large-magnitude targets, compute scaling statistics on the training targets,
fit on the transformed targets, and apply the inverse transform to predictions:

```js
const mean = yTrain.reduce((sum, value) => sum + value, 0) / yTrain.length
const variance = yTrain.reduce((sum, value) => sum + (value - mean) ** 2, 0) / yTrain.length
const scale = Math.sqrt(variance) || 1
const yScaled = yTrain.map(value => (value - mean) / scale)

model.fit(XTrain, yScaled)
const predictions = Float64Array.from(
  model.predict(XTest), value => value * scale + mean
)
```

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

Use `.dispose()` when creating and discarding many models so WASM memory is released promptly.

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

- **Deterministic shuffling**: Emscripten's parameterless `std::random_shuffle`
  does not consume the state set by `srand()`. The WASM build uses
  `std::shuffle` with an explicit `std::mt19937` seeded from the public `seed`
  parameter, making repeated LR/FM/FFM fits reproducible within a runtime.

## License

Apache-2.0 (same as upstream xLearn)
