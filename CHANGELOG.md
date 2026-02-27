# Changelog

## 0.1.0 (unreleased)

- Initial release
- xLearn v0.44 compiled to WASM via Emscripten with SSE-to-WASM SIMD translation
- Six model classes: LR, FM, FFM classifiers and regressors
- Unified sklearn-style API: `create()`, `fit()`, `predict()`, `score()`, `save()`, `dispose()`
- `predictProba()` for binary classifiers (sigmoid-based)
- `decisionFunction()` for raw decision values
- Dense and CSR sparse input support
- FFM field mapping via `featureFields: Int32Array`
- WLRN bundle format with xLearn binary model artifact
- `getParams()`/`setParams()` for AutoML integration
- `defaultSearchSpace()` for hyperparameter search
- `FinalizationRegistry` safety net for leak detection
- Registry loaders for `@wlearn/core` global `load()` dispatch
- 44 tests covering all model types, save/load, CSR, FFM fields, error handling
- Apache-2.0 license (same as upstream xLearn)
