const { loadXLearn, getWasm } = require('./wasm.js')
const { XLearnLRClassifier, XLearnLRRegressor } = require('./lr.js')
const { XLearnFMClassifier, XLearnFMRegressor } = require('./fm.js')
const { XLearnFFMClassifier, XLearnFFMRegressor } = require('./ffm.js')
const { createModelClass } = require('@wlearn/core')

const XLearnLR = createModelClass(XLearnLRClassifier, XLearnLRRegressor, { name: 'XLearnLR', load: loadXLearn })
const XLearnFM = createModelClass(XLearnFMClassifier, XLearnFMRegressor, { name: 'XLearnFM', load: loadXLearn })
const XLearnFFM = createModelClass(XLearnFFMClassifier, XLearnFFMRegressor, { name: 'XLearnFFM', load: loadXLearn })

module.exports = {
  loadXLearn, getWasm,
  // Unified classes (recommended)
  XLearnLR, XLearnFM, XLearnFFM,
  // Original split classes (backward compat)
  XLearnLRClassifier, XLearnLRRegressor,
  XLearnFMClassifier, XLearnFMRegressor,
  XLearnFFMClassifier, XLearnFFMRegressor
}
