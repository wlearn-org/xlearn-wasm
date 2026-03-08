const { loadXLearn, getWasm } = require('./wasm.js')
const { XLearnLRClassifier, XLearnLRRegressor } = require('./lr.js')
const { XLearnFMClassifier, XLearnFMRegressor } = require('./fm.js')
const { XLearnFFMClassifier, XLearnFFMRegressor } = require('./ffm.js')

module.exports = {
  loadXLearn, getWasm,
  XLearnLRClassifier, XLearnLRRegressor,
  XLearnFMClassifier, XLearnFMRegressor,
  XLearnFFMClassifier, XLearnFFMRegressor
}
