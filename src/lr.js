import { XLearnBase, LOAD_SENTINEL } from './base.js'
import { register } from '@wlearn/core'

export class XLearnLRClassifier extends XLearnBase {
  static async create(params = {}) {
    return XLearnBase._create('linear', 'binary', params, XLearnLRClassifier)
  }

  static async load(bytes) {
    return XLearnBase._load(bytes, XLearnLRClassifier)
  }

  static async _fromBundle(manifest, toc, blobs) {
    return XLearnBase._fromBundle(manifest, toc, blobs, XLearnLRClassifier)
  }

  get _typeId() { return 'wlearn.xlearn.lr.classifier@1' }

  get capabilities() {
    return {
      classifier: true, regressor: false, predictProba: true,
      decisionFunction: true, sampleWeight: false, csr: true,
      earlyStopping: false
    }
  }

  static defaultSearchSpace() {
    return {
      lr: { type: 'log_uniform', low: 1e-4, high: 1.0 },
      lambda: { type: 'log_uniform', low: 1e-6, high: 1e-1 },
      opt: { type: 'categorical', values: ['adagrad', 'ftrl', 'sgd'] },
      epoch: { type: 'int_uniform', low: 5, high: 50 }
    }
  }
}

export class XLearnLRRegressor extends XLearnBase {
  static async create(params = {}) {
    return XLearnBase._create('linear', 'reg', params, XLearnLRRegressor)
  }

  static async load(bytes) {
    return XLearnBase._load(bytes, XLearnLRRegressor)
  }

  static async _fromBundle(manifest, toc, blobs) {
    return XLearnBase._fromBundle(manifest, toc, blobs, XLearnLRRegressor)
  }

  get _typeId() { return 'wlearn.xlearn.lr.regressor@1' }

  get capabilities() {
    return {
      classifier: false, regressor: true, predictProba: false,
      decisionFunction: true, sampleWeight: false, csr: true,
      earlyStopping: false
    }
  }

  static defaultSearchSpace() {
    return {
      lr: { type: 'log_uniform', low: 1e-4, high: 1.0 },
      lambda: { type: 'log_uniform', low: 1e-6, high: 1e-1 },
      opt: { type: 'categorical', values: ['adagrad', 'ftrl', 'sgd'] },
      epoch: { type: 'int_uniform', low: 5, high: 50 }
    }
  }
}

register('wlearn.xlearn.lr.classifier@1', (m, t, b) => XLearnLRClassifier._fromBundle(m, t, b))
register('wlearn.xlearn.lr.regressor@1', (m, t, b) => XLearnLRRegressor._fromBundle(m, t, b))
