import { XLearnBase, LOAD_SENTINEL } from './base.js'
import { register } from '@wlearn/core'

export class XLearnFMClassifier extends XLearnBase {
  static async create(params = {}) {
    return XLearnBase._create('fm', 'binary', params, XLearnFMClassifier)
  }

  static async load(bytes) {
    return XLearnBase._load(bytes, XLearnFMClassifier)
  }

  static async _fromBundle(manifest, toc, blobs) {
    return XLearnBase._fromBundle(manifest, toc, blobs, XLearnFMClassifier)
  }

  get _typeId() { return 'wlearn.xlearn.fm.classifier@1' }

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
      k: { type: 'int_uniform', low: 2, high: 16 },
      opt: { type: 'categorical', values: ['adagrad', 'ftrl', 'sgd'] },
      epoch: { type: 'int_uniform', low: 5, high: 50 }
    }
  }
}

export class XLearnFMRegressor extends XLearnBase {
  static async create(params = {}) {
    return XLearnBase._create('fm', 'reg', params, XLearnFMRegressor)
  }

  static async load(bytes) {
    return XLearnBase._load(bytes, XLearnFMRegressor)
  }

  static async _fromBundle(manifest, toc, blobs) {
    return XLearnBase._fromBundle(manifest, toc, blobs, XLearnFMRegressor)
  }

  get _typeId() { return 'wlearn.xlearn.fm.regressor@1' }

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
      k: { type: 'int_uniform', low: 2, high: 16 },
      opt: { type: 'categorical', values: ['adagrad', 'ftrl', 'sgd'] },
      epoch: { type: 'int_uniform', low: 5, high: 50 }
    }
  }
}

register('wlearn.xlearn.fm.classifier@1', (m, t, b) => XLearnFMClassifier._fromBundle(m, t, b))
register('wlearn.xlearn.fm.regressor@1', (m, t, b) => XLearnFMRegressor._fromBundle(m, t, b))
