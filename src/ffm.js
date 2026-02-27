import { XLearnBase, LOAD_SENTINEL } from './base.js'
import { register } from '@wlearn/core'

export class XLearnFFMClassifier extends XLearnBase {
  static async create(params = {}) {
    return XLearnBase._create('ffm', 'binary', params, XLearnFFMClassifier)
  }

  static async load(bytes) {
    return XLearnBase._load(bytes, XLearnFFMClassifier)
  }

  static async _fromBundle(manifest, toc, blobs) {
    return XLearnBase._fromBundle(manifest, toc, blobs, XLearnFFMClassifier)
  }

  get _typeId() { return 'wlearn.xlearn.ffm.classifier@1' }

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
      k: { type: 'int_uniform', low: 2, high: 8 },
      opt: { type: 'categorical', values: ['adagrad', 'ftrl', 'sgd'] },
      epoch: { type: 'int_uniform', low: 5, high: 50 }
    }
  }
}

export class XLearnFFMRegressor extends XLearnBase {
  static async create(params = {}) {
    return XLearnBase._create('ffm', 'reg', params, XLearnFFMRegressor)
  }

  static async load(bytes) {
    return XLearnBase._load(bytes, XLearnFFMRegressor)
  }

  static async _fromBundle(manifest, toc, blobs) {
    return XLearnBase._fromBundle(manifest, toc, blobs, XLearnFFMRegressor)
  }

  get _typeId() { return 'wlearn.xlearn.ffm.regressor@1' }

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
      k: { type: 'int_uniform', low: 2, high: 8 },
      opt: { type: 'categorical', values: ['adagrad', 'ftrl', 'sgd'] },
      epoch: { type: 'int_uniform', low: 5, high: 50 }
    }
  }
}

register('wlearn.xlearn.ffm.classifier@1', (m, t, b) => XLearnFFMClassifier._fromBundle(m, t, b))
register('wlearn.xlearn.ffm.regressor@1', (m, t, b) => XLearnFFMRegressor._fromBundle(m, t, b))
