import os
import inspect
import json
import importlib
import pickle
from itertools import chain
from typing import List

import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import SampleForecast
from gluonts.dataset.util import forecast_start

from data_loader import convert_tsf_to_dataframe as load_data
from mlprops.util import read_json


class HorizonFCEnsemble:

    def __init__(self, freq, context_length, prediction_length, samples_per_series=100):
        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.samples_per_series = samples_per_series
        self.lead_time = 0

    def fit(self, model_idx, X, y):
        self.models[model_idx].fit(X, y)
        
    def train(self, training_data):
        sampled_window_starts = []
        # identify context windows to train on
        for ser in training_data:
            valid_starts = np.arange(ser['target'].size - self.context_length - self.prediction_length + 1)
            if valid_starts.size == 0:
                continue
            if valid_starts.size > self.samples_per_series:
                sampled_window_starts.append( (ser['target'], np.random.choice(valid_starts, size=self.samples_per_series)) )
            else:
                sampled_window_starts.append( (ser['target'], valid_starts) )
        n_samples = int(np.sum([starts.size for _, starts in sampled_window_starts]))
        # run a training step for each individual model
        for pred_idx in range(self.prediction_length):
            print(f'Training AutoLearn regressor {pred_idx+1} / {self.prediction_length}')
            y = np.zeros((n_samples), dtype=training_data[0]['target'].dtype)
            X = np.zeros((n_samples, self.context_length), dtype=training_data[0]['target'].dtype)
            idx = 0
            for ser, starts in sampled_window_starts:
                for start in starts:
                    X[idx,:] = ser[start:(start + self.context_length)]
                    y[idx] = ser[start + self.context_length + pred_idx]
                    idx += 1
            self.fit(pred_idx, X, y)
        return self
    
    def predict(self, dataset, num_samples):
        if num_samples:
            print("Forecast is not sample based. Ignoring parameter `num_samples` from predict method.")
        X_test = np.array([ts['target'][-self.context_length:] for ts in dataset])
        ys = np.array([self.predict_single(model, X_test) for model in self.models])
        fc = [ SampleForecast(samples=np.expand_dims(ys[:,i], 0), start_date=forecast_start(ts)) for i, ts in enumerate(dataset) ]
        return fc
    
    def predict_single(self, model, X_test):
        return model.predict(X_test)


class AutoGluon(HorizonFCEnsemble):
    
    def __init__(self, freq, context_length, prediction_length, samples_per_series=100, time_budget=1000, output_dir=None):
        if output_dir is None:
            output_dir = os.environ['TMPDIR']
        super().__init__(freq, context_length, prediction_length, samples_per_series)
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        self.time_budget = time_budget
        self.data_class = TimeSeriesDataFrame
        self.model = TimeSeriesPredictor(prediction_length=self.prediction_length, path=output_dir, target="target", eval_metric="MASE")

    def as_tsdf(self, data):
        data_list = []
        for idx, series in enumerate(data):
            data_list.append(pd.DataFrame(series['target'], columns=['target']))
            data_list[-1]['item_id'] = idx
            try:
                data_list[-1]['timestamp'] = pd.date_range(start=series['start'], periods=len(series['target']), freq=self.freq)
            except ValueError: # start might also be a period (e.g., a day)
                data_list[-1]['timestamp'] = pd.date_range(start=series['start'].start_time, periods=len(series['target']), freq=self.freq)
        return self.data_class( pd.concat(data_list) )

    def train(self, training_data):
        train_data = self.as_tsdf(training_data)
        self.model.fit(train_data, presets="medium_quality", time_limit=self.time_budget)
        return self

    def predict(self, dataset, num_samples):
        test_data = self.as_tsdf(dataset)
        predictions = self.model.predict(test_data)
        ys = predictions['mean'].values.reshape(len(dataset), self.prediction_length)
        fc = [ SampleForecast(samples=np.expand_dims(ys[i,:], 0), start_date=forecast_start(ts)) for i, ts in enumerate(dataset) ]
        return fc
    
    def get_param_count(self):
        return -1

    def get_fsize(self, output_path=None):
        return sum([os.path.getsize(os.path.join(dp, f)) for dp, _, filenames in os.walk(self.model.path) for f in filenames])


class AutoSklearn(HorizonFCEnsemble):

    def __init__(self, freq, context_length, prediction_length, samples_per_series=100, time_budget=1000):
        super().__init__(freq, context_length, prediction_length, samples_per_series)
        from autosklearn.regression import AutoSklearnRegressor
        per_regr = max(30, time_budget // self.prediction_length)
        self.models = [AutoSklearnRegressor(per_regr) for _ in range(self.prediction_length)]

    def get_param_count(self):
        param_lookup = {
            'MyDummyRegressor': lambda clf: 1,
            'AdaboostRegressor': lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_]),
            'DecisionTree': lambda clf: sum([clf.tree_.node_count * 2]),
            'ExtraTreesRegressor': lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_]),
            'KNearestNeighborsRegressor': lambda clf: clf.n_features_in_ * clf.n_samples_fit_,
            'MLPRegressor': lambda clf: sum([layer_w.size for layer_w in clf.coefs_] + [layer_i.size for layer_i in clf.intercepts_]),
            'RandomForest': lambda clf: sum([tree.tree_.node_count * 2 for tree in clf.estimators_]),
            'SGD': lambda clf: sum([clf.coef_.size, clf.intercept_.size]),
            'GradientBoosting': lambda clf: sum([len(tr[0].nodes[0]) * tr[0].nodes.size for tr in clf.estimator._predictors])
        }
        params = 0
        try:
            for model in self.models:
                for _, ens_mod in model.get_models_with_weights():
                    params += 1 # ensemble member weight
                    if hasattr(ens_mod, 'named_steps'): # filter away the preprocessing
                        ens_mod = ens_mod.named_steps['regressor'].choice
                    params += param_lookup[ens_mod.__class__.__name__](ens_mod)
        except Exception:
            params = -1
        return params

    def get_fsize(self, output_path):
        fsize = 0
        for idx, mod in enumerate(self.models):
            fname = os.path.join(output_path, f"model_{idx}")
            with open(fname, 'wb') as modelfile:
                pickle.dump(mod, modelfile)
            fsize += os.path.getsize(fname)
        return fsize


class AutoKeras(HorizonFCEnsemble):

    def __init__(self, freq, context_length, prediction_length, samples_per_series=100, epochs=100, max_trials=2):
        super().__init__(freq, context_length, prediction_length, samples_per_series)
        self.epochs = epochs
        self.max_trials = max_trials
        import autokeras as ak
        self.models = [ak.StructuredDataRegressor(max_trials=self.max_trials, directory=os.environ['TMPDIR']) for _ in range(self.prediction_length)]

    def fit(self, model_idx, X, y):
        self.models[model_idx].fit(X, y, epochs=self.epochs)

    def predict_single(self, model, X_test):
        return model.predict(X_test)[:,0]

    def get_param_count(self):
        params = sum( [ mod.export_model().count_params() for mod in self.models ] )
        return params

    def get_fsize(self, output_path):
        fsize = 0
        for idx, mod in enumerate(self.models):
            fname = os.path.join(output_path, f"model_{idx}")
            mod.export_model().save(fname, save_format="tf")
            fsize += os.path.getsize(fname)
        return fsize


class AutoPyTorch(HorizonFCEnsemble):
    
    def __init__(self, freq, context_length, prediction_length, samples_per_series=100):
        super().__init__(freq, context_length, prediction_length, samples_per_series)
        

    def train(self, training_data):
        sampled_window_starts = []
        # identify context windows to train on
        for ser in training_data:
            valid_starts = np.arange(ser['target'].size - self.context_length - self.prediction_length + 1)
            if valid_starts.size == 0:
                continue
            if valid_starts.size > self.samples_per_series:
                sampled_window_starts.append( (ser['target'], np.random.choice(valid_starts, size=self.samples_per_series)) )
            else:
                sampled_window_starts.append( (ser['target'], valid_starts) )
        n_samples = int(np.sum([starts.size for _, starts in sampled_window_starts]))
        # run a training step for each individual model
        print(f'Training AutoLearn regressor {pred_idx+1} / {self.prediction_length}')
        y = np.zeros((n_samples), dtype=training_data[0]['target'].dtype)
        X = np.zeros((n_samples, self.context_length), dtype=training_data[0]['target'].dtype)
        idx = 0
        for ser, starts in sampled_window_starts:
            for start in starts:
                X[idx,:] = ser[start:(start + self.context_length)]
                y[idx] = ser[start + self.context_length + pred_idx]
                idx += 1
        self.fit(pred_idx, X, y)

        print(1)

    def predict(self, dataset, num_samples):
        print(2)


with open('meta_model.json', 'r') as meta:
    MODELS = json.load(meta)


def init_model_and_data(args):
    dataset, model, lag, epochs = args.dataset, args.model, args.lag, args.epochs

    model_props = MODELS[model]
    if model_props['module'] is None:
        if not model_props['class'] in globals():
            raise RuntimeError
        else:
            model_cls = globals()[model_props['class']]
    else:
        module = importlib.import_module(model_props['module'])
        # bugfix for deepstate model frequency map
        if hasattr(module, "FREQ_LONGEST_PERIOD_DICT") and 'Q' not in module.FREQ_LONGEST_PERIOD_DICT:
            module.FREQ_LONGEST_PERIOD_DICT['Q'] = 4
        model_cls = getattr(module, model_props['class'])

    full_path = os.path.join(args.datadir, dataset + '.tsf')

    hc = None if not hasattr(args, "external_forecast_horizon") else args.external_forecast_horizon
    ds, freq, seasonality, forecast_horizon, contain_missing_values, contain_equal_length = load_data(full_path, ds_sample_seed=args.ds_seed, ext_fc_horizon=hc)

    all_train_ts = []
    all_fcast_ts = []

    for _, row in ds.iterrows():

        ts_start = row["start_timestamp"]
        ts_data = row["series_value"]

        # use gluonts data format
        all_train_ts.append( {
            FieldName.TARGET: ts_data[:len(ts_data) - forecast_horizon],
            FieldName.START: pd.Timestamp(ts_start)
        } )
        all_fcast_ts.append( {
            FieldName.TARGET: ts_data,
            FieldName.START: pd.Timestamp(ts_start)
        } )

    train_gluonts_ds = ListDataset(all_train_ts, freq=freq)
    fcast_gluonts_ds = ListDataset(all_fcast_ts, freq=freq)

    model_args = {
        'freq': freq,
        'context_length': lag,
        'prediction_length': forecast_horizon,
        'epochs': epochs
    }
    exp_args = inspect.signature(model_cls).parameters
    for key in list(model_args.keys()):
        if key not in exp_args:
            del(model_args[key])

    if model == 'deepstate': # described in https://github.com/awslabs/gluonts/issues/794
        model_args['cardinality'] = [1]
        model_args['use_feat_static_cat'] = False

    if model == 'gpforecaster':
        model_args['cardinality'] = len(train_gluonts_ds)

    if model == 'deeprenewal': # following the paper experiments setup
        model_args['num_layers'] = 1
        model_args['num_cells'] = 10

    if model == 'autosklearn' or model == 'autogluon':
        ds_meta = read_json('meta_dataset.json')
        model_args['time_budget'] = ds_meta[dataset]['budget']

    if model == 'autogluon':
        model_args['output_dir'] = os.path.join(args.train_logdir, 'autogluon')
        
    if model in ['rotbaum', 'naiveseasonal']: # for some reason the args are not included in sugnature of these methods
        model_args['freq'] = freq
        model_args['prediction_length'] = forecast_horizon

    # already init estimator & predictor for early stopping callback
    estimator = model_cls(**model_args)

    if not isinstance(estimator, HorizonFCEnsemble):
        from early_stopping import MetricInferenceEarlyStopping
        from gluonts.mx import Trainer
        from gluonts.mx.trainer.callback import TrainingHistory
        early_stopper = MetricInferenceEarlyStopping(validation_dataset=fcast_gluonts_ds, estimator=estimator, metric="RMSE", verbose=False)
        history = TrainingHistory()
        trainer = Trainer(epochs=epochs, callbacks=[history, early_stopper])
        estimator.trainer = trainer
    else:
        history = None


    return train_gluonts_ds, history, fcast_gluonts_ds, estimator


def run_validation(predictor, dataset, num_samples=100):
    # TODO check for integer_conversion in args and round?
    return make_evaluation_predictions(dataset=dataset, predictor=predictor, num_samples=num_samples)


def evaluate(forecast, groundtruth):
    try:
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9], num_workers=None)
        agg_metrics, item_metrics = evaluator(groundtruth, forecast)
        contained_nan = False
    except ValueError:
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9], num_workers=None, allow_nan_forecast=True) # nan can happen for tempfus
        agg_metrics, item_metrics = evaluator(groundtruth, forecast)
        contained_nan = True

    return {'aggregated': agg_metrics, "contained_nan": contained_nan}
