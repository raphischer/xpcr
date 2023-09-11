import os
import inspect
import json
import importlib
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

        # fc = []
        # for i, ts in enumerate( dataset ):
        #     print('           ', i, '/', len(dataset))
        #     starting_index = len(ts["target"]) - self.context_length
        #     end_index = starting_index + self.context_length
        #     time_series_window = ts["target"][starting_index:end_index]
        #     prediction = np.array(
        #         list(
        #             chain.from_iterable(
        #                 model.predict([time_series_window])
        #                 for model in self.models
        #             )
        #         )
        #     )
        #     fc.append( SampleForecast(samples=np.swapaxes(prediction, 0, 1), start_date=forecast_start(ts)) )
        #     if i > 4:
        #         break

        # muuuuch faster than the weird gluonts code above (every nn is already run once)
        X_test = np.array([ts['target'][-self.context_length:] for ts in dataset])
        ys = np.array([model.predict(X_test)[:,0] for model in self.models])
        fc = [ SampleForecast(samples=np.expand_dims(ys[:,i], 0), start_date=forecast_start(ts)) for i, ts in enumerate(dataset) ]

        return fc
        


class AutoSklearn(HorizonFCEnsemble):

    def __init__(self, freq, context_length, prediction_length, samples_per_series=100):
        super().__init__(freq, context_length, prediction_length, samples_per_series)
        from autosklearn.regression import AutoSklearnRegressor
        per_regr = 1200 // self.prediction_length # 4500
        self.models = [AutoSklearnRegressor(per_regr) for _ in range(self.prediction_length)]


class AutoKeras(HorizonFCEnsemble):

    def __init__(self, freq, context_length, prediction_length, samples_per_series=100, epochs=100, max_trials=2):
        super().__init__(freq, context_length, prediction_length, samples_per_series)
        self.epochs = epochs
        self.max_trials = max_trials
        import autokeras as ak
        self.models = [ak.StructuredDataRegressor(max_trials=self.max_trials, directory=os.environ['TMPDIR']) for _ in range(self.prediction_length)]

    def fit(self, model_idx, X, y):
        self.models[model_idx].fit(X, y, epochs=self.epochs)


class AutoKerasForecaster(HorizonFCEnsemble):

    def __init__(self, freq, context_length, prediction_length, samples_per_series=100, epochs=5, max_trials=1):
        super().__init__(freq, context_length, prediction_length, samples_per_series)
        self.epochs = epochs
        self.max_trials = max_trials
        import autokeras as ak
        self.model = ak.TimeseriesForecaster(
            lookback=self.context_length,
            predict_from=self.lead_time + 1,
            predict_until=self.prediction_length,
            max_trials=self.max_trials,
            objective="val_loss",
        )

    def train(self, training_data):
        min_length = min([len(ser['target']) for ser in training_data])
        X = np.swapaxes(np.array([ser['target'][:min_length] for ser in training_data]), 0, 1)
        self.model.fit(X, X, epochs=self.epochs)
        print('TRAIN DATA', X.shape, len(training_data))
        return self

    def predict(self, dataset, num_samples):
        if num_samples:
            print("Forecast is not sample based. Ignoring parameter `num_samples` from predict method.")

        min_length = min([len(ser['target']) for ser in dataset])
        X = np.swapaxes(np.array([ser['target'][:min_length] for ser in dataset]), 0, 1)
        for ts in dataset:
            starting_index = len(ts["target"]) - self.context_length
            end_index = starting_index + self.context_length
            X.append(ts["target"][starting_index:end_index])
        X = np.swapaxes(np.array(X), 0, 1)
        X = np.concatenate([X, np.zeros((self.prediction_length, len(dataset)))])
        print('TEST DATA', X.shape, len(dataset))
        prediction = self.model.predict(X)
        print(1)

    


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

    args = {
        'freq': freq,
        'context_length': lag,
        'prediction_length': forecast_horizon,
        'epochs': epochs
    }
    exp_args = inspect.signature(model_cls).parameters
    for key in list(args.keys()):
        if key not in exp_args:
            del(args[key])

    if model == 'deepstate': # described in https://github.com/awslabs/gluonts/issues/794
        args['cardinality'] = [1]
        args['use_feat_static_cat'] = False

    if model == 'gpforecaster':
        args['cardinality'] = len(train_gluonts_ds)

    if model == 'deeprenewal': # following the paper experiments setup
        args['num_layers'] = 1
        args['num_cells'] = 10

    if model in ['rotbaum', 'naiveseasonal']: # for some reason the args are not included in sugnature of these methods
        args['freq'] = freq
        args['prediction_length'] = forecast_horizon

    # already init estimator & predictor for early stopping callback
    estimator = model_cls(**args)

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
