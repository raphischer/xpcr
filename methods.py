import os
import pandas as pd
import inspect
import json
from datetime import datetime
import importlib
from typing import Dict, Optional

from gluonts.model.r_forecast import RForecastPredictor

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.mx import Trainer
from gluonts.mx.trainer.callback import TrainingHistory
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator


from data_loader import convert_tsf_to_dataframe as load_data

class ARIMAWrapper(RForecastPredictor):

    def __init__(
        self,
        freq: str,
        prediction_length: int,
        period: int = None,
        trunc_length: Optional[int] = None,
        params: Optional[Dict] = None,
    ) -> None:

        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            method_name='arima',
            period=period,
            trunc_length=trunc_length,
            params=params
        )


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
        model_cls = getattr(importlib.import_module(model_props['module']), model_props['class'])

    full_path = os.path.join(args.datadir, dataset + '.tsf')
    ds, freq, seasonality, forecast_horizon, contain_missing_values, contain_equal_length = load_data(full_path)

    # forecast horizon might not be available from tsf
    if forecast_horizon is None:
        if not hasattr(args, "external_forecast_horizon"):
            raise Exception("Please provide the required forecast horizon")
        else:
            forecast_horizon = args.external_forecast_horizon

    all_train_ts = []
    all_fcast_ts = []

    for _, row in ds.iterrows():

        ts_start = row["start_timestamp"] if "start_timestamp" in ds.columns else datetime(1900, 1, 1, 0, 0, 0)
        ts_data = row["series_value"]

        # use gluonts data format
        all_train_ts.append( {
            FieldName.TARGET: ts_data[:len(ts_data) - forecast_horizon],
            FieldName.START: pd.Timestamp(ts_start, freq=freq)
        } )
        all_fcast_ts.append( {
            FieldName.TARGET: ts_data,
            FieldName.START: pd.Timestamp(ts_start, freq=freq)
        } )

    train_gluonts_ds = ListDataset(all_train_ts, freq=freq)
    fcast_gluonts_ds = ListDataset(all_fcast_ts, freq=freq)

    args = {
        'freq': freq,
        'context_length': lag,
        'prediction_length': forecast_horizon
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

    if model in ['rotbaum', 'naiveseasonal']: # for some reason the args are not in 
        args['freq'] = freq
        args['prediction_length'] = forecast_horizon

    # already init estimator & predictor for early stopping callback
    estimator = model_cls(**args)

    early_stopping = MetricInferenceEarlyStopping(validation_dataset=fcast_gluonts_ds, estimator=estimator, metric="RMSE", verbose=False)
    history = TrainingHistory()
    trainer = Trainer(epochs=epochs, callbacks=[history, early_stopping])

    estimator.trainer = trainer    
    return train_gluonts_ds, history, fcast_gluonts_ds, estimator


def evaluate(predictor, ts_test):

    # TODO check for integer_conversion in args and round?

    forecast, groundtruth = make_evaluation_predictions(dataset=ts_test, predictor=predictor, num_samples=100)
    
    forecast = list(forecast)
    groundtruth = list(groundtruth)
    
    # evaluate
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(groundtruth, forecast)

    return {'aggregated': agg_metrics} # , 'item': item_metrics}




### EARLY STOPPING adapted from https://gist.github.com/pbruneau/04c0dce4bdfb66ffac3f554f1b98c706
import numpy as np
import mxnet as mx

from gluonts.model.estimator import Estimator
from gluonts.dataset.common import Dataset
from gluonts.mx import copy_parameters, GluonPredictor
from gluonts.mx.trainer.callback import Callback


class MetricInferenceEarlyStopping(Callback):
    """
    Early Stopping mechanism based on the prediction network.
    Can be used to base the Early Stopping directly on a metric of interest, instead of on the training/validation loss.
    In the same way as test datasets are used during model evaluation,
    the time series of the validation_dataset can overlap with the train dataset time series,
    except for a prediction_length part at the end of each time series.
    Parameters
    ----------
    validation_dataset
        An out-of-sample dataset which is used to monitor metrics
    predictor
        A gluon predictor, with a prediction network that matches the training network
    evaluator
        The Evaluator used to calculate the validation metrics.
    metric
        The metric on which to base the early stopping on.
    patience
        Number of epochs to train on given the metric did not improve more than min_delta.
    min_delta
        Minimum change in the monitored metric counting as an improvement
    verbose
        Controls, if the validation metric is printed after each epoch.
    minimize_metric
        The metric objective.
    restore_best_network
        Controls, if the best model, as assessed by the validation metrics is restored after training.
    num_samples
        The amount of samples drawn to calculate the inference metrics.
    """

    def __init__(
        self,
        validation_dataset: Dataset,
        estimator: Estimator,
        evaluator: Evaluator = Evaluator(num_workers=None),
        metric: str = "MSE",
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: bool = True,
        minimize_metric: bool = True,
        restore_best_network: bool = True,
        num_samples: int = 100,
    ):
        assert (
            patience >= 0
        ), "EarlyStopping Callback patience needs to be >= 0"
        assert (
            min_delta >= 0
        ), "EarlyStopping Callback min_delta needs to be >= 0.0"
        assert (
            num_samples >= 1
        ), "EarlyStopping Callback num_samples needs to be >= 1"

        self.validation_dataset = list(validation_dataset)
        self.estimator = estimator
        self.evaluator = evaluator
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_network = restore_best_network
        self.num_samples = num_samples

        if minimize_metric:
            self.best_metric_value = np.inf
            self.is_better = np.less
        else:
            self.best_metric_value = -np.inf
            self.is_better = np.greater

        self.validation_metric_history = []
        self.best_network = None
        self.n_stale_epochs = 0

    def on_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: mx.gluon.nn.HybridBlock,
        trainer: mx.gluon.Trainer,
        best_epoch_info: dict,
        ctx: mx.Context
    ) -> bool:
        should_continue = True
        
        transformation = self.estimator.create_transformation()
        predictor = self.estimator.create_predictor(transformation=transformation, trained_network=training_network)

        from gluonts.evaluation.backtest import make_evaluation_predictions

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.validation_dataset,
            predictor=predictor,
            num_samples=self.num_samples,
        )

        agg_metrics, item_metrics = self.evaluator(ts_it, forecast_it)
        current_metric_value = agg_metrics[self.metric]
        self.validation_metric_history.append(current_metric_value)

        if self.verbose:
            print(
                f"Validation metric {self.metric}: {current_metric_value}, best: {self.best_metric_value}"
            )

        if self.is_better(current_metric_value, self.best_metric_value):
            self.best_metric_value = current_metric_value

            if self.restore_best_network:
                training_network.save_parameters("best_network.params")

            self.n_stale_epochs = 0
        else:
            self.n_stale_epochs += 1
            if self.n_stale_epochs == self.patience:
                should_continue = False
                print(
                    f"EarlyStopping callback initiated stop of training at epoch {epoch_no}."
                )

                if self.restore_best_network:
                    print(
                        f"Restoring best network from epoch {epoch_no - self.patience}."
                    )
                    training_network.load_parameters("best_network.params")

        return should_continue
