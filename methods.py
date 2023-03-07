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

    trainer = Trainer(epochs=epochs)
    args = {
        'freq': freq,
        'context_length': lag,
        'prediction_length': forecast_horizon,
        'trainer': trainer
    }

    exp_args = inspect.signature(model_cls).parameters
    for key in list(args.keys()):
        if key not in exp_args:
            del(args[key])

    estimator = model_cls(**args)

    return train_gluonts_ds, fcast_gluonts_ds, estimator


def evaluate(predictor, ts_test):

    # TODO check for integer_conversion in args and round?

    forecast, groundtruth = make_evaluation_predictions(dataset=ts_test, predictor=predictor, num_samples=100)
    
    forecast = list(forecast)
    groundtruth = list(groundtruth)
    
    # evaluate
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(groundtruth, forecast)

    return {'aggregated': agg_metrics} # , 'item': item_metrics}
