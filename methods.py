import pandas as pd

from gluonts.model.deepar import DeepAREstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.mx import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator


from data_loader import load_data


METHODS = {
    "feed_forward": SimpleFeedForwardEstimator,
    "deepar": DeepAREstimator,
    "nbeats": NBEATSEstimator,
    "wavenet": WaveNetEstimator,
    "transformer": TransformerEstimator
}


def init_model_and_data(args):
    dataset, method, lag, epochs = args.dataset, args.model, args.lag, args.epochs

    ds, freq, seasonality, forecast_horizon, contain_missing_values, contain_equal_length = load_data(dataset)

    all_train_ts = []
    all_fcast_ts = []

    for _, row in ds.iterrows():

        ts_start = row["start_timestamp"] # if "start_timestamp" in ds.columns else datetime(1900, 1, 1, 0, 0, 0)
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
    try:
        estimator = METHODS[method](freq=freq, context_length=lag, prediction_length=forecast_horizon, trainer=trainer)
    except TypeError:
        estimator = METHODS[method](context_length=lag, prediction_length=forecast_horizon, trainer=trainer)

    return train_gluonts_ds, fcast_gluonts_ds, estimator


def evaluate(predictor, ts_test):
    forecast, groundtruth = make_evaluation_predictions(dataset=ts_test, predictor=predictor, num_samples=100)
    
    forecast = list(forecast)
    groundtruth = list(groundtruth)
    
    # evaluate
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(groundtruth, forecast)

    return {'aggregated': agg_metrics, 'item': item_metrics}
