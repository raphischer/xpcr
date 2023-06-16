import os

import numpy as np
import mxnet as mx

from gluonts.model.estimator import Estimator
from gluonts.dataset.common import Dataset
from gluonts.mx.trainer.callback import Callback
from gluonts.evaluation import Evaluator

### EARLY STOPPING adapted from https://gist.github.com/pbruneau/04c0dce4bdfb66ffac3f554f1b98c706
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
        evaluator: Evaluator = Evaluator(num_workers=None, allow_nan_forecast=True), # nan can happen for tempfus
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
        self.written_first = False

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
        with self.estimator.trainer.ctx:
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

        if self.is_better(current_metric_value, self.best_metric_value) or not self.written_first:
            self.written_first = True
            self.best_metric_value = current_metric_value

            if self.restore_best_network:
                training_network.save_parameters(os.path.join(os.getenv('TMPDIR'), "best_network.params"))

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
                    training_network.load_parameters(os.path.join(os.getenv('TMPDIR'), "best_network.params"))

        return should_continue
