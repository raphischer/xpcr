import json
import pandas as pd

from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.renewal import DeepRenewalProcessEstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.seq2seq import MQRNNEstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.model.rotbaum import TreeEstimator 
from gluonts.model.tft import TemporalFusionTransformerEstimator

from gluonts.model.seasonal_naive import SeasonalNaiveEstimator


from methods import ARIMAWrapper

MODELS = {
    "deepar": DeepAREstimator,
    "deepstate": DeepStateEstimator,
    "deepfactor": DeepFactorEstimator,
    "deeprenewalprocesses": DeepRenewalProcessEstimator,
    "gpforecaster": GaussianProcessEstimator,
    "mqcnn": MQCNNEstimator,
    "mqrnn": MQRNNEstimator,
    "nbeats": NBEATSEstimator,
    "rotbaum": TreeEstimator,
    "temporalfusiontransformer": TemporalFusionTransformerEstimator,
    "transformer": TransformerEstimator,
    "wavenet": WaveNetEstimator,
    "simplefeedforward": SimpleFeedForwardEstimator,
    "naiveseasonal": SeasonalNaiveEstimator,
    "arima": ARIMAWrapper
}

csv = pd.read_csv('gluonts_models.csv', sep=';')

meta = {}

for (_, row) in csv.iterrows():
    info = row.to_dict()
    for key, val in list(info.items()):
        info[key.lower()] = val
        del(info[key])
    key = info['model + paper'].lower().replace('-', '').replace(' ', '')
    if key not in MODELS:
        info['module'] = None
        info['class'] = None
    else:
        info['module'] = MODELS[key].__module__
        info['class'] = MODELS[key].__name__
    meta[key] = info

with open('meta_model2.json', 'w') as mf:
    json.dump(meta, mf, indent=4)
