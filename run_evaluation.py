import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from strep.index_and_rate import rate_database, find_relevant_metrics
from strep.util import load_meta
from strep.elex.app import Visualization

from run_log_processing import DB_COMPLETE, DB_BL, DB_META
from run_meta_learning import FreqTransformer # import needed for loading the MASE models for paper result creation

DATABASES = ['autokeras.pkl', 'autosklearn.pkl', 'autogluon.pkl', 'dnns.pkl']

# parser = argparse.ArgumentParser()

# parser.add_argument("--mode", default='interactive', choices=['interactive', 'paper'])
# parser.add_argument("--boundaries", default="boundaries.json")
# parser.add_argument("--dropsubsampled", default=False, type=bool)
# args = parser.parse_args()

database = pd.concat(pd.read_pickle(db) for db in [DB_COMPLETE, DB_BL]).reset_index(drop=True)
# GluonTS has no num params -> interpolate from file size and inf running time
to_use_for_regr = ['fsize', 'running_time']
invalid_num_params = database[database['parameters'] == -1]
valid_num_params = database.drop(invalid_num_params.index)
lin_regr = LinearRegression()
lin_regr.fit(valid_num_params[to_use_for_regr], valid_num_params['parameters'])
interp_params = lin_regr.predict(invalid_num_params[to_use_for_regr])
database.loc[database['parameters'] == -1,'parameters'] = interp_params.astype(int)

meta = load_meta()
for ds in pd.unique(database['dataset']):
    rows = database[database['dataset'] == ds]
    if ds not in meta['dataset']:
        # if args.dropsubsampled:
        #     database = database[database['dataset'] != ds]
        # else: # store meta information for all subsampled datasets!
            orig = rows['dataset_orig'].iloc[0]
            meta['dataset'][ds] = meta['dataset'][orig].copy()
            meta['dataset'][ds]['name'] = meta['dataset'][ds]['name'] + ds.replace(orig + '_', '') # append the ds seed to name

database, metrics, xaxis_default, yaxis_default = find_relevant_metrics(database, meta)
rated_database, boundaries, real_boundaries, references = rate_database(database, given_meta=meta['properties'], boundaries="boundaries.json")
print(f'Database constructed from logs has {rated_database.shape} entries')

# if args.mode == 'paper':
#     from create_paper_results import create_all
#     create_all(rated_database, pd.read_pickle(DB_META), meta)
#     sys.exit(0)

# else interactive
db = {'DB': (rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references)}
app = Visualization(db)
server = app.server

if __name__ == '__main__':
    app.run_server(debug=False)
