import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from strep.index_and_rate import rate_database, find_relevant_metrics
from strep.util import load_meta

from run_log_processing import DB_COMPLETE, DB_BL, DB_META
from run_meta_learning import FreqTransformer # import needed for loading the MASE models for paper result creation

DATABASES = ['autokeras.pkl', 'autosklearn.pkl', 'autogluon.pkl', 'dnns.pkl']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='paper', choices=['interactive', 'paper', 'stats'])
    parser.add_argument("--boundaries", default="boundaries.json")
    parser.add_argument("--dropsubsampled", default=False, type=bool)
    args = parser.parse_args()

    database = pd.concat(pd.read_pickle(db) for db in [DB_COMPLETE, DB_BL]).reset_index(drop=True)
    # GluonTS has no num params -> interpolate from file size and inf running time
    to_use_for_regr = ['fsize', 'running_time']
    invalid_num_params = database[database['parameters'] == -1]
    valid_num_params = database.drop(invalid_num_params.index)
    lin_regr = LinearRegression()
    lin_regr.fit(valid_num_params[to_use_for_regr], valid_num_params['parameters'])
    interp_params = lin_regr.predict(invalid_num_params[to_use_for_regr])
    database.loc[database['parameters'] == -1,'parameters'] = interp_params.astype(int)

    if args.mode == 'stats':
        grouped_by = database.groupby(['environment', 'dataset'])
        ds_stats, ds_details = [], []
        max_shape = [0, 0]
        for i, ((env, ds), data) in enumerate( iter(grouped_by) ):
            na_vals = np.count_nonzero((data != data).values)
            na_vals_rel = na_vals / data.select_dtypes('number').size * 100
            ds_stats.append(f'{i:<3} {env[:10]:<10} {ds[:15]:<15}..{ds[-6:]:<6} shape {data.shape}, {na_vals:<2} N/A values ({na_vals_rel:5.2f}%)')
            cols = ['model'] + list(data.select_dtypes('number').columns)
            ds_details.append( '\n' + f'{env} {ds} shape {data.shape}, {na_vals:<2} N/A values ({na_vals_rel:5.2f}%)' + '\n' + str(data[cols]) )
        for stats in ds_details:
            print(stats)
        for stats in ds_stats:
            print(stats)

        results = []
        for group, data in database.groupby('dataset_orig'):
            results.append((np.sum([d["value"] for d in data["train_running_time"]])/3600, group, np.sum([d["value"] for d in data["running_time"]])))
        for train, ds, infer in sorted(results):
            print(f'{ds:<40} {train:6.2f}h training {infer:6.2f}s per inference')

        sys.exit(0)

    meta = load_meta()
    for ds in pd.unique(database['dataset']):
        rows = database[database['dataset'] == ds]
        if ds not in meta['dataset']:
            if args.dropsubsampled:
                database = database[database['dataset'] != ds]
            else: # store meta information for all subsampled datasets!
                orig = rows['dataset_orig'].iloc[0]
                meta['dataset'][ds] = meta['dataset'][orig].copy()
                meta['dataset'][ds]['name'] = meta['dataset'][ds]['name'] + ds.replace(orig + '_', '') # append the ds seed to name

    database, metrics, xaxis_default, yaxis_default = find_relevant_metrics(database, meta)
    rated_database, boundaries, real_boundaries, references = rate_database(database, given_meta=meta['properties'], boundaries=args.boundaries)
    print(f'Database constructed from logs has {rated_database.shape} entries')

    if args.mode == 'interactive':
        from strep.elex.app import Visualization
        db = {'DB': (rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references)}
        app = Visualization(db)
        app.run_server(debug=False)

    if args.mode == 'paper':
        from create_paper_results import create_all
        create_all(rated_database, pd.read_pickle(DB_META), meta)
