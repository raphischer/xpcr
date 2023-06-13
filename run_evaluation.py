import argparse
import os
import pandas as pd
import numpy as np
import re

from mlprops.index_and_rate import rate_database
from mlprops.util import load_meta

from data_lookup_info import LOOKUP
from data_loader import convert_tsf_to_dataframe as load_data
from data_loader import subsampled_to_orig


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='interactive', choices=['meta', 'interactive', 'paper', 'label', 'stats'])
    parser.add_argument("--property-extractors-module", default="properties", help="python file with PROPERTIES dictionary, which maps properties to executable extractor functions")
    parser.add_argument("--database-path", default="results/database22.pkl", help="filename for database, or directories with databases inside")
    parser.add_argument("--boundaries", default="boundaries.json")
    parser.add_argument("--dropsubsampled", default=False, type=bool)
    # interactive exploration params
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")

    args = parser.parse_args()

    if os.path.isfile(args.database_path): # read just the single database
        database = pd.read_pickle(args.database_path)
    elif os.path.isdir(args.database_path): # directory with multiple databases
        databases = []
        for fname in os.listdir(args.database_path):
            if 'pkl' in fname:
                databases.append(pd.read_pickle(os.path.join(args.database_path, fname)))
                print(f'{fname:<20} shape {databases[-1].shape}, env {pd.unique(databases[-1]["environment"])} {len(pd.unique(databases[-1]["dataset"]))} datasets')
        database = pd.concat(databases)
    # merge infer and train tasks
    merged_database = []
    for group_field_vals, data in database.groupby(['dataset', 'environment', 'model']):
        assert data.shape[0] < 3, "found too many values"
        merged = data.fillna(method='bfill').head(1)
        merged.loc[:,'task'] = 'Train and Test'
        merged_database.append(merged)
    database = pd.concat(merged_database)
    # # retrieve original dataset from all subsampled versions, and recalculate the configuration
    database['dataset_orig'] = database['dataset'].map(subsampled_to_orig)
    database['configuration'] = database.aggregate(lambda row: ' - '.join([row['task'], row['dataset'], row['model']]), axis=1)
    database.reset_index()

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

    rated_database, boundaries, real_boundaries, _ = rate_database(database, properties_meta=meta['properties'], boundaries=args.boundaries)
    print(f'Database constructed from logs has {rated_database.shape} entries')

    if args.mode == 'interactive':
        from mlprops.elex.app import Visualization
        app = Visualization(rated_database, boundaries, real_boundaries, meta)
        app.run_server(debug=args.debug, host=args.host, port=args.port)

    if args.mode == 'paper':
        from create_paper_results import create_all
        create_all(rated_database, boundaries, real_boundaries, meta)

    if args.mode == 'label':
        from mlprops.labels.label_generation import PropertyLabel
        from mlprops.elex.util import fill_meta
        summary = fill_meta(rated_database.iloc[0].to_dict(), meta)
        pdf_doc = PropertyLabel(summary, 'optimistic median')
        pdf_doc.save('label.pdf')

    if args.mode == 'stats':
        grouped_by = rated_database.groupby(['environment', 'dataset'])
        ds_stats = []
        max_shape = [0, 0]
        for i, ((env, ds), data) in enumerate( iter(grouped_by) ):
            print(f'{i:<3} {env[:15]:<15} {ds[:20]:<20}..{ds[-6:]:<6} {len(pd.unique(data["model"])):<2} models, shape {data.shape}')
        #     stats = {'ds': ds}
        #     for col in data.columns:
        #         if data[col].dropna().size == 0:
        #             data = data.drop(col, axis=1)
        #     stats['entries'] = data.shape
        #     if data.shape[0] > max_shape[0] or data.shape[1] > max_shape[1]:
        #         max_shape = [data.shape[0], data.shape[1]]
        #     if "running_time" in data:
        #         stats['time_infer'] = sum([val['value'] for val in data["running_time"] if isinstance(val, dict)])
        #     else:
        #         stats['time_infer'] = -1
        #     if "train_running_time" in data:
        #         stats['time_train'] = sum([val['value'] for val in data["train_running_time"] if isinstance(val, dict)])
        #     else:
        #         stats['time_train'] = -1
        #     stats['time_total'] = stats['time_train'] + stats['time_infer']
        #     ds_stats.append(stats)
        # print(max_shape)
        # sorted_ds_stats = sorted(ds_stats, key=lambda d: d['ds'])
        # for idx, ds_stat in enumerate(sorted_ds_stats):
        #     if ds_stat["entries"][0] == max_shape[0] and ds_stat["entries"][1] == max_shape[1]: # TODO remove -1
        #         success = 'FULL RESULTS   '
        #     else:
        #         success = 'MISSING RESULTS'
        #     ds_print = f'{ds_stat["ds"][:20]:<20}' + '....' + ds_stat["ds"][-6:]
        #     print(f'{idx:<3} {ds_print} {success} {str(ds_stat["entries"]):<8} entries {len(ds_)}, time total {ds_stat["time_total"] / 3600:6.2f} h (train {ds_stat["time_train"] / 3600:6.2f} h, infer {ds_stat["time_infer"] / 3600:6.2f} h)')
    
    if args.mode == 'meta':
        meta_database_path = "results/database_for_meta.pkl"
        if os.path.isfile(meta_database_path): # read just the single database
            rated_database = pd.read_pickle(meta_database_path)
        else:
            for idx, ((ds), data) in enumerate(iter(rated_database.groupby(['dataset']))):
                # store dataset specific meta features
                ds_name = data['dataset_orig'].iloc[0]
                subsample_str = ds.replace(ds_name, '').replace('_', '')
                ds_seed = -1 if len(subsample_str) == 0 else int(subsample_str)

                full_path = os.path.join('mnt_data/data', ds_name + '.tsf')
                ts_data, freq, seasonality, forecast_horizon, contain_missing_values, contain_equal_length = load_data(full_path, ds_sample_seed=ds_seed)
                if forecast_horizon is None:
                    forecast_horizon = LOOKUP[ds_name][1]
                # rated_database.loc[data.index,'orig_dataset'] = ds_name # ensure no bleeding across subsampled datasets in cross-validation
                rated_database.loc[data.index,'num_ts'] = ts_data.shape[0]
                rated_database.loc[data.index,'avg_ts_len'] = ts_data['series_value'].map(lambda ts: len(ts)).mean()
                rated_database.loc[data.index,'std_ts_len'] = ts_data['series_value'].map(lambda ts: len(ts)).std()
                rated_database.loc[data.index,'avg_ts_mean'] = ts_data['series_value'].map(lambda ts: np.mean(ts)).mean()
                rated_database.loc[data.index,'min_ts_mean'] = ts_data['series_value'].map(lambda ts: np.mean(ts)).min()
                rated_database.loc[data.index,'max_ts_mean'] = ts_data['series_value'].map(lambda ts: np.mean(ts)).max()
                rated_database.loc[data.index,'avg_ts_min'] = ts_data['series_value'].map(lambda ts: np.min(ts)).mean()
                rated_database.loc[data.index,'min_ts_min'] = ts_data['series_value'].map(lambda ts: np.min(ts)).min()
                rated_database.loc[data.index,'avg_ts_max'] = ts_data['series_value'].map(lambda ts: np.max(ts)).mean()
                rated_database.loc[data.index,'max_ts_max'] = ts_data['series_value'].map(lambda ts: np.max(ts)).max()
                rated_database.loc[data.index,'seasonality'] = seasonality
                rated_database.loc[data.index,'freq'] = freq
                rated_database.loc[data.index,'forecast_horizon'] = forecast_horizon
                rated_database.loc[data.index,'contain_missing_values'] = contain_missing_values
                rated_database.loc[data.index,'contain_equal_length'] = contain_equal_length
            rated_database.to_pickle(meta_database_path)

        # TODO check why there is an issue here?
        rated_database = rated_database[rated_database['dataset_orig'] != 'bitcoin_dataset_without_missing_values']
        models = pd.unique(rated_database["model"])
        shape = rated_database.shape
        ds = pd.unique(rated_database["dataset"])
        print(f'Meta learning to be run on {shape} database entries, with a total of {len(ds)} datasets and {len(models)} models!')
        from run_model_recommendation import evaluate_recommendation
        evaluate_recommendation(rated_database)
