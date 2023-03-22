import argparse
import os
import pandas as pd
import numpy as np
import re

from exprep.index_and_rate import rate_database
from exprep.util import load_meta

from exprep.elex.util import fill_meta
from exprep.elex.app import Visualization
from exprep.labels.label_generation import PropertyLabel

from data_loader import convert_tsf_to_dataframe as load_data
from create_paper_results import create_all


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='stats', choices=['meta', 'interactive', 'paper', 'label', 'stats'])
    parser.add_argument("--property-extractors-module", default="properties", help="python file with PROPERTIES dictionary, which maps properties to executable extractor functions")
    parser.add_argument("--database-path", default="results", help="filename for database, or directories with databases inside")
    parser.add_argument("--boundaries", default="boundaries.json")
    parser.add_argument("--drop-subsampled", default=False, type=bool)
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
        database = pd.concat(databases)
        database.reset_index()
    # merge infer and train tasks
    merged_database = []
    for group_field_vals, data in database.groupby(['dataset', 'environment', 'model']):
        assert data.shape[0] < 3, "found too many values"
        merged = data.fillna(method='bfill').head(1)
        merged['task'] = 'Train and Test'
        merged_database.append(merged)
    database = pd.concat(merged_database)
    # # retrieve original dataset from all subsampled versions, and recalculate the configuration
    database['dataset_orig'] = database['dataset'].map(lambda ds: re.match(r'(.*)_(\d*)', ds).group(1) if re.match(r'(.*)_(\d*)', ds) else ds)
    database['configuration'] = database.aggregate(lambda row: ' - '.join([row['task'], row['dataset'], row['model']]), axis=1)

    meta = load_meta()
    if args.drop_subsampled:
        raise NotImplementedError
    else: # store meta information for all subsampled datasets!
        for ds in pd.unique(database['dataset']):
            if ds not in meta['dataset']:
                orig = database[database['dataset'] == ds]['dataset_orig'].iloc[0]
                meta['dataset'][ds] = meta['dataset'][orig].copy()
                meta['dataset'][ds]['name'] = meta['dataset'][ds]['name'] + ds.replace(orig + '_', '') # append the ds seed to name

    # load meta learn data if already processed!
    # TODO rename
    meta_learn_data_fname = 'meta_learn.pkl'
    if args.mode == 'meta' and os.path.isfile(meta_learn_data_fname):
        meta_learn_data = pd.read_pickle(meta_learn_data_fname)
    else:        

        if args.mode == 'paper': # TODO REMOVE LATER
            create_all(None, meta)

        rated_database, boundaries, real_boundaries, _ = rate_database(database, properties_meta=meta['properties'], boundaries=args.boundaries)
        print(f'Database constructed from logs has {rated_database.shape} entries')

        if args.mode == 'interactive':
            app = Visualization(rated_database, boundaries, real_boundaries, meta)
            app.run_server(debug=args.debug, host=args.host, port=args.port)

        if args.mode == 'paper':
            create_all(rated_database, meta)

        if args.mode == 'label':
            summary = fill_meta(rated_database.iloc[0].to_dict(), meta)
            pdf_doc = PropertyLabel(summary, 'optimistic median')
            pdf_doc.save('label.pdf')

        if args.mode == 'stats':
            grouped_by = rated_database.groupby(['dataset'])
            ds_stats = []
            max_shape = [0, 0]
            for (ds), data in iter(grouped_by):
                stats = {'ds': ds}
                for col in data.columns:
                    if data[col].dropna().size == 0:
                        data = data.drop(col, axis=1)
                stats['entries'] = data.shape
                if data.shape[0] > max_shape[0] or data.shape[1] > max_shape[1]:
                    max_shape = [data.shape[0], data.shape[1]]
                if "running_time" in data:
                    stats['time_infer'] = sum([val['value'] for val in data["running_time"] if isinstance(val, dict)])
                else:
                    stats['time_infer'] = -1
                if "train_running_time" in data:
                    stats['time_train'] = sum([val['value'] for val in data["train_running_time"] if isinstance(val, dict)])
                else:
                    stats['time_train'] = -1
                stats['time_total'] = stats['time_train'] + stats['time_infer']
                ds_stats.append(stats)
            print(max_shape)
            sorted_ds_stats = sorted(ds_stats, key=lambda d: d['time_total'])
            for idx, ds_stat in enumerate(sorted_ds_stats):
                if ds_stat["entries"][0] == max_shape[0] and ds_stat["entries"][1] == max_shape[1]: # TODO remove -1
                    success = 'FULL RESULTS ON   '
                else:
                    success = 'MISSING RESULTS ON'
                print(f'{idx:<2} {success} {ds_stat["ds"]:<45} {str(ds_stat["entries"]):<8} entries, time total {ds_stat["time_total"] / 3600:6.2f} h (train {ds_stat["time_train"] / 3600:6.2f} h, infer {ds_stat["time_infer"] / 3600:6.2f} h)')
        
        if args.mode == 'meta':
            # only look at inference results
            rated_database = rated_database.drop(rated_database[rated_database['task'] != 'infer'].index)
            grouped_by = rated_database.groupby(['dataset'])
            # check for completeness of results
            max_shape = (0, 0)
            for idx, ((ds), data) in enumerate(iter(grouped_by)):
                shape = data.shape
                if shape[0] > max_shape[0] or shape[1] > max_shape[1]:
                    max_shape = shape
            for idx, ((ds), data) in enumerate(iter(grouped_by)):
                if data.shape != max_shape:
                    # discard these results due to incompleteness
                    mod_missing = " ".join([mod for mod in pd.unique(rated_database["model"]) if mod not in pd.unique(data['model'])])
                    print(f'Dropping {ds:<40} - shape {str(data.shape):<8} does not match expected {str(max_shape):<8} - missing models: {mod_missing}')
                    rated_database = rated_database.drop(data.index)
                else:
                    # store dataset specific meta features
                    match = re.match(r'(.*)_(\d*)', ds)
                    if match is None:
                        ds_name, ds_seed = ds, -1
                    else:
                        ds_name, ds_seed = match.group(1), int(match.group(2))
                    full_path = os.path.join('mnt_data/data', ds_name + '.tsf')
                    ts_data, freq, seasonality, forecast_horizon, contain_missing_values, contain_equal_length = load_data(full_path, ds_sample_seed=ds_seed)
                    rated_database.loc[data.index,'orig_dataset'] = ds_name # ensure no bleeding across subsampled datasets in cross-validation
                    rated_database.loc[data.index,'num_ts'] = ts_data.shape[0]
                    rated_database.loc[data.index,'avg_ts_len'] = ts_data['series_value'].map(lambda ts: len(ts)).mean()
                    rated_database.loc[data.index,'avg_ts_mean'] = ts_data['series_value'].map(lambda ts: np.mean(ts)).mean()
                    rated_database.loc[data.index,'avg_ts_min'] = ts_data['series_value'].map(lambda ts: np.min(ts)).mean()
                    rated_database.loc[data.index,'avg_ts_max'] = ts_data['series_value'].map(lambda ts: np.max(ts)).mean()
                    rated_database.loc[data.index,'seasonality'] = seasonality
                    rated_database.loc[data.index,'freq'] = freq
                    rated_database.loc[data.index,'forecast_horizon'] = forecast_horizon
                    rated_database.loc[data.index,'contain_missing_values'] = contain_missing_values
                    rated_database.loc[data.index,'contain_equal_length'] = contain_equal_length
            rated_database.to_pickle(meta_learn_data_fname)
            meta_learn_data = rated_database
        
    if args.mode == 'meta':

        from run_model_recommendation import evaluate_recommendation
        models = pd.unique(meta_learn_data["model"])
        shape = meta_learn_data.shape
        ds = pd.unique(meta_learn_data["dataset"])
        print(f'Meta learning to be run on {shape} database entries, with a total of {len(ds)} datasets and {len(models)} models!')
        print('Available datasets:', ' '.join(ds))
        
        evaluate_recommendation(meta_learn_data)
