import argparse
import os
import pandas as pd
import numpy as np

from exprep.load_experiment_logs import load_database
from exprep.index_and_rate import rate_database
from exprep.util import load_meta

from exprep.elex.util import fill_meta
from exprep.elex.app import Visualization
from exprep.labels.label_generation import PropertyLabel

from data_loader import convert_tsf_to_dataframe as load_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir-root", default="mnt_data/results", type=str, help="directory with experimental result sub directories")
    parser.add_argument("--output-logdir-merged", default="results/merged", type=str, help="directory where merged experiment logs (json format) are created")
    parser.add_argument("--property-extractors-module", default="properties", help="python file with PROPERTIES dictionary, which maps properties to executable extractor functions")
    parser.add_argument("--database-fname", default="results/database.pkl", help="filename for the database that shall be created")
    parser.add_argument("--boundaries", default="boundaries.json")
    parser.add_argument("--clean", action="store_true", help="set to first delete all content in given output directories")
    parser.add_argument("--mode", default='stats', choices=['meta', 'interactive', 'paper_results', 'label', 'stats'])
    # interactive exploration
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")

    args = parser.parse_args()

    meta_learn_data_fname = 'meta_learn.pkl'

    # load meta learn data if already processed!
    if args.mode == 'meta' and os.path.isfile(meta_learn_data_fname):
            meta_learn_data = pd.read_pickle(meta_learn_data_fname)

    else:

        meta = load_meta()
        if os.path.isfile(args.database_fname): # if available, read from disc
            database = pd.read_pickle(args.database_fname)
        else:
            database = load_database(args.logdir_root, args.output_logdir_merged, None, args.property_extractors_module, args.clean)
            database.to_pickle(args.database_fname)

        rated_database, boundaries, real_boundaries, _ = rate_database(database, properties_meta=meta['properties'], boundaries=args.boundaries)
        print(f'Database constructed from logs has {rated_database.shape} entries')

        if args.mode == 'interactive':
            app = Visualization(rated_database, boundaries, real_boundaries, meta)
            app.run_server()#debug=args.debug, host=args.host, port=args.port)

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
                    stats['time_infer'] = np.inf
                if "train_running_time" in data:
                    stats['time_train'] = sum([val['value'] for val in data["train_running_time"] if isinstance(val, dict)]) / 2 * 40 # TODO remove the epoch hack
                else:
                    stats['time_train'] = np.inf
                stats['time_total'] = stats['time_train'] + stats['time_infer']
                ds_stats.append(stats)
            print(max_shape)
            sorted_ds_stats = sorted(ds_stats, key=lambda d: d['time_total'])
            to_rerun = []
            for idx, ds_stat in enumerate(sorted_ds_stats):
                if ds_stat["entries"][0] >= max_shape[0] - 1 and ds_stat["entries"][1] == max_shape[1]: # TODO remove -1
                    to_rerun.append(ds_stat["ds"])
                    print(f'SUCCESS ON {idx:<2} {ds_stat["ds"]:<45} {str(ds_stat["entries"]):<8} entries, time total {ds_stat["time_total"] / 60:6.2f} m (train {ds_stat["time_train"] / 60:6.2f} m, infer {ds_stat["time_infer"] / 60:6.2f} m)')
                # else:
                #     print(f'ERRORS ON  {idx:<2} {ds_stat["ds"]:<45} {str(ds_stat["entries"]):<8} entries, time total {ds_stat["time_total"] / 3600:6.2f} h (train {ds_stat["time_train"] / 3600:6.2f} h, infer {ds_stat["time_infer"] / 3600:6.2f} h)')
            print('"' + '" "'.join(to_rerun) + '"')

        if args.mode == 'paper_results':
            pass
        
        if args.mode == 'meta':
            # only look at infer results
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
                    full_path = os.path.join('mnt_data/data', ds + '.tsf')
                    ts_data, freq, seasonality, forecast_horizon, contain_missing_values, contain_equal_length = load_data(full_path)
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

        from meta_learn import run_meta_learn

        models = pd.unique(meta_learn_data["model"])
        shape = meta_learn_data.shape
        ds = pd.unique(meta_learn_data["dataset"])
        print(f'Meta learning to be run on {shape} database entries, with a total of {len(ds)} datasets and {len(models)} models!')
        print('Available datasets:', ' '.join(ds))
        
        run_meta_learn(meta_learn_data)
