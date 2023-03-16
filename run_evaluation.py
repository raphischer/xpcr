import argparse
import os
import pandas as pd

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
    parser.add_argument("--mode", default='meta', choices=['meta', 'interactive', 'paper_results', 'label', 'stats'])
    # interactive exploration
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")

    args = parser.parse_args()

    meta = load_meta()

    if os.path.isfile(args.database_fname): # if available, read from disc
        database = pd.read_pickle(args.database_fname)
    else:
        database = load_database(args.logdir_root, args.output_logdir_merged, None, args.property_extractors_module, args.clean)
        database.to_pickle(args.database_fname)

    rated_database, boundaries, real_boundaries, references = rate_database(database, properties_meta=meta['properties'], boundaries=args.boundaries)

    print(f'Database constructed from logs has {rated_database.shape} entries')

    if args.mode == 'interactive':
        app = Visualization(rated_database, boundaries, real_boundaries, meta, references)
        app.run_server()#debug=args.debug, host=args.host, port=args.port)

    if args.mode == 'label':
        summary = fill_meta(rated_database.iloc[0].to_dict(), meta)
        pdf_doc = PropertyLabel(summary, 'optimistic median')
        pdf_doc.save('label.pdf')

    if args.mode == 'stats':
        grouped_by = rated_database.groupby(['dataset', 'task'])
        for idx, ((ds, task), data) in enumerate(iter(grouped_by)):
            for col in data.columns:
                if data[col].dropna().size == 0:
                    data = data.drop(col, axis=1)
            runtime_field = "running_time" if task == "infer" else "train_running_time"
            if runtime_field in data.columns:
                time = sum([val['value'] for val in data[runtime_field] if isinstance(val, dict)])
                time_n_nan = sum([1 for val in data[runtime_field] if not isinstance(val, dict)])
            else:
                time, time_n_nan = -1, -1
            print(f'{idx:<2} {task:<5} {ds:<45} {str(data.shape):<8} entries, processing time {time / 3600:6.2f} h ({time_n_nan} missing time infos)')

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
                print(f'Dropping {ds} - shape {data.shape} does not match expected {max_shape}')
                rated_database = rated_database.drop(data.index)
            else:
                full_path = os.path.join('mnt_data/data', ds + '.tsf')
                _, freq, seasonality, forecast_horizon, contain_missing_values, contain_equal_length = load_data(full_path)
                rated_database.loc[data.index,'seasonality'] = seasonality
                rated_database.loc[data.index,'freq'] = freq
                rated_database.loc[data.index,'forecast_horizon'] = forecast_horizon
                rated_database.loc[data.index,'contain_missing_values'] = contain_missing_values
                rated_database.loc[data.index,'contain_equal_length'] = contain_equal_length
        rated_database.to_pickle('meta_learn.pkl')


    if args.mode == 'paper_results':
        pass

    # for task, task_agg_logs in aggregated_logs.items():
    #     print('Results for', task)
    #     total_ds = pd.unique(task_agg_logs['dataset']).size
    #     best_per_ds = task_agg_logs.groupby('dataset').apply( lambda grp: grp.iloc[grp['RMSE'].argmin()].drop('dataset') )
    #     print(best_per_ds)
