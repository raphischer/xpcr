import os
import argparse
import json
import shutil
import tarfile
import importlib

import pandas as pd
import numpy as np

from monitoring import aggregate_monitoring_log
from util import basename, PatchedJSONEncoder


#############################
####     log loading     ####
#############################

def read_json(filepath):
    with open(filepath, 'r') as logf:
        return json.load(logf)


def read_txt(filepath):
    with open(filepath, 'r') as reqf:
        return [line.strip() for line in reqf.readlines()]


def read_monitoring(filepath):
    return aggregate_monitoring_log(filepath)


def read_log_directory(directory):
    reader_methods = {name: func for name, func in globals().items() if name.startswith('read_')}
    res = {'directory_name': basename(directory)}
    # read all files
    for filename in os.listdir(directory):
        fbase, ext = os.path.splitext(filename)
        reader_method = f'read_{ext[1:]}'
        if reader_method in reader_methods:
            res[fbase] = reader_methods[reader_method](os.path.join(directory, filename))
    return res


def process_directory(directory, merged_log_dir=None, output_tar_dir=None):
    print('Processing', directory)
    # create summary
    if merged_log_dir is not None: 
        if not os.path.isdir(merged_log_dir):
            os.makedirs(merged_log_dir)
        agglog_name = os.path.join(merged_log_dir, basename(directory) + '.json')
        # load if already exists
        if os.path.isfile(agglog_name):
            res = read_json(agglog_name)
        else:
            res = read_log_directory(directory)
            if merged_log_dir is not None:
                res['full_log'] = os.path.join(merged_log_dir, basename(directory) + '.tar.gz')
            with open(agglog_name, 'w') as agglog:
                json.dump(res, agglog, indent=4, cls=PatchedJSONEncoder)
    else:
        res = read_log_directory(directory)
    # create tar
    if output_tar_dir is not None:
        if not os.path.isdir(output_tar_dir):
            os.makedirs(output_tar_dir)
        log_tar_name = os.path.join(output_tar_dir, basename(directory) + '.tar.gz')
        if not os.path.exists(log_tar_name):
            with tarfile.open(log_tar_name, 'w:gz') as tar:
                for fname in os.listdir(directory):
                    tar.add(os.path.join(directory, fname))
    return res


#############################
####   log aggregation   ####
#############################

def aggregate_log(log, property_extractors):
    log_name = log['directory_name']
    agg_log = {'log_name': log_name}
    for task, extractors in property_extractors.items():
        if log_name.startswith(task) or task == 'meta':
            # use extractor functions
            for key, func in extractors.items():
                try:
                    agg_log[key] = func(log)
                except KeyError:
                    print(f'Error in accessing {key} from {log_name}!')
                    agg_log[key] = np.nan
    return agg_log


def merge_database(database):
    database['configuration'] = database.aggregate(lambda row: ' - '.join([row['task'], row['dataset'], row['model']]), axis=1)
    database['environment'] = database.aggregate(lambda row: ' - '.join([row['architecture'], row['software']]), axis=1)
    grouped = database.groupby(['configuration', 'environment'])
    grouped_results = grouped.first() # take first occurence as a start
    mean_values = grouped.mean()
    grouped_results.update(mean_values)
    grouped_results['n_results'] = grouped.size()
    return grouped_results.reset_index()


def aggregate_logs(logs, property_extractors_module, database_fname):
    # import extractors
    if property_extractors_module is None:
        property_extractors_module = 'properties'
    try:
        property_extractors = importlib.import_module(property_extractors_module).PROPERTIES
    except (AttributeError, ImportError) as e:
        raise RuntimeError(f'Error when trying to import aggregators from {args.experiment_properties} module!')

    if os.path.isfile(database_fname): # if available, read from disc
        aggregated_logs = pd.read_pickle(database_fname)
    else:
        if not isinstance(logs, list):
            # directory given instead of logs
            if not os.path.isdir(logs):
                raise RuntimeError('Please pass a list of merged logs, or the directory where respective .json logs can be found!')
            # parse all logs in directory
            lognames = [os.path.join(logs, fname) for fname in os.listdir(logs) if fname.endswith('.json')]
            logs = [read_json(fname) for fname in lognames]
        
        # aggregate properties per log, build database, and merge to single entries per (configuration X environment) combination
        agg_logs = [ aggregate_log(log, property_extractors) for log in logs ]
        aggregated_logs = pd.DataFrame.from_records(agg_logs)
        merged_database = merge_database(aggregated_logs)
        merged_database.to_pickle(database_fname)

    return merged_database


def load_database(results_dir, merged_log_dir=None, output_tar_dir=None, property_extractors_module=None, database_fname='database.pkl', clean=False):
    if clean: # remove all subdirectory contents
        if os.path.isfile(database_fname):
            os.remove(database_fname)
        for rootdir in [merged_log_dir, output_tar_dir]:
            if rootdir is not None and os.path.isdir(rootdir):
                shutil.rmtree(rootdir)
    # process
    logs = []
    for dir in sorted(os.listdir(results_dir)):
        log = process_directory(os.path.join(results_dir, dir), merged_log_dir, output_tar_dir)
        logs.append(log)

    # aggregate
    aggregated_logs = aggregate_logs(logs, property_extractors_module, database_fname)

    return aggregated_logs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--results-dir", default="mnt_data/results", type=str, help="directory with experimental result sub directories")
    parser.add_argument("--merged-log-dir", default="mnt_data/results_merged", type=str, help="directory where merged experiment logs (json format) are created")
    parser.add_argument("--output-tar-dir", default=None, type=str, help="directory where the full logs shall be stored (.tar.gz archives)")
    parser.add_argument("--property-extractors-module", default="properties", help="python file with PROPERTIES dictionary, which maps properties to executable extractor functions")
    parser.add_argument("--database-fname", default="database.pkl", help="filename for the database that shall be created")
    parser.add_argument("--clean", action="store_true", help="set to first delete all content in given output directories")

    args = parser.parse_args()

    # parse results directories
    database = load_database(args.results_dir, args.merged_log_dir, args.output_tar_dir, args.property_extractors_module, args.database_fname, args.clean)

    print(f'Database constructed from logs has {database.shape} entries')

    # for task, task_agg_logs in aggregated_logs.items():
    #     print('Results for', task)
    #     total_ds = pd.unique(task_agg_logs['dataset']).size
    #     best_per_ds = task_agg_logs.groupby('dataset').apply( lambda grp: grp.iloc[grp['RMSE'].argmin()].drop('dataset') )
    #     print(best_per_ds)
