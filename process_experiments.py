import os
import argparse
import json
import shutil
import tarfile
import importlib

import pandas as pd

from monitoring import aggregate_log
from util import basename, PatchedJSONEncoder


def read_json(filepath):
    with open(filepath, 'r') as logf:
        return json.load(logf)


def read_txt(filepath):
    with open(filepath, 'r') as reqf:
        return [line.strip() for line in reqf.readlines()]


def read_monitoring(filepath):
    return aggregate_log(filepath)


def aggregate_results(directory):
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
            with open(agglog_name, 'r') as agglog:
                res = json.load(agglog)
        else:
            res = aggregate_results(directory)
            if merged_log_dir is not None:
                res['full_log'] = os.path.join(merged_log_dir, basename(directory) + '.tar.gz')
            with open(agglog_name, 'w') as agglog:
                json.dump(res, agglog, indent=4, cls=PatchedJSONEncoder)
    else:
        res = aggregate_results(directory)
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


def aggregate_logs(results, database_fname, log_metric_extractors):

    if not isinstance(results, list):
        if not os.path.isdir(results):
            raise RuntimeError('Please pass a list of merged results, or the directory where respective .json logs can be found!')
        merged = []
        for fname in os.listdir(results):
            if fname.endswith('.json'):
                # parse results file
                with open(os.path.join(results, fname), 'r') as f:
                    merged.append( json.load(f) )
        results = merged

    aggregated_logs = {}
    for task, extractors in log_metric_extractors.items():
        merged_db = database_fname.format(task=task)

        if os.path.isfile(database_fname): # if available, read from disc
            aggregated_logs[task] = pd.read_pickle(merged_db)

        else:
            agg_tasklogs = []
            for merged_log in results:
                log_name = merged_log['directory_name']
                if log_name.startswith(task):
                    agg_log = {}
                    # use extractor functions
                    for key, func in extractors.items():
                        try:
                            agg_log[key] = func(merged_log)
                        except KeyError:
                            print(f'Error in accessing {key} from {log_name}!')
                    agg_tasklogs.append(agg_log)

            aggregated_logs[task] = pd.DataFrame.from_records(agg_tasklogs)
            aggregated_logs[task].to_pickle(merged_db)

    return aggregated_logs


def main(results_dir, merged_log_dir=None, output_tar_dir=None, clean=False):
    if clean: # remove all subdirectory contents
        for rootdir in [merged_log_dir, output_tar_dir]:
            if rootdir is not None and os.path.isdir(rootdir):
                for subdir in os.listdir(rootdir):
                    if os.path.isfile(os.path.join(rootdir, subdir)):
                        os.remove(os.path.join(rootdir, subdir))
                    else:
                        shutil.rmtree(os.path.join(rootdir, subdir))
    # process
    logs = []
    for dir in sorted(os.listdir(results_dir)):
        log = process_directory(os.path.join(results_dir, dir), merged_log_dir, output_tar_dir)
        logs.append(log)
    return logs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--results-dir", default="mnt_data/results", type=str, help="directory with experimental result directories")
    parser.add_argument("--merged-log-dir", default="mnt_data/results_merged", type=str, help="directory where experiments log aggregates (json format) are created")
    parser.add_argument("--output-tar-dir", default=None, type=str, help="directory where the full logs shall be stored (.tar.gz archives)")
    parser.add_argument('--metric-aggregators', default='metric_aggregators')
    parser.add_argument('--database-fname', default='{task}.pkl')
    parser.add_argument("--clean", action='store_true', help="set to first delete all content in given output directories")

    args = parser.parse_args()

    # parse results directories
    logs = main(args.results_dir, args.merged_log_dir, args.output_tar_dir, args.clean)

    # aggregate metrics
    try:
        extractors = importlib.import_module(args.metric_aggregators).AGGREGATORS
    except (AttributeError, ImportError) as e:
        raise RuntimeError(f'Error when trying to import aggregators from {args.metric_aggregators} module!')
    database_fname = os.path.join(args.merged_log_dir, args.database_fname)
    aggregated_logs = aggregate_logs(logs, database_fname, extractors)

    print('Processed', len(logs), 'directories!')

    for task, task_agg_logs in aggregated_logs.items():
        print('Results for', task)
        total_ds = pd.unique(task_agg_logs['dataset']).size
        best_per_ds = task_agg_logs.groupby('dataset').apply( lambda grp: grp.iloc[grp['RMSE'].argmin()].drop('dataset') )
        print(best_per_ds)
