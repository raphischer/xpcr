import os
import json
import argparse

import pandas as pd

log_metric_extractors = {
    'infer': {
        'method': lambda log: log['config']['model'],
        'dataset': lambda log: log['config']['dataset'],
        'runtime': lambda log: log['validation_pyrapl']['total']['total_duration'],
        'power_draw': lambda log: log['validation_pyrapl']['total']['total_power_draw'],
        'RMSE': lambda log: log['validation_results']['metrics']['aggregated']['RMSE'],
        'RMSE': lambda log: log['validation_results']['metrics']['aggregated']['MAPE'],
        'sMAPE': lambda log: log['validation_results']['metrics']['aggregated']['sMAPE']
    }
}


def load_logs(args):

    logs = {}
    for task, extractors in log_metric_extractors.items():
        merged_db = os.path.join(args.database_dir, args.database_fname.format(task=task))

        if os.path.isfile(merged_db): # if available, read from disc
            logs[task] = pd.read_pickle(merged_db)

        else:
            tasklogs = []
            for fname in os.listdir(args.res_dir):
                if fname.endswith('.json') and fname.startswith(task):
                    # parse results file
                    results = {}
                    with open(os.path.join(args.res_dir, fname), 'r') as f:
                        merged_log = json.load(f)
                    # use extractor functions
                    for key, func in extractors.items():
                        try:
                            results[key] = func(merged_log)
                        except KeyError:
                            print(f'Error in accessing {key} from {fname}!')
                    tasklogs.append(results)

            logs[task] = pd.DataFrame.from_records(tasklogs)
            logs[task].to_pickle(merged_db)

    return logs

def main(args):
    all_logs = load_logs(args)

    for task, logs in all_logs.items(): 

        total_ds = pd.unique(logs['dataset']).size

        best_per_ds = logs.groupby('dataset').apply( lambda grp: grp.iloc[grp['RMSE'].argmin()].drop('dataset') )
        print(best_per_ds)


# detailed results per dataset
# for idx, ds in enumerate(pd.unique(log_df['dataset'])):
#     print(f'\nDataset {idx:<2} / {total_ds:<2} - \n{ds} ::::::: ::::: :::: ::: :: :')
#     results = log_df[log_df['dataset'] == ds]
#     means = results.groupby('method').mean(numeric_only=True).sort_values('RMSE')
#     print(means)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir',    default='mnt_data/results_merged')
    parser.add_argument('--database-dir', default='mnt_data')
    parser.add_argument('--database-fname', default='{task}_merged.pkl')

    args = parser.parse_args()

    main(args)
