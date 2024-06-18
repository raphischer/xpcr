import argparse
import os

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

from strep.load_experiment_logs import assemble_database
from strep.util import format_software, format_hardware, write_json

PROPERTIES = {
    'meta': {
        'task': lambda log: log['directory_name'].split('_')[0],
        'dataset': lambda log: log['config']['dataset'] + '_' + str(log['config']['ds_seed']) if log['config']['ds_seed'] != -1 else log['config']['dataset'],
        'model': lambda log: log['config']['model'],
        'architecture': lambda log: format_hardware(log['execution_platform']['Processor']),
        'software': lambda log: format_software('GluonTS', log['requirements']),
    },

    'train': {
        'train_running_time': lambda log: log['emissions']['duration']['0'],
        'train_power_draw': lambda log: log['emissions']['energy_consumed']['0'] * 3.6e6
    },
    
    'infer': {
        'running_time': lambda log: log['emissions']['duration']['0'] / log['validation_results']['num_samples'],
        'power_draw': lambda log: log['emissions']['energy_consumed']['0'] * 3.6e6 / log['validation_results']['num_samples'],
        'RMSE': lambda log: log['validation_results']['metrics']['aggregated']['RMSE'],
        'MAPE': lambda log: log['validation_results']['metrics']['aggregated']['MAPE'],
        'MASE': lambda log: log['validation_results']['metrics']['aggregated']['MASE'],
        'parameters': lambda log: log['validation_results']['model']['params'],
        'fsize': lambda log: log['validation_results']['model']['fsize'],
    }
}

RES_DIR = 'results'
DB_COMPLETE = os.path.join(RES_DIR, 'dnns.pkl')
DB_BL = os.path.join(RES_DIR, 'baselines.pkl')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # data and model input
    parser.add_argument("--output-dir", default='/data/d1/xpcr/logs', type=str, help="path to saved outputs")
    parser.add_argument("--merged-dir", default='results/merged')

    args = parser.parse_args()

    basename = os.path.basename(args.output_dir)
    mergedir = os.path.join(args.merged_dir, basename)
    db_file = os.path.join(RES_DIR, f'{basename}.pkl')
    if not os.path.isdir(RES_DIR):
        os.makedirs(RES_DIR)

    database = assemble_database(args.output_dir, mergedir, None, PROPERTIES)
    # database = pd.read_pickle(db_file)
    # merge for having a single task
    merged_database = []
    for group_field_vals, data in database.groupby(['dataset', 'environment', 'model']):
        assert data.shape[0] < 3, "found too many values"
        data = data.sort_values('task')
        merged = data.bfill().head(1)
        merged.loc[:,'task'] = 'Train and Test'
        merged_database.append(merged)
    database = pd.concat(merged_database)
    database = database[~database['dataset'].str.contains('austra')]
    database = database.reset_index(drop=True)
    database.to_pickle(db_file)

    # # load all dbs
    # dbs = []
    # for fname in sorted(os.listdir(DB_DIR)):
    #     if '.pkl' in fname and fname not in [os.path.basename(fname) for fname in [DB_COMPLETE, DB_BL, DB_SUB]]:
    #         dbs.append( pd.read_pickle(os.path.join(DB_DIR, fname)) )
    #         # print(f'{fname:<20} no nan {str(dbs[-1].dropna().shape):<12} original {str(dbs[-1].shape):<12}')

    # # merge all dbs and do some small fixes
    # complete = pd.concat(dbs)
    # complete['parameters'] = complete['parameters'].fillna(0) # nan parameters are okay (occurs for PFN)
    # complete = complete.dropna().reset_index(drop=True) # get rid of failed PFN evals
    # # for some weird AGL experiments, codecarbon logged extreeemely high and unreasonable consumed energy (in the thousands and even millions of Watt)
    # # we discard these outliers (assuming a max draw of 400 Watt) and do a simple kNN gap filling
    # complete.loc[complete['train_power_draw'] / complete['train_running_time'] > 400,'train_power_draw'] = np.nan
    # complete.loc[complete['power_draw'] / complete['running_time'] > 400,'power_draw'] = np.nan
    # imputer = KNNImputer(n_neighbors=10, weights="uniform")
    # numeric = complete.select_dtypes('number').columns
    # complete.loc[:,numeric] = imputer.fit_transform(complete[numeric])
    
    # # split into metaqure, baselines and subset
    # baselines = complete[complete['model'].isin(['PFN', 'AGL', 'NAM', 'PFN4', 'PFN16', 'PFN64', 'PFN32'])]
    # complete = complete.drop(baselines.index, axis=0)
    # baselines.reset_index(drop=True).to_pickle(DB_BL)
    # complete.reset_index(drop=True).to_pickle(DB_COMPLETE)
    # subset = complete[complete['dataset'].isin(pd.unique(complete['dataset'])[5:15].tolist() + ['credit-g'])]
    # subset.reset_index(drop=True).to_pickle(DB_SUB)


    # # print some statistics
    # for env, data in complete.groupby('environment'):
    #     print(env)
    #     print('  complete:', data.dropna().shape)
    #     print('  baselines:', baselines[baselines['environment'] == env].dropna().shape)
    #     print(f'  time per baseline: {sum(budgets[env.split(" - ")[0]].values())}')
