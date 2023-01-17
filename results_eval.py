import os
import json

import numpy as np
import pandas as pd

res_dir = 'mnt_data/results_merged'

log_metric_extractor = {
    'method': lambda log: log['config']['model'],
    'dataset': lambda log: log['config']['dataset'],
    'runtime': lambda log: log['validation_pyrapl']['total']['total_duration'],
    'RMSE': lambda log: log['validation_results']['metrics']['aggregated']['RMSE'],
    'RMSE': lambda log: log['validation_results']['metrics']['aggregated']['MAPE'],
    'sMAPE': lambda log: log['validation_results']['metrics']['aggregated']['sMAPE']
}

logs = []
for fname in os.listdir(res_dir):
    if not fname.startswith('infer_'):
        continue
    with open(os.path.join(res_dir, fname), 'r') as f:
        merged_log = json.load(f)
    logs.append({key: func(merged_log) for key, func in log_metric_extractor.items()})

log_df = pd.DataFrame.from_records(logs)
print(log_df)

for ds in pd.unique(log_df['dataset']):
    print(f'\n{ds} ::::::: ::::: :::: ::: :: :')
    results = log_df[log_df['dataset'] == ds]
    means = results.groupby('method').mean(numeric_only=True).sort_values('RMSE')
    print(means)
