import json

import pandas as pd

from mlprops.util import read_json

ds_meta = 'meta_dataset.json'
dataset_meta = read_json(ds_meta)
database = pd.read_pickle('results/new03.pkl')

for ds in dataset_meta.keys():
    ds_results = database[database['dataset'] == ds]
    train_times = ds_results['train_running_time'].dropna()
    if train_times.size > 0:
        dataset_meta[ds]['budget'] = int(train_times.mean())
    else:
        dataset_meta[ds]['budget'] = int(database['train_running_time'].dropna().mean())

with open(ds_meta, 'w') as meta:
    json.dump(dataset_meta, meta, indent=4)