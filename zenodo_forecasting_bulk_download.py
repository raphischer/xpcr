import requests
import os
import subprocess
import json

directory='mnt_data/data'

response = requests.get('https://zenodo.org/api/records',
                        params={'communities': 'forecasting', 'size': 1000, 'status': 'published',
                                'access_token': 'ANON'})
resp = response.json()

dataset_meta = {}

for hit in resp['hits']['hits']:
    if len(hit['files']) > 1:
        raise Exception
    ds_name = hit['files'][0]['key'].replace('.zip', '')
    hit_id = hit['id']
    ds_name_tsf = f'{ds_name}.tsf'

    full_path = os.path.join(directory, ds_name_tsf)
    if not os.path.isfile(full_path):
        cwd = os.getcwd()
        os.chdir(directory)
        subprocess.call(["wget", f"https://zenodo.org/record/{hit_id}/files/{ds_name}.zip"])
        subprocess.call(["unzip", f"{ds_name}.zip"])
        subprocess.call(["rm", f"{ds_name}.zip"])
        os.chdir(cwd)
    if os.path.isfile(full_path):
        print('SUCCESS   ', ds_name)
        dataset_meta[ds_name] = {
            'name': hit['metadata']['title'].replace(' Dataset', ''),
            'doi': hit['metadata']['doi'],
            'license': hit['metadata']['license'],
            'publication_date': hit['metadata']['publication_date']
        }
    else:
        print('ERROR   ', ds_name)
    print('--------------------------------------')

with open('meta_datasets.json', 'w') as meta:
    json.dump(dataset_meta, meta, indent=4)
