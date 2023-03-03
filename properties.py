import argparse
import json


PROPERTIES = {
    'meta': {
        'task': lambda log: log['directory_name'].split('_')[0],
        'dataset': lambda log: log['config']['dataset'],
        'model': lambda log: log['config']['model'],
        'architecture': lambda log: extract_architecture(log),
        'software': lambda log: extract_software(log),
    },

    'train': {
        'train_running_time': lambda log: log['pyrapl']['total']['total_duration'],
        'train_power_draw': lambda log: log['pyrapl']['total']['total_power_draw']
    },
    
    'infer': {
        'running_time': lambda log: log['validation_pyrapl']['total']['total_duration'],
        'power_draw': lambda log: log['validation_pyrapl']['total']['total_power_draw'],
        'RMSE': lambda log: log['validation_results']['metrics']['aggregated']['RMSE'],
        'RMSE': lambda log: log['validation_results']['metrics']['aggregated']['MAPE'],
        'sMAPE': lambda log: log['validation_results']['metrics']['aggregated']['sMAPE']
    }
}


def extract_architecture(log):
    with open('meta_info.json', 'r') as meta:
        processor_shortforms = json.load(meta)['processor_shortforms']
    if 'GPU' in log['execution_platform']:
        n_gpus = len(log['execution_platform']['GPU'])
        gpu_name = processor_shortforms[log['execution_platform']['GPU']['0']['Name']]
        name = f'{gpu_name} x{n_gpus}' if n_gpus > 1 else gpu_name
    else:
        name = processor_shortforms[log['execution_platform']['Processor']]
    return name


def extract_software(log):
    with open('meta_info.json', 'r') as meta:
        ml_backends = json.load(meta)['ml_backends']
    if 'backend' in log['config']:
        backend_name = log['config']['backend']
    else:
        backend_name = list(ml_backends.keys())[0]
    backend_meta = ml_backends[backend_name]
    backend_version = 'n.a.'
    for package in backend_meta["Packages"]:
        for req in log['requirements']:
            if req.split('==')[0].replace('-', '_') == package.replace('-', '_'):
                backend_version = req.split('==')[1]
                break
        else:
            continue
        break
    return f'{backend_name} {backend_version}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", default="mnt_data/results_merged/train_2023_02_27_16_31_23.json")
    args = parser.parse_args()

    with open(args.logfile, 'r') as f:
        log = json.load(f)

    print(args.logfile)
    for task, metrics in PROPERTIES.items():
        if task == 'meta' or log['directory_name'].startswith(task):
            for metric_key, func in metrics.items():
                print(f'{task:<10} - {metric_key:<30} - {func(log)}')