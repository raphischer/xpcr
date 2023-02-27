AGGREGATORS = {
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
