from mlprops.load_experiment_logs import load_database

RESULT_PATHS = [
    ('mnt_data/results_dnns', 'results/merged_dnns', 'results/dnns.pkl')
    ('mnt_data/autokeras', 'results/merged_autokeras', 'results/autokeras.pkl')
    ('mnt_data/autosklearn', 'results/merged_autosklearn', 'results/autosklearn.pkl'),
    ('mnt_data/autogluon', 'results/merged_autogluon', 'results/autogluon.pkl')
]

for logdir, mergedir, database_name in RESULT_PATHS:
    database = load_database(logdir, mergedir, None, "properties")
    database.to_pickle(database_name)
