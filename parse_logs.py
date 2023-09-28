from mlprops.load_experiment_logs import load_database

RESULT_PATHS = [
    ('mnt_data/autokeras_new', 'results/merged_autokeras_new', 'results/autokeras.pkl')
    ('mnt_data/results_new03', 'results/merged_new03', 'results/new03.pkl')
]

for logdir, mergedir, database_name in RESULT_PATHS:
    database = load_database(logdir, mergedir, None, "properties")
    database.to_pickle(database_name)
