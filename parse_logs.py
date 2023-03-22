from exprep.load_experiment_logs import load_database

RESULT_PATHS = [
    ('mnt_data/results', 'results/merged22', 'results/database22.pkl')
    ('mnt_data/results05', 'results/merged05', 'results/database05.pkl')
]

for logdir, mergedir, database_name in RESULT_PATHS:
    database = load_database(logdir, mergedir, None, "properties")
    database.to_pickle(database_name)
