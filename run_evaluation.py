import argparse
import os
import pandas as pd

from exprep.load_experiment_logs import load_database
from exprep.index_and_rate import rate_database
from exprep.util import load_meta

from exprep.elex.util import fill_meta
from exprep.elex.app import Visualization
from exprep.labels.label_generation import PropertyLabel


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir-root", default="mnt_data/results", type=str, help="directory with experimental result sub directories")
    parser.add_argument("--output-logdir-merged", default="mnt_data/results_merged", type=str, help="directory where merged experiment logs (json format) are created")
    parser.add_argument("--property-extractors-module", default="properties", help="python file with PROPERTIES dictionary, which maps properties to executable extractor functions")
    parser.add_argument("--database-fname", default="database.pkl", help="filename for the database that shall be created")
    parser.add_argument("--clean", action="store_true", help="set to first delete all content in given output directories")
    parser.add_argument("--mode", default='interactive', choices=['interactive', 'paper_results', 'label'])
    # interactive exploration
    parser.add_argument("--host", default='localhost', type=str, help="default host") # '0.0.0.0'
    parser.add_argument("--port", default=8888, type=int, help="default port")
    parser.add_argument("--debug", default=False, type=bool, help="debugging")

    args = parser.parse_args()

    meta = load_meta()

    if os.path.isfile(args.database_fname): # if available, read from disc
        database = pd.read_pickle(args.database_fname)
    else:
        database = load_database(args.logdir_root, args.output_logdir_merged, None, args.property_extractors_module, args.clean)
        database.to_pickle(args.database_fname)

    rated_database, boundaries, real_boundaries = rate_database(database, properties_meta=meta['properties'])

    print(f'Database constructed from logs has {rated_database.shape} entries')

    if args.mode == 'interactive':
        app = Visualization(rated_database, boundaries, real_boundaries, meta)
        app.run_server()#debug=args.debug, host=args.host, port=args.port)

    if args.mode == 'label':
        summary = fill_meta(rated_database.iloc[0].to_dict(), meta)
        pdf_doc = PropertyLabel(summary, 'optimistic median')
        pdf_doc.save('label.pdf')

    if args.mode == 'paper_results':
        pass

    # for task, task_agg_logs in aggregated_logs.items():
    #     print('Results for', task)
    #     total_ds = pd.unique(task_agg_logs['dataset']).size
    #     best_per_ds = task_agg_logs.groupby('dataset').apply( lambda grp: grp.iloc[grp['RMSE'].argmin()].drop('dataset') )
    #     print(best_per_ds)
