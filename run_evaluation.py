from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

app = Dash()
server = app.server

app.layout = (
    html.Div([
        html.H1(children='Minimal Dash App', style={'textAlign':'center'}),
        dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
        dcc.Graph(id='graph-content'),
        html.Button('Click me!', id='test-button'),
        html.Div(id='button-out', children='Press Button to test!')
    ])
)

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    dff = df[df.country==value]
    return px.line(dff, x='year', y='pop')

@callback(
    Output('button-out', 'children'),
    Input('test-button', 'n_clicks'),
    prevent_initial_call=True
)
def on_click(n_clicks):
    return f'Button was pressed {n_clicks} times!'

if __name__ == '__main__':
    app.run(debug=True)





# import argparse
# import sys
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression

# from strep.index_and_rate import rate_database, find_relevant_metrics
# from strep.util import load_meta

# from run_log_processing import DB_COMPLETE, DB_BL, DB_META
# from run_meta_learning import FreqTransformer # import needed for loading the MASE models for paper result creation

# DATABASES = ['autokeras.pkl', 'autosklearn.pkl', 'autogluon.pkl', 'dnns.pkl']

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()

#     parser.add_argument("--mode", default='interactive', choices=['interactive', 'paper'])
#     parser.add_argument("--boundaries", default="boundaries.json")
#     parser.add_argument("--dropsubsampled", default=False, type=bool)
#     args = parser.parse_args()

#     database = pd.concat(pd.read_pickle(db) for db in [DB_COMPLETE, DB_BL]).reset_index(drop=True)
#     # GluonTS has no num params -> interpolate from file size and inf running time
#     to_use_for_regr = ['fsize', 'running_time']
#     invalid_num_params = database[database['parameters'] == -1]
#     valid_num_params = database.drop(invalid_num_params.index)
#     lin_regr = LinearRegression()
#     lin_regr.fit(valid_num_params[to_use_for_regr], valid_num_params['parameters'])
#     interp_params = lin_regr.predict(invalid_num_params[to_use_for_regr])
#     database.loc[database['parameters'] == -1,'parameters'] = interp_params.astype(int)

#     meta = load_meta()
#     for ds in pd.unique(database['dataset']):
#         rows = database[database['dataset'] == ds]
#         if ds not in meta['dataset']:
#             if args.dropsubsampled:
#                 database = database[database['dataset'] != ds]
#             else: # store meta information for all subsampled datasets!
#                 orig = rows['dataset_orig'].iloc[0]
#                 meta['dataset'][ds] = meta['dataset'][orig].copy()
#                 meta['dataset'][ds]['name'] = meta['dataset'][ds]['name'] + ds.replace(orig + '_', '') # append the ds seed to name

#     database, metrics, xaxis_default, yaxis_default = find_relevant_metrics(database, meta)
#     rated_database, boundaries, real_boundaries, references = rate_database(database, given_meta=meta['properties'], boundaries=args.boundaries)
#     print(f'Database constructed from logs has {rated_database.shape} entries')

#     if args.mode == 'paper':
#         from create_paper_results import create_all
#         create_all(rated_database, pd.read_pickle(DB_META), meta)
#         sys.exit(0)

#     # else interactive
    
#     from strep.elex.app import Visualization
#     db = {'DB': (rated_database, meta, metrics, xaxis_default, yaxis_default, boundaries, real_boundaries, references)}
#     app = Visualization(db)
#     server = app.server
#     app.run_server(debug=False)
