import argparse
import base64
import json

import numpy as np
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc

from exprep.index_and_rate import rate_database, load_boundaries, save_boundaries, calculate_optimal_boundaries, save_weights, update_weights
from exprep.elex.pages import create_page
from exprep.elex.util import summary_to_html_tables, toggle_element_visibility, fill_meta
from exprep.elex.graphs import create_scatter_graph, create_bar_graph, add_rating_background
from exprep.labels.label_generation import PropertyLabel
from exprep.unit_reformatting import CustomUnitReformater
from exprep.load_experiment_logs import find_sub_database


class Visualization(dash.Dash):

    def __init__(self, rated_database, boundaries, real_boundaries, meta, dark_mode=True, **kwargs):
        self.dark_mode = dark_mode
        if dark_mode:
            kwargs['external_stylesheets'] = [dbc.themes.DARKLY]
        super().__init__(__name__, **kwargs)
        
        # init some values
        self.database, self.boundaries, self.boundaries_real, self.meta = rated_database, boundaries, real_boundaries, meta
        self.update_necessary, self.rating_mode = False, 'optimistic median'
        self.datasets = pd.unique(self.database['dataset'])

        # init dicts to find restrictions between dataset, task and environments more easily
        self.tasks = {ds: pd.unique(find_sub_database(self.database, ds)['task']) for ds in self.datasets}
        self.environments = {(ds, task): pd.unique(find_sub_database(self.database, ds, task)['environment']) for ds, tasks in self.tasks.items() for task in tasks}
        self.unit_fmt = CustomUnitReformater()

        self.curr_data = {
            'ds': self.datasets[0],
            'task': self.tasks[self.datasets[0]][0],
            'env': self.environments[(self.datasets[0], self.tasks[self.datasets[0]][0])][0],
            'model': None,
            'label': None
        }

        # create a dict with all metrics for any dataset & task combination, and a map of metric unit symbols
        self.metrics, self.metric_units = {}, {}
        self.xaxis_default = {}
        self.yaxis_default = {}
        for ds in self.datasets:
            for task in self.tasks[ds]:
                subd = find_sub_database(self.database, ds, task)
                metrics = []
                for col in subd.columns:
                    for val in subd[col]:
                        if isinstance(val, dict):
                            metrics.append(col)
                            if col not in self.metric_units:
                                self.metric_units[col] = val['unit']
                            else:
                                if not self.metric_units[col] == val['unit']:
                                    raise RuntimeError(f'Unit of metric {col} not consistent in database!')
                            # set axis defaults for dataset / task combo
                            if val['group'] == 'Resources' and (ds, task) not in self.xaxis_default:
                                self.xaxis_default[(ds, task)] = col
                            if val['group'] == 'Quality' and (ds, task) not in self.yaxis_default:
                                self.yaxis_default[(ds, task)] = col
                            break
                self.metrics[(ds, task)] = metrics
        
        # setup page and create callbacks
        self.layout = create_page(self.datasets, self.meta['dataset'])
        self.callback(
            [Output('x-weight', 'value'), Output('y-weight', 'value')],
            [Input('xaxis', 'value'), Input('yaxis', 'value'), Input('weights-upload', 'contents')]
        ) (self.update_metric_fields)
        self.callback(
            [Output('task-switch', 'options'), Output('task-switch', 'value')],
            Input('ds-switch', 'value')
        ) (self.update_ds_changed)
        self.callback(
            [Output('environments', 'options'), Output('environments', 'value'), Output('xaxis', 'options'), Output('xaxis', 'value'), Output('yaxis', 'options'),  Output('yaxis', 'value'), Output('select-reference', 'options'), Output('select-reference', 'value')],
            Input('task-switch', 'value')
        ) (self.update_task_changed)
        self.callback(
            [Output(sl_id, prop) for sl_id in ['boundary-slider-x', 'boundary-slider-y'] for prop in ['min', 'max', 'value', 'marks']],
            [Input('xaxis', 'value'), Input('yaxis', 'value'), Input('boundaries-upload', 'contents'), Input('btn-calc-boundaries', 'n_clicks'), Input('select-reference', 'value')]
        ) (self.update_boundary_sliders)
        self.callback(
            Output('graph-scatter', 'figure'),
            [Input('environments', 'value'), Input('scale-switch', 'value'), Input('rating', 'value'), Input('x-weight', 'value'), Input('y-weight', 'value'), Input('boundary-slider-x', 'value'), Input('boundary-slider-y', 'value')]
        ) (self.update_scatter_graph)
        self.callback(
            Output('graph-bars', 'figure'),
            Input('graph-scatter', 'figure')
        ) (self.update_bars_graph)
        self.callback(
            [Output('model-table', 'children'), Output('metric-table', 'children'), Output('model-label', "src"), Output('label-modal-img', "src"), Output('btn-open-paper', "href"), Output('info-hover', 'is_open')],
            Input('graph-scatter', 'hoverData'), State('environments', 'value'), State('rating', 'value')
        ) (self.display_model)
        self.callback(Output('save-label', 'data'), [Input('btn-save-label', 'n_clicks'), Input('btn-save-label2', 'n_clicks'), Input('btn-save-summary', 'n_clicks'), Input('btn-save-logs', 'n_clicks')]) (self.save_label)
        self.callback(Output('save-boundaries', 'data'), Input('btn-save-boundaries', 'n_clicks')) (self.save_boundaries)
        self.callback(Output('save-weights', 'data'), Input('btn-save-weights', 'n_clicks')) (self.save_weights)
        # offcanvas and modals
        self.callback(Output("exp-config", "is_open"), Input("btn-open-exp-config", "n_clicks"), State("exp-config", "is_open")) (toggle_element_visibility)
        self.callback(Output("graph-config", "is_open"), Input("btn-open-graph-config", "n_clicks"), State("graph-config", "is_open")) (toggle_element_visibility)
        self.callback(Output('label-modal', 'is_open'), Input('model-label', "n_clicks"), State('label-modal', 'is_open')) (toggle_element_visibility)


    def update_scatter_graph(self, env_names=None, scale_switch=None, rating_mode=None, xweight=None, yweight=None, *slider_args):
        triggered_prop = dash.callback_context.triggered[0]['prop_id']
        if xweight is not None and 'x-weight' in triggered_prop:
            self.summaries = update_weights(self.summaries, xweight, self.curr_data['xaxis'])
        if yweight is not None and 'y-weight' in triggered_prop:
            self.summaries = update_weights(self.summaries, yweight, self.curr_data['yaxis'])
        update_db = False
        scale_switch = 'index' if scale_switch is None else scale_switch
        if any(slider_args) and 'slider' in triggered_prop:
            # check if sliders were updated from selecting axes, or if value was changed
            if 'x' in triggered_prop:
                axis = self.curr_data['xaxis']
                values = slider_args[0]
            else:
                axis = self.curr_data['yaxis']
                values = slider_args[1]
            print(self.boundaries[axis])
            print(self.boundaries['sMAPE'])
            for sl_idx, sl_val in enumerate(values):
                self.boundaries[axis][4 - sl_idx][0] = sl_val
                self.boundaries[axis][3 - sl_idx][1] = sl_val
            update_db = True
        if rating_mode != self.rating_mode: # check if rating mode was changed
            self.rating_mode = rating_mode
            update_db = True
        if update_db:
            self.update_database(only_current=False)
        self.plot_data = {}
        # assemble data for plotting
        env_names = self.environments[self.curr_data['task']] if env_names is None else env_names
        for env in env_names:
            env_data = { 'names': [], 'ratings': [], 'x': [], 'y': [] }
            for _, log in find_sub_database(self.curr_data['sub_database'], environment=env).iterrows():
                env_data['ratings'].append(log['compound'])
                env_data['names'].append(log['model'])
                for xy_axis, metric in zip(['x', 'y'], [self.curr_data['xaxis'], self.curr_data['yaxis']]):
                    if isinstance(log[metric], dict): # either take the value or the index of the metric
                        env_data[xy_axis].append(log[metric][scale_switch])
                    else: # error during value aggregation
                        env_data[xy_axis].append(0)
            self.plot_data[env] = env_data
        axis_names = [f'{ax} {self.metric_units[ax]}' for ax in [self.curr_data['xaxis'], self.curr_data['yaxis']]] # TODO pretty print, use name of axis?
        if scale_switch == 'index':
            rating_pos = [self.boundaries[self.curr_data['xaxis']], self.boundaries[self.curr_data['yaxis']]]
            axis_names = [name.split('[')[0].strip() + ' Index' for name in axis_names]
        else:
            current = (self.curr_data['ds'], self.curr_data['task'], env_names[0])
            rating_pos = [self.boundaries_real[current][self.curr_data['xaxis']], self.boundaries_real[current][self.curr_data['yaxis']]]
        scatter = create_scatter_graph(self.plot_data, axis_names, dark_mode=self.dark_mode)
        add_rating_background(scatter, rating_pos, self.rating_mode, dark_mode=self.dark_mode)
        return scatter
    
    def update_database(self, only_current=True):
        if not only_current: # remark for making a full update when task / data set is changed
            self.update_necessary = True
        # update the data currently displayed to user
        self.curr_data['sub_database'], self.boundaries, self.boundaries_real = rate_database(self.curr_data['sub_database'], self.boundaries, {self.curr_data['ds']: self.curr_data['reference']}, self.meta['properties'], self.unit_fmt, self.rating_mode)
        self.database.loc[self.curr_data['sub_database'].index] = self.curr_data['sub_database']

    def update_bars_graph(self, scatter_graph=None, discard_y_axis=False):
        bars = create_bar_graph(self.plot_data, self.dark_mode, discard_y_axis)
        return bars

    def update_boundary_sliders(self, xaxis=None, yaxis=None, uploaded_boundaries=None, calculated_boundaries=None, reference=None):
        if uploaded_boundaries is not None:
            raise RuntimeError
            boundaries_dict = json.loads(base64.b64decode(uploaded_boundaries.split(',')[-1]))
            self.boundaries = load_boundaries(boundaries_dict)
            self.summaries, self.boundaries, self.boundaries_real = rate_database(self.database, self.boundaries, self.meta['properties'], self.rating_mode)
        if calculated_boundaries is not None and 'calc' in dash.callback_context.triggered[0]['prop_id']:
            raise RuntimeError
            self.boundaries = calculate_optimal_boundaries(self.summaries, [0.8, 0.6, 0.4, 0.2])
            self.summaries, self.boundaries, self.boundaries_real = rate_database(self.database, boundaries=self.boundaries)
        if reference is not None and reference != self.curr_data['reference']:
            # reference changed, so re-index the current sub database
            self.curr_data['reference'] = reference
            self.update_database()
        self.curr_data['xaxis'] = xaxis or self.curr_data['xaxis']
        self.curr_data['yaxis'] = yaxis or self.curr_data['yaxis']
        values = []
        for axis in [self.curr_data['xaxis'], self.curr_data['yaxis']]:
            all_ratings = [metric['index'] for metric in self.curr_data['sub_database'][axis].dropna()]
            min_v = min(all_ratings)
            max_v = max(all_ratings)
            value = [entry[0] for entry in reversed(self.boundaries[axis][1:])]
            marks={ val: {'label': str(val)} for val in np.round(np.linspace(min_v, max_v, 10), 2)}
            values.extend([min_v, max_v, value, marks])
        return values

    def update_ds_changed(self, ds=None):
        self.curr_data['ds'] = ds or self.curr_data['ds']
        tasks = [{"label": task.capitalize(), "value": task} for task in self.tasks[self.curr_data['ds']]]
        return tasks, tasks[0]['value']

    def update_task_changed(self, task=None):
        if self.update_necessary:
            self.database, self.boundaries, self.boundaries_real = rate_database(self.database, self.boundaries, {self.curr_data['ds']: self.curr_data['reference']}, self.meta['properties'], self.unit_fmt, self.rating_mode)
            self.update_necessary = False
        self.curr_data['task'] = task or self.curr_data['task']
        avail_envs = [{"label": env, "value": env} for env in self.environments[(self.curr_data['ds'], self.curr_data['task'])]]
        axis_options = [{'label': metr, 'value': metr} for metr in self.metrics[(self.curr_data['ds'], self.curr_data['task'])]]
        self.curr_data['xaxis'] = self.xaxis_default[(self.curr_data['ds'], self.curr_data['task'])]
        self.curr_data['yaxis'] = self.yaxis_default[(self.curr_data['ds'], self.curr_data['task'])]
        self.curr_data['sub_database'] = find_sub_database(self.database, self.curr_data['ds'], self.curr_data['task'])
        models = self.curr_data['sub_database']['model'].values
        ref_options = [{'label': mod, 'value': mod} for mod in models]
        
        # find current reference model
        for _, row in self.curr_data['sub_database'].iterrows():
            metric_index_one = [metric['index'] == 1 for metric in row[self.metrics[(self.curr_data['ds'], self.curr_data['task'])]]]
            if all(metric_index_one):
                self.curr_data['reference'] = row['model']
                break
        else:
            raise RuntimeError

        return avail_envs, [avail_envs[0]['value']], axis_options, self.curr_data['xaxis'], axis_options, self.curr_data['yaxis'], ref_options, self.curr_data['reference']

    def display_model(self, hover_data=None, env_names=None, rating_mode=None):
        if hover_data is None:
            self.curr_data['model'] = None
            self.curr_data['label'] = None
            model_table, metric_table,  enc_label, link, open = None, None, None, "/", True
        else:
            print('RATING MODE', rating_mode)
            point = hover_data['points'][0]
            env_name = env_names[point['curveNumber']]
            model = find_sub_database(self.curr_data['sub_database'], environment=env_name).iloc[point['pointNumber']].to_dict()
            self.curr_data['model'] = fill_meta(model, self.meta)
            self.curr_data['label'] = PropertyLabel(self.curr_data['model'])

            model_table, metric_table = summary_to_html_tables(self.curr_data['model'])
            enc_label = self.curr_data['label'].to_encoded_image()
            link = self.curr_data['model']['model']['url']
            open = False
        return model_table, metric_table,  enc_label, enc_label, link, open

    def save_boundaries(self, save_labels_clicks=None):
        if save_labels_clicks is not None:
            return dict(content=save_boundaries(self.boundaries, None), filename='boundaries.json')

    def update_metric_fields(self, xaxis=None, yaxis=None, upload=None):
        if upload is not None:
            weights = json.loads(base64.b64decode(upload.split(',')[-1]))
            self.summaries = update_weights(self.summaries, weights)
        any_summary = self.curr_data['sub_database'].iloc[0]
        return any_summary[self.curr_data['xaxis']]['weight'], any_summary[self.curr_data['yaxis']]['weight']

    def save_weights(self, save_weights_clicks=None):
        if save_weights_clicks is not None:
            return dict(content=save_weights(self.summaries, None), filename='weights.json')

    def save_label(self, lbl_clicks=None, lbl_clicks2=None, sum_clicks=None, log_clicks=None):
        if (lbl_clicks is None and lbl_clicks2 is None and sum_clicks is None and log_clicks is None) or self.curr_data['model'] is None:
            return # callback init
        f_id = f'{self.curr_data["summary"]["name"]}_{self.curr_data["summary"]["environment"]}'
        if 'label' in dash.callback_context.triggered[0]['prop_id']:
            return dcc.send_bytes(self.curr_data['label'].write(), filename=f'energy_label_{f_id}.pdf')
        elif 'sum' in dash.callback_context.triggered[0]['prop_id']:
            return dict(content=json.dumps(self.curr_data['model'], indent=4), filename=f'energy_summary_{f_id}.json')
        else: # full logs
            # TODO load logs
            raise NotImplementedError
