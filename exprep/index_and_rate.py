import os
import json

import numpy as np
import pandas as pd

from exprep.unit_reformatting import CustomUnitReformater
from exprep.load_experiment_logs import find_sub_database


def calculate_compound_rating(ratings, mode='optimistic median', meanings=None):
    if isinstance(ratings, pd.DataFrame): # full database to rate
        for idx, log in ratings.iterrows():
            try:
                ratings.loc[idx,'compound'] = calculate_single_compound_rating(log, mode, meanings)
            except RuntimeError:
                ratings.loc[idx,'compound'] = -1
        ratings['compound'] = ratings['compound'].astype(int)
        return ratings
    return calculate_single_compound_rating(ratings, mode, meanings)


def calculate_single_compound_rating(ratings, mode, meanings=None):
    if isinstance(ratings, pd.Series):
        ratings = ratings.to_dict()
    if isinstance(ratings, dict): # model summary given instead of list of ratings
        weights, ratings_gathered = [], []
        for val in ratings.values():
            if isinstance(val, dict) and 'weight' in val and val['weight'] > 0:
                weights.append(val['weight'])
                ratings_gathered.append(val['rating'])
        weights = [w / sum(weights) for w in weights]
        ratings = ratings_gathered
    else:
        weights = [1.0 / len(ratings) for _ in ratings]
    if len(ratings) == 0:
        raise RuntimeError
    if meanings is None:
        meanings = np.arange(np.max(ratings) + 1, dtype=int)
    round_m = np.ceil if 'pessimistic' in mode else np.floor # optimistic
    if mode == 'best':
        return meanings[min(ratings)] # TODO no weighting here
    if mode == 'worst':
        return meanings[max(ratings)] # TODO no weighting here
    if 'median' in mode:
        asort = np.argsort(ratings)
        weights = np.array(weights)[asort]
        ratings = np.array(ratings)[asort]
        cumw = np.cumsum(weights)
        for i, (cw, r) in enumerate(zip(cumw, ratings)):
            if cw == 0.5:
                return meanings[int(round_m(np.average([r, ratings[i + 1]])))]
            if cw > 0.5 or (cw < 0.5 and cumw[i + 1] > 0.5):
                return meanings[r]
    if 'mean' in mode:
        return meanings[int(round_m(np.average(ratings, weights=weights)))]
    if mode == 'majority':
        return meanings[np.argmax(np.bincount(ratings))]
    raise NotImplementedError('Rating Mode not implemented!', mode)


def value_to_index(value, ref, higher_better):
    #      i = v / r                     OR                i = r / v
    try:
        return value / ref if higher_better else ref / value
    except:
        return 0


def index_to_value(index, ref, higher_better):
    if index == 0:
        index = 10e-4
    #      v = i * r                            OR         v = r / i
    return index * ref  if higher_better else ref / index


def index_to_rating(index, scale):
    for i, (upper, lower) in enumerate(scale):
        if index <= upper and index > lower:
            return i
    return 4 # worst rating if index does not fall in boundaries


def process_property(value, reference_value, meta, boundaries, higher_better, unit_fmt):
    if isinstance(value, dict): # re-indexing
        value = value['value']
    returned_dict = meta.copy()
    if pd.isna(value):
        return value
    returned_dict['value'] = value
    if 'unit' in returned_dict: # TODO is this condition for indexable metrics
        returned_dict['index'] = value_to_index(value, reference_value, higher_better)
        returned_dict['rating'] = index_to_rating(returned_dict['index'], boundaries)
        fmt_val, fmt_unit = unit_fmt.reformat_value(value, meta['unit'])
        returned_dict.update({'fmt_val': fmt_val, 'fmt_unit': fmt_unit})
    return returned_dict


def find_optimal_reference(database, pre_rating_use_meta=None):
    model_names = database['model'].values
    metric_values = {}
    if pre_rating_use_meta is not None:
        metrics = [col for col in pre_rating_use_meta.keys() if any([not np.isnan(entry) for entry in database[col]])]
    else:
        metrics = [col for col in database.columns if any([isinstance(entry, dict) for entry in database[col]])]
    # aggregate index values for each metric
    for metric in metrics:
        if pre_rating_use_meta is not None:
            meta = pre_rating_use_meta[metric]
            higher_better = 'maximize' in meta and meta['maximize']
            weight = meta['weight']
            values = {model: val for _, (model, val) in database[['model', metric]].iterrows()}
        else:
            weight, values = 0, {}
            for idx, entry in enumerate(database[metric]):
                if isinstance(entry, dict):
                    higher_better = 'maximize' in entry and entry['maximize']
                    weight = max([entry['weight'], weight])
                    values[model_names[idx]] = entry['value']
        # assess the reference for each individual metric
        ref = np.median(list(values.values())) # TODO allow to change the rating mode
        values = {name: value_to_index(val, ref, higher_better) for name, val in values.items()}
        metric_values[metric] = values, weight
    # calculate model-specific scores based on metrix index values
    scores = {}
    for model in model_names:
        scores[model] = 0
        for values, weight in metric_values.values():
            if model in values:
                scores[model] += values[model] * weight
    # take the most average scored model
    ref_model_idx = np.argsort(list(scores.values()))[len(scores)//2]
    return model_names[ref_model_idx]


def calculate_optimal_boundaries(database, quantiles):
    boundaries = {'default': [1.5, 1.0, 0.5, 0.25]}
    for col in database.columns:
        index_values = [ val['index'] for val in database[col] if isinstance(val, dict) and 'index' in val ]
        if len(index_values) > 0:
            try:
                boundaries[col] = np.quantile(index_values, quantiles)
            except Exception as e:
                print(e)
    return load_boundaries(boundaries)


def load_boundaries(content=None):
    if content is None:
        content = {'default': [1.5, 1.0, 0.5, 0.25]}
    if isinstance(content, str):
        with open(content, "r") as file:
            content = json.load(file)

    # Convert boundaries to dictionary
    min_value, max_value = 0, 100000
    boundary_intervals = {}
    for key, boundaries in content.items():
        intervals = [[max_value, boundaries[0]]]
        for i in range(len(boundaries)-1):
            intervals.append([boundaries[i], boundaries[i+1]])
        intervals.append([boundaries[-1], min_value])
        boundary_intervals[key] = intervals

    return boundary_intervals


def save_boundaries(boundary_intervals, output="boundaries.json"):
    scale = {}
    for key in boundary_intervals.keys():
        scale[key] = [sc[0] for sc in boundary_intervals[key][1:]]
    if output is not None:
        with open(output, 'w') as out:
            json.dump(scale, out, indent=4)
    
    return json.dumps(scale, indent=4)


def save_weights(summaries, output="weights.json"):
    weights = {}
    for task_summaries in summaries.values():
        any_summary = list(task_summaries.values())[0][0]
        for key, vals in any_summary.items():
            if isinstance(vals, dict) and 'weight' in vals:
                weights[key] = vals['weight']
    if output is not None:
        with open(output, 'w') as out:
            json.dump(weights, out, indent=4)
    
    return json.dumps(weights, indent=4)


def update_weights(summaries, weights, axis=None):
    for task_summaries in summaries.values():
        for env_summaries in task_summaries.values():
            for model_sum in env_summaries:
                if isinstance(weights, dict):
                    for key, values in model_sum.items():
                        if key in weights:
                            values['weight'] = weights[key]
                else: # only update a single metric weight
                    if axis in model_sum:
                        model_sum[axis]['weight'] = weights
    return summaries


def rate_database(database, boundaries=None, references=None, properties_meta=None, unit_fmt=None, rating_mode='optimistic median'):
    # load defaults
    boundaries = boundaries or load_boundaries()
    references = references or {}
    properties_meta = properties_meta or {}
    unit_fmt = unit_fmt or CustomUnitReformater()
    real_boundaries = {}

    # group each dataset, task and environment combo
    database['old_index'] = database.index # store index for mapping the groups later on
    fixed_fields = ['dataset', 'task', 'environment']
    grouped_by = database.groupby(fixed_fields)

    for group_field_vals, data in grouped_by:
        real_boundaries[group_field_vals] = {}
        # get reference values
        group_fields = {field: val for (field, val) in zip(fixed_fields, group_field_vals)}
        if group_fields['dataset'] in references:
            reference_name = references[group_fields['dataset']]
        else: # find optimal
            reference_name = find_optimal_reference(data, properties_meta) # data['model'].iloc[0]
            references[group_fields['dataset']] = reference_name
        reference = data[data['model'] == reference_name]
        if reference.shape[0] > 1:
            raise RuntimeError(f'Found multiple results for reference {reference_name} in {group_field_vals} results!')

        # index and rate metrics based on reference
        for prop, meta in properties_meta.items():
            # TODO currently this assumes that all metrics given in properties should be indexed and rated!
            higher_better = 'maximize' in meta and meta['maximize']
            ref_val = reference[prop].values[0]
            if isinstance(ref_val, dict): # if database was already indexed before
                ref_val = ref_val['value']
            prop_boundaries = boundaries[prop] if prop in boundaries else boundaries['default']
            boundaries[prop] = [bound.copy() for bound in prop_boundaries] # not using explicit copies leads to problems on default!
            # extract meta, project on index values and rate
            data[prop] = data[prop].map(lambda value: process_property(value, ref_val, meta, prop_boundaries, higher_better, unit_fmt))
            # calculate real boundary values
            real_boundaries[group_field_vals][prop] = [(index_to_value(start, ref_val, higher_better), index_to_value(stop, ref_val, higher_better)) for (start, stop) in prop_boundaries]
        # store results back to database
        database.loc[data['old_index']] = data

    # make certain model metrics available across all tasks
    for prop, meta in properties_meta.items():
        if 'independent_of_task' in meta and meta['independent_of_task']:
            fixed_fields = ['dataset', 'environment', 'model']
            grouped_by = database.groupby(fixed_fields)
            for group_field_vals, data in grouped_by:
                valid = data[prop].dropna()
                if valid.shape[0] != 1:
                    print(f'{valid.shape[0]} not-NA values found for {prop} across all tasks on {group_field_vals}!')
                if valid.shape[0] > 0:
                    data[prop] = [valid.values[0]] * data.shape[0]
                    database.loc[data['old_index']] = data
    
    database.drop('old_index', axis=1, inplace=True) # drop the tmp index info
    # calculate compound ratings
    database = calculate_compound_rating(database, rating_mode)
    return database, boundaries, real_boundaries, references


def find_relevant_metrics(database):
    all_metrics, metric_units = {}, {}
    most_imp_res, most_imp_qual = {}, {}
    for ds in pd.unique(database['dataset']):
        for task in pd.unique(database[database['dataset'] == ds]['task']):
            subd = find_sub_database(database, ds, task)
            metrics = []
            for col in subd.columns:
                for val in subd[col]:
                    if isinstance(val, dict):
                        metrics.append(col)
                        if col not in metric_units:
                            metric_units[col] = val['unit']
                        else:
                            if not metric_units[col] == val['unit']:
                                raise RuntimeError(f'Unit of metric {col} not consistent in database!')
                        # set axis defaults for dataset / task combo
                        if val['group'] == 'Resources' and (ds, task) not in most_imp_res:
                            most_imp_res[(ds, task)] = col
                        if val['group'] == 'Quality' and (ds, task) not in most_imp_qual:
                            most_imp_qual[(ds, task)] = col
                        break
            all_metrics[(ds, task)] = metrics
    return all_metrics, metric_units, most_imp_res, most_imp_qual


if __name__ == '__main__':

    experiment_database = pd.read_pickle('database.pkl')
    rate_database(experiment_database)
