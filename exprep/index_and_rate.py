import os
import json

import numpy as np
import pandas as pd

from exprep.unit_reformatting import CustomUnitReformater


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
        weights = [val['weight'] for val in ratings.values() if isinstance(val, dict) and 'rating' in val if val['weight'] > 0]
        weights = [w / sum(weights) for w in weights]
        ratings = [val['rating'] for val in ratings.values() if isinstance(val, dict) and 'rating' in val if val['weight'] > 0]
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


def calculate_optimal_boundaries(summaries, quantiles):
    boundaries = {}
    for metric in METRICS_INFO.keys():
        index_values = []
        for sum_ds in summaries.values():
            task = 'training' if 'training' in metric else 'inference'
            for sum_env in sum_ds[task].values():
                index_values += [ summary[metric]['index'] for summary in sum_env if metric in summary and summary[metric]['index'] is not None ]
        try:
            boundaries[metric] = np.quantile(index_values, quantiles)
        except Exception as e:
            print(e)
    return load_boundaries(boundaries)


def load_boundaries(content=None):
    if content is None:
        content = {'default': [1.5, 1.0, 0.5, 0.25]}
    if isinstance(content, dict):
        boundary_json = content
    elif isinstance(content, str):
        with open(content, "r") as file:
            boundary_json = json.load(file)

    # Convert boundaries to dictionary
    max_value = 10000
    min_value = 0

    boundary_intervals = {}

    for key, boundaries in boundary_json.items():
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
    if boundaries is None:
        boundaries = load_boundaries()
    if references is None:
        references = {}
    if properties_meta is None:
        properties_meta = {}
    if unit_fmt is None:
        unit_fmt = CustomUnitReformater()
    real_boundaries = {}

    # group each dataset, task and environment combo
    database['old_index'] = database.index # store index for mapping the groups later on
    fixed_fields = ['dataset', 'task', 'environment']
    grouped_by = database.groupby(fixed_fields)

    for group_field_vals, data in grouped_by:
        # get reference values
        group_fields = {field: val for (field, val) in zip(fixed_fields, group_field_vals)}
        if group_fields['dataset'] in references:
            reference_name = references[group_fields['dataset']]
        else:
            # TODO implement to take the most avg model
            reference_name = data['model'].iloc[0]
            references[group_fields['dataset']] = reference_name
        reference = data[data['model'] == reference_name]
        real_boundaries[group_field_vals] = {}

        # rate metrics based on reference
        for prop, meta in properties_meta.items():
            higher_better = False # TODO encode higher better info in meta
            ref_val = reference[prop].values[0]
            if isinstance(ref_val, dict): # re-indexing
                ref_val = ref_val['value']
            if prop in boundaries:
                prop_boundaries = boundaries[prop]
            else:
                prop_boundaries = boundaries['default']
                 # store so that they can be changed in returned boundaries
                boundaries[prop] = [bound.copy() for bound in prop_boundaries] # make explicit copies!
            data[prop] = data[prop].map(lambda value: process_property(value, ref_val, meta, prop_boundaries, higher_better, unit_fmt))
            # calculate real boundary values
            if 'unit' in meta: # TODO is this condition for indexable metrics 
                real_boundaries[group_field_vals][prop] = [(index_to_value(start, ref_val, higher_better), index_to_value(stop, ref_val, higher_better)) for (start, stop) in prop_boundaries]
        
        # store results back to database
        database.loc[data['old_index']] = data
    
    database.drop('old_index', axis=1, inplace=True) # drop the tmp index info
    # calculate compound ratings
    database = calculate_compound_rating(database, rating_mode)
    return database, boundaries, real_boundaries


if __name__ == '__main__':

    experiment_database = pd.read_pickle('database.pkl')
    rate_database(experiment_database)