import os
import json

import pandas as pd
import numpy as np

# preprocessing, metrics & co
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import check_scoring

# ML methods
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.dummy import DummyRegressor

from exprep.util import PatchedJSONEncoder


# suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def prepare_features(df, features):
    fnames = []
    X = []
    fitted_transform = {}
    for feat_name, transform in features.items():
        feat = df[feat_name]
        if transform is None:
            feat = np.expand_dims(feat.values, 1)
            fnames.append(feat_name)
        else:
            if isinstance(transform, OneHotEncoder):
                feat = transform.fit_transform(np.expand_dims(feat.values, 1)).toarray()
                fnames = fnames + [f'{feat_name}_{cat}' for cat in transform.categories_[0]]
                fitted_transform[feat_name] = transform
            elif isinstance(transform, OrdinalEncoder):
                feat = transform.fit_transform(np.expand_dims(feat.values, 1))
                fitted_transform[feat_name] = transform
                fnames.append(feat_name)
            else:
                feat = np.expand_dims(feat.map(transform).values, 1)
                fnames.append(feat_name)
        X.append(feat)
    X = np.concatenate(X, axis=1)
    return X, fnames, fitted_transform


SHORTFORMS = {
    'mean_absolute_error': 'MAE',
    'max_error': 'MaxE',
    'accuracy': 'ACC',
    'balanced_accuracy': 'BAL ACC'
}


def print_cv_scoring_results(model_name, scoring, scores):
    results = {} # 'fit time': (f'{np.mean(scores["fit_time"]):7.2f}'[:6], f'{np.std(scores["fit_time"]):6.1f}'[:5])}
    for split in ['train', 'test']:
        for score in scoring:
            res = scores[split + '_' + score]
            mean_res, std_res = np.mean(res), np.std(res)
            if mean_res < 0:
                score = score.replace('neg_', '')
                mean_res *= -1
            for lf, sf in SHORTFORMS.items():
                if score == lf:
                    score = score.replace(lf, sf)
            results[f'{split:<5} {score}'] = (f'{mean_res:7.5f}'[:6], f'{std_res:6.4f}'[:5])
    print(f'{model_name:<20}' + ' - '.join([f'{metric:<10} {mean_v} +- {std_v}' for metric, (mean_v, std_v) in results.items()]))


FEATURES = {
    'freq':                     OneHotEncoder(),
    'forecast_horizon':         OneHotEncoder(),
    'contain_missing_values':   OneHotEncoder(),
    'contain_equal_length':     OneHotEncoder(),
    'model':                    OneHotEncoder(),
    'num_ts':                   None,
    'avg_ts_len':               None,
    'avg_ts_mean':              None,
    'avg_ts_min':               None,
    'avg_ts_max':               None
}


REGRESSORS = {
    'Global Mean': DummyRegressor(),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet': ElasticNet(),
    # 'SVM': LinearSVR(),
    'Random Forest': RandomForestRegressor(),
    'Extra RF': ExtraTreesRegressor(),
}


SCORING = ['neg_mean_absolute_error', 'max_error']
# meta learn config
N_SPLITS =          5
# TARGET_COLUMNS =    ['compound_index', 'running_time', 'parameters', 'RMSE']
CV =                GroupKFold(n_splits=N_SPLITS) # KFold StratifiedKFold
GROUP_BY =          'dataset_orig' # 'model'
METRIC_FIELD =      'index'


def evaluate_recommendation(database, only_print_better_than_random=False):
    BEST_SCORER = f'test_{SCORING[0]}'
    # load features and prepare ML pipeline
    X, fnames, fitted_transform = prepare_features(database, FEATURES)
    group_info = LabelEncoder().fit_transform(database[GROUP_BY].values)

    ##############################################################
    ##### 1. TRAIN AND EVALUATE A BUNCH OF REGRESSION MODELS #####
    ##############################################################
    print(f'\n\n:::::::::::::::::::::::::::: {"RUNNING ON " + str(X.shape) + " FEATURES":^35} ::::::::::::::::::::::::::::\n\n')

    target_cols = []
    for col in database.columns:
        
        y = np.array([val[METRIC_FIELD] if isinstance(val, dict) else np.nan for val in database[col]])
        if np.any(np.isnan(y)):
            print('SKIPPING', col)
        else:
            target_cols.append(col)
            task = f'Regression of {col}'
            print(f'\n\n:::::::::::::::::::::::::::: {task:^35} ::::::::::::::::::::::::::::')
            
            predictions, true, proba = {}, {}, {}
            best_name, best_score, best_scores = '', -np.inf, None
            cv_splitted = list(CV.split(X, y, group_info))
            split_index = np.zeros((database.shape[0], 1))

            for model_name, model_cls in REGRESSORS.items():
                # custom CV to ensure splitting y labels only on training y
                # scores = cross_validate(cls_, X, y, scoring=SCORING, cv=CV, groups=group_info, return_train_score=True)
                scores = {}

                for split_idx, (train_idx, test_idx) in enumerate(cv_splitted):
                    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
                    split_index[test_idx] = split_idx

                    model_cls.fit(X_train, y_train)
                    for score in SCORING:
                        scorer = check_scoring(model_cls, score)
                        for split, X_pred, y_true in zip(['train', 'test'], [X_train, X_test], [y_train, y_test]):
                            score_name = f'{split}_{score}'
                            if score_name not in scores:
                                scores[score_name] = []
                            scores[score_name].append(scorer(model_cls, X_pred, y_true))

                    # safe the predictions per model for later usage
                    if model_name not in predictions:
                        predictions[model_name] = np.zeros_like(y, dtype=np.float)
                        true[model_name] = np.zeros_like(y, dtype=np.float)
                        proba[model_name] = np.zeros_like(y, dtype=np.float)
                    predictions[model_name][test_idx] = model_cls.predict(X_test)
                    true[model_name][test_idx] = y_test
                    has_prob = hasattr(model_cls, "predict_proba")
                    if has_prob:
                        proba[model_name] = model_cls.predict_proba(X_test)
                
                # print scoring and best method
                if not only_print_better_than_random or np.mean(scores[BEST_SCORER]) > 0.5:
                    print_cv_scoring_results(model_name, SCORING, scores)
                if np.mean(scores[BEST_SCORER]) > np.mean(best_score):
                    best_name = model_name
                    best_score = scores[BEST_SCORER]
                    best_scores = scores
            print('----------- BEST METHOD:')
            print_cv_scoring_results(best_name, SCORING, best_scores)

            # store true label and prediction in database
            database[f'{col}_pred'] = predictions[best_name]
            database[f'{col}_true'] = true[best_name]
            database[f'{col}_prob'] = proba[best_name]
            database[f'{col}_pred_model'] = best_name
            database[f'{col}_pred_error'] = np.abs(database[f'{col}_pred'] - database[f'{col}_true'])
            database['split_index'] = split_index

    
    ##############################################################
    ##### 2. EVALUATE THE PREDICTION ACROSS CV AND DATASETS  #####
    ##############################################################
    k_best = 5
    error_threshold = 0.1

    result_scores = {
        'top_1_acc': {},
        'top_k_acc': {},
        'top_k_overlap': {},
        'pred_error': {},
        'pred_thresholderror_acc': {}
    }
    for col in target_cols:
        for key in result_scores.keys():
            result_scores[key][col] = []

        for _, sub_data in iter(database.groupby(['split_index'])):
            top_1, top_k, overlap, error, error_acc = [], [], [], [], []

            for (ds), data in iter(sub_data.groupby(['dataset'])):
                sorted_by_true = data.sort_values(f'{col}_true', ascending=False)
                sorted_by_pred = data.sort_values(f'{col}_pred', ascending=False)
                top_5_true = sorted_by_true.iloc[:k_best]['model'].values
                top_5_pred = sorted_by_pred.iloc[:k_best]['model'].values
                best_model, err = sorted_by_true.iloc[0][['model', f'{col}_pred_error']]

                top_1.append(best_model == sorted_by_pred.iloc[0]['model'])
                top_k.append(best_model in top_5_pred)
                overlap.append(len(set(top_5_true).intersection(set(top_5_pred))))
                error.append(err)
                error_acc.append(err < error_threshold)
            
            result_scores['top_1_acc'][col].append(np.mean(top_1) * 100)
            result_scores['top_k_acc'][col].append(np.mean(top_k) * 100)
            result_scores['top_k_overlap'][col].append(np.mean(overlap))
            result_scores['pred_error'][col].append(np.mean(error))
            result_scores['pred_thresholderror_acc'][col].append(np.mean(error_acc) * 100)
    
    with open(os.path.join('paper_results', 'model_selection_results.json'), 'w') as rf:
        json.dump(result_scores, rf, indent=4, cls=PatchedJSONEncoder)

        # print(f'\n\nStatistics for predicting the best model on {col}')
        # print(f'Top-1 Accuracy of finding the best model:  {np.mean(top_1_acc):5.1f} +- {np.std(top_1_acc):5.2f}')
        # print(f'Top-{k_best} Accuracy of finding the best model:  {np.mean(top_k_acc):5.1f} +- {np.std(top_k_acc):5.2f}')
        # print(f'Overlap of top-{k_best} scored and predicted:     {np.mean(top_k_overlap):5.1f} +- {np.std(top_k_overlap):5.2f}')
        # print(f'Prediction Error:                          {np.mean(pred_error):5.1f} +- {np.std(pred_error):5.2f}')
        # print(f'Accuracy of being below {error_threshold:<4} error:        {np.mean(pred_thresholderror_acc):5.1f} +- {np.std(pred_thresholderror_acc):5.2f}')
