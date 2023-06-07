import os
import json

import pandas as pd
import numpy as np

# preprocessing, metrics & co^
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# ML methods
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.dummy import DummyRegressor

from mlprops.util import PatchedJSONEncoder


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


def print_cv_scoring_results(model_name, scoring, scores):
    results = {} # 'fit time': (f'{np.mean(scores["fit_time"]):7.2f}'[:6], f'{np.std(scores["fit_time"]):6.1f}'[:5])}
    for split in ['train', 'test']:
        for score in scoring:
            res = scores[split + '_' + score]
            mean_res, std_res = np.mean(res), np.std(res)
            results[f'{split:<5} {score}'] = (f'{mean_res:7.5f}'[:6], f'{std_res:6.4f}'[:5])
    print(f'{model_name:<20}' + ' - '.join([f'{metric:<10} {mean_v} +- {std_v}' for metric, (mean_v, std_v) in results.items()]))


FEATURES_CAT = {
    'freq':                     True,
    'forecast_horizon':         True,
    'contain_missing_values':   True,
    'contain_equal_length':     True,
    'model':                    True,
    'num_ts':                   False,
    'avg_ts_len':               False,
    'avg_ts_mean':              False,
    'avg_ts_min':               False,
    'avg_ts_max':               False
}


REGRESSORS = {
    # 'Global Mean':              DummyRegressor(),
    'Linear Regression':        LinearRegression(),
    'Ridge A1':                 Ridge(alpha=1.0),
    'Ridge A0.1':               Ridge(alpha=0.1),
    'Lasso A1':                 Lasso(alpha=1.0),
    'Lasso A0.1':               Lasso(alpha=0.1),
    'ElasticNet A1':            ElasticNet(alpha=1.0),
    'ElasticNet A0.1':          ElasticNet(alpha=0.1),
    'LinearSVR':                LinearSVR(),
    'SVR':                      SVR(),
    'Random Forest':            RandomForestRegressor(n_estimators=10),
    'Extra RF':                 ExtraTreesRegressor(n_estimators=10),
}


SCORING = {
    'MAE': mean_absolute_error,
    'MaxE': max_error
}

# meta learn config
N_SPLITS =          5
# TARGET_COLUMNS =    ['compound_index', 'running_time', 'parameters', 'RMSE']
CV =                GroupKFold(n_splits=N_SPLITS) # KFold StratifiedKFold
GROUP_BY =          'dataset_orig' # 'model'
METRIC_FIELD =      'index'

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
# clf = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('classifier', LogisticRegression(solver='lbfgs'))])



def evaluate_recommendation(database):
    BEST_SCORER = f'test_{next(iter(SCORING.keys()))}'
    # load features and prepare ML pipeline
    # X, fnames, fitted_transform = prepare_features(database, FEATURES)
    # custom CV to ensure splitting y labels only on training y
    group_info = LabelEncoder().fit_transform(database[GROUP_BY].values)
    cv_splitted = list(CV.split(np.zeros((database.shape[0], 1)), None, group_info))

    ##############################################################
    ##### 1. TRAIN AND EVALUATE A BUNCH OF REGRESSION MODELS #####
    ##############################################################
    # print(f'\n\n:::::::::::::::::::::::::::: {"RUNNING ON " + str(X.shape) + " FEATURES":^35} ::::::::::::::::::::::::::::\n\n')

    for col in database.columns:
        y = np.array([val[METRIC_FIELD] if isinstance(val, dict) else np.nan for val in database[col]])
        if np.any(np.isnan(y)):
            print('SKIPPING', col)
        else:
            task = f'Regression of {col}'
            print(f'\n\n:::::::::::::::::::::::::::: {task:^35} ::::::::::::::::::::::::::::')
            
            predictions, true, proba = {}, {}, {}
            best_name, best_score, best_scores = '', np.inf, None
            split_index = np.zeros((database.shape[0], 1))

            for model_name, model_cls in REGRESSORS.items():
                # for models with intercept, onehot enocded features need to have one column dropped due to collinearity
                # https://stackoverflow.com/questions/44712521/very-large-values-predicted-for-linear-regression
                drop_first = 'first'#  if hasattr(model_cls, 'fit_intercept') else None
                numeric_transformer = Pipeline(steps=[ ('scaler', StandardScaler()) ])
                categories = [ sorted(pd.unique(database[feat]).tolist()) for feat, cat in FEATURES_CAT.items() if cat ]
                categoric_transformer = Pipeline(steps=[ ('onehot', OneHotEncoder(categories=categories, drop=drop_first)) ])
                preprocessor = ColumnTransformer(transformers=[
                    ('num', numeric_transformer, [feat for feat, cat in FEATURES_CAT.items() if not cat]),
                    ('cat', categoric_transformer, [feat for feat, cat in FEATURES_CAT.items() if cat]) 
                ])

                predictions[model_name] = np.zeros_like(y, dtype=np.float)
                true[model_name] = np.zeros_like(y, dtype=np.float)
                proba[model_name] = np.zeros_like(y, dtype=np.float)
                
                # init scoring dict
                scores = {}
                for score in SCORING.keys():
                    scores[f'train_{score}'] = []
                    scores[f'test_{score}'] = []
                
                # fit and predict for each split
                for split_idx, (train_idx, test_idx) in enumerate(cv_splitted):
                    split_index[test_idx] = split_idx
                    X_train, X_test, y_train, y_test = database.iloc[train_idx], database.iloc[test_idx], y[train_idx], y[test_idx]
                    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', clone(model_cls))])
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    # safe the predictions per model for later usage
                    predictions[model_name][test_idx] = model.predict(X_test)
                    true[model_name][test_idx] = y_test
                    if hasattr(model, "predict_proba"):
                        proba[model_name] = model.predict_proba(X_test)
                    # calculate scoring
                    for score_name, score in SCORING.items():
                        scores[f'train_{score_name}'].append(score(y_train, y_train_pred))
                        scores[f'test_{score_name}'].append(score(y_test, predictions[model_name][test_idx]))

                # print scoring and best method
                print_cv_scoring_results(model_name, SCORING.keys(), scores)
                if np.mean(scores[BEST_SCORER]) < np.mean(best_score):
                    best_name = model_name
                    best_score = scores[BEST_SCORER]
                    best_scores = scores
            print('----------- BEST METHOD:')
            print_cv_scoring_results(best_name, SCORING.keys(), best_scores)

            # store true label and prediction in database
            high_errors = predictions['Linear Regression'] - true['Linear Regression'] > 100
            print(database.loc[high_errors].shape[0], pd.unique(database.loc[high_errors]['dataset_orig']))
            database[f'{col}_pred'] = predictions[best_name]
            database[f'{col}_true'] = true[best_name]
            database[f'{col}_prob'] = proba[best_name]
            database[f'{col}_pred_model'] = best_name
            database[f'{col}_pred_error'] = np.abs(database[f'{col}_pred'] - database[f'{col}_true'])
            database['split_index'] = split_index

    database.to_pickle('results/meta_learn_results.pkl')
