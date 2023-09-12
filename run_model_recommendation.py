import os
import pickle

import pandas as pd
import numpy as np

# preprocessing, metrics & co
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, TransformerMixin

# ML methods
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor

from create_paper_results import COL_SEL
from data_loader import TIMEDELTA_MAP, FREQUENCY_MAP
from mlprops.util import fix_seed
FREQ_TO_SECS = {v: TIMEDELTA_MAP[k].total_seconds() for k, v in FREQUENCY_MAP.items()}


class FreqTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.applymap(lambda val: FREQ_TO_SECS[val])


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


FEATURES_ENCODING = {
    'contain_missing_values':   'cat',
    'contain_equal_length':     'cat',
    'model':                    'cat',
    'freq':                     'freq',
    'forecast_horizon':         'num',
    'num_ts':                   'num',
    'avg_ts_len':               'num',
    'avg_ts_mean':              'num',
    'avg_ts_min':               'num',
    'avg_ts_max':               'num'
}


ENCODERS = {
    'num': lambda config: Pipeline(steps=[ ('scaler', StandardScaler()) ]),
    'cat': lambda config: Pipeline(steps=[ ('onehot', OneHotEncoder(drop=config[0], categories=config[1])) ]),
    'freq': lambda config: Pipeline(steps=[ ('freq_trans', FreqTransformer()), ('scaler', StandardScaler()) ])
}


REGRESSORS = {
    # 'Global Mean':              DummyRegressor(),
    'Linear Regression':        LinearRegression(),
    'Ridge A1':                 Ridge(alpha=1.0),
    'Ridge A0.1':               Ridge(alpha=0.1),
    'Ridge A00.1':              Ridge(alpha=0.01),
    'Lasso A1':                 Lasso(alpha=1.0),
    'Lasso A0.1':               Lasso(alpha=0.1),
    'Lasso A0.01':              Lasso(alpha=0.01),
    'ElasticNet A1':            ElasticNet(alpha=1.0),
    'ElasticNet A0.1':          ElasticNet(alpha=0.1),
    'ElasticNet A0.01':         ElasticNet(alpha=0.01),
    'LinearSVR C1':             LinearSVR(),
    'LinearSVR C10':            LinearSVR(C=10.0),
    'LinearSVR C100':           LinearSVR(C=100.0),
    'SVR rbf':                  SVR(),
    'SVR poly':                 SVR(kernel='poly'),
    'SVR sigmoid':              SVR(kernel='sigmoid'),
    'DecisionTree':             DecisionTreeRegressor(max_depth=5),
    'FriedmanTree':             DecisionTreeRegressor(max_depth=5, criterion='friedman_mse'),
    'PoissonTree':              DecisionTreeRegressor(max_depth=5, criterion='poisson'),
    # 'Random Forest':            RandomForestRegressor(n_estimators=10),
    # 'Extra RF':                 ExtraTreesRegressor(n_estimators=10),
}


SCORING = {
    'MAE': mean_absolute_error,
    'MaxE': max_error
}
CV_SCORER = f'test_{next(iter(SCORING.keys()))}'

# meta learn config
N_SPLITS =          5
# TARGET_COLUMNS =    ['compound_index', 'running_time', 'parameters', 'RMSE']
CV =                GroupKFold(n_splits=N_SPLITS) # KFold StratifiedKFold
GROUP_BY =          'dataset_orig' # 'model'
METRIC_FIELD =      'index'


def evaluate_recommendation(database, seed=0):
    fix_seed(seed)
    # custom CV to ensure splitting y labels only on training y
    group_info = LabelEncoder().fit_transform(database[GROUP_BY].values)
    cv_splitted = list(CV.split(np.zeros((database.shape[0], 1)), None, group_info))

    ##############################################################
    ##### 1. TRAIN AND EVALUATE A BUNCH OF REGRESSION MODELS #####
    ##############################################################

    for col in database.columns:
        y = np.array([val[METRIC_FIELD] if isinstance(val, dict) else np.nan for val in database[col]])
        if col == 'compound_index':
            y = database[col].to_numpy()
            col = 'compound_index_direct'
        if np.any(np.isnan(y)):
            print('SKIPPING', col)
        else:
            task = f'Regression of {col}'
            print(f'\n\n:::::::::::::::::::::::::::: {task:^35} ::::::::::::::::::::::::::::')
            
            predictions, true, proba = {}, {}, {}
            best_models, best_name, best_score, best_scores = None, '', np.inf, None
            split_index = np.zeros((database.shape[0], 1))

            for model_name, model_cls in REGRESSORS.items():
                # create the feature preprocessing pipeline
                # for models with intercept, onehot enocded features need to have one column dropped due to collinearity
                # https://stackoverflow.com/questions/44712521/very-large-values-predicted-for-linear-regression
                # drop_first = 'first' # if hasattr(model_cls, 'fit_intercept') else None
                categories = [ sorted(pd.unique(database[feat]).tolist()) for feat, enc in FEATURES_ENCODING.items() if enc == 'cat' ]
                config = ('first', categories)
                transformers = {}
                for enc in FEATURES_ENCODING.values():
                    if enc not in transformers:
                        transformers[enc] = (enc, ENCODERS[enc](config), [ft for ft, enc_ in FEATURES_ENCODING.items() if enc_ == enc])
                preprocessor = ColumnTransformer(transformers=list(transformers.values()))

                predictions[model_name] = np.zeros_like(y, dtype=np.float)
                true[model_name] = np.zeros_like(y, dtype=np.float)
                proba[model_name] = np.zeros_like(y, dtype=np.float)
                
                # init scoring dict
                scores = {}
                for score in SCORING.keys():
                    scores[f'train_{score}'] = []
                    scores[f'test_{score}'] = []
                
                # fit and predict for each split
                models = []
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
                    models.append(model)

                # print scoring and best method
                print_cv_scoring_results(model_name, SCORING.keys(), scores)
                if np.mean(scores[CV_SCORER]) < np.mean(best_score):
                    best_models = models
                    best_name = model_name
                    best_score = scores[CV_SCORER]
                    best_scores = scores
            print('----------- BEST METHOD:')
            print_cv_scoring_results(best_name, SCORING.keys(), best_scores)

            # store true label and prediction in database
            database[f'{col}_pred'] = predictions[best_name]
            database[f'{col}_true'] = true[best_name]
            database[f'{col}_prob'] = proba[best_name]
            database[f'{col}_pred_model'] = best_name
            database[f'{col}_pred_error'] = np.abs(database[f'{col}_pred'] - database[f'{col}_true'])
            database['split_index'] = split_index
            
            # write models in order to check feature importance later on
            if col == COL_SEL:
                path = os.path.join('results', f'{COL_SEL}_models')
                if not os.path.isdir(path):
                    os.makedirs(path)
                for idx, model in enumerate(best_models):
                    with open(os.path.join(path, f'model{idx}.pkl'), 'wb') as outfile:
                        pickle.dump(model, outfile)

    # train regressors to predict compound rating based on the predicted metrics
    database['compound_index_true'] = database['compound_index']
    pred_cols = [col for col in database.columns if col.endswith('_pred')]
    X = database[pred_cols].values
    y = database['compound_index_true'].values
    y_pred = np.zeros_like(y)

    for split, data in database.groupby('split_index'):
        anti_index = np.array(list(set(database.index.tolist()) - set(data.index)))
        X_train = X[anti_index]
        y_train = y[anti_index]
        X_test = X[data.index]
        y_test = y[data.index]
        model = Pipeline(steps=[ ('scaler', StandardScaler()), ('regressor', LinearRegression()) ])
        model.fit(X_train, y_train)
        y_pred[data.index] = model.predict(X_test)
        print(split, np.mean(np.abs(y_test - y_pred[data.index])), np.mean(data['compound_index_direct_pred_error']))
    database['compound_index_ensemble_true'] = database['compound_index_true']
    database['compound_index_pred'] = y_pred
    database['compound_index_pred_error'] = np.abs(database['compound_index_pred'] - database['compound_index_true'] )

    database.to_pickle('results/meta_learn_results.pkl')
