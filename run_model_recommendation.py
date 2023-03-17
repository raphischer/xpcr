import pandas as pd
import numpy as np

# preprocessing, metrics & co
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import check_scoring

# ML methods
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


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


# def retransform_encoding(X, fitted_transform, fnames):
#     X_retransformed = {}
#     for transform_name, transform in fitted_transform.items():
#         f_idc = np.array([idx for idx, fname in enumerate(fnames) if transform_name in fname], dtype=int)
#         X_retransformed[transform_name] = transform.inverse_transform(X[:, f_idc])
#     return X_retransformed


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


CLASSIFIERS = {
    "Random":               DummyClassifier(strategy='uniform'),
    # "Ridge":                RidgeClassifier(),
    "Nearest Neighbors":    KNeighborsClassifier(3),
    # "Linear SVM":           SVC(kernel="linear", C=0.025),
    # "RBF SVM":              SVC(gamma=2, C=1),
    "Gaussian Process":     GaussianProcessClassifier(1.0 * RBF(1.0)),
    "Decision Tree":        DecisionTreeClassifier(max_depth=5),
    "Random Forest":        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Neural Net":           MLPClassifier(alpha=1, max_iter=1000),
    "AdaBoost":             AdaBoostClassifier(),
    "Naive Bayes":          GaussianNB(),
    "QDA":                  QuadraticDiscriminantAnalysis(),
}


# meta learn arguments
N_SPLITS =          5
TARGET_COLUMNS =    ['compound', 'running_time', 'parameters', 'RMSE']
SCORING =           ['accuracy', 'f1']
CV =                GroupKFold(n_splits=N_SPLITS) # KFold StratifiedKFold
GROUP_BY =          'dataset' # 'model'
METRIC_FIELD =      'index'


def split_y(y_train, y_test=None, quantile=None):
    quant = np.quantile(y_train, [0.5])
    y_train = [0 if y_ < quant else 1 for y_ in y_train]
    if y_test is not None:
        y_test = [0 if y_ < quant else 1 for y_ in y_test]
    return y_train, y_test


def evaluate_recommendation(database, only_print_better_than_random=False):
    # load features and prepare ML pipeline
    X, fnames, fitted_transform = prepare_features(database, FEATURES)
    group_info = LabelEncoder().fit_transform(database[GROUP_BY].values)
    print(f'\n\n:::::::::::::::::::::::::::: {"RUNNING ON " + str(X.shape) + " FEATURES":^35} ::::::::::::::::::::::::::::\n\n')

    for col in TARGET_COLUMNS:
        
        # make binary class problem
        y = np.array([val[METRIC_FIELD] if isinstance(val, dict) else val for val in database[col]])
        print(f'\n\n:::::::::::::::::::::::::::: {"Prediction of " + col:^35} ::::::::::::::::::::::::::::')
        # labels, counts = np.unique(y, return_counts=True)
        # label_header = f"{counts[0]}x{labels[0]} and {counts[1]}x{labels[1]}, split at {quant[0]:5.3f}"
        # print(f':::::::::::::::::::::::::::: {label_header:^35} ::::::::::::::::::::::::::::\n')

        best_name, best_acc, best_scores = '', 0, None
        for cls_name, cls_ in CLASSIFIERS.items():
            # scores = cross_validate(cls_, X, y, scoring=SCORING, cv=CV, groups=group_info, return_train_score=True)
            scores = {}
            # custom CV to ensure splitting y labels only on training y
            for idx, (train_idx, test_idx) in enumerate(CV.split(X, y, group_info)):
                X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
                y_train, y_test = split_y(y_train, y_test)
                cls_.fit(X_train, y_train)
                for score in SCORING:
                    scorer = check_scoring(cls_, score)
                    for split, X_pred, y_true in zip(['train', 'test'], [X_train, X_test], [y_train, y_test]):
                        score_name = f'{split}_{score}'
                        if score_name not in scores:
                            scores[score_name] = []
                        scores[score_name].append(scorer(cls_, X_pred, y_true))
            
            # print scoring and best method
            if not only_print_better_than_random or np.mean(scores['test_f1']) > 0.5:
                print_cv_scoring_results(cls_name, SCORING, scores)
            if np.mean(scores['test_f1']) > np.mean(best_acc):
                best_name = cls_name
                best_acc = scores['test_f1']
                best_scores = scores
        print('----------- BEST METHOD:')
        print_cv_scoring_results(best_name, SCORING, best_scores)
