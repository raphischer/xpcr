import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate


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


features = {
    'seasonality':              OneHotEncoder(),
    'freq':                     OneHotEncoder(),
    'forecast_horizon':         OneHotEncoder(),
    'contain_missing_values':   OneHotEncoder(),
    'contain_equal_length':     OneHotEncoder(),
    'model':                    OneHotEncoder()
}

classifiers = {
    'Random': DummyClassifier(strategy='uniform'),
    'Linear': LogisticRegression(penalty='none'),
    'Ridge': RidgeClassifier(),
    'Lasso': LogisticRegression(penalty='l1', solver='liblinear'),
    # 'SGD': SGDClassifier(),
    'SVM': SVC(),
    'AdaBoost': AdaBoostClassifier(),
    'Random Forest': RandomForestClassifier(),
    'MLP': MLPClassifier(max_iter=500, early_stopping=True)
}

database = pd.read_pickle('meta_learn.pkl')

for feat in features.keys():
    print(f'{feat:<30} {pd.unique(database[feat])}')

X, fnames, fitted_transform = prepare_features(database, features)
y = database['compound'].values
print(np.unique(y, return_counts=True))
y[y <= 1] = 1
y[y > 1] = 0
print(np.unique(y, return_counts=True))

scoring = ['accuracy', 'balanced_accuracy']
cv = StratifiedKFold(n_splits=5)

for cls_name, cls_ in classifiers.items():
    scores = cross_validate(cls_, X, y, scoring=scoring, cv=cv, return_train_score=True)
    print_cv_scoring_results(cls_name, scoring, scores)

# def retransform_encoding(X, fitted_transform, fnames):
#     X_retransformed = {}
#     for transform_name, transform in fitted_transform.items():
#         f_idc = np.array([idx for idx, fname in enumerate(fnames) if transform_name in fname], dtype=int)
#         X_retransformed[transform_name] = transform.inverse_transform(X[:, f_idc])
#     return X_retransformed
