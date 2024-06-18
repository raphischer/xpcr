def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import os
import time

import pandas as pd
import numpy as np
from seedpy import fixedseed

from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer

from strep.index_and_rate import rate_database, load_database, index_to_value, calculate_single_compound_rating
from strep.util import load_meta, prop_dict_to_val

from data_lookup_info import LOOKUP
from data_loader import convert_tsf_to_dataframe as load_data
from data_loader import subsampled_to_orig, TIMEDELTA_MAP, FREQUENCY_MAP

FREQ_TO_SECS = {v: TIMEDELTA_MAP[k].total_seconds() for k, v in FREQUENCY_MAP.items()}

class FreqTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.applymap(lambda val: FREQ_TO_SECS[val])

SCORING = {
    'MAE': mean_absolute_error,
    'MaxE': max_error
}
CV_SCORER = 'test_MAE'

FEATURES_ENCODING = {
    'contain_missing_values':   'cat',
    'contain_equal_length':     'cat',
    'environment':              'cat',
    'freq':                     'frq',
    'forecast_horizon':         'num',
    'num_ts':                   'num',
    'avg_ts_len':               'num',
    'avg_ts_mean':              'num',
    'avg_ts_min':               'num',
    'avg_ts_max':               'num'
}

REGRESSORS = {
    # 'Global Mean':              (DummyRegressor, {}),
    'Linear Regression':        (LinearRegression, {}),
    'Ridge A1':                 (Ridge, {'alpha': 1.0}),
    'Ridge A0.1':               (Ridge, {'alpha': 0.1}),
    'Ridge A00.1':              (Ridge, {'alpha': 0.01}),
    'Lasso A1':                 (Lasso, {'alpha': 1.0}),
    'Lasso A0.1':               (Lasso, {'alpha': 0.1}),
    'Lasso A0.01':              (Lasso, {'alpha': 0.01}),
    'ElasticNet A1':            (ElasticNet, {'alpha': 1.0}),
    'ElasticNet A0.1':          (ElasticNet, {'alpha': 0.1}),
    'ElasticNet A0.01':         (ElasticNet, {'alpha': 0.01}),
    'LinearSVR C1':             (LinearSVR, {}),
    'LinearSVR C10':            (LinearSVR, {'C': 10.0}),
    'LinearSVR C100':           (LinearSVR, {'C': 100.0}),
    'SVR rbf':                  (SVR, {}),
    'SVR poly':                 (SVR, {'kernel': 'poly'}),
    'SVR sigmoid':              (SVR, {'kernel': 'sigmoid'}),
    'DecisionTree':             (DecisionTreeRegressor, {'max_depth': 5}),
    'FriedmanTree':             (DecisionTreeRegressor, {'max_depth': 5, 'criterion': 'friedman_mse'})
    # 'PoissonTree':              (DecisionTreeRegressor, {'max_depth': 5, 'criterion': 'poisson'})
}


def evaluate_regressor(regr, X, y, cv_splitted, seed, scale):
    results = pd.DataFrame(index=X.index, columns=['train_pred', 'test_pred', 'train_err', 'test_err'])
    model_cls, params = REGRESSORS[regr]
    if model_cls not in [DummyRegressor, LinearRegression, SVR]:
        params['random_state'] = seed
    clsf = model_cls(**params)
    drop = 'first' if hasattr(clsf, 'fit_intercept') else None
    preprocessor = ColumnTransformer(transformers=[
        ('cat', Pipeline(steps=[ ('onehot', OneHotEncoder(drop=drop)) ]), [k for k, v in FEATURES_ENCODING.items() if v=='cat']),
        ('num', Pipeline(steps=[ ('scaler', StandardScaler()) ]), [k for k, v in FEATURES_ENCODING.items() if v=='num']),
        ('frq', Pipeline(steps=[ ('freq_trans', FreqTransformer()), ('scaler', StandardScaler()) ]), ['freq'])
    ])
    # fit and predict for each split
    for split_idx, (train_idx, test_idx) in enumerate(cv_splitted):
        X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
        model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', clsf)])
        model.fit(X_train, y_train)
        results.loc[X.iloc[train_idx].index,f'train_pred{split_idx}'] = model.predict(X_train)
        results.loc[X.iloc[test_idx].index,'test_pred'] = model.predict(X_test)
    results['train_pred'] = results[[c_ for c_ in results.columns if 'train_pred' in c_]].mean(axis=1)
    results = results.drop([f'train_pred{split}' for split in range(len(cv_splitted))], axis=1)
    if scale == 'index': 
        results['train_pred'] = results['train_pred'].clip(lower=0, upper=1)
        results['test_pred'] = results['test_pred'].clip(lower=0, upper=1)
    results['train_err'], results['test_err'] = results['train_pred'] - y, results['test_pred'] - y
    return results


def recalculate_original_values(results, ref_vals, value_db, col, col_meta):
    higher_better = 'maximize' in col_meta and col_meta['maximize']
    recalc_results = pd.DataFrame(index=results.index)
    recalc_results['train_pred'] = index_to_value(results['train_pred'], ref_vals, higher_better)
    recalc_results['test_pred'] = index_to_value(results['test_pred'], ref_vals, higher_better)
    recalc_results['train_err'] = value_db.loc[results.index,col] - recalc_results['train_pred']
    recalc_results['test_err'] = value_db.loc[results.index,col] - recalc_results['test_pred']
    return recalc_results


def error_info_as_string(row, col):
    return ' - '.join([f'{c.replace(f"{col}_", "")}: {row[c].abs().mean():10.6f} +- {row[c].abs().std():10.2f}' for c in row.columns if 'err' in c])


def load_meta_features(datasets, datadir):
    meta_database_path = "results/meta_features.csv"
    if not os.path.isfile(meta_database_path):
        all_metafeatures, ds_index = [], []
        for ds in datasets:
            ds_name = subsampled_to_orig(ds)
            subsample_str = ds.replace(ds_name, '').replace('_', '')            
            ds_seed = -1 if len(subsample_str) == 0 else int(subsample_str)
            full_path = os.path.join(datadir, ds_name + '.tsf')
            hc = LOOKUP[ds_name][1] if len(LOOKUP[ds_name]) > 1 else None
            ts_data, freq, seasonality, fh, miss_values, equal_len = load_data(full_path, ds_sample_seed=ds_seed, ext_fc_horizon=hc)
            features = {
                'num_ts': ts_data.shape[0],
                'avg_ts_len': ts_data['series_value'].map(lambda ts: len(ts)).mean(),
                'std_ts_len': ts_data['series_value'].map(lambda ts: len(ts)).std(),
                'avg_ts_mean': ts_data['series_value'].map(lambda ts: np.mean(ts)).mean(),
                'min_ts_mean': ts_data['series_value'].map(lambda ts: np.mean(ts)).min(),
                'max_ts_mean': ts_data['series_value'].map(lambda ts: np.mean(ts)).max(),
                'avg_ts_min': ts_data['series_value'].map(lambda ts: np.min(ts)).mean(),
                'min_ts_min': ts_data['series_value'].map(lambda ts: np.min(ts)).min(),
                'avg_ts_max': ts_data['series_value'].map(lambda ts: np.max(ts)).mean(),
                'max_ts_max': ts_data['series_value'].map(lambda ts: np.max(ts)).max(),
                'seasonality': seasonality,
                'freq': freq,
                'forecast_horizon': fh,
                'contain_missing_values': miss_values,
                'contain_equal_length': equal_len
            }
            all_metafeatures.append( pd.Series(features) )
            ds_index.append(ds)
        pd.DataFrame(all_metafeatures, index=ds_index).to_csv(meta_database_path)
    return pd.read_csv(meta_database_path).set_index('Unnamed: 0').fillna(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed to use")
    parser.add_argument('--datadir', default='/data/d1/xpcr/data')
    args = parser.parse_args()

    meta = load_meta()
    weights = {col: val['weight'] for col, val in meta['properties'].items()}
    database = load_database('results/dnns_merged.pkl')
    database['dataset_orig'] = database['dataset'].map(subsampled_to_orig)
    # database['MASE'] = database['MASE'].clip(upper=100)
    numeric = database.select_dtypes('number')
    numeric[np.isinf(numeric)] = np.nan
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    database.loc[:,numeric.columns] = imputer.fit_transform(numeric)
    assert not np.any(np.isnan(database.select_dtypes('number')))

    rated_db = rate_database(database, meta, indexmode='best')[0]
    index_db, value_db = prop_dict_to_val(rated_db, 'index'), prop_dict_to_val(rated_db, 'value')
    meta_features = load_meta_features(pd.unique(index_db['dataset']), args.datadir)
    meta_ft_cols = list(meta_features.columns) + ['environment']

    best_regr = {}
    
    with fixedseed(np, seed=args.seed):
        all_results = {'index': pd.DataFrame(index=database.index), 'value': pd.DataFrame(index=database.index), 'recalc_value': pd.DataFrame(index=database.index)}
        compound_direct = pd.DataFrame(index=database.index)
        # already collect reference values (only done once, for faster recalculation)
        for (d, t, e), sub_db in index_db.groupby(['dataset', 'task', 'environment']):
            for col in meta['properties'].keys():
                ref_idx = sub_db[col].idxmax() # reference value always has highest index (== 1)
                index_db.loc[(index_db['dataset'] == d) & (index_db['task'] == t) & (index_db['environment'] == e),f'{col}_ref_val'] = value_db.loc[ref_idx,col]
        split_index = {'index': np.zeros((index_db.shape[0], 1)), 'value': np.zeros((index_db.shape[0], 1))}
        for db, scale in zip([index_db, value_db], ['index', 'value']):
            # store meta feature in respective dataframe
            for ds, sub_db in db.groupby('dataset'):
                if ds in meta_features.index:
                    db.loc[sub_db.index,meta_features.columns] = meta_features[meta_features.index == ds].values
            # train meta-learners for each individual algorithm
            for algo, algo_db in db.groupby('model'):
                # prepare grouped cross-validation
                group_info = LabelEncoder().fit_transform(algo_db['dataset_orig'])
                cv_splits = list(GroupKFold(n_splits=5).split(np.zeros((group_info.size, 1)), None, group_info))
                algo_split_index = np.zeros((algo_db.shape[0], 1))
                for split_idx, (_, test_idx) in enumerate(cv_splits):
                    algo_split_index[test_idx] = split_idx
                split_index[scale][algo_db.index] = algo_split_index
                # first cv train split used for finding optimal model choice
                opt_find_db = algo_db.iloc[cv_splits[0][0]]
                opt_find_db_groups = LabelEncoder().fit_transform(opt_find_db['dataset_orig'])
                opt_find_cv_splits = list(GroupKFold(n_splits=5).split(np.zeros((opt_find_db_groups.size, 1)), None, opt_find_db_groups))
                print(f'\n\n\n\n:::::::::::::::: META LEARN FOR {algo} with {scale}\n')
                for col, col_meta in meta['properties'].items():
                    # 1. find optimal model choice on subset of the data (first cv train split)
                    regr_results = {}
                    for regr in REGRESSORS.keys():
                        res = evaluate_regressor(regr, opt_find_db[meta_ft_cols], opt_find_db[col], opt_find_cv_splits, args.seed, scale)
                        if scale == 'index': # make the selection based on MAE of REAL measurements
                            res = recalculate_original_values(res, opt_find_db[f'{col}_ref_val'], value_db, col, col_meta)
                        regr_results[regr] = res
                    sorted_results = list(sorted([(res['test_err'].abs().mean(), regr) for regr, res in regr_results.items()]))
                    best_model = sorted_results[0][1]
                    if best_model not in best_regr:
                        best_regr[best_model] = 0
                    best_regr[best_model] += 1
                    # TODO also store sorted_models.index[0] (best model name?)

                    # 2. train and evaluate best model on full data!
                    best_model_results = evaluate_regressor(best_model, algo_db[meta_ft_cols], algo_db[col], cv_splits, args.seed, scale)
                    best_results_renamed = best_model_results.rename(lambda c_ : f'{col}_{c_}', axis=1)
                    all_results[scale].loc[algo_db.index,best_results_renamed.columns] = best_results_renamed
                    if scale == 'index': # recalculate index predictions to real value predictions
                        recalc = recalculate_original_values(best_model_results, algo_db[f'{col}_ref_val'], value_db, col, col_meta).rename(lambda c_ : f'{col}_{c_}', axis=1)
                        all_results['recalc_value'].loc[algo_db.index,recalc.columns] = recalc
                    err_data = recalc if scale == 'index' else best_results_renamed
                    print(f'{col:<18} - {str(algo_db[meta_ft_cols].shape):<10} - Best Model: {best_model:<17} - {error_info_as_string(err_data, col)}')

                # also directly try to estimate the compound index
                if scale == 'index':
                    regr_results = {regr: evaluate_regressor(regr, opt_find_db[meta_ft_cols], opt_find_db['compound_index'], opt_find_cv_splits, args.seed, scale)}
                    sorted_results = list(sorted([(res['test_err'].abs().mean(), regr) for regr, res in regr_results.items()]))
                    best_model = sorted_results[0][1]
                    best_model_results = evaluate_regressor(best_model, algo_db[meta_ft_cols], algo_db[col], cv_splits, args.seed, scale).rename(lambda c_ : f'compound_index_direct_{c_}', axis=1)
                    compound_direct.loc[algo_db.index,best_model_results.columns] = best_model_results
        # recalculate the compound index score
        index_pred = pd.DataFrame(index=db.index)
        for split in ['_train_pred', '_test_pred']:
            # retrieve the index predictions for all property columns
            results = all_results['index'][[col for col in all_results['index'].columns if split in col]].rename(lambda c: c.replace(split, ''), axis=1)
            index_pred[f'compound_index{split}'] = [calculate_single_compound_rating(vals, custom_weights=weights) for _, vals in results.iterrows()]
            index_pred[f'compound_index{split.replace("_pred", "_err")}'] = db['compound_index'] - index_pred[f'compound_index{split}']
        print(f'{"compound":<18} - {str(db[meta_ft_cols].shape):<10} -                               - {error_info_as_string(index_pred, "compound_index")}')
        all_results['index'].loc[:,index_pred.columns] = index_pred
        all_results['index'].loc[:,compound_direct.columns] = compound_direct

        final_results = pd.concat(all_results.values(), keys=all_results.keys(), axis=1)
        final_results['split_index'] = split_index['index']
        final_results['split_index_value'] = split_index['value']
        final_results = pd.concat([final_results, database], axis=1)
        final_results.to_pickle('results/meta_learn_results_new.pkl')

    print(best_regr)