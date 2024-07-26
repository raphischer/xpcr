import os
import time
import pickle

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.express.colors import sample_colorscale

from strep.util import fix_seed
from strep.index_and_rate import rate_database
from strep.elex.graphs import create_scatter_graph, add_rating_background
from strep.elex.util import RATING_COLORS, RATING_COLOR_SCALE
from strep.util import prop_dict_to_val

from run_autoxpcr import DS_SEL, COL_SEL, MOD_SEL


PLOT_WIDTH = 900
PLOT_HEIGHT = PLOT_WIDTH // 4


FT_NAMES = {
    'model_choice':                 'Model Choice',
    'freq':                         'Seasoniality',
    'forecast_horizon':             'Fc Horizon',
    'num_ts':                       'Num Series',
    'avg_ts_len':                   'Avg Length',
    'avg_ts_mean':                  'Avg Mean',
    'avg_ts_min':                   'Avg Min',
    'avg_ts_max':                   'Avg Max',
    'contain_equal_length_True':    'Equal length',
    'contain_equal_length_False':   'Div length',
    'contain_missing_values_True':  'With NAN',
    'contain_missing_values_False':  'No NAN',
}


def get_ds_short(ds_name):
    return ds_name[:4] + '..' + ds_name[-3:] if len(ds_name) > 9 else ds_name


def format_val(value):
    if value >= 1000000:
        return f'{np.round(value/1000000, 1)}e6'
    if value >= 100000:
        return f'{np.round(value/100000, 1)}e5'
    if value >= 10000:
        return f'{np.round(value/10000, 1)}e4'
    if value >= 100:
        return str(np.round(value))[:-2]
    return f'{value:4.2f}'[:4]


def rgb_to_rgba(rgb, alpha):
    return rgb.replace('rgb', 'rgba').replace(')', f',{alpha:3.1f})')


def create_all(database, meta_learned_db, meta, seed=0):

    monash = pd.read_csv('monash.csv', delimiter=';', index_col='Dataset')
    monash = monash.replace('-', np.nan).astype(float)

    fix_seed(seed)
    dnns = sorted(pd.unique(meta_learned_db['model']).tolist())
    pred_cols = list(meta['properties'].keys())
    pred_col_shortnames = [meta['properties'][col]['shortname'] for col in pred_cols]
    os.chdir('paper_results')

    print('Generating tables')

    TEX_TABLE_GENERAL = r'''
    \begin{tabular}$ALIGN
        \toprule 
        $DATA
        \bottomrule
    \end{tabular}'''

    #### PROPERTY TABLE
    rows = [r'Property & Group & Weight \\' + '\n' + r'        \midrule']
    for i, p_meta in enumerate(meta['properties'].values()):
        row = [p_meta[field] if isinstance(p_meta[field], str) else f'{p_meta[field]:4.3f}' for field in ['name', 'group', 'weight']]
        rows.append(' & '.join(row) + r' \\')
    final_text = TEX_TABLE_GENERAL.replace('$DATA', '\n        '.join(rows))
    final_text = final_text.replace('$ALIGN', r'{lcc}')
    with open('properties.tex', 'w') as outf:
        outf.write(final_text)
    
    #### MODEL X DATA PERFORMANCE
    # rows = ['Data set & ' + ' & '.join(meta['model'][mod]['short'] for mod in dnns) + r' \\' + '\n' + r'        \midrule']
    # for ds, data in database.groupby('dataset_orig'):
    #     meta_sub = meta_learned_db[meta_learned_db['dataset'] == ds]
    #     results = meta_sub.sort_values(['model'])[('index', 'compound_index_test_pred')].tolist()
    #     row = [met]
    #     for res in results:
    #         if res == max(results):
    #             row.append(r'\textbf{' + f'{res:3.2f}' + r'}')
    #         else:
    #             row.append(f'{res:3.2f}')
    #     rows.append(' & '.join(row) + r' \\')
    # final_text = TEX_TABLE_GENERAL.replace('$DATA', '\n        '.join(rows))
    # final_text = final_text.replace('$ALIGN', r'{l|ccccccccccc}')
    # with open('ds_model_pred_pcr.tex', 'w') as outf:
    #     outf.write(final_text)

    ######### METHOD COMPARISON TABLE
    rows = [
        ' & '.join(['Data set'] + [r'\multicolumn{3}{c}{' + mod + '}' for mod in ['AutoXPCR', 'AutoXP ($\approx$ \cite{abdallah/etal/22})', 'AutoGluonTS \cite{shchur_autogluon-timeseries_2023}', 'AutoKeras \cite{jin2019auto}', 'AutoSklearn \cite{JMLR:v23:21-0992}', 'Exhaustive']]) + r' \\',
        ' & '.join([ ' ' ] + ['PCR', r'$f_{\text{MASE}}$', 'kWh'] * 6) + r' \\',
        r'\midrule',
    ]
    for ds, data in meta_learned_db.groupby('dataset'):
        if data['dataset_orig'].iloc[0] == ds:
            row = [ get_ds_short(meta['dataset'][ds]['name']) ]

            # best XPCR
            xpcr_rec_config = data.sort_values(('index', 'compound_index_test_pred'), ascending=False).iloc[0]
            xpcr = database[(database['dataset'] == xpcr_rec_config['dataset']) & (database['model'] == xpcr_rec_config['model'])].iloc[0]
            values = [ (xpcr['compound_index'], xpcr[COL_SEL]['index'], xpcr['train_power_draw']['value'] / 3.6e3) ]

            # AutoForecast / simulates metsa-learned selection based on best estimated error
            aufo_rec_config = data.sort_values(('index', f'{COL_SEL}_test_pred'), ascending=False).iloc[0]
            aufo = database[(database['dataset'] == aufo_rec_config['dataset']) & (database['model'] == aufo_rec_config['model'])].iloc[0]
            values.append( (aufo['compound_index'], aufo[COL_SEL]['index'], aufo['train_power_draw']['value'] / 3.6e3) )
            # random
            # sel_rand = sort_xpcr.iloc[np.random.randint(1, sort_xpcr.shape[0])]
            # values.append( (sel_rand['compound_index_true'], sel_rand[COL_SEL]['value'], sel_rand['train_power_draw']['value'] / 3.6e3) )
    
            # autokeras & autosklearn
            for auto in ['autogluon', 'autokeras', 'autosklearn']:
                try:
                    auto_res = database[(database['dataset'] == ds) & (database['model'] == auto)].iloc[0]
                    qual = auto_res[COL_SEL]['index'] if isinstance(auto_res[COL_SEL], dict) else np.nan
                    powr = auto_res['train_power_draw']['value'] / 3.6e3 if isinstance(auto_res['train_power_draw'], dict) else np.nan
                except:
                    qual, powr = 0, 0
                values.append( (auto_res['compound_index'], qual, powr) )
            
            # testing all - except automl search models, hence filter for "auto" keyword
            exha_all = database[(database['dataset'] == ds) & ~(database['model'].str.contains('auto'))].sort_values(f'compound_index', ascending=False)
            exha = exha_all.iloc[0]
            exha_power = np.sum([val['value'] / 3.6e3 for val in exha_all['train_power_draw']])
            values.append((exha['compound_index'], exha[COL_SEL]['index'], exha_power))

            # sel_exha = data.sort_values('compound_index_true', ascending=False).iloc[0]
            # values.append( (sel_exha['compound_index_true'], sel_exha[COL_SEL]['index'],  ) )

            # bold print best error
            best_idx = np.max([val[0] for val in values[:(len(values)-1)]])
            best_err = np.max([val[1] for val in values[:(len(values)-1)]])
            best_ene = np.min([val[2] for val in values[:(len(values)-1)]])
            for idx, results in enumerate(values):
                format_results = [ format_val(res) for res in results ]
                format_results[0] = f'{results[0]:3.2f}'
                format_results[1] = f'{results[1]:3.2f}'
                for val_idx, best in enumerate([best_idx, best_err, best_ene]):
                    if idx < len(values) - 1: # search competitor best result?
                        if (val_idx < 2 and results[val_idx] == best) or (val_idx == 2 and results[val_idx] == best):
                            format_results[val_idx] = r'\textbf{' + format_results[val_idx] + r'}'
                    if idx == len(values) - 1: # exhaustive search (last column) best result?
                        if (val_idx < 2 and results[val_idx] > best) or (val_idx == 2 and results[val_idx] < best):
                            format_results[val_idx] = r'\textbf{' + format_results[val_idx] + r'}'
                row += format_results
            rows.append(' & '.join(row) + r' \\')
    final_text = TEX_TABLE_GENERAL.replace('$DATA', '\n        '.join(rows))
    final_text = final_text.replace('$ALIGN', r'{l|ccc|ccc|ccc|ccc|ccc||ccc}')
    with open('method_comparison.tex', 'w') as outf:
        outf.write(final_text)


    print('Generating figures')
    ####### DUMMY OUTPUT #######
    # for setting up pdf export of plotly
    fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(0.5)
    os.remove("dummy.pdf")



    ### SOTA COMPARISON
    dnn_map = {meta['model'][mod]['name']: (mod, meta['model'][mod]['short']) for mod in dnns}
    mod_map = {meta['model'][mod]['name']: (mod, meta['model'][mod]['short']) for mod in pd.unique(database['model'])}
    ds_overlap = list(reversed([ds for ds in pd.unique(database['dataset']) if ds in monash.index]))
    ds_short = [get_ds_short(meta['dataset'][ds]['name']) for ds in ds_overlap]

    for name, (code_name, short) in mod_map.items():
        for ds in ds_overlap:
            subdb = database[(database['model'] == code_name) & (database['dataset'] == ds)]
            monash.loc[ds,f'{name}_mase'] = subdb['MASE'].iloc[0]['value']
            monash.loc[ds,f'{name}_compound'] = subdb['compound_index'].iloc[0]
            if name in monash.columns:
                monash.loc[ds, f'{name}_mase_diff'] = np.abs(subdb[COL_SEL].iloc[0]['value'] - monash.loc[ds,name])

    min_max = {}
    all_rel_cols = [[col for col in monash.columns if '_mase' in col], [col for col in monash.columns if '_compound' in col], [col for col in monash.columns if '_mase_diff' in col]]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.1, subplot_titles=([COL_SEL, 'PCR Score', 'Diff to Monash MASE']))
    for idx, rel_cols in enumerate(all_rel_cols):
        sub_monash = monash.loc[ds_overlap,rel_cols]
        colorbar = {'x': 0.367*(idx+1)-0.105}
        if idx==0:
            sub_monash = np.log(sub_monash)
            colorbar.update( dict(tick0=0, tickmode= 'array', tickvals=[-10, -5, 0, 5, 10, 15], ticktext=["e-10", "e-5", "1", "e5", "e10", "e15"]) )
        min_max[idx] = sub_monash.values.min(), sub_monash.values.max()
        fig.add_trace(go.Heatmap(z=sub_monash.values, x=[mod_map[mod.split('_')[0]][1] for mod in sub_monash], y=ds_short, colorscale=RATING_COLOR_SCALE, reversescale=idx==1, colorbar=colorbar, zmin=min_max[idx][0], zmax=min_max[idx][1]), row=1, col=idx+1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*1.5, margin={'l': 0, 'r': 0, 'b': 0, 't': 18})
    fig.show()

    # autoxpcr pcr estimation
    for name, (code_name, short) in dnn_map.items():
        for ds in ds_overlap:
            sub_meta = meta_learned_db[(meta_learned_db['model'] == code_name) & (meta_learned_db['dataset'] == ds)]
            monash.loc[ds,f'{name}_est_mase'] = sub_meta[('value', 'MASE_test_pred')].iloc[0]
            monash.loc[ds,f'{name}_est_compound'] = sub_meta[('index', 'compound_index_test_pred')].iloc[0]
            monash.loc[ds,f'{name}_est_compound_err'] = sub_meta[('index', 'compound_index_test_err')].iloc[0]
    all_rel_cols2 = [[c for c in monash.columns if c.endswith('_est_mase')], [c for c in monash.columns if c.endswith('_est_compound')], [c for c in monash.columns if c.endswith('_est_compound_err')]]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.1, subplot_titles=(['Estimated MASE', 'Estimated PCR Score', 'Estimation Error']))
    for idx, rel_cols in enumerate(all_rel_cols2):
        sub_monash = monash.loc[ds_overlap,rel_cols]
        colorbar = {'x': 0.367*(idx+1)-0.105}
        if idx==0:
            sub_monash = np.log(sub_monash)
            colorbar.update( dict(tick0=0, tickmode= 'array', tickvals=[-10, -5, 0, 5, 10, 15], ticktext=["e-10", "e-5", "1", "e5", "e10", "e15"]) )
        if idx < 1:
            zmin, zmax = min_max[idx]
        else:
            zmin, zmax = sub_monash.values.min(), sub_monash.values.max()
        
        fig.add_trace(go.Heatmap(z=sub_monash.values, x=[mod_map[mod.split('_')[0]][1] for mod in sub_monash], y=ds_short, colorscale=RATING_COLOR_SCALE, reversescale=idx>0, colorbar=colorbar, zmin=zmin, zmax=zmax), row=1, col=idx+1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*1.5, margin={'l': 0, 'r': 0, 'b': 0, 't': 18})
    fig.show()
    fig.write_image("pcr_estimation.pdf")


    # ## EXPLANATIONS
    data = meta_learned_db[meta_learned_db['dataset'] == DS_SEL]
    best_model = meta_learned_db[(meta_learned_db['dataset'] == DS_SEL) & (meta_learned_db['model'] == MOD_SEL)].iloc[0]
    contrib = np.array([best_model[('index', f'{col}_test_pred')] * meta['properties'][col]['weight'] for col in pred_cols])
    contrib = contrib / contrib.sum() * 100
    # why recommendation?
    title = f"Q1: Why use {meta['model'][best_model['model']]['short']} on {meta['dataset'][DS_SEL]['name']}?"
    fig=px.bar(
    title=title, x=[meta['properties'][col]['shortname'] for col in pred_cols],
    y=contrib, color=np.array(contrib) * -1.0, color_continuous_scale=RATING_COLOR_SCALE
    )
    fig.update_yaxes(title='E1 - Importance [%]')
    fig.update_xaxes(title='Property', tickangle=90)
    fig.update_layout(title_x=0.5, title_y=0.99, width=PLOT_WIDTH / 2, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 24})
    fig.update(layout_coloraxis_showscale=False)
    fig.write_image("why_recommended.pdf")
    # why ERROR estimate?
    mod_path = os.path.join(os.path.dirname(os.getcwd()), 'results', f'{COL_SEL}_models', f'model{int(best_model[("split_index", "")])}.pkl')
    with open(mod_path, 'rb') as modfile:
        model = pickle.load(modfile)
    ft_imp = model.named_steps['regressor'].feature_importances_
    ft_names = [None] * ft_imp.size
    transf_ind = model.named_steps['preprocessor'].output_indices_
    transf = model.named_steps['preprocessor'].named_transformers_
    ft_names[transf_ind['num']] = transf['num'].feature_names_in_ # numerical feature names
    ft_names[transf_ind['frq']] = ['freq']
    cat_names = []
    for ft, cat in zip(transf['cat'].feature_names_in_, transf['cat'].named_steps['onehot'].categories_):
        cat_names = cat_names + ([f'{ft}_{val}' for val in cat[1:]] if hasattr(model, 'fit_intercept') else [f'{ft}_{val}' for val in cat])
    ft_names[transf_ind['cat']] = cat_names # categorical feature names
    title = f"Q2: Reasons for {COL_SEL} estimate?"
    # summarize the importance of onehot encoded model choices
    ft_names = [FT_NAMES[ft] for i, ft in enumerate(ft_names)]
    fig=px.bar(title=title, x=ft_names, y=ft_imp, color=ft_imp * -1.0, color_continuous_scale=RATING_COLOR_SCALE)
    fig.update_yaxes(title='E2 - Importance [%]')
    fig.update_xaxes(title='Meta-feature', tickangle=90)
    fig.update_layout(title_x=0.5, title_y=0.99, width=PLOT_WIDTH / 2, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 24})
    fig.update(layout_coloraxis_showscale=False)
    fig.write_image("why_error.pdf")



    ### PCR trade-offs scatters
    for xaxis, yaxis in [['running_time', COL_SEL], ['train_power_draw', 'RMSE']]:#, ['parameters', 'MAPE']]:
        for idx, (ds, data) in enumerate(database.groupby('dataset_orig')):
            # if ds == DS_SEL:
            subd = data[data['dataset'] == ds]
            plot_data = {}
            env_data = { 'names': [], 'ratings': [], 'x': [], 'y': [], 'index': [] }
            for _, log in subd.iterrows():
                env_data['ratings'].append(log['compound_rating'])
                env_data['index'].append(log['compound_index'])
                env_data['names'].append(meta['model'][log['model']]['short'])
                for xy_axis, metric in zip(['x', 'y'], [xaxis, yaxis]):
                    if isinstance(log[metric], dict): # either take the value or the index of the metric
                        env_data[xy_axis].append(log[metric]['index'])
                    else: # error during value aggregation
                        env_data[xy_axis].append(0)
            max_index, min_index = max(env_data['index']), min(env_data['index'])
            env_data['index'] = [(val - min_index) / (max_index - min_index) for val in env_data['index']]
            plot_data[subd['environment'].iloc[0]] = env_data
            rating_pos2 = []
            for ax in ['x', 'y']:
                q = np.quantile(plot_data[subd['environment'].iloc[0]][ax], [0.8, 0.6, 0.4, 0.2])
                rating_pos2.append([[10000, q[0]], [q[0], q[1]], [q[1], q[2]], [q[2], q[3]], [q[3], -100]])
            axis_names = [meta['properties'][ax]['name'].split('[')[0].strip() + ' Index' for ax in [xaxis, yaxis]]
            scatter = create_scatter_graph(plot_data, axis_names, False, ax_border=0.15, marker_width=PLOT_WIDTH / 90)
            scatter.update_traces(textposition='top center', textfont_size=16, textfont_color='black')
            add_rating_background(scatter, rating_pos2, 'optimistic mean', False)
            scatter.update_layout(width=PLOT_WIDTH / 2, height=PLOT_HEIGHT*1.2, margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
            scatter.write_image(f'landscape_{ds}_{xaxis}_{yaxis}.pdf')



    ######## META LEARN RESULTS #######
    star_cols = pred_cols + [pred_cols[0]]
    star_cols_short = pred_col_shortnames + [pred_col_shortnames[0]]

    # access statistics per data set
    top_increasing_k_stats = {}
    for ds, data in database.groupby('dataset'):
        meta_data = meta_learned_db[meta_learned_db['dataset'] == ds]
        sorted_by_pred_err = meta_data.sort_values(('index', f'{COL_SEL}_test_pred'), ascending=False)
        lowest_err = min([entry['value'] for entry in data[COL_SEL]])
        lowest_ene = sum([entry['value'] for entry in data['train_power_draw']])
        # save ene and err for increasing k
        if not np.isinf(lowest_err) and not np.isinf(lowest_ene):
            top_increasing_k_stats[ds] = {'err': [], 'ene': []}
            for k in range(1, meta_data.shape[0] + 1):
                model_results = [data[data['model']==m].iloc[0] for m in sorted_by_pred_err.iloc[:k]['model']]
                top_increasing_k_stats[ds]['err'].append( lowest_err / min([res[COL_SEL]['value'] for res in model_results]))
                top_increasing_k_stats[ds]['ene'].append( sum([res['train_power_draw']['value'] for res in model_results]) / lowest_ene )

        if data['dataset_orig'].iloc[0] == ds:
            ######### STAR PLOTS with recommendation vs best
            pred_afo = meta_data.sort_values(('index', f'{COL_SEL}_test_pred'), ascending=False).iloc[0]['model']
            pred_xpcr = meta_data.sort_values(('index', 'compound_index_test_pred'), ascending=False).iloc[0]['model']
            fig = go.Figure()
            for model, m_str, col_idx in zip([pred_xpcr, pred_afo, 'autogluon'], ['XPCR', 'XP', 'AGl'], [0, 2, 4]):
                name = m_str if meta['model'][model]['short'] == m_str else f"{m_str} / {meta['model'][model]['short']}"
                row = database[(database['dataset'] == ds) & (database['model'] == model)].iloc[0]
                fig.add_trace(go.Scatterpolar(
                    r=[row[col]['index'] for col in star_cols], line={'color': RATING_COLORS[col_idx]}, fillcolor=rgb_to_rgba(RATING_COLORS[col_idx], 0.1),
                    theta=star_cols_short, fill='toself', name=f'Score ({name}): {row["compound_index"]:4.2f}'
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)), width=PLOT_WIDTH*0.25, height=PLOT_HEIGHT*1.2, title_y=0.99, title_x=0.5, title_text=meta['dataset'][ds]['name'],
                legend=dict( yanchor="bottom", y=1.22, xanchor="center", x=0.5), margin={'l': 0, 'r': 0, 'b': 17, 't': 120}
            )
            fig.write_image(f'true_best_{ds}.pdf')

    

    ###### increasing top k curves
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, x_title='k (testing top-k recommendations)', y_title="Relative value", subplot_titles=[f'Best possible {COL_SEL}', 'Total power draw'], horizontal_spacing=0.05)
    for ds, values in top_increasing_k_stats.items():
        err = values['err']
        ene = values['ene']
        k = np.arange(1, len(err) + 1)
        fig.add_trace(go.Scatter(x=k, y=err, mode='lines', line=dict(color='rgba(229,36,33,0.3)')), row=1, col=1)
        fig.add_trace(go.Scatter(x=k, y=ene, mode='lines', line=dict(color='rgba(229,36,33,0.3)')), row=1, col=2)
    avg_err = np.array([np.array(val['err']) for val in top_increasing_k_stats.values()]).mean(axis=0)
    avg_ene = np.array([np.array(val['ene']) for val in top_increasing_k_stats.values()]).mean(axis=0)
    fig.add_trace( go.Scatter(x=k, y=avg_err, mode='lines', line=dict(color='rgba(0,0,0,1.0)')), row=1, col=1)
    fig.add_trace( go.Scatter(x=k, y=avg_ene, mode='lines', line=dict(color='rgba(0,0,0,1.0)')), row=1, col=2)
    fig.update_layout( width=PLOT_WIDTH, height=PLOT_HEIGHT*1.2, showlegend=False, margin={'l': 57, 'r': 0, 'b': 60, 't': 17} )
    fig.write_image(f'increasing_k.pdf')


    # ERRORS OF PREDICTION
    meta['properties']['compound_index'] = {'shortname': 'XPCR Score', 'weight': 1.0}
    meta['properties']['compound_index_direct'] = {'shortname': 'Direct Score', 'weight': 1.0}
    pred_cols = ['compound_index', 'compound_index_direct'] + pred_cols
    result_scores = {
        'Error (a)': {},
        'Thresh (b)': {},
        'Top-1 (c)': {},
        'Top-5 (d)': {},
        'Inters (e)': {}
    }

    rated_db = rate_database(meta_learned_db.loc[:,database.drop(['compound_index', 'quality_index', 'resource_index', 'compound_rating', 'quality_rating', 'resource_rating'], axis=1).columns], meta['properties'], indexmode='best')[0]
    index_db = prop_dict_to_val(rated_db, 'index')
    k_best = 5
    error_threshold = 0.1
    for col in pred_cols:
        for key in result_scores.keys():
            result_scores[key][col] = []

        for _, sub_data in iter(meta_learned_db.groupby(('split_index', ''))):
            top_1, top_k, overlap, error, error_acc = [], [], [], [], []

            for ds, data in sub_data.groupby('dataset'):
                sorted_by_true = index_db.loc[data.index].sort_values('compound_index' if 'direct' in col else col, ascending=False)
                sorted_by_pred = data.sort_values(('index', f'{col}_test_pred'), ascending=False)
                top_5_true = sorted_by_true.iloc[:k_best]['model'].values
                top_5_pred = sorted_by_pred.iloc[:k_best]['model'].values
                best_model = sorted_by_true.iloc[0]['model']
                err = np.abs(data[('index', f'{col}_test_err')]).mean()

                top_1.append(best_model == sorted_by_pred.iloc[0]['model'])
                top_k.append(best_model in top_5_pred)
                overlap.append(len(set(top_5_true).intersection(set(top_5_pred))))
                error.append(err)
                error_acc.append(err < error_threshold)
            result_scores['Error (a)'][col].append(np.mean(error))
            result_scores['Thresh (b)'][col].append(np.mean(error_acc) * 100)
            result_scores['Top-1 (c)'][col].append(np.mean(top_1) * 100)
            result_scores['Top-5 (d)'][col].append(np.mean(top_k) * 100)
            result_scores['Inters (e)'][col].append(np.mean(overlap))

    fig = make_subplots(rows=1, cols=len(result_scores), subplot_titles=[key for key in result_scores.keys()], shared_yaxes=True, horizontal_spacing=0.02)
    max_x = []
    for plot_idx, results in enumerate(result_scores.values()):
        x, y, e, w = zip(*reversed([(np.mean(vals), meta['properties'][key]['shortname'], np.std(vals)
        , meta['properties'][key]['weight']) for key, vals in results.items()]))
        w = np.array(w) * -2 + 0.5
        w = list(w[:-2]) + [1.0, 1.0]
        c = sample_colorscale(RATING_COLOR_SCALE, w)
        fig.add_trace(go.Bar(
                x=x, y=y, error_x=dict(type='data', array=e), orientation='h', marker_color=c
            ), row=1, col=plot_idx + 1
        )
        max_x.append(max(x) + (max(x) / 10))
    fig.update_layout(width=PLOT_WIDTH, title_y=0.99, title_x=0.5, height=PLOT_HEIGHT*1.2, 
                      showlegend=False, margin={'l': 0, 'r': 0, 'b': 0, 't': 17},
                      xaxis_range=[0, max_x[0]], xaxis2_range=[0, max_x[1]], xaxis3_range=[0, max_x[2]],
                      xaxis4_range=[0, max_x[3]], xaxis5_range=[0, max_x[4]],
    )
    fig.write_image(f'quality_of_model_recommendation.pdf')
