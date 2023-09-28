import os
import time
import pickle

import pandas as pd
import numpy as np

from mlprops.util import fix_seed

PLOT_WIDTH = 900
PLOT_HEIGHT = PLOT_WIDTH // 4

DS_SEL = 'car_parts_dataset_without_missing_values'
COL_SEL = 'MASE'


FT_NAMES = {
    'model_choice': 'Model Choice',
    'freq': 'Seasoniality',
    'forecast_horizon': 'Forecast Horizon',
    'num_ts': 'Number of Series',
    'avg_ts_len': 'Avg Series Length',
    'avg_ts_mean': 'Avg Series Mean',
    'avg_ts_min': 'Avg Series Min',
    'avg_ts_max': 'Avg Series Max',
    'contain_equal_length_True': 'Equally long?'
}

def create_all(database, meta, seed=0):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from plotly.express.colors import sample_colorscale
    from mlprops.elex.graphs import create_scatter_graph, add_rating_background
    from mlprops.elex.util import RATING_COLORS, RATING_COLOR_SCALE

    fix_seed(seed)
    meta_learned_db = pd.read_pickle('results/meta_learn_results.pkl')
    autokeras = pd.read_pickle('results/autokeras.pkl')
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
    models = pd.unique(database['model'])
    rows = ['Data set & ' + ' & '.join(meta['model'][mod]['short'] for mod in models) + r' \\' + '\n' + r'        \midrule']
    for ds, data in database.groupby(['dataset_orig']):
        subd = data[data['dataset'] == ds]
        ds_name = meta['dataset'][ds]['name']
        ds_name_short = ds_name[:5] + '..' + ds_name[-3:] if len(ds_name) > 10 else ds_name
        row = [ds_name_short]
        results = [subd[subd['model'] == mod]['compound_index'].iloc[0] for mod in models]
        max_r = max(results)
        for res in results:
            if res == max_r:
                row.append(r'\textbf{' + f'{res:3.2f}' + r'}')
            else:
                row.append(f'{res:3.2f}')
        rows.append(' & '.join(row) + r' \\')
    final_text = TEX_TABLE_GENERAL.replace('$DATA', '\n        '.join(rows))
    final_text = final_text.replace('$ALIGN', r'{l|ccccccccccc}')
    with open('ds_model_index.tex', 'w') as outf:
        outf.write(final_text)

    ######### METHOD COMPARISON TABLE
    models = pd.unique(database['model'])
    rows = [
        ' & '.join(['Data set', r'\multicolumn{2}{c}{X-PCR}', r'\multicolumn{2}{c}{Random}', r'\multicolumn{2}{c}{AutoKeras}', r'\multicolumn{2}{c}{Exhaustive} \\']),
        ' & '.join([ ' ' ] + [f'{COL_SEL}', 'kWh'] * 4) + r' \\',
        r'\midrule',
    ]
    for idx, ((ds), data) in enumerate(meta_learned_db.groupby(['dataset'])):
        if data['dataset_orig'].iloc[0] == ds:
            ds_name = meta['dataset'][ds]['name']
            ds_name_short = ds_name[:5] + '..' + ds_name[-3:] if len(ds_name) > 10 else ds_name
            row = [ds_name_short]
            # best recommendation
            sort_rec = data.sort_values(f'{COL_SEL}_pred', ascending=False)
            sel = sort_rec.iloc[0]
            values = [ (sel[COL_SEL]['value'], sel['train_power_draw']['value'] / 3.6e3) ]
            # random
            sel = sort_rec.iloc[np.random.randint(1, sort_rec.shape[0])]
            values.append( (sel[COL_SEL]['value'], sel['train_power_draw']['value'] / 3.6e3) )
            # autokeras
            auto = autokeras[autokeras['dataset'] == ds]
            auto = auto.sort_values('task')
            # auto = auto.fillna(method='bfill').head(1)
            values.append( (auto[COL_SEL].values[0], auto['train_power_draw'].values[1] / 3.6e3) )
            # testing all
            lowest_err = min([e['value'] for e in data[COL_SEL]])
            values.append( (lowest_err, np.sum([val['value'] / 3.6e3 for val in data['train_power_draw']]) ) )
            # bold print best error
            best_err = np.min([val[0] for val in values[:-1] if not np.isnan(val[0])])
            best_ene = np.min([val[1] for val in values if not np.isnan(val[1])])
            for idx, (err, ene) in enumerate(values):
                if err == np.inf or np.isnan(err):
                    row = row + ['N.A.', 'N.A.']
                else:
                    if err == best_err and idx < len(values) - 1:
                        row.append(r'\textbf{' + f'{err:6.3f}'[:6] + r'}')
                    else:
                        row.append(f'{err:6.3f}'[:6])
                    if ene == best_ene and idx < len(values) - 1:
                        row.append(r'\textbf{' + f'{ene:6.3f}'[:6] + r'}')
                    else:
                        row.append(f'{ene:6.3f}'[:6])
            rows.append(' & '.join(row) + r' \\')
    final_text = TEX_TABLE_GENERAL.replace('$DATA', '\n        '.join(rows))
    final_text = final_text.replace('$ALIGN', r'{l|cc|cc|cc||cc}')
    with open('method_comparison.tex', 'w') as outf:
        outf.write(final_text)


    print('Generating figures')
    ####### DUMMY OUTPUT #######
    # for setting up pdf export of plotly
    fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(0.5)
    os.remove("dummy.pdf")



    ## EXPLANATIONS
    data = meta_learned_db[meta_learned_db['dataset'] == DS_SEL]
    best_model = data.iloc[np.argmax(data['compound_index_pred'])]
    cols_to_plot = [col for col in data.columns if col.endswith('_pred') and 'compound' not in col]
    contrib = np.array([best_model[col] * best_model[col.replace('_pred', '')]['weight'] for col in cols_to_plot])
    contrib = contrib / contrib.sum() * 100
    # why recommendation?
    title = f"Why use {meta['model'][best_model['model']]['short']} on {meta['dataset'][DS_SEL]['name']}?"
    fig=px.bar(
    title=title, x=[meta['properties'][col.replace('_pred', '')]['shortname'] for col in cols_to_plot],
    y=contrib, color=np.array(contrib) * -1.0, color_continuous_scale=RATING_COLOR_SCALE
    )
    fig.update_yaxes(title='Contribution to compound [%]')
    fig.update_xaxes(title='', tickangle=90)
    fig.update_layout(title_x=0.5, width=PLOT_WIDTH / 2, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 24})
    fig.update(layout_coloraxis_showscale=False)
    fig.write_image("why_recommended.pdf")
    # why ERROR estimate?
    mod_path = os.path.join(os.path.dirname(os.getcwd()), 'results', f'{COL_SEL}_models', f'model{int(best_model["split_index"])}.pkl')
    with open(mod_path, 'rb') as modfile:
        model = pickle.load(modfile)
    ft_imp = model.named_steps['regressor'].feature_importances_
    ft_names = [None] * ft_imp.size
    transf_ind = model.named_steps['preprocessor'].output_indices_
    transf = model.named_steps['preprocessor'].named_transformers_
    ft_names[transf_ind['num']] = transf['num'].feature_names_in_ # numerical feature names
    ft_names[transf_ind['freq']] = ['freq']
    cat_names = []
    for ft, cat in zip(transf['cat'].feature_names_in_, transf['cat'].named_steps['onehot'].categories):
        cat_names = cat_names + [f'{ft}_{val}' for val in cat[1:]]
    ft_names[transf_ind['cat']] = cat_names # categorical feature names
    title = f"Reasons for {COL_SEL} estimate?"
    model_idc = [i for i, ft in enumerate(ft_names) if ft.startswith('model')]
    # summarize the importance of onehot encoded model choices
    ft_names = ['Model Choice'] + [FT_NAMES[ft] for i, ft in enumerate(ft_names) if i not in model_idc]
    ft_imp = np.array( [np.sum([ft_imp[i] for i in model_idc])] + [imp for i, imp in enumerate(ft_imp) if i not in model_idc] )
    fig=px.bar(title=title, x=ft_names, y=ft_imp, color=ft_imp * -1.0, color_continuous_scale=RATING_COLOR_SCALE)
    fig.update_yaxes(title='Feature importance')
    fig.update_xaxes(title='', tickangle=90)
    fig.update_layout(title_x=0.5, width=PLOT_WIDTH / 2, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 24})
    fig.update(layout_coloraxis_showscale=False)
    fig.write_image("why_error.pdf")


    # ### PCR trade-offs scatters
    for xaxis, yaxis in [['running_time', COL_SEL], ['train_power_draw', 'RMSE']]:#, ['parameters', 'MAPE']]:
        for idx, (ds, data) in enumerate(database.groupby(['dataset_orig'])):
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
            scatter.update_layout(width=PLOT_WIDTH / 2, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
            scatter.write_image(f'landscape_{ds}_{xaxis}_{yaxis}.pdf')


    ######## META LEARN RESULTS #######
    pred_cols = [col.replace('_true', '') for col in meta_learned_db.columns if '_true' in col and 'compound' not in col]
    pred_col_shortnames = [meta['properties'][col]['shortname'] for col in pred_cols]

    # access statistics per data set
    top_increasing_k_stats = {}
    for idx, ((ds), data) in enumerate(meta_learned_db.groupby(['dataset'])):
        sorted_by_pred_err = data.sort_values(f'{COL_SEL}_pred', ascending=False)
        lowest_err = min([entry['value'] for entry in data[COL_SEL]])
        lowest_ene = sum([entry['value'] for entry in data['train_power_draw']]) / 3.6e3
        # save ene and err for increasing k
        if not np.isinf(lowest_err) and not np.isinf(lowest_ene):
            top_increasing_k_stats[ds] = {'err': [], 'ene': []}
            for k in range(1, data.shape[0] + 1):
                subd = sorted_by_pred_err.iloc[:k]
                top_increasing_k_stats[ds]['err'].append( lowest_err / min([entry['value'] for entry in subd[COL_SEL]]))
                top_increasing_k_stats[ds]['ene'].append( sum([entry['value'] for entry in subd['train_power_draw']]) / (3.6e3 * lowest_ene) )

        if data['dataset_orig'].iloc[0] == ds:
            ######### STAR PLOTS with recommendation vs best
            true_best = data.sort_values(f'compound_index_true', ascending=False).iloc[0]
            pred_best = data.sort_values(f'compound_index_pred', ascending=False).iloc[0]
            fig = go.Figure()
            # the first pred col gives the real assessed compound index
            for model, col, m_str in zip([true_best, pred_best], [RATING_COLORS[0], RATING_COLORS[2]], ['Best', 'Pred']):
                name = meta['model'][model['model']]['short']
                fig.add_trace(go.Scatterpolar(
                    r=[model[col + '_true'] for col in pred_cols], line={'color': col},
                    theta=pred_col_shortnames, fill='toself', name=f'Score ({name} - {m_str}): {model["compound_index"]:4.2f}'
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)), width=PLOT_WIDTH*0.33, height=PLOT_HEIGHT, title_y=1.0, title_x=0.5, title_text=meta['dataset'][ds]['name'],
                legend=dict( yanchor="bottom", y=1.06, xanchor="center", x=0.5), margin={'l': 0, 'r': 0, 'b': 15, 't': 70}
            )
            fig.write_image(f'true_best_{ds}.pdf')

    
    

    ###### increasing top k curves
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, x_title='k (testing top-k recommendations)', y_title="Relative value", subplot_titles=[COL_SEL, 'Power Draw'], horizontal_spacing=0.02)

    for ds, values in top_increasing_k_stats.items():
        err = values['err']
        ene = values['ene']
        k = np.arange(1, len(err) + 1)
        fig.add_trace(go.Scatter(x=k, y=err, mode='lines', line=dict(color='rgba(229,36,33,0.3)')), row=1, col=1)
        fig.add_trace(go.Scatter(x=k, y=ene, mode='lines', line=dict(color='rgba(229,36,33,0.3)')), row=1, col=2)
    
    avg_err = np.array([np.array(val['err']) for val in top_increasing_k_stats.values()]).mean(axis=0)
    avg_ene = np.array([np.array(val['ene']) for val in top_increasing_k_stats.values()]).mean(axis=0)
    fig.add_trace(go.Scatter(
        x=k, y=avg_err, mode='lines', line=dict(color='rgba(0,0,0,1.0)')), row=1, col=1
    )
    fig.add_trace(go.Scatter(
        x=k, y=avg_ene, mode='lines', line=dict(color='rgba(0,0,0,1.0)')), row=1, col=2
    )
    fig.update_layout(
        width=PLOT_WIDTH, height=PLOT_HEIGHT,
        showlegend=False, margin={'l': 60, 'r': 0, 'b': 60, 't': 25}
    )
    fig.write_image(f'increasing k.pdf')


    # ERRORS OF PREDICTION
    meta['properties']['compound_index'] = {'shortname': 'Compound', 'weight': 1.0}
    meta['properties']['compound_index_direct'] = {'shortname': 'Direct comp', 'weight': 1.0}
    pred_cols = ['compound_index', 'compound_index_direct'] + pred_cols
    result_scores = {
        'Error (a)': {},
        'Thresh (b)': {},
        'Top-1 (c)': {},
        'Top-5 (d)': {},
        'Inters (e)': {}
    }

    k_best = 5
    error_threshold = 0.1
    for col in pred_cols:
        for key in result_scores.keys():
            result_scores[key][col] = []

        for _, sub_data in iter(meta_learned_db.groupby(['split_index'])):
            top_1, top_k, overlap, error, error_acc = [], [], [], [], []

            for (ds), data in iter(sub_data.groupby(['dataset'])):
                sorted_by_true = data.sort_values(f'{col}_true', ascending=False)
                sorted_by_pred = data.sort_values(f'{col}_pred', ascending=False)
                top_5_true = sorted_by_true.iloc[:k_best]['model'].values
                top_5_pred = sorted_by_pred.iloc[:k_best]['model'].values
                best_model = sorted_by_true.iloc[0]['model']
                err = data[f'{col}_pred_error'].mean()

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
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, showlegend=False, margin={'l': 0, 'r': 0, 'b': 0, 't': 24})
    fig.update_layout(
        xaxis_range = [0, max_x[0]],
        xaxis2_range = [0, max_x[1]],
        xaxis3_range = [0, max_x[2]],
        xaxis4_range = [0, max_x[3]],
        xaxis5_range = [0, max_x[4]],
    )
    fig.write_image(f'quality_of_model_recommendation.pdf')
