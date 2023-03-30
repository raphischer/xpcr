import os
import time

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.express.colors import sample_colorscale

from exprep.index_and_rate import calculate_single_compound_rating
from exprep.elex.graphs import create_scatter_graph, add_rating_background
from exprep.elex.util import RATING_COLORS, RATING_COLOR_SCALE

PLOT_WIDTH = 1000


def create_all(database, boundaries, boundaries_real, meta):
    meta_learned_db = pd.read_pickle('results/meta_learn_results.pkl')
    os.chdir('paper_results')

    print('Generating tables')

    TEX_TABLE_GENERAL = r'''
    \begin{tabular}$ALIGN
        \toprule 
        $DATA
        \bottomrule
    \end{tabular}'''

    #### PROPERTY TABLE
    rows = [r'Function & Property & Group & Weight \\' + '\n' + r'        \midrule']
    weights = {prop: val['weight'] for prop, val in meta['properties'].items()}
    weights_sum = np.sum(np.array(list(weights.values())))
    for prop in meta['properties'].keys():
        meta['properties'][prop]['weight'] = meta['properties'][prop]['weight'] / weights_sum
    for i, p_meta in enumerate(meta['properties'].values()):
        row = [r'$f_{' + str(i + 1) + r'}$']
        row += [p_meta[field] if isinstance(p_meta[field], str) else f'{p_meta[field]:4.3f}' for field in ['name', 'group', 'weight']]
        rows.append(' & '.join(row) + r' \\')
    final_text = TEX_TABLE_GENERAL.replace('$DATA', '\n        '.join(rows))
    final_text = final_text.replace('$ALIGN', r'{llcc}')
    with open('properties.tex', 'w') as outf:
        outf.write(final_text)
    
    #### MODEL X DATA PERFORMANCE
    models = pd.unique(database['model'])
    rows = ['Data set & ' + ' & '.join(meta['model'][mod]['short'] for mod in models) + r' \\' + '\n' + r'        \midrule']
    for ds, data in database.groupby(['dataset_orig']):
        subd = data[data['dataset'] == ds]
        ds_name = meta['dataset'][ds]['name']
        ds_name_short = ds_name[:13] + '..' + ds_name[-5:] if len(ds_name) > 20 else ds_name
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
    with open('ds_model_index.tex', 'w') as outf:        outf.write(final_text)


    print('Generating figures')
    ####### DUMMY OUTPUT #######
    # for setting up pdf export of plotly
    fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(1)
    os.remove("dummy.pdf")


    #### PCR trade-offs scatters
    for xaxis, yaxis in [['running_time', 'RMSE'], ['parameters', 'power_draw']]:
        for idx, (ds, data) in enumerate(database.groupby(['dataset_orig'])):
            subd = data[data['dataset'] == ds]
            plot_data = {}
            scale_switch = 'index'
            env_data = { 'names': [], 'ratings': [], 'x': [], 'y': [] }
            for _, log in subd.iterrows():
                env_data['ratings'].append(log['compound_rating'])
                env_data['names'].append(meta['model'][log['model']]['short'])
                for xy_axis, metric in zip(['x', 'y'], [xaxis, yaxis]):
                    if isinstance(log[metric], dict): # either take the value or the index of the metric
                        env_data[xy_axis].append(log[metric][scale_switch])
                    else: # error during value aggregation
                        env_data[xy_axis].append(0)
            plot_data[subd['environment'].iloc[0]] = env_data
            axis_names = [meta['properties'][ax]['name'] for ax in [xaxis, yaxis]] # TODO pretty print, use name of axis?
            if scale_switch == 'index':
                rating_pos = [boundaries[ax] for ax in [xaxis, yaxis]]
                axis_names = [name.split('[')[0].strip() + ' Index' for name in axis_names]
            else:
                current = (ds, subd['task'].iloc[0], subd['environment'].iloc[0])
                rating_pos = [boundaries_real[current][ax] for ax in [xaxis, yaxis]]
            scatter = create_scatter_graph(plot_data, axis_names, False)
            add_rating_background(scatter, rating_pos, 'optimistic median', False)
            scatter.update_layout(width=PLOT_WIDTH / 2, height=PLOT_WIDTH / 2, margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
            scatter.write_image(f'landscape_{ds}_{xaxis}_{yaxis}.pdf')
            if idx == 0:
                scatter.show()


    ######## META LEARN RESULTS #######
    meta['properties']['compound_index'] = {'name': 'Compound score', 'weight': 1.0}
    k_best = 5
    error_threshold = 0.1
    pred_cols = [col.replace('_true', '') for col in meta_learned_db.columns if '_true' in col]
    pred_col_names = [meta['properties'][col]['name'] for col in pred_cols]
    pred_col_shortnames = [meta['properties'][col]['shortname'] for col in pred_cols]
    # assess the overal compound index prediction
    meta_learned_db['compound_index_true'] = meta_learned_db['compound_index']
    compounds_pred = []
    for _, row in meta_learned_db.iterrows():
        to_rate = {}
        for col in pred_cols:
            to_rate[col] = {'rating': 0} # TODO also assess compound rating with help of boundaries
            to_rate[col]['weight'] = row[col]['weight']
            to_rate[col]['index'] = row[pred_cols[0] + '_pred']
        compounds_pred.append(calculate_single_compound_rating(to_rate, 'optimistic median')['index'])
    meta_learned_db['compound_index_pred'] = compounds_pred
    meta_learned_db['compound_index_pred_error'] = np.abs(meta_learned_db['compound_index_pred'] - meta_learned_db['compound_index_true'] )
    pred_cols = ['compound_index'] + pred_cols
    pred_col_names = ['Compound index'] + pred_col_names


    ###### star plots with recommendation vs best
    best5_ene, best5_rmse, opti_ene, opti_rmse, rec_ene, rec_rmse = [], [], [], [], [], []
    for idx, ((ds), data) in enumerate(meta_learned_db.groupby(['dataset'])):
        true_best = data.sort_values(f'compound_index_true', ascending=False).iloc[0]
        pred_best = data.sort_values(f'compound_index_pred', ascending=False).iloc[0]

        # safe energy draw and rmses for scatter plot
        if data['dataset_orig'].iloc[0] == ds:
            pred_best_rmse = data.sort_values(f'RMSE_pred', ascending=False).iloc[0]
            random = data.iloc[np.random.choice(data.shape[0], 1)[0]]
            best5 = data.sort_values(f'RMSE_pred', ascending=False).iloc[:4]
            opti_ene.append(sum([entry['value'] for entry in data['train_power_draw']]) / 3.6e3)
            opti_rmse.append(min([entry['value'] for entry in data['RMSE']]))
            rec_ene.append(pred_best_rmse['train_power_draw']['value'] / 3.6e3)
            rec_rmse.append(pred_best_rmse['RMSE']['value'])
            best5_ene.append(sum([entry['value'] for entry in best5['train_power_draw']]) / 3.6e3)
            best5_rmse.append(min([entry['value'] for entry in best5['RMSE']]))
            assert rec_rmse[-1] >= opti_rmse[-1]
            assert best5_rmse[-1] >= opti_rmse[-1]
            assert best5_ene[-1] < opti_ene[-1]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[true_best[col + '_true'] for col in pred_cols[1:]], line={'color': RATING_COLORS[0]},
                theta=pred_col_shortnames, fill='toself', name=f'Compound Index (Actual Best): {true_best[pred_cols[0]]:4.2f}'
            ))
            fig.add_trace(go.Scatterpolar(
                r=[pred_best[col + '_true'] for col in pred_cols[1:]], line={'color': RATING_COLORS[2]},
                theta=pred_col_shortnames, fill='toself', name=f'Compound Index (Predicted):   {pred_best[pred_cols[0]]:4.2f}'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)), title_x=0.5, title_y=1,
                title_text=meta['dataset'][ds]['name'], width=PLOT_WIDTH*0.33, height=350,
                legend=dict( yanchor="bottom", y=1.1, xanchor="center", x=0.5 ))
            fig.write_image(f'true_best_{ds}.pdf')
            if idx < 2:
                fig.show()

    ####### energy consumption VS RMSE scatter comparison
    fig = go.Figure()
    connections_x, connections_y = [], []
    for i in range(len(rec_ene)):
        connections_x = connections_x + [None, rec_ene[i], best5_ene[i], opti_ene[i], None]
        connections_y = connections_y + [None, rec_rmse[i], best5_rmse[i], opti_rmse[i], None]

    fig.add_trace(go.Scatter(x=connections_x, y=connections_y,
                        mode='lines',
                        line=dict(color='grey', width=1, dash='dash'),
                        name='Connect', showlegend=False))
    fig.add_trace(go.Scatter(x=rec_ene, y=rec_rmse,
                        mode='markers',
                        marker=dict(size=15, color=RATING_COLORS[0], line=dict(width=3, color='white')),
                        name='Training best recommended'))
    fig.add_trace(go.Scatter(x=best5_ene, y=best5_rmse,
                        mode='markers',
                        marker=dict(size=15, color=RATING_COLORS[2], line=dict(width=3, color='white')),
                        name='Training top-5 recommended'))
    fig.add_trace(go.Scatter(x=opti_ene, y=opti_rmse,
                        mode='markers',
                        marker=dict(size=15, color=RATING_COLORS[4], line=dict(width=3, color='white')),
                        name='Training all models'))
    
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 50000])
    fig.update_layout(
        legend=dict( yanchor="top", y=0.99, xanchor="right", x=0.99 ),
        xaxis_title="Power Draw [Wh]",
        yaxis_title="RMSE",
        width=PLOT_WIDTH, height=500
    )
    fig.write_image(f'energy savings.pdf')
    fig.show()



    # ERRORS OF PREDICTION
    result_scores = {
        'Prediction error (a)': {},
        'Error < 0.1 (b)': {},
        'Top-1 acc (c)': {},
        'Top-5 acc (d)': {},
        'Top-5 intersect (e)': {}
    }

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
                best_model, err = sorted_by_true.iloc[0][['model', f'{col}_pred_error']]

                top_1.append(best_model == sorted_by_pred.iloc[0]['model'])
                top_k.append(best_model in top_5_pred)
                overlap.append(len(set(top_5_true).intersection(set(top_5_pred))))
                error.append(err)
                error_acc.append(err < error_threshold)
            
            result_scores['Prediction error (a)'][col].append(np.mean(error))
            result_scores['Error < 0.1 (b)'][col].append(np.mean(error_acc) * 100)
            result_scores['Top-1 acc (c)'][col].append(np.mean(top_1) * 100)
            result_scores['Top-5 acc (d)'][col].append(np.mean(top_k) * 100)
            result_scores['Top-5 intersect (e)'][col].append(np.mean(overlap))

    fig = make_subplots(rows=1, cols=len(result_scores), subplot_titles=[key for key in result_scores.keys()], shared_yaxes=True, horizontal_spacing=0.02)

    max_x = []
    for plot_idx, results in enumerate(result_scores.values()):
        x, y, e, w = zip(*reversed([(np.mean(vals), meta['properties'][key]['name'], np.std(vals), meta['properties'][key]['weight']) for key, vals in results.items()]))
        w = np.array(w) * -2 + 0.5
        w = list(w[:-1]) + [1.0]
        c = sample_colorscale(RATING_COLOR_SCALE, w)
        fig.add_trace(go.Bar(
                x=x, y=y, error_x=dict(type='data', array=e), orientation='h', marker_color=c
            ), row=1, col=plot_idx + 1
        )
        max_x.append(max(x) + (max(x) / 10))
    fig.update_layout(width=PLOT_WIDTH, height=500)
    fig.update_layout(showlegend=False)
    fig.update_layout(
        xaxis_range = [0, max_x[0]],
        xaxis2_range = [0, max_x[1]],
        xaxis3_range = [0, max_x[2]],
        xaxis4_range = [0, max_x[3]],
        xaxis5_range = [0, max_x[4]],
    )
    fig.update_coloraxes(colorscale=RATING_COLOR_SCALE)
    fig.update_layout(coloraxis_colorbar=dict(
        title="Contribution to compound",
    ))
    fig.write_image(f'quality_of_model_recommendation.pdf')
    fig.show()
