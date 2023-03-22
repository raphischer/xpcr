import os
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from exprep.util import read_json

PLOT_WIDTH = 1000


def create_all(database, meta):
    os.chdir('paper_results')
    meta['properties']['compound_index'] = {'name': 'Overall Efficiency'}

    ####### DUMMY OUTPUT #######
    # for setting up pdf export of plotly
    fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(1)
    os.remove("dummy.pdf")



    ######## META LEARN RESULTS #######
    titles = {
        'top_1_acc': 'Top-1 Accuracy',
        'top_k_acc': 'Top-5 Accuracy',
        'top_k_overlap': 'Top-5 Overlap',
        'pred_error': 'Prediction Error',
        'pred_thresholderror_acc': 'Error < 0.1'
    }
    meta_results = read_json('model_selection_results.json')

    fig = make_subplots(rows=1, cols=len(meta_results), subplot_titles=[titles[key] for key in meta_results.keys()], shared_yaxes=True, horizontal_spacing=0.02)
    
    colors = ['lightslategray',] * len(list(meta_results.values())[0])
    colors[-1] = 'crimson'

    for plot_idx, results in enumerate(meta_results.values()):

        x, y, e = zip(*reversed([(np.mean(vals), meta['properties'][key]['name'], np.std(vals)) for key, vals in results.items()]))
        
        fig.add_trace(go.Bar(
                x=x, y=y, error_x=dict(type='data', array=e), orientation='h', marker_color=colors
            ), row=1, col=plot_idx + 1
        )
        # for metric, values in results.items():
        #     fig.add_trace(go.Bar(
        #         name=metric,
        #         x=[0], y=[np.mean(values)],
        #         error_y=dict(type='data', array=[np.std(values)])
        #     ))
        # fig.update_layout(barmode='group')
    fig.update_layout(width=PLOT_WIDTH, height=500)
    fig.update_layout(showlegend=False)
    fig.write_image(f'quality_of_model_recommendation.pdf')
    fig.show()
    print(1)