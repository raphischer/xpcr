from plotly import colors
from plotly.validators.scatter.marker import SymbolValidator
from dash import html


ENV_SYMBOLS = [SymbolValidator().values[i] for i in range(0, len(SymbolValidator().values), 12)]
RATING_COLOR_SCALE = colors.make_colorscale(['rgb(99,155,48)', 'rgb(184,172,43)', 'rgb(248,184,48)', 'rgb(239,125,41)', 'rgb(229,36,33)'])
RATING_COLORS = ['rgb(99,155,48)', 'rgb(184,172,43)', 'rgb(248,184,48)', 'rgb(239,125,41)', 'rgb(229,36,33)', 'rgb(36,36,36)']
PATTERNS = ["", "/", ".", "x", "-", "\\", "|", "+", "."]
RATING_MEANINGS = 'ABCDE'


# def summary_to_str(summary, rating_mode):
#     environment = f"({summary['environment']} Environment)"
#     ret_str = [f'Name: {summary["name"]:17} {environment:<34} - Final Rating {final_rating}']
#     for key, val in summary.items():
#         if isinstance(val, dict) and "value" in val:
#             if val["value"] is None:
#                 value, index = f'{"n.a.":<13}', "n.a."
#             else:
#                 value, index = f'{val["value"]:<13.3f}', f'{val["index"]:4.2f}'
#             ret_str.append(f'{AXIS_NAMES[key]:<30}: {value} - Index {index} - Rating {val["rating"]}')
#     full_str = '\n'.join(ret_str)
#     return full_str


def fill_meta(summary, meta):
    for property, value in list(summary.items()):
        try:
            summary[property] = meta[property][value]
        except KeyError:
            pass
    return summary


def summary_to_html_tables(summary):
    final_rating = f"{summary['compound_index']:5.3f} ({RATING_MEANINGS[summary['compound_rating']]})"
    info_header = [
        html.Thead(html.Tr([html.Th("Task"), html.Th("Model Name"), html.Th("Environment"), html.Th("Final Rating")]))
    ]
    
    task = f"{summary['task']} on {summary['dataset']['name']}"
    mname = summary['model']['name'] if isinstance(summary['model'], dict) else summary['model']
    info_row = [html.Tbody([html.Tr([html.Td(field) for field in [task, mname, summary['environment'], final_rating]])])]

    metrics_header = [
        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value"), html.Th("Index"), html.Th("Rating")]))
    ]
    metrics_rows = []
    for key, val in summary.items():
        if isinstance(val, dict) and "value" in val:
            value, index = val["fmt_val"], f'{val["index"]:6.4f}'[:6]
            table_cells = [f'{val["name"]} {val["fmt_unit"]}', value, index, val["rating"]]
            metrics_rows.append(html.Tr([html.Td(field) for field in table_cells]))

    model = info_header + info_row
    metrics = metrics_header + [html.Tbody(metrics_rows)]
    return model, metrics


def toggle_element_visibility(n1, is_open):
    if n1:
        return not is_open
    return is_open

