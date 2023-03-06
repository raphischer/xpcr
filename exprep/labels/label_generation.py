import json
import os
import base64
import argparse

from PIL import Image
import numpy as np
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import black, white
import fitz # install via PyMuPDF
import qrcode

from exprep.index_and_rate import calculate_compound_rating

C_SIZE = (1560, 2411)

POS_GENERAL = {
    # infos that are directly taken from summary via keys
    "model":       (.04,  .855, 'drawString',        90, '-Bold'),
    "task":        (.04,  .815, 'drawString',        90, ''),
    "environment": (.04,  .42,  'drawString',        68, ''),
    "dataset":     (.95,  .815, 'drawRightString',   90, ''),
}

POS_RATINGS = { char: (.66, y) for char, y in zip('ABCDE', reversed(np.linspace(.461, .727, 5))) }

POS_METRICS = {
    'upper_left': {
        'icon':  (0.25, 0.33)
    },
    'upper_right': {
        'icon':  (0.75, 0.33)
    },
    'lower_left': {
        'icon':  (0.25, 0.137)
    },
    'lower_right': {
        'icon':  (0.75, 0.137)
    },
}

PARTS_DIR = os.path.join(os.path.dirname(__file__), "label_design", "parts")

ICONS = { f.split('_0.png')[0]: os.path.join(PARTS_DIR, f.replace('_0.', '_$.')) for f in os.listdir(PARTS_DIR) if f.endswith('_0.png') }


def place_relatively(canvas, rel_x, rel_y, draw_method, content, fontstyle='', font_size=None):
    image = 'Image' in draw_method
    draw_method = getattr(canvas, draw_method)
    x, y = int(C_SIZE[0] * rel_x), int(C_SIZE[1] * rel_y)
    if image:
        img = Image.open(content)
        draw_method(content, x - img.width // 2, y - img.height // 2)
    else:
        canvas.setFont('Helvetica' + fontstyle, font_size)
        draw_method(x, y, content)


def format_power_draw_sources(summary):
    sources = 'Sources:'
    for key, vals in summary['power_draw_sources'].items():
        if len(vals) > 0:
            sources += f' {key},'
    return sources[:-1]


def create_qr(url):
    qr = qrcode.QRCode(
        version=1, box_size=1, border=0,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img


def draw_qr(canvas, qr, x, y, width):
    qr_pix = np.array(qr)
    width //= qr_pix.shape[0]
    for (i, j), v in np.ndenumerate(qr_pix):
        if v:
            canvas.setFillColor(white)
        else:
            canvas.setFillColor(black)
        canvas.rect(x + (i * width), y + int(width * qr_pix.shape[0]) - ((j + 1) * width), width, width, fill=1, stroke=0)


def find_icon(metric_name, metric_group, icons):
    # check for exact name match
    for key, path in icons.items():
        if key == metric_name:
            return path
    # check for exact group match
    for key, path in icons.items():
        if key == metric_group:
            return path
    # check for similar name match
    for key, path in icons.items():
        if key in metric_name:
            return path
    # check for similar group match
    for key, path in icons.items():
        if key in metric_group:
            return path
    # TODO implement option to pass a mapping dictionary?
    return next(iter(icons.values()))


class PropertyLabel(fitz.Document):

    def __init__(self, summary, rating_mode, metric_map=None, custom_icons=None):
        if metric_map is None: # display highest weighted metrics
            weights = {prop: vals['weight'] for prop, vals in summary.items() if isinstance(vals, dict) and 'weight' in vals}
            metrics_sorted_by_weight = list(reversed(sorted(weights, key=weights.get)))
            if len(metrics_sorted_by_weight) < 4:
                raise RuntimeError('Please pass at least four rated metrics to Label Creation')
            metric_map = { position: metrics_sorted_by_weight[idx] for idx, position in enumerate(POS_METRICS.keys()) }
        if custom_icons is None:
            custom_icons = {}
        custom_icons.update(ICONS)
        canvas = Canvas("result.pdf", pagesize=C_SIZE)
        # background
        place_relatively(canvas, 0.5, 0.5, 'drawInlineImage', os.path.join(PARTS_DIR, f"bg.png"))
        # Final Rating & QR
        frate = calculate_compound_rating(summary, rating_mode, 'ABCDE')
        pos = POS_RATINGS[frate]
        place_relatively(canvas, pos[0], pos[1], 'drawInlineImage', os.path.join(PARTS_DIR, f"rating_{frate}.png"))
        # qr = create_qr(summary['model_info']['url'])
        # draw_qr(canvas, qr, 0.84 * C_SIZE[0], 0.896 * C_SIZE[1], 175)

        # Add stroke to make even bigger letters
        canvas.setFillColor(black)
        canvas.setLineWidth(3)
        canvas.setStrokeColor(black)
        text=canvas.beginText()
        text.setTextRenderMode(2)
        canvas._code.append(text.getCode())

        # general text
        for key, (rel_x, rel_y, draw_method, fsize, style) in POS_GENERAL.items():
            place_relatively(canvas, rel_x, rel_y, draw_method, summary[key], style, fsize)

        # rated pictograms & values
        for location, positions in POS_METRICS.items():
            metric_name = metric_map[location]
            metric = summary[metric_name]
            # print icon
            icon = find_icon(metric_name, metric['group'].lower(), custom_icons)
            rating = metric['rating']
            icon = icon.replace('_$.', f'_{rating}.')
            rel_x, rel_y = positions['icon']
            place_relatively(canvas, rel_x, rel_y, 'drawInlineImage', icon)
            # print texts
            # TODO improve this by looking at the absolute height of the placed icon
            place_relatively(canvas, rel_x, rel_y - 0.08, 'drawCentredString', metric['fmt_val'] + metric['fmt_unit'], '', 56)
            place_relatively(canvas, rel_x, rel_y - 0.11, 'drawCentredString', metric_name, '', 56)
        
        super().__init__(stream=canvas.getpdfdata(), filetype='pdf')
    
    def to_encoded_image(self):
        label_bytes = self.load_page(0).get_pixmap().tobytes()
        base64_enc = base64.b64encode(label_bytes).decode('ascii')
        return 'data:image/png;base64,{}'.format(base64_enc)


# if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Generate an energy label (.pdf) for tasks on ImageNet data")

    # # data and model input
    # parser.add_argument("--task", "-t", default="inference", choices=['inference', 'training'])
    # parser.add_argument("--model", "-m", default="ResNet101", type=str)
    # parser.add_argument("--environment", "-e", default='A100 x8 - TensorFlow 2.8.0', type=str)
    # parser.add_argument("--directory", "-d", default='results', type=str, help="Directory with .json result files")
    # parser.add_argument("--filename", "-f", default="", type=str, help="name of json logfile")
    # parser.add_argument("--output", "-o", default="label.pdf", type=str, help="name of output file")
      
    # args = parser.parse_args()

    # _, summaries = load_results(args.directory)
    # summaries, _, _ = rate_results(summaries)

    # # generate label for given filename
    # if os.path.isfile(os.path.join(args.directory, args.filename)):
    #     with open(os.path.join(args.directory, args.filename), 'r') as rf:
    #         log = json.load(rf)
    #         environment = get_environment_key(log)
    #         task = TASK_TYPES[args.filename.split('_')[0]]
    #         model = log['config']['model']
    # else:
    #     task, model, environment = args.task, args.model, args.environment
    
    # for summary in summaries[task][environment]:
    #     if summary['name'] == model:
    #         pdf_doc = PropertyLabel(summary, 'optimistic median')
    #         pdf_doc.save(args.output)
