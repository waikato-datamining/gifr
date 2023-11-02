import gradio as gr
import json
import logging
import numpy as np
import os
import sys
import traceback

from datetime import datetime
from PIL import Image, ImageDraw
from typing import Tuple

from opex import ObjectPredictions, BBox
from gifr.common import init_logging, set_logging_level, create_parser, init_state, State, make_prediction
from gifr.colors import default_colors, text_color
from gifr.fonts import load_font, DEFAULT_FONT_FAMILY


PROG: str = "gifr-objdet"

_logger = logging.getLogger(PROG)

state: State = None


def next_default_color(state: State):
    """
    Returns the next default color.

    :param state: the state
    :type state: State
    :return: the color tuple
    :rtype: tuple
    """
    if state.params["default_colors_index"] >= len(state.params["default_colors"]):
        state.params["default_colors_index"] = 0
    result = state.params["default_colors"][state.params["default_colors_index"]]
    state.params["default_colors_index"] += 1
    return result


def get_color(state: State, label: str):
    """
    Returns the color for the label.

    :param state: the state
    :type state: State
    :param label: the label to get the color for
    :type label: str
    :return: the RGB color tuple
    :rtype: tuple
    """
    if label not in state.params["colors"]:
        state.params["colors"][label] = next_default_color(state)
    return state.params["colors"][label]


def get_outline_color(state, label):
    """
    Generates the color for the outline.

    :param state: the state
    :type state: State
    :param label: the label to get the color for
    :type label: str
    :return: the RGBA color tuple
    :rtype: tuple
    """
    r, g, b = get_color(state, label)
    return r, g, b, state.params["outline_alpha"]


def get_fill_color(state, label):
    """
    Generates the color for the filling.

    :param state: the state
    :type state: State
    :param label: the label to get the color for
    :type label: str
    :return: the RGBA color tuple
    :rtype: tuple
    """
    r, g, b = get_color(state, label)
    return r, g, b, state.params["fill_alpha"]


def expand_label(state: State, label: str, score: float) -> str:
    """
    Expands the label text.

    :param state: the state
    :type state: State
    :param label: the current label
    :type label: str
    :param score: the prediction score
    :type score: float
    :return: the expanded label text
    :rtype: str
    """
    result = state.params["text_format"].replace("{label}", label)
    result = result.replace("{score}", ("%." + str(state.params["num_decimals"]) + "f") % float(score))
    return result


def text_coords(state: State, draw: ImageDraw, text: str, rect: BBox) -> Tuple[int, int, int, int]:
    """
    Determines the text coordinates in the image.

    :param state: the state
    :type state: State
    :param draw: the ImageDraw instance
    :type draw: ImageDraw
    :param text: the text to output
    :type text: str
    :param rect: the rectangle to use as reference
    :return: the x, y, w, h tuple
    :rtype: tuple
    """
    try:
        w, h = draw.textsize(text, font=state.params["font"])
    except:
        # newer versions of Pillow deprecated ImageDraw.textsize
        # https://levelup.gitconnected.com/how-to-properly-calculate-text-size-in-pil-images-17a2cc6f51fd
        ascent, descent = state.params["font"].getmetrics()
        w = state.params["font"].getmask(text).getbbox()[2]
        h = state.params["font"].getmask(text).getbbox()[3] + descent

    horizontal = state.params["horizontal"]
    vertical = state.params["vertical"]

    # x
    if horizontal == "L":
        x = rect.left
    elif horizontal == "C":
        x = rect.left + (rect.right - rect.left - w) // 2
    elif horizontal == "R":
        x = rect.right - w
    else:
        raise Exception("Unhandled horizontal text position: %s" % horizontal)

    # y
    if vertical == "T":
        y = rect.top
    elif vertical == "C":
        y = rect.top + (rect.bottom - rect.top - h) // 2
    elif vertical == "B":
        y = rect.bottom - h
    else:
        raise Exception("Unhandled horizontal text position: %s" % horizontal)

    return x, y, w, h


def predict(img_file: str) -> np.ndarray:
    """
    Sends the image to the model and returns the result.

    :param img_file: the image to send
    :type img_file: str
    :return: the prediction result
    :rtype: str
    """
    global state
    state.logger.info("Loading: %s" % img_file)
    with open(img_file, "rb") as f:
        content = f.read()
    img = Image.open(img_file)

    data = make_prediction(state, content)
    if data is None:
        preds_str = json.dumps({
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S.%f"),
            "id": os.path.basename(img_file),
            "objects": []
        })
    else:
        preds_str = data.decode()
    state.logger.info("Prediction: %s" % preds_str)
    preds = ObjectPredictions.from_json_string(preds_str)
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i, obj in enumerate(preds.objects):
        if state.params["vary_colors"]:
            color_label = "object-%d" % i
        else:
            color_label = obj.label

        # assemble polygon
        points = []
        if not state.params["force_bbox"]:
            d = obj.polygon.to_array_pair_polygon()
            for x, y in zip(d["x"], d["y"]):
                points.append((x, y))
        else:
            rect = obj.bbox
            points.append((rect.left, rect.top))
            points.append((rect.right, rect.top))
            points.append((rect.right, rect.bottom))
            points.append((rect.left, rect.bottom))
        if state.params["fill"]:
            draw.polygon(tuple(points), outline=get_outline_color(state, color_label), fill=get_fill_color(state, color_label), width=state.params["outline_thickness"])
        else:
            draw.polygon(tuple(points), outline=get_outline_color(state, color_label), width=state.params["outline_thickness"])

        # output text
        if len(state.params["text_format"]) > 0:
            text = expand_label(state, obj.label, obj.score)
            rect = obj.bbox
            x, y, w, h = text_coords(state, draw, text, rect)
            draw.rectangle((x, y, x+w, y+h), fill=get_outline_color(state, color_label))
            draw.text((x, y), text, font=state.params["font"], fill=text_color(get_color(state, color_label)))

    img.paste(overlay, (0, 0), mask=overlay)

    return np.asarray(img)


def create_interface() -> gr.Interface:
    """
    Generates the interface.
    """
    return gr.Interface(
        title="Object detection",
        description="Sends the selected image to the model and overlays the predicted objects on it in the output.",
        fn=predict,
        inputs=[
            gr.Image(type="filepath", label="Input"),
        ],
        outputs=[
            gr.Image(label="Predictions"),
        ],
        allow_flagging="never")


def finish_init_state(state: State):
    """
    Finalizes the initialization of the state.

    :param state: the state to update
    :type state: State
    """
    state.logger = _logger
    state.params["colors"] = dict()
    state.params["default_colors"] = default_colors()
    state.params["default_colors_index"] = 0
    state.params["font"] = load_font(_logger, state.params["font_family"], state.params["font_size"])
    anchors = state.params["text_placement"].split(",")
    state.params["vertical"] = anchors[0]
    state.params["horizontal"] = anchors[1]


def main(args=None):
    """
    The main method for parsing command-line arguments.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    global state
    init_logging()
    parser = create_parser("Object detection interface. Allows the user to select an image "
                           + "and overlay the predictions that the model generated.",
                           PROG, model_channel_in="images", model_channel_out="predictions", timeout=1.0)
    parser.add_argument("--min_score", metavar="FLOAT", help="The minimum score a prediction must have (0-1).", default=0.0, type=float, required=False)
    parser.add_argument("--text_format", metavar="FORMAT", help="The format for the text, placeholders: {label}, {score}.", default="{label}", type=str, required=False)
    parser.add_argument("--text_placement", metavar="V,H", help="Comma-separated list of vertical (T=top, C=center, B=bottom) and horizontal (L=left, C=center, R=right) anchoring.", default="T,L", type=str, required=False)
    parser.add_argument("--font_family", metavar="NAME", help="The name of the font family.", default=DEFAULT_FONT_FAMILY, type=str, required=False)
    parser.add_argument("--font_size", metavar="SIZE", help="The size of the font.", default=14, type=int, required=False)
    parser.add_argument("--num_decimals", metavar="NUM", help="The number of decimals to use for the score.", default=3, type=int, required=False)
    parser.add_argument("--outline_thickness", metavar="NUM", help="The line thickness to use for the outline, <1 to turn off.", default=3, type=int, required=False)
    parser.add_argument("--outline_alpha", metavar="NUM", help="The alpha value to use for the outline (0: transparent, 255: opaque).", default=255, type=int, required=False)
    parser.add_argument("--fill", action="store_true", help="Whether to fill the bounding boxes/polygons", required=False)
    parser.add_argument("--fill_alpha", metavar="NUM", help="The alpha value to use for the filling (0: transparent, 255: opaque).", default=128, type=int, required=False)
    parser.add_argument("--vary_colors", action="store_true", help="Whether to vary the colors of the outline/filling regardless of label", required=False)
    parser.add_argument("--force_bbox", action="store_true", help="Whether to force a bounding box even if there is a polygon available", required=False)
    parsed = parser.parse_args(args=args)
    set_logging_level(_logger, parsed.logging_level)
    state = init_state(parsed)
    finish_init_state(state)
    ui = create_interface()
    ui.launch(show_api=False, share=parsed.share_interface, inbrowser=parsed.launch_browser)


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.

    :return: 0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except Exception:
        traceback.print_exc()
        print("options: %s" % str(sys.argv[1:]), file=sys.stderr)
        return 1


if __name__ == '__main__':
    main()
