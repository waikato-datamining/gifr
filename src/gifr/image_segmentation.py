import io
import logging
import numpy as np
import sys
import traceback

import gradio as gr

from PIL import Image
from gifr.common import init_logging, set_logging_level, create_parser, init_state, State, make_prediction
from gifr.colors import default_colors

PROG: str = "gifr-imgseg"

_logger = logging.getLogger(PROG)

state: State = None

PREDICTION_TYPE_AUTO = "auto"
PREDICTION_TYPE_BLUECHANNEL = "blue-channel"
PREDICTION_TYPE_GRAYSCALE = "grayscale"
PREDICTION_TYPE_INDEXEDPNG = "indexed-png"
PREDICTION_TYPES = [
    PREDICTION_TYPE_AUTO,
    PREDICTION_TYPE_BLUECHANNEL,
    PREDICTION_TYPE_GRAYSCALE,
    PREDICTION_TYPE_INDEXEDPNG,
]


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
        state.logger.error("No data received. Timeout or error?")
        return None

    # mask: num classes and turn into palette image
    mask = Image.open(io.BytesIO(data))
    pred_type = state.params["prediction_type"]
    if pred_type == PREDICTION_TYPE_AUTO:
        if mask.mode == "RGB":
            pred_type = PREDICTION_TYPE_BLUECHANNEL
        elif mask.mode == "L":
            pred_type = PREDICTION_TYPE_GRAYSCALE
        elif mask.mode == "P":
            pred_type = PREDICTION_TYPE_INDEXEDPNG
        else:
            raise Exception("Unhandled image mode: %s" % mask.mode)
        state.logger.info("Prediction type determined: %s" % pred_type)
    if pred_type == PREDICTION_TYPE_BLUECHANNEL:
        arr = np.asarray(mask)
        arr = arr[:, :, 2]  # blue channel
        num_classes = len(np.unique(arr)) - 1
        mask = Image.fromarray(arr, "P")
    elif pred_type == PREDICTION_TYPE_GRAYSCALE:
        arr = np.asarray(mask)
        num_classes = len(np.unique(arr)) - 1
        mask = Image.fromarray(arr, "P")
    elif pred_type == PREDICTION_TYPE_INDEXEDPNG:
        arr = np.asarray(mask)
        num_classes = len(np.unique(arr)) - 1
    else:
        raise Exception("Unhandled prediction type: %s" % state.params["prediction_type"])
    state.logger.info("# classes: %d" % num_classes)

    # new palette for mask
    palette = [0, 0, 0]  # background
    for i in range(num_classes):
        color = next_default_color(state)
        palette.extend(color)
    mask.putpalette(palette)
    state.logger.debug("palette: %s" % str(palette))

    # overlay mask
    mask = mask.convert("RGBA")
    if state.params["only_mask"]:
        combined = mask
    else:
        arr = np.asarray(mask)
        a = state.params["alpha"] * np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.uint8)
        arr = np.concatenate([arr[:, :, 0:3], a], axis=2, dtype=np.uint8)
        mask = Image.fromarray(arr, "RGBA")
        combined = img.convert("RGBA")
        combined.paste(mask, (0, 0), mask=mask)

    return np.asarray(combined)


def create_interface(state: State) -> gr.Interface:
    """
    Generates the interface.
    """
    return gr.Interface(
        title=state.title,
        description=state.description,
        fn=predict,
        inputs=[
            gr.Image(type="filepath", label="Input"),
        ],
        outputs=[
            gr.Image(label="Prediction"),
        ],
        allow_flagging="never")


def post_init_state(state: State):
    """
    Finalizes the initialization of the state.

    :param state: the state to update
    :type state: State
    """
    state.logger = _logger
    state.params["colors"] = dict()
    state.params["default_colors"] = default_colors()
    state.params["default_colors_index"] = 0


def main(args=None):
    """
    The main method for parsing command-line arguments.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    global state
    init_logging()
    parser = create_parser("Image segmentation interface. Allows the user to select an image "
                           + "and display the generated pixel mask overlayed.",
                           PROG, model_channel_in="images", model_channel_out="predictions",
                           timeout=2.0, ui_title="Image segmentation",
                           ui_desc="Sends the selected image to the model and shows the result (overlay or pixel mask).")
    parser.add_argument("--prediction_type", choices=PREDICTION_TYPES, default=PREDICTION_TYPE_AUTO, help="The type of image that the model returns")
    parser.add_argument("--alpha", metavar="NUM", help="The alpha value to use for the overlay (0: transparent, 255: opaque).", default=128, type=int, required=False)
    parser.add_argument("--only_mask", action="store_true", help="Whether to show only the predicted mask rather than overlaying it.")
    parsed = parser.parse_args(args=args)
    set_logging_level(_logger, parsed.logging_level)
    state = init_state(parsed)
    post_init_state(state)
    ui = create_interface(state)
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
