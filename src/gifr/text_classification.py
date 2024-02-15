import json
import logging
import sys
import traceback

from typing import Tuple

import gradio as gr

from gifr.common import init_logging, set_logging_level, create_parser, init_state, State, make_prediction

PROG: str = "gifr-textclass"

_logger = logging.getLogger(PROG)

state: State = None


def predict(text: str) -> Tuple[str, float]:
    """
    Sends the text to the model and returns the label and score.

    :param text: the text to send
    :type text: str
    :return: the prediction result
    :rtype: str
    """
    global state
    state.logger.info("Classifying: %s" % text)
    d = {"text": text}
    prediction = make_prediction(state, json.dumps(d))
    if prediction is None:
        label = ""
        score = -1
    else:
        try:
            d = json.loads(prediction.decode())
            label = d["label"]
            score = d["score"]
        except:
            label = prediction.decode()
            score = -1

    state.logger.info("Prediction: %s -> label=%s, score=%f" % (prediction, label, score))
    return label, score


def create_interface(state: State) -> gr.Interface:
    """
    Generates the interface.

    :param state: the state to use
    :type state: State
    """
    return gr.Interface(
        title=state.title,
        description=state.description,
        fn=predict,
        inputs=[
            gr.Textbox(label="Input"),
        ],
        outputs=[
            gr.Textbox(label="Label"),
            gr.Textbox(label="Score"),
        ],
        allow_flagging="never")


def post_init_state(state: State):
    """
    Finalizes the initialization of the state.

    :param state: the state to update
    :type state: State
    """
    state.logger = _logger


def main(args=None):
    """
    The main method for parsing command-line arguments.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    global state
    init_logging()
    parser = create_parser("Text classification interface. Allows the user to enter text "
                           + "and display the predicted label and score returned by the model.",
                           PROG, model_channel_in="text", model_channel_out="prediction",
                           timeout=1.0, ui_title="Text classification",
                           ui_desc="Sends the entered text to the model to complete and displays the predicted label and score.")
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
