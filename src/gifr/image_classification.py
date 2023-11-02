import json
import logging
import sys
import traceback

import gradio as gr

from gifr.common import init_logging, set_logging_level, create_parser, init_state, State, make_prediction

PROG: str = "gifr-imgcls"

_logger = logging.getLogger(PROG)

state: State = None


def predict(img_file: str) -> str:
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

    data = make_prediction(state, content)
    if data is None:
        result = {"no result": 0.0}
    else:
        result = json.loads(data.decode())
    state.logger.info("Prediction: %s" % result)
    return result


def create_interface() -> gr.Interface:
    """
    Generates the interface.
    """
    return gr.Interface(
        title="Image classification",
        description="Sends the selected image to the model and displays the generated prediction results.",
        fn=predict,
        inputs=[
            gr.Image(type="filepath", label="Input"),
        ],
        outputs=[
            gr.Label(label="Prediction"),
        ],
        allow_flagging="never")


def finish_init_state(state: State):
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
    parser = create_parser("Image classification interface. Allows the user to select an image "
                           + "and display the probabilities per label that the model generated.",
                           PROG, model_channel_in="images", model_channel_out="predictions", timeout=1.0)
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
