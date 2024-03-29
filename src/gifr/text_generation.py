import json
import logging
import sys
import traceback

import gradio as gr

from gifr.common import init_logging, set_logging_level, create_parser, init_state, State, make_prediction

PROG: str = "gifr-textgen"

_logger = logging.getLogger(PROG)

state: State = None


def predict(text: str) -> str:
    """
    Sends the text to the model and returns the completed text.

    :param text: the text to send
    :type text: str
    :return: the prediction result
    :rtype: str
    """
    global state
    state.logger.info("Completing: %s" % text)

    # build query
    d = {state.params["send_text"]: text}
    if state.params["history_on"]:
        if state.params["send_history"] is not None:
            d[state.params["send_history"]] = state.history
        if state.params["send_turns"] is not None:
            d[state.params["send_turns"]] = state.turns

    # perform query
    result = make_prediction(state, json.dumps(d))

    # parse response
    if result is None:
        result = "no result"
    else:
        if state.params["json_response"]:
            try:
                d = json.loads(result.decode())
                result = d[state.params["receive_prediction"]]
                if state.params["history_on"]:
                    if state.params["receive_history"] in d:
                        state.history = d[state.params["receive_history"]]
                        _logger.info("History: %s" % str(state.history))
                    if state.params["receive_turns"] in d:
                        state.turns = d[state.params["receive_turns"]]
                        _logger.info("Turns: %s" % str(state.turns))
            except:
                result = "Failed to parse: %s" % str(result)
        else:
            try:
                result = result.decode()
            except:
                result = "Failed to parse: %s" % str(result)

    state.logger.info("Prediction: %s" % result)
    if state.params["clean_response"]:
        result = result.strip()
        if result.endswith("</s>"):
            result = result[0:-4]
            result = result.strip()

    return result


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
            gr.Textbox(label="Prediction"),
        ],
        allow_flagging="never")


def post_init_state(state: State):
    """
    Finalizes the initialization of the state.

    :param state: the state to update
    :type state: State
    """
    state.logger = _logger
    state.history = ""
    state.turns = 0


def main(args=None):
    """
    The main method for parsing command-line arguments.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    global state
    init_logging()
    parser = create_parser("Text generation interface. Allows the user to enter text "
                           + "and display the text generated by the model.",
                           PROG, model_channel_in="text", model_channel_out="prediction",
                           timeout=1.0, ui_title="Text generation",
                           ui_desc="Sends the entered text to the model to complete and displays the result.")
    parser.add_argument("--send_text", metavar="FIELD", help="The field name in the JSON prompt used for sending the text, ignored if not provided.", default="prompt", type=str, required=False)
    parser.add_argument("--json_response", action="store_true", help="Whether the reponse is a JSON object.")
    parser.add_argument("--receive_prediction", metavar="FIELD", help="The field name in the JSON response used for receiving the predicted text, ignored if not provided.", default="text", type=str, required=False)
    parser.add_argument("--history_on", action="store_true", help="Whether to keep track of the interactions.")
    parser.add_argument("--send_history", metavar="FIELD", help="The field name in the JSON query to use for sending the input history, ignored if not provided.", default=None, type=str, required=False)
    parser.add_argument("--send_turns", metavar="FIELD", help="The field name in the JSON query to use for sending the number of turns in the interaction, ignored if not provided.", default=None, type=str, required=False)
    parser.add_argument("--receive_history", metavar="FIELD", help="The field name in the JSON response used for receiving the input history, ignored if not provided.", default=None, type=str, required=False)
    parser.add_argument("--receive_turns", metavar="FIELD", help="The field name in the JSON response used for receiving the number of turns in the interaction, ignored if not provided.", default=None, type=str, required=False)
    parser.add_argument("--clean_response", action="store_true", help="Whether to clean up the response.")
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
