import logging
import sys
import traceback
from typing import Tuple

import gradio as gr
import gifr.asr
import gifr.text_generation

from gifr.asr import predict as predict_asr
from gifr.common import init_logging, set_logging_level, create_parser, init_state, State
from gifr.text_generation import predict as predict_text_generation

PROG: str = "gifr-asr-textgen"

_logger = logging.getLogger(PROG)

state: State = None


def predict(audio) -> Tuple[str, str]:
    """
    Transcribes the audio and generates text from it.

    :param audio: the audio to transcribe
    :return: the transcribed audio and the generated text
    :rtype: tuple
    """
    global state
    gifr.asr.state = state
    gifr.text_generation.state = state
    transcript = predict_asr(audio, channel_in=state.params["audio_channel_in"], channel_out=state.params["audio_channel_out"])
    text = predict_text_generation(transcript, channel_in=state.params["text_channel_in"], channel_out=state.params["text_channel_out"])
    return transcript, text


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
            gr.Audio(label="Input", waveform_options={"show_recording_waveform": True}),
        ],
        outputs=[
            gr.Textbox(label="Transcription"),
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
    parser = create_parser("Combined Automatic Speech Recognition (ASR) and text generation interface. Allows the user to record/upload audio, "
                           + "which gets transcribed and the transcription fed into the text generation model. The generated text is then displayed.",
                           PROG, timeout=1.0, ui_title="ASR+Text generation",
                           ui_desc="First transcribes the recorded/uploaded audio and then sends the transcript to the model to complete and displays the result.")
    parser.add_argument("--audio_channel_in", metavar="CHANNEL", help="The channel to send the audio to for transcribing.", default="audio", type=str, required=False)
    parser.add_argument("--audio_channel_out", metavar="CHANNEL", help="The channel to receive the transcriptions on.", default="transcription", type=str, required=False)
    parser.add_argument("--text_channel_in", metavar="CHANNEL", help="The channel to send the text to for making predictions.", default="text", type=str, required=False)
    parser.add_argument("--text_channel_out", metavar="CHANNEL", help="The channel to receive the text predictions on.", default="prediction", type=str, required=False)
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
