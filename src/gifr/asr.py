import io
import logging
import numpy as np
import sys
import traceback

import gradio as gr
from scipy.io.wavfile import write

from gifr.common import init_logging, set_logging_level, create_parser, init_state, State, make_prediction

PROG: str = "gifr-asr"

_logger = logging.getLogger(PROG)

state: State = None


def predict(audio, channel_out: str = None, channel_in: str = None) -> str:
    """
    Sends the audio file to the model and returns the transcribed text.

    :param audio: the audio file to send
    :type audio: str
    :param channel_out: for overriding the state's out channel
    :type channel_out: str
    :param channel_in: for overriding the state's in channel
    :type channel_in: str
    :return: the transcription result
    :rtype: str
    """
    global state

    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    buf = io.BytesIO()
    write(buf, sr, y)

    state.logger.info("Transcribing...")

    # perform query
    result = make_prediction(state, buf.getvalue(), channel_in=channel_in, channel_out=channel_out)

    # parse response
    if result is None:
        result = "no result"
    else:
        try:
            result = result.decode()
        except:
            result = "Failed to parse: %s" % str(result)

    state.logger.info("Transcription: %s" % result)

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
            gr.Audio(label="Input", waveform_options={"show_recording_waveform": True}),
        ],
        outputs=[
            gr.Textbox(label="Transcription"),
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
    parser = create_parser("Automatic Speech Recognition (ASR) interface. Allows the user to record/upload audio "
                           + "and display the text transcribed by the model.",
                           PROG, model_channel_in="audio", model_channel_out="transcription",
                           timeout=2.0, ui_title="Automatic Speech Recognition (ASR)",
                           ui_desc="Sends the recorded/uploaded audio to the model to transcribe and displays the result.")
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
