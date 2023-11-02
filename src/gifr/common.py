import argparse
import logging
import os
import redis

from dataclasses import dataclass, field
from datetime import datetime
from time import sleep


LOGGING_DEBUG = "DEBUG"
LOGGING_INFO = "INFO"
LOGGING_WARN = "WARN"
LOGGING_ERROR = "ERROR"
LOGGING_CRITICAL = "CRITICAL"
LOGGING_LEVELS = [
    LOGGING_DEBUG,
    LOGGING_INFO,
    LOGGING_WARN,
    LOGGING_ERROR,
    LOGGING_CRITICAL,
]

ENV_GIFR_LOGLEVEL = "GIFR_LOGLEVEL"
""" environment variable for the global default logging level. """


@dataclass
class State:
    connection: redis.Redis = None
    pubsub = None
    channel_out: str = "redis_in"
    channel_in: str = "redis_out"
    timeout: float = 5.0
    data = None
    logger: logging.Logger = None
    params: dict = field(default_factory=dict)


def str_to_logging_level(level: str) -> int:
    """
    Turns a logging level string into the corresponding integer constant.

    :param level: the level to convert
    :type level: str
    :return: the int level
    :rtype: int
    """
    if level not in LOGGING_LEVELS:
        raise Exception("Invalid logging level (%s): %s" % ("|".join(LOGGING_LEVELS), level))
    if level == LOGGING_CRITICAL:
        return logging.CRITICAL
    elif level == LOGGING_ERROR:
        return logging.ERROR
    elif level == LOGGING_WARN:
        return logging.WARN
    elif level == LOGGING_INFO:
        return logging.INFO
    elif level == LOGGING_DEBUG:
        return logging.DEBUG
    else:
        raise Exception("Unhandled logging level: %s" % level)


def init_logging():
    """
    Initializes the logging.
    """
    level = logging.WARNING
    if os.getenv(ENV_GIFR_LOGLEVEL) is not None:
        level = str_to_logging_level(os.getenv(ENV_GIFR_LOGLEVEL))
    logging.basicConfig(level=level)


def set_logging_level(logger: logging.Logger, level: str):
    """
    Sets the logging level of the logger.

    :param logger: the logger to update
    :type logger: logging.Logger
    :param level: the level string, see LOGGING_LEVELS
    :type level: str
    """
    logger.setLevel(str_to_logging_level(level))


def create_parser(description: str, prog: str, host: str = "localhost", port: int = 6379, db: int = 0,
                  model_channel_in: str = "model_channel_in", model_channel_out: str = "model_channel_out",
                  timeout: float = 5.0) -> argparse.ArgumentParser:
    """
    Creates a base parser with options for redis.

    :param description: the description of the program
    :type description: str
    :param prog: the command-line program
    :type prog: str
    :param host: the default redis host
    :type host: str
    :param port: the default redis port
    :type port: int
    :param db: the redis database to use
    :type db: int
    :param model_channel_in: the redis channel to send the data to for making predictions
    :type model_channel_in: str
    :param model_channel_out: the redis channel to receive the predictions on
    :type model_channel_out: str
    :param timeout: the number of seconds to wait for a prediction
    :type timeout: float
    """
    parser = argparse.ArgumentParser(
        description=description, prog=prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--redis_host", metavar="HOST", help="The host with the redis server.", default=host, type=str, required=False)
    parser.add_argument("--redis_port", metavar="PORT", help="The port of the redis server.", default=port, type=int, required=False)
    parser.add_argument("--redis_db", metavar="DB", help="The redis database to use.", default=db, type=int, required=False)
    parser.add_argument("--model_channel_in", metavar="CHANNEL", help="The channel to send the data to for making predictions.", default=model_channel_in, type=str, required=False)
    parser.add_argument("--model_channel_out", metavar="CHANNEL", help="The channel to receive the predictions on.", default=model_channel_out, type=str, required=False)
    parser.add_argument("--timeout", metavar="SECONDS", help="The number of seconds to wait for a prediction.", default=timeout, type=float, required=False)
    parser.add_argument("--launch_browser", action="store_true", help="Whether to automatically launch the interface in a new tab of the default browser.")
    parser.add_argument("--share_interface", action="store_true", help="Whether to publicly share the interface at https://XYZ.gradio.live/.")
    parser.add_argument("--logging_level", choices=LOGGING_LEVELS, default=LOGGING_WARN, help="The logging level to use")
    return parser


def init_state(ns: argparse.Namespace) -> State:
    """
    Initializes the redis state container with the supplied parameters.

    :param ns: the parsed options
    :type ns: argparse.Namespace
    :return: the state container
    :rtype: State
    """
    result = State(
        connection=redis.Redis(host=ns.redis_host, port=ns.redis_port, db=ns.redis_db),
        channel_in=ns.model_channel_in,
        channel_out=ns.model_channel_out,
        timeout=ns.timeout,
    )
    return result


def make_prediction(state: State, data):
    """
    Makes a prediction by broadcasting the data and waiting for a result coming through.

    :param state: the state to use to broadcasting/listening
    :type state: State
    :param data: the data to send
    :return: the received data, None if failed or timeout
    """
    def anon_handler(message):
        data = message['data']
        state.data = data
        state.pubsub_thread.stop()
        state.pubsub.close()
        state.pubsub = None

    state.pubsub = state.connection.pubsub()
    state.pubsub.psubscribe(**{state.channel_out: anon_handler})
    state.pubsub_thread = state.pubsub.run_in_thread(sleep_time=0.01)
    state.connection.publish(state.channel_in, data)

    # wait for data to show up
    start = datetime.now()
    end = start
    no_data = False
    while state.pubsub is not None:
        sleep(0.01)
        end = datetime.now()
        if state.timeout > 0:
            if (end - start).total_seconds() >= state.timeout:
                msg = "Timeout reached!"
                if state.logger is None:
                    print(msg)
                else:
                    state.logger.error(msg)
                no_data = True
                break

    if no_data:
        return None
    else:
        msg = "Time for prediction: %0.3f seconds" % (end - start).total_seconds()
        if state.logger is None:
            print(msg)
        else:
            state.logger.info(msg)
        return state.data
