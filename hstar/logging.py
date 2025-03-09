import logging
import sys
import traceback
from logging import Formatter, Logger, StreamHandler

FORMAT = "%(levelname)s\t%(filename)s:%(lineno)d %(message)s"


def log_exception(logger: Logger, e: Exception) -> None:
    """Like logger.exception(e) but with full stack trace."""
    stack_trace = traceback.format_exception(type(e), e, e.__traceback__)

    # Join the stack trace strings into a single string
    stack_trace_str = "".join(stack_trace)

    # Log the exception information along with the entire stack trace
    logger.error(f"Caught exception:\n{stack_trace_str}")


class LogColors:
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BRIGHT_RED = "\033[91m"


class ColorFormatter(Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.CYAN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BRIGHT_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = self.LEVEL_COLORS.get(record.levelno)
        if color:
            levelname_colored = color + levelname + LogColors.RESET
        else:
            levelname_colored = levelname

        # Replace the original levelname with the colored one
        record.levelname = levelname_colored
        message = super().format(record)
        # Restore the original levelname to avoid affecting other loggers
        record.levelname = levelname
        return message


def setup_color_logging(format: str = FORMAT, level: int = logging.INFO) -> None:
    """Sets up colorful logging a la uvicorn."""
    handler = StreamHandler()
    formatter: Formatter
    if sys.stdout.isatty():
        formatter = ColorFormatter(format)
    else:
        formatter = Formatter(format)
    handler.setFormatter(formatter)
    for name in ["hstar", "__main__"]:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(level)
