import logging
from pathlib import Path
from typing import Optional

_logger = logging.getLogger("MiniGPT")
_DEFAULT_LOGFILE = Path.cwd() / "minigpt.log"


def get_logger() -> logging.Logger:
    global _logger
    return _logger


def setup_logging(
    level: str = "INFO",
    logfile: Optional[Path] = _DEFAULT_LOGFILE,
) -> None:
    logger = get_logger()
    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s|%(name)s|%(levelname)s] %(message)s")
    # Clear any existing handlers
    logger.handlers = []
    # Add a handler for stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # Add a handler for the logfile
    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
