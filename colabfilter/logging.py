from .utils import typing as _t


import sys as _sys
import logging as _logging


def _get_logger(
        level: str='INFO',
        stdout: bool=True,
        file_path: str | None=None,
    ) -> _t.Logger:
    logger = _logging.getLogger(__name__)
    logger.setLevel(level)
    formatter = _logging.Formatter(
        fmt='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt=None,
    )

    if not logger.hasHandlers():
        if stdout:
            handler = _logging.StreamHandler(_sys.stdout)
            handler.setLevel(level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if file_path:
            handler = _logging.FileHandler(file_path)
            handler.setLevel(level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger


logger: _t.Logger = _get_logger()
