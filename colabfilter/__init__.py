import sys as _sys
from functools import wraps as _wraps, partial as _partial


from tqdm.std import tqdm as _tqdm


from .config import config
from .logging import logger
from . import utils


class _ConfiguredPipeLine(utils.PipeLine):
    @_wraps(utils.PipeLine.__init__)
    def __init__(self, *args, **kwargs):
        _kwargs = dict(
            default_num_procs=config.num_processes,
            default_tqdm_kwargs=_configured_tqdm_kwargs(),
            logger=logger,
        )
        _kwargs.update(kwargs)
        super().__init__(*args, **_kwargs)


class _ConfiguredPathsToPyTablesConverter(utils.PathsToPyTablesConverter):
    @_wraps(utils.PathsToPyTablesConverter.__init__)
    def __init__(self, **kwargs):
        super().__init__(logger=logger, **kwargs)


@_wraps(utils.print_stores)
def configured_print_stores(*args, **kwargs):
    return utils.print_stores(config.path_pytables, logger, *args, **kwargs)


@_wraps(utils.query_store_attributes)
def configured_query_store_attributes(*args, **kwargs):
    return utils.query_store_attributes(config.path_pytables, *args, **kwargs)


def _configured_tqdm(*args, **kwargs):
    _kwargs = _configured_tqdm_kwargs()
    _kwargs.update(kwargs)
    return _tqdm(*args, **_kwargs)


def _configured_tqdm_kwargs():
    return dict(
        file=_sys.stdout, colour='#808080', position=0, ncols=79,
        leave=config.pbar_leave, disable=not config.pbar_show,
    )
