from .utils import typing as _t


import os as _os
import shutil as _shutil
import contextlib as _contextlib
import dataclasses as _dataclasses


import numpy as _np


from . import utils as _utils
from .logging import logger as _logger


_EnvConfigSpec = _t.ParamSpec('_EnvConfig')


@_dataclasses.dataclass
class _EnvConfig:
    '''Env Configurations.

    Args:
        path_data_folder
            Path to a folder that will save generated data files.
        path_raw_dataset
            Path to the folder that contains the CSV files.
        expected_rows
            A dict of {csv_file: number_of_rows, ...}.
        chunk_size
            Determine how much data is read each time by data processing
            and analytics functions
        num_processes
            For running data processing concurrently
        pbar_show, pbar_leave
            Whether to show the progress bar and, if shown, whether to
            leave the bar on screen on complete
        default_int, default_float
            Default datatype to be used by data processing functions
        seed
            The default seed value whenever needed

    Attributes:
        path_pytables, path_tfrecords_train, path_tfrecords_test
            Paths to created data files and folders.
    '''
    path_data_folder: str | None = None
    path_raw_dataset: str | None = None
    expected_rows: _t.Dict[str, int] = _dataclasses.field(default_factory=dict)
    chunk_size: int = 1000
    num_processes: int = 8
    pbar_show: bool = True
    pbar_leave: bool = True
    default_int: _np.signedinteger = _np.int64 #TODO: add to utils.typing
    default_float: _np.floating = _np.float32 #TODO: add to utils.typing
    default_seed: int = 10

    @property
    def path_pytables(self):
        return _os.path.join(self.path_data_folder, 'pytables.h5')

    @property
    def path_tfrecords(self):
        return _os.path.join(self.path_data_folder, 'tfrecords')

    def preset(self, *, small_or_full: str, clear_files: bool) -> _t.Self:
        '''Set `small_or_full` to either "small" or "full".
        Set `clear_files` to True or False for whether to delete the
        PyTables and TFRecord files.
        '''
        match small_or_full:
            case 'small':
                self.path_data_folder = './data_small'
                self.path_raw_dataset = './data_small/ml-latest-small'
                self.expected_rows = {'movies': 10_000, 'ratings': 100_000}
            case 'full':
                self.path_data_folder = './data_full'
                self.path_raw_dataset = './data_full/ml-25m'
                self.expected_rows = {'movies': 100_000, 'ratings': 25_000_000}
            case _:
                msg = f'Only accepts {"small", "full"}. Got {small_of_full}'
                _logger.error(msg)
                raise ValueError(msg)

        if clear_files:
            if _os.path.exists(path := self.path_pytables):
                _os.remove(path)
                _logger.info(f'Removed {path}')
            if _os.path.exists(path := self.path_tfrecords):
                _shutil.rmtree(path)
                _logger.info(f'Removed {path}')

        if not _os.path.exists(path := self.path_tfrecords):
            _os.makedirs(path, exist_ok=True)
            _logger.info(f'Created {path}')

        return self

    def print(self) -> None:
        a = ['path_pytables', 'path_tfrecords']
        a = ((k, str(getattr(self, k))) for k in a)
        b = ((k, str(v)) for k, v in _dataclasses.asdict(self).items())
        s = _utils.simple_tabulate(sorted([*a, *b]))
        _logger.info(f'Configurations:\n{s}')

    @_contextlib.contextmanager
    def __call__(self, **kwargs: _EnvConfigSpec.kwargs):
        self._original_config = _dataclasses.replace(self)
        try:
            for key, val in kwargs.items():
                setattr(self, key, val)
            yield
        finally:
            for key, val in _dataclasses.asdict(self._original_config).items():
                setattr(self, key, val)
            del self._original_config


config = _EnvConfig().preset(small_or_full='small', clear_files=False)
