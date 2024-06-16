from . import typing as _t


import gc as _gc
import ctypes as _ctypes
import inspect as _inspect
import contextlib as _contextlib
from functools import partial as _partial, wraps as _wraps
from itertools import zip_longest as _zip_longest, product as _product


class Slices:
    '''A generator of slices.

    Slice(start=1, stop=6, step=2) will generate `slice(1, 3, None)`,
    `slice(3, 5, None)`, and `slice(5, 6, None)`.
    '''

    def __init__(self, start: int, stop: int, step: int = 1):
        self.start = start
        self.stop = stop
        self.step = step
        self._range = range(start, stop, step)

    def __len__(self):
        return len(self._range)

    def __iter__(self):
        self._iter = iter(self._range)
        return self

    def __next__(self):
        return slice(i := next(self._iter), min(i + self.step, self.stop))

    def copy(self):
        return Slices(self.start, self.stop, self.step)


class TopLevelObjectRegister:
    '''With this declared Top-level-wise, enlisted `objs` can be
    picklable.'''
    @classmethod
    def add_objs(self, *objs):
        for obj in objs:
            setattr(self, obj.__name__, obj)
            obj.__qualname__ = self.__qualname__ + '.' + obj.__name__


def simple_tabulate(s: _t.List[_t.List[str]]) -> str:
    '''Print a 2D List with aligned columns' widths.'''
    _zip = _partial(_zip_longest, fillvalue='')
    widths = [max(map(len, col)) for col in _zip(*s)]
    formatter = lambda cols: (f'{c:{w}s}' for c, w in _zip(cols, widths))
    return '\n'.join(' | '.join(formatter(cols)) for cols in s)


def get_func_default_kwargs(func: _t.Callable) -> _t.Dict[str, _t.Any]:
    '''Get arguments with default values.'''
    sig = _inspect.signature(func).parameters.items()
    return {k: d for k, v in sig if (d := v.default) is not _inspect._empty}


def enum_docstring(enum_class: _t.Enum) -> _t.Enum:
    '''Append key-value pairs of a Enum to its docstring.'''
    enum_class.__doc__ = '\n'.join((
        enum_class.__doc__,
        f'',
        f'Key-value pairs:',
        f'----------------',
        *(f'{k}: {v.value}' for k, v in enum_class.__members__.items())
    ))
    return enum_class


def chunk_iterator(iterator: _t.Iterable, chunk_size: int) -> _t.Generator:
    '''Collect data into chunks.'''
    dummy = [] # for a tracable unique ID.
    iterators = [iter(iterator)] * chunk_size
    for items in _zip_longest(*iterators, fillvalue=dummy):
        yield list(filter(lambda x: x is not dummy, items))


@_contextlib.contextmanager
def try_release_free_memory_from_heap():
    try:
        yield
    finally:
        _gc.collect()
        try:
            _ctypes.CDLL("libc.so.6").malloc_trim(0)
        except:
            pass


def simple_grid_search(params: _t.Dict[str, _t.List]) -> _t.Generator:
    '''Simple Grid Search that respects dictionary order.'''
    func = lambda vals: dict(zip(params.keys(), vals))
    yield from enumerate(map(func, _product(*params.values())))
