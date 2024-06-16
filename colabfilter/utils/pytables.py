'''All PyTables-related functions. Other places should not need to
import tables.
'''


from . import typing as _t


import os as _os
import inspect as _inspect
import dataclasses as _dataclasses
from enum import Enum as _Enum
from math import nan as _nan
from functools import partial as _partial, wraps as _wraps


import numpy as _np
import tables as _tb


from . import misc as _misc


@_dataclasses.dataclass
class _TableDType:
    '''For pretty-printing dtype.'''
    dtype: _t.Dict[str, type]

    def __str__(self):
        it = zip(*self.dtype.items())
        return ' '.join(map(lambda x, y: f'{x}({y})', *it))


@_misc.enum_docstring
class StoreMode(_Enum):
    '''Storage mode enumeration.'''
    R = 'Read-Only Mode'
    W = 'Create Or Overwrite Mode'
    A = 'Append-Only Mode'


@_misc.enum_docstring
class StoreType(_Enum):
    '''Pytables storage types enumeration.'''
    TABLE = 'table'
    ARRAY = 'array'
    CARRAY = 'carray'
    EARRAY = 'earray'
    VLARRAY = 'vlarray'


class Tables:
    '''Pytables' object types.'''
    Col = _tb.Col
    Row = _tb.tableextension.Row
    Table = _tb.Table
    Array = _tb.Array
    CArray = _tb.CArray
    EArray = _tb.EArray
    VLArray = _tb.VLArray


class PathsToPyTablesConverter:
    '''Decorate a function so that when the function is called, it
    converts the function's path arguments to PyTables storage objects
    (StoreMode.R or StoreMode.A) or functions that can create storages
    (StoreMode.W).

    Examples:
    ```
        @PathsToPyTablesConverter(
            source=(StoreMode.R, StoreType.TABLE),
            target=(StoreMode.W, StoreType.CARRAY),
            dummy=(StoreMode.W, StoreType.CARRAY),
        )
        # Same arguments are required in the function.
        def func(
            *,
            source: tables.Table,
            target: tables.CArray,
            dummy: tables.CArray | None = None,
            other_thing: int = 1,
        ) -> None:
            pass

        func(source='/path/to/source', target='/path/to/target')
    ```

    This decorator will copy, to the decorated function, the DocString
    and the Signature of `func` but replace the relevant annotations
    (`source`, `target` and `dummy`) to `str` as they are expected to be
    paths that will be converted into PyTables objects.

    Args:
        **kwargs: Tuple[StoreMode, StoreType]
    '''

    def __init__(
            self,
            logger: _t.Logger | None = None,
            **kwargs: _t.Tuple[StoreMode, StoreType],
            ) -> None:
        self._logger = logger
        self._kwargs = kwargs

    def __call__(self, func: _t.Callable) -> _t.Callable:
        default_kwargs = _misc.get_func_default_kwargs(func)

        @_wraps(func)
        def _wrapper(pytables_file_path, **kwargs: _t.Any):
            # To combine & replace default arguments with the provided ones
            kwargs = dict(default_kwargs, **kwargs)
            with (
                    _tb.File(pytables_file_path, 'a') as handle,
                    _misc.try_release_free_memory_from_heap(),
                ):
                # Track all stores created
                self._store_paths = []
                # Update, in place, paths in kwargs to PyTables objects
                self._update_path_to_pytables(handle, kwargs)
                # Execute the function with PyTables objects
                res = func(**kwargs)
                # Print the created stores
                print_stores(
                    pytables_file_path, self._logger, self._store_paths)
                # Flush makes sure data is written &
                # prevents Exception (`mode` not an attribute) at __del__
                handle.flush()
                return res

        new_signature = _modify_annotation(func, self._kwargs.keys())
        _wrapper.__signature__ = new_signature
        return _wrapper

    def _update_path_to_pytables(
            self, handle: _tb.File, kwargs: _t.Dict[str, _t.Any]) -> None:
        '''Convert paths specified in self._kwargs to PyTables'''
        for key, (mode, stype) in self._kwargs.items():
            if (path := kwargs[key]) is None:
                continue

            match mode, (path_exists := path in handle):
                case StoreMode.R | StoreMode.A, True:
                    store = handle.root[path]
                    s1 = store.attrs['CLASS'].lower()
                    s2 = stype.value
                    if s1 != s2:
                        msg = f'"{path}" should be a {s2}. Got {s1}.'
                        getattr(self._logger, 'error', print)(msg)
                        raise TypeError(msg)
                    kwargs[key] = store

                case StoreMode.W, _:
                    if path_exists:
                        handle.remove_node(path, recursive=True)
                    kwargs[key] = self._create_function(handle, path, stype)

                case StoreMode.R | StoreMode.A, False:
                    msg = f'"{path}" not found in "{handle.filename}".'
                    getattr(self._logger, 'error', print)(msg)
                    raise IndexError(msg)

                case _:
                    raise NotImplementedError # shouldn't trigger.

    def _create_function(
            self, handle: _tb.File, path: str, stype: StoreType
        ) -> _t.Callable:
        # This wraps a tables.File.create_xxxxx() function and is returned
        # for use later. xxxxx is one of `StoreType` values, such as,
        # {'table', 'carray', 'vlarray', ...}
        name = f'create_{stype.value}'
        func = getattr(handle, name)

        @_wraps(func)
        def _wrapper(*args, dtype=None, dflt=None, **kwargs):
            if 'atom' not in kwargs and dtype:
                dtype = _np.dtype(dtype)
                kwargs['atom'] = (
                    _tb.Atom.from_type(
                        dtype.name, shape=kwargs.pop('shape', ()), dflt=dflt)
                    if stype.value == 'vlarray' else
                    _tb.Atom.from_dtype(dtype, dflt=dflt)
                )
            s = func(
                *_os.path.split(path), *args, createparents=True, **kwargs)
            self._store_paths.append(s._v_pathname)
            return s

        _wrapper.__doc__ = (
            f'This is a wrapped version of tables.{name}. If `atom` is not '
            f'among the input arguments but `dtype` (e.g. np.float32) and '
            f'`dflt` (for default values, optional) are provided, then it '
            f'attempts to build an atom. Below is the Doc for tables.{name}.'
            f'\n\n{_wrapper.__doc__}'
        )
        return _wrapper


def print_stores(
        pytables_file_path: str,
        logger: _t.Logger | None = None,
        paths: _t.List[str] | None = None,
    ) -> None:
    '''Print details of stores listed in `paths` or all stores if
    `paths is None`.
    '''
    if paths is None:
        paths = list(_list_all_stores_paths(pytables_file_path))

    if len(paths) == 0:
        return

    def _format_attr(attr):
        _spec = {'Disk(MB)': ' 8.1f', 'Memory(MB)': ' 8.1f'}
        for k, v in attr.items():
            yield f'{k} {format(v, _spec.get(k, ""))}'

    iterator = zip(
        paths,
        map(_partial(query_store_attributes, pytables_file_path), paths)
    )
    table_str = sorted((path, *_format_attr(attr)) for path, attr in iterator)
    msg = f'Created stores:\n{_misc.simple_tabulate(table_str)}\n'
    getattr(logger, 'info', print)(msg)


def query_store_attributes(
        pytables_file_path: str,
        path: str,
    ) -> _t.Dict:
    '''Query disk size (MB), memory size (if fully loaded, MB), shape,
    and data type of a storage at `path`.
    '''
    with _tb.File(pytables_file_path, 'r') as f:
        store = f.root[path]
        store_type = store.attrs['CLASS'].lower()
        is_table = store_type == StoreType.TABLE.value
        is_vlarray = store_type == StoreType.VLARRAY.value
        return {
            'Disk(MB)': _nan if is_vlarray else store.size_on_disk/1024/1024,
            'Memory(MB)': store.size_in_memory/1024/1024,
            'Shape': (
                (store.nrows, len(store.coldtypes)) if is_table else
                store.shape
            ),
            'Type': _TableDType(store.coltypes) if is_table else store.dtype
        }


def get_tables_object(attr: str) -> _t.Any:
    return getattr(_tb, attr)


def _list_all_stores_paths(
        pytables_file_path: str,
    ) -> _t.Generator[str, None, None]:
    with _tb.File(pytables_file_path, 'r') as f:
        for node in f.walk_nodes('/'):
            if node._v_attrs.CLASS in StoreType.__members__:
                yield node._v_pathname


def _modify_annotation(
        func: _t.Callable, keys: _t.KeysView[str]) -> _t.Signature:
    '''Return a copy of signature of `func` with the copy's annotations
    for arguments in `keys` set to `str`.
    '''
    sig = _inspect.signature(func)
    parameters = [
        _inspect.Parameter(
            'pytables_file_path',
            _inspect.Parameter.POSITIONAL_ONLY,
            annotation=str)
    ]
    for k, v in sig.parameters.items():
        if k in keys:
            _args = _t.get_args(v.annotation)
            _orig = _t.get_origin(v.annotation)
            if (
                (_orig is _t.types.UnionType and type(None) in _args) or
                (_orig is _t.Union and type(None) in _args)
            ):
                new_v = v.replace(annotation=str | None)
            else:
                new_v = v.replace(annotation=str)
        else:
            new_v = v
        parameters.append(new_v)
    return sig.replace(parameters=parameters)
