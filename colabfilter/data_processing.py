'''Data processing functions that create PyTables' objects.'''
from .utils import typing as _t


import dataclasses as _dataclasses
from functools import partial as _partial, reduce as _reduce


import numpy as _np
from pandas import read_csv as _read_csv


from . import (
    config as _config,
    utils as _utils,
    _ConfiguredPathsToPyTablesConverter,
    _ConfiguredPipeLine,
)
# from .utils import TopLevelObjectRegister


_PreProcessTask = _t.Callable[[_t.DataFrame], _t.DataFrame]


@_ConfiguredPathsToPyTablesConverter(
    target_store=(_utils.StoreMode.W, _utils.StoreType.CARRAY),
)
def save_array(*, target_store: _t.Callable, array: _t.Array) -> None:
    '''Save `array` to `target_store`.'''
    target_store(obj=array)


@_ConfiguredPathsToPyTablesConverter(
    target_table=(_utils.StoreMode.W, _utils.StoreType.TABLE),
    target_test_table=(_utils.StoreMode.W, _utils.StoreType.TABLE),
)
def load_csv_to_table(
        *,
        csv_path: str,
        create_table_kwargs: _t.Dict[str, _t.Any],
        indexing_columns: _t.Tuple[str] = (),
        preprocess_tasks: _t.Tuple[_PreProcessTask, ...] = (),
        target_table: _t.Callable,
        target_test_table: _t.Callable | None = None,
        test_split_size: _t.Annotated[float, _t.ValueRange(0, 1)] = .2,
        test_split_id: int = 0,
    ) -> _t.Dict:
    '''Load a CSV file to a PyTables Table.

    Chunks (`config.chunk_size) of rows from the CSV file at `csv_path`
    are read, and applied with the `preprocess_tasks`, before getting
    train-test split randomly (`config.default_seed`) if
    a path is provided in `target_test_table`.

    The ratio of test samples is controlled by `test_split_size`, which,
    for example, if set to `0.2`, one out of the 1 / 0.2 = 5 possible
    slices may be used as test. `test_split_id` chooses which one to
    use and thus enable the creation of k-fold train/test sets.

    The train (and test) samples are then stored in a Table created at
    the path in `target_table` (and `target_test_table`) following the
    specifications in `create_table_kwargs` as required by
    `tables.File.create_table()`. Among the requirements, `description`
    must be provided.

    `indexing_columns` will be indexed to enable faster query, and the
    unqiue indexed values will be returned in a dict.

    Returns:
        dict
            Contains the unique values of each indexed column.
    '''
    # Create _utils.Tables
    create_table_kwargs['description'] = {
        k: _utils.get_tables_object(v)()
        for k, v in create_table_kwargs['description'].items()
    }
    target_table = target_table(**create_table_kwargs)
    if (do_test_split := target_test_table is not None):
        target_test_table = target_test_table(**create_table_kwargs)

    # Helpers
    rng = _np.random.default_rng(_config.default_seed)

    def _target_test_table_append(x: _np.recarray) -> _np.recarray:
        is_test = (rng.random(len(x)) // test_split_size) == test_split_id
        target_test_table.append(x[is_test])
        return x[~is_test]

    def _index_column(
            table: _utils.Tables.Table, buffer: dict, col: str) -> None:
        table.flush()
        table.colinstances[col].create_index(optlevel=9, kind='full')
        buffer[col] = _np.unique(table.colindexes[col].read_sorted())

    # Buffer for unique index values
    buffer = {'train': {}, 'test': {}}

    # ConfiguredPipeLines
    (
        _ConfiguredPipeLine(
            source=_read_csv(csv_path, chunksize=_config.chunk_size)
        )
        .map(
            lambda df: _reduce(lambda y, f: f(y), preprocess_tasks, df),
            doc='Apply preprocessing tasks',
        )
        .map(
            lambda df: df[target_table.colnames],
            doc='Filter & reorder columns for proper typing',
        )
        .map(
            lambda df: df.to_records(index=False).astype(target_table.dtype),
            doc='Convert to records and change type'
        )
        .map(
            _target_test_table_append,
            chain_if=do_test_split,
            doc='Load data to `target_test_table`',
        )
        .map(
            target_table.append,
            doc='Load data to `target_table`',
        )
        .tqdm(
            desc=f'Loading {csv_path}', unit='chunks',
        )
        .consume()
        .execute()
    )
    (
        _ConfiguredPipeLine(source := indexing_columns)
        .peek(
            _partial(_index_column, target_table, buffer['train']),
            doc='Index columns for `target_table`',
        )
        .peek(
            _partial(_index_column, target_test_table, buffer['test']),
            chain_if=do_test_split,
            doc='Index columns for `target_test_table`',
        )
        .tqdm(
            total=len(source),
            desc='Creating index',
        )
        .consume()
        .execute()
    )
    return buffer


@_ConfiguredPathsToPyTablesConverter(
    source_table=(_utils.StoreMode.R, _utils.StoreType.TABLE),
    target_array=(_utils.StoreMode.W, _utils.StoreType.VLARRAY),
)
def interaction_from_table(
        *,
        source_table: _utils.Tables.Table,
        target_array: _t.Callable,
        sort_by: str,
        field_1: str,
        field_2: str | None,
        sort_by_max: int,
    ) -> _t.IntegralArray:
    '''Create sorted interaction arrays.

    To explain in context, given the following table at `source_table`:

    | userId | itemId | rating |
    | ------ | ------ | ------ |
    |      0 |      1 |    1.5 |
    |      2 |      0 |    2.0 |
    |      0 |      2 |    3.0 |

    When sort_by='userId', field_1='itemId', field_2='rating', and
    sort_by_max=4, the resulting array at `target_array` will be:
    [
        array([[1.0, 1.5], [2.0, 3.0]]), # userId = 0
        array([]),                       # userId = 1
        array([[0.0, 2.0]]),             # userId = 2
        array([]),                       # userId = 3
        array([]),                       # userId = 4
    ]

    If, in above, field_2=None, then, the result will be:
    [
        array([[1.0], [2.0]]), # userId = 0
        array([]),             # userId = 1
        array([[0.0]]),        # userId = 2
        array([]),             # userId = 3
        array([]),             # userId = 4
    ]

    Equivalent to:
    ```
    (
        source_table_df
        .groupby(sort_by)[[field_1, field_2]]
        .apply(np.array)
        .reindex(np.arange(sort_by_max + 1))
        .apply(lambda x: x if isinstance(x, np.ndarray) else empty_array)
        .to_list()
    )
    ```

    Args:
        source_table, target_array
            Paths. target_array is floating.
        sort_by
            Table's field name. The field's values must be integral.
        field_1
            Table's field name.
        field_2
            Table's field name. Optional.
        sort_by_max
            The maximum value in the sort_by field.

    Returns:
        integral, 1d array
            Distribution of lengths of arrays per sort_by value.
    '''
    has_field_2 = field_2 in source_table.colnames

    # Create VLArray
    target_array = target_array(
        dtype=_config.default_float,
        shape=(2, ) if has_field_2 else (1, ),
        expectedrows=sort_by_max + 1,
    )

    # Helpers
    def _fill_missing(
            datas: _t.Iterable[_t.Tuple[int, _t.Array]]
        ) -> _t.Iterable[_t.Array]:
        last_val = -1
        for sort_val, field_vals in datas:
            for _ in range(last_val + 1, last_val := sort_val):
                yield _np.array([])
            else:
                yield field_vals
        else:
            for _ in range(last_val + 1, sort_by_max + 1):
                yield _np.array([])

    # Buffer for counts
    sort_val_sizes = []

    # _ConfiguredPipeLine
    (
        _ConfiguredPipeLine(
            source=source_table.itersorted(sort_by)
        )
        .tqdm(
            total=source_table.nrows,
            desc='Table to Interaction',
        )
        .map_where(
            has_field_2,
            lambda r: (r[sort_by], (r[field_1], r[field_2])),
            lambda r: (r[sort_by], (r[field_1], )),
            doc='Extract values from a Table Row',
        )
        .groupby(
            lambda x: x[0],
            doc='Group consecutive elements with the same sort value',
        )
        .starmap(
            lambda sort_val, grp: (sort_val, _np.array(list(zip(*grp))[1])),
            doc='Convert group to arrays',
        )
        .call(
            _fill_missing,
            doc='Fill missing sort values with empty arrays'
        )
        .peek(
            lambda field_vals: sort_val_sizes.append(len(field_vals)),
            doc='Count array sizes',
        )
        .map(
            target_array.append,
            doc='Save results to file',
        )
        .consume()
        .execute()
    )
    return _np.bincount(sort_val_sizes)


def _read_interaction(
        store: _utils.Tables.VLArray | _t.List,
        has_col2: bool,
        idxs: _t.List[int] | slice | None,
    ) -> _t.Tuple[_t.IntegralArray, _t.IntegralArray, _t.FloatingArray]:
    if idxs is None:
        data = store[:]
    elif isinstance(idxs, slice):
        data = store[idxs]
    elif isinstance(idxs, _t.Iterable):
        data = [store[idx] for idx in idxs]
    else:
        raise NotImplementedError

    data_lens = _np.array(list(map(len, data)))
    if len(data) == 0:
        data_col1 = _np.array([])
        data_col2 = _np.array([])
    else:
        data = _np.concatenate(data)
        data_col1 = data[:, 0]
        data_col2 = data[:, 1] if has_col2 else _np.array([])

    return (
        data_lens.astype(_config.default_int),
        data_col1.astype(_config.default_int),
        data_col2.astype(_config.default_float),
    )


@_ConfiguredPathsToPyTablesConverter(
    source_item_supp=(_utils.StoreMode.R, _utils.StoreType.VLARRAY),
    source_supp_item=(_utils.StoreMode.R, _utils.StoreType.VLARRAY),
    target_corr_array=(_utils.StoreMode.W, _utils.StoreType.CARRAY),
    target_k_th_corr_array=(_utils.StoreMode.W, _utils.StoreType.CARRAY),
)
def corr_from_interaction(
        *,
        source_item_supp: _utils.Tables.CArray,
        source_supp_item: _utils.Tables.CArray,
        target_corr_array: _t.Callable,
        target_k_th_corr_array: _t.Callable | None,
        min_support: _t.Annotated[int, _t.ValueRange(1, None, True)] = 1,
        K_th: _t.Annotated[int, _t.ValueRange(0, None, True)] = 100,
        self_correlation: float | None = None,
    ) -> _t.IntegralArray:
    '''Compute item-item correlations via their common supports, and the
    K-th largest correlation value for each item.

    Item may be movie, and support may be user or genre.
    If user, and user-item ratings are provided, two items' correlation
    is calculated with their common users' ratings. If genre, since it
    is a binary relation with movie, 0 and 1 are automatically generated
    to take the place of ratings in the calculation of two items'
    correlation.

    It uses a pair of interactions (source_item_supp, source_supp_item)
    created with `interaction_from_table()`. The first one takes item
    as "sort_by" field and support as "field_1", whereas the second one
    does the reverse. The pair enables fast access to two items' common
    supports.

    Meaning of arguments' shape variables:

        Ni: number of items
        Ns: number of supports (e.g. users, genres)

    Roughly equivalent to:
    ```
    (
        source_table_df
        .set_index([supp, item])
        .reindex(
            pd.MultiIndex.from_product([
                np.arange(max(supp) + 1),
                np.arange(max(item) + 1)
            ])
        )
        [value]
        .unstack()
        .corr(min_periods=min_support)
    )
    ```

    Args:
        source_item_supp: floating, (Ni, )
        source_supp_item: floating, (Ns, )
        target_corr_array: floating, (Ni, Ni)
            Paths to a pair of interaction and the correlation arrays.
        target_k_th_corr_array: floating, (Ni, )
            Stores the K-th largest correlation value for each item.
            The value of `K_th` will also be stored.
        min_support: int, >=1, default 1
            Minimum number of common support for a non-NaN correlation.
        K_th: int >=0, default 100
            Which K-th largest correlation value to get.
        self_correlation: float or None, default None
            If provided, a value to replace self-correlation. Setting it
            to NaN avoids self as self's neighbours.

    Return:
        integral, 1d array
            Distribution of counts of Non-NaN correlations per item.
    '''
    num_items = source_item_supp.nrows
    num_supps = source_supp_item.nrows
    # True: two columns --> second column may be ratings by user to item
    # False: one column --> can be a genre-movie relation
    has_values = source_item_supp.dtype.shape[0] == 2

    # Create arrays
    target_corr_array = target_corr_array(
        shape=(num_items, num_items),
        dflt=_np.nan,
        dtype=_np.dtype(_config.default_float),
        chunkshape=(_config.chunk_size, num_items),
    )
    target_k_th_corr_array = target_k_th_corr_array(
        shape=(num_items, ),
        dflt=_np.nan,
        dtype=_np.dtype(_config.default_float),
        chunkshape=(num_items, ),
    )
    target_k_th_corr_array._f_setattr('K_th', K_th)

    # Helpers
    source_supp_item = source_supp_item[:] # preload for faster access

    _read_supp = _partial(_read_interaction, source_supp_item, has_values)

    if has_values:
        def _load_process(item_idx: int):
            item = source_item_supp[item_idx]
            supp_idxs = item[:, 0].astype(_config.default_int)
            item_vals = item[:, 1]
            corr = _utils.corr_via_supports(
                item_vals,
                *_read_supp(supp_idxs),
                min_support,
                N = num_items,
            )
            return item_idx, corr
    else:
        other_item = _read_supp(None)
        def _load_process(item_idx: int):
            item = source_item_supp[item_idx]
            supp_idxs = item[:, 0].astype(_config.default_int)
            item_vals = _np.zeros(num_supps, dtype=_config.default_float)
            item_vals[supp_idxs] = 1.
            corr = _utils.corr_via_supports(
                item_vals,
                *other_item,
                min_support,
                N = num_items,
            )
            return item_idx, corr

    # Buffer for counting non-NaN correlations
    nonnan_counts = []

    # Enable parallel processing
    # TopLevelObjectRegister.add_objs(_load_process)

    # _ConfiguredPipeLine
    (
        _ConfiguredPipeLine(
            range(num_items)
        )
        .tqdm(
            total=num_items,
            desc='Correlations',
        )
        # .parallel_map(
        .map(
            _load_process,
            doc='Load data and compute correlations',
        )
        .starpeek(
            lambda idx, corr: corr.__setitem__(idx, self_correlation),
            chain_if=self_correlation is not None,
            doc='Replace self correlation values',
        )
        .starpeek(
            lambda _, corr: nonnan_counts.append((~_np.isnan(corr)).sum()),
            doc='Count number of NaNs',
        )
        .starpeek(
            # Slower (~20ns) than target_corr_array[idx] = corr
            lambda idx, corr: target_corr_array.__setitem__(idx, corr),
            doc='Save correlation values',
        )
        .starmap(
            lambda idx, corr: (
                idx, _np.fmin.reduce(corr[_np.argsort(-corr)[:K_th]])
            ),
            doc='Compute k-th correlation values',
        )
        .batch(
            _config.chunk_size,
            doc='Batch many k_th_corr values up before saving to file'
        )
        .map(
            lambda batch: (
                target_k_th_corr_array.__setitem__(*map(list, zip(*batch)))
            ),
            doc='Save K-th correlation values',
        )
        .consume()
        .execute()
    )
    return _np.bincount(nonnan_counts)


@_ConfiguredPathsToPyTablesConverter(
    source_item_corr=(_utils.StoreMode.R, _utils.StoreType.CARRAY),
    source_item_k_th_corr=(_utils.StoreMode.R, _utils.StoreType.CARRAY),
    source_supp_item=(_utils.StoreMode.R, _utils.StoreType.VLARRAY),
    target_soft_array=(_utils.StoreMode.W, _utils.StoreType.CARRAY),
    target_hard_array=(_utils.StoreMode.W, _utils.StoreType.CARRAY),
)
def neighbours(
        *,
        source_item_corr: _utils.Tables.CArray,
        source_item_k_th_corr: _utils.Tables.VLArray,
        source_supp_item: _utils.Tables.VLArray,
        target_soft_array: _t.Callable,
        target_hard_array: _t.Callable,
    ) -> _t.Tuple[_t.IntegralArray, _t.IntegralArray]:
    '''Compute the top-K soft and hard neighbours for each support-item
    pair.

    Neighbours are items that are close to the item in the support-item
    pair. To explain in context, consider the case that (1) supports are
    users, (2) the user-item relation is by ratings, and (3) closeness
    is measured by the correlations of items:

        Soft: Among items rated by the user, get the K closest items
        Hard: Among the K closest items, get those rated by the user

    Pre-computed and stored at `source_item_k_th_corr`, the K-th highest
    correlation values for each item are used to filter hard neighbour
    candidates. The `source_supp_item` is the interaction sorted by
    users so that a user's rated item can be retrieved easily. Together
    with the items' correlations at `source_item_corr`, both soft and
    hard neighbours can be computed.

    Meaning of arguments' shape variables:

        Ni: number of items
        Ns: number of supports (e.g. users)

    Args:
        source_item_corr
            Path. Created with corr_from_interaction().
            The content is floating, (Ni, Ni)
        source_item_k_th_corr
            Path. Created with corr_from_interaction().
            The content is floating, (Ni, )
        source_supp_item
            Path. Created with interaction_from_table().
            The content is floating, (Ns, )
        target_soft_array
            Path. The content will be floating, (Ns, Ni, K, 2). The
            last axis's shape is 2 because each k (out of K) has two
            element - the k-th neighbour's item ID and the user's
            rating to the k-th neighbour.
        target_hard_array
            Path. The content will be integral, (Ns, Ni). For each
            support-item part, it has a length value to indicate the
            first N soft neighbours are the hard neighbours.

    Returns:
        integral, (2, K + 1)
            Distributions of counts of soft and hard neighbours.
    '''
    K = int(source_item_k_th_corr._f_getattr('K_th'))
    num_items = source_item_corr.nrows
    num_supps = source_supp_item.nrows

    # Create arrays
    target_soft_array = target_soft_array(
        dflt=-1,
        shape=(num_supps, num_items, K, 2),
        dtype=_np.dtype(_config.default_float),
        chunkshape=(_config.chunk_size, _config.chunk_size, K, 2),
    )
    target_hard_array = target_hard_array(
        dflt=0,
        shape=(num_supps, num_items),
        dtype=_np.dtype(_config.default_int),
        chunkshape=(_config.chunk_size, num_items),
    )

    # Helpers
    # pre-load for faster access
    source_supp_item = source_supp_item[:]
    source_item_k_th_corr = source_item_k_th_corr[:]
    _read_supp = _partial(_read_interaction, source_supp_item, True)

    @_dataclasses.dataclass
    class _Data:
        item_slice: slice
        supp_slice: slice
        items_corrs: _t.FloatingArray | None
        out_soft: _t.IntegralArray | None = None # (S, I, K, 2)
        out_hard: _t.IntegralArray | None = None # (S, I)
        out_bincount: _t.IntegralArray | None = None # (S, I)

        def add_results(self, *args: _t.Array) -> _t.Self:
            self.out_soft, self.out_hard, self.out_bincount = args
            return self

    def _load(item_slices: _utils.Slices, supp_slices: _utils.Slices) -> _Data:
        for item_slice in item_slices:
            items_corrs = source_item_corr[item_slice] # (I, N)
            for supp_slice in supp_slices:
                yield _Data(item_slice, supp_slice, items_corrs)

    # _ConfiguredPipeLine
    counts = (
        _ConfiguredPipeLine(
            _load(
                item_slices := _utils.Slices(0, num_items, _config.chunk_size),
                supp_slices := _utils.Slices(0, num_supps, _config.chunk_size),
            )
        )
        .map(
            lambda data: data.add_results(
                *_utils.get_neighbours(
                    data.items_corrs,
                    *_read_supp(data.supp_slice),
                    K=K,
                    items_k_th_corrs=source_item_k_th_corr,
                    return_bincount=True,
                    num_threads=_config.num_processes,
                )
            ),
            doc='Compute neighbours',
        )
        .peek(
            lambda data: target_soft_array.__setitem__(
                (data.supp_slice, data.item_slice), data.out_soft
            ),
            doc='Save soft neighbours to file',
        )
        .peek(
            lambda data: target_hard_array.__setitem__(
                (data.supp_slice, data.item_slice), data.out_hard
            ),
            doc='Save hard neighbours to file',
        )
        .map(
            lambda data: data.out_bincount,
            doc='Only pass bincounts through for reducion',
        )
        .tqdm(
            total=len(supp_slices) * len(item_slices),
            desc='Neighbours',
        )
        .reduce(
            lambda x, y: x + y
        )
        .execute()
    )
    return counts
