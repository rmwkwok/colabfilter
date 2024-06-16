'''Create TFRecords and assemble Dataset pipeline.'''


from .utils import typing as _t


import os as _os
import dataclasses as _dataclasses
from functools import partial as _partial
from contextlib import ExitStack as _ExitStack


import numpy as _np
import tensorflow as _tf


from . import (
    config as _config,
    utils as _utils,
    _ConfiguredPathsToPyTablesConverter,
    _ConfiguredPipeLine,
)
from .utils import TopLevelObjectRegister


@_dataclasses.dataclass
class _TFFeaturesBase:
    def serializer_spec(self) -> _t.Dict:
        return dict(self._serializer_spec())

    @classmethod
    def parser_spec(self) -> _t.Dict:
        return dict(self._parser_spec())

    @classmethod
    def tensor_spec(self, batch_size: int | None = None) -> _t.Dict:
        return dict(self._tensor_spec(batch_size))

    def _serializer_spec(self, prefix: str = '') -> _t.Generator:
        for key, anno in self.__annotations__.items():
            val = getattr(self, key)
            if issubclass(anno, _TFFeaturesBase):
                yield from val._serializer_spec(f'{key}_')
            else:
                yield f'{prefix}{key}', _get_tfrecord_features(anno)(val)

    @classmethod
    def _parser_spec(self, prefix: str = '') -> _t.Generator:
        for key, anno in self.__annotations__.items():
            if issubclass(anno, _TFFeaturesBase):
                yield from anno._parser_spec(f'{key}_')
            else:
                yield f'{prefix}{key}', _get_tfrecord_feature_parser(anno)

    @classmethod
    def _tensor_spec(
            self,
            batch_size: int | None = None,
            prefix: str = '',
        ) -> _t.Generator:
        for key, anno in self.__annotations__.items():
            name = f'{prefix}{key}'
            if issubclass(anno, _TFFeaturesBase):
                yield from anno._tensor_spec(batch_size, f'{key}_')
            else:
                yield name, _get_tensor_spec(anno, name, batch_size)


@_dataclasses.dataclass
class _ImplicitFeedback(_TFFeaturesBase):
    indices: _t.IntegralArray
    scaling: _t.Float


@ _dataclasses.dataclass
class _Neighbourhood(_TFFeaturesBase):
    indices: _t.IntegralArray
    ratings: _t.FloatingArray
    scaling: _t.Float


@ _dataclasses.dataclass
class _Siam(_TFFeaturesBase):
    indices: _t.IntegralArray
    ratings: _t.FloatingArray


@_dataclasses.dataclass
class _TFRecordFeatures(_TFFeaturesBase):
    user: _t.Int
    item: _t.Int
    rating: _t.Float
    fdbk_rating: _ImplicitFeedback
    nbhd_soft_genre: _Neighbourhood
    nbhd_hard_genre: _Neighbourhood
    nbhd_soft_rating: _Neighbourhood
    nbhd_hard_rating: _Neighbourhood
    siam: _Siam


def _get_tfrecord_features(anno: type):
    match anno:
        case _t.Int:
            return lambda v: _tf.train.Feature(
                int64_list=_tf.train.Int64List(value=[v])
            )
        case _t.Float:
            return lambda v: _tf.train.Feature(
                float_list=_tf.train.FloatList(value=[v])
            )
        case _t.IntegralArray:
            return lambda v: _tf.train.Feature(
                int64_list=_tf.train.Int64List(value=v)
            )
        case _t.FloatingArray:
            return lambda v: _tf.train.Feature(
                float_list=_tf.train.FloatList(value=v)
            )
        case _:
            raise NotImplementedError


def _get_tfrecord_feature_parser(anno: type):
    tf_int = _tf.dtypes.as_dtype(_config.default_int)
    tf_float = _tf.dtypes.as_dtype(_config.default_float)
    match anno:
        case _t.Int:
            return _tf.io.FixedLenFeature([], tf_int)
        case _t.Float:
            return _tf.io.FixedLenFeature([], tf_float)
        case _t.IntegralArray:
            return _tf.io.VarLenFeature(tf_int)
        case _t.FloatingArray:
            return _tf.io.VarLenFeature(tf_float)
        case _:
            raise NotImplementedError


def _get_tensor_spec(anno: type, name:str, batch_size: int):
    tf_int = _tf.dtypes.as_dtype(_config.default_int)
    tf_float = _tf.dtypes.as_dtype(_config.default_float)
    match anno:
        case _t.Int:
            return _tf.TensorSpec(
                shape=[batch_size], dtype=tf_int)
        case _t.Float:
            return _tf.TensorSpec(
                shape=[batch_size], dtype=tf_float)
        case _t.IntegralArray:
            return _tf.SparseTensorSpec(
                shape=[batch_size, None], dtype=tf_int)
        case _t.FloatingArray:
            return _tf.SparseTensorSpec(
                shape=[batch_size, None], dtype=tf_float)
        case _:
            raise NotImplementedError


def get_tensor_spec(batch_size: int | None = None) -> _t.Dict:
    return _TFRecordFeatures.tensor_spec(batch_size)


@_ConfiguredPathsToPyTablesConverter(
    src_ratings=(_utils.StoreMode.R, _utils.StoreType.TABLE),
    src_fdbk_rating=(_utils.StoreMode.R, _utils.StoreType.VLARRAY),
    src_nbhd_soft_genre=(_utils.StoreMode.R, _utils.StoreType.CARRAY),
    src_nbhd_hard_genre=(_utils.StoreMode.R, _utils.StoreType.CARRAY),
    src_nbhd_soft_rating=(_utils.StoreMode.R, _utils.StoreType.CARRAY),
    src_nbhd_hard_rating=(_utils.StoreMode.R, _utils.StoreType.CARRAY),
)
def create_tfrecords(
        *,
        src_ratings: _utils.Tables.Table,
        src_fdbk_rating: _utils.Tables.VLArray,
        src_nbhd_soft_genre: _utils.Tables.CArray,
        src_nbhd_hard_genre: _utils.Tables.CArray,
        src_nbhd_soft_rating: _utils.Tables.CArray,
        src_nbhd_hard_rating: _utils.Tables.CArray,
        file_prefix: str,
        number_of_files: _t.Annotated[int, _t.ValueRange(1, None, True)],
    ) -> None:
    '''Create TFRecords.

    Each record consists of:
        item: int, user: int, rating: float

        Implicit feedback, naming format: fdbk_{type}_{attrribute}
            fdbk_rating_indices: 1d array, int
            fdbk_rating_scaling: float

        Neighbourhood, naming format: nbhd_{type}_{attrribute}
            nbhd_hard_genre_indices: 1d array, int
            nbhd_hard_genre_ratings: 1d array, float
            nbhd_hard_genre_scaling: float
            nbhd_hard_rating_indices: 1d array, int
            nbhd_hard_rating_ratings: 1d array, float
            nbhd_hard_rating_scaling: float
            nbhd_soft_genre_indices: 1d array, int
            nbhd_soft_genre_ratings: 1d array, float
            nbhd_soft_genre_scaling: float
            nbhd_soft_rating_indices: 1d array, int
            nbhd_soft_rating_ratings: 1d array, float
            nbhd_soft_rating_scaling: float

        SIAM, naming format: siam_{attribute}
            siam_indices: 1d array, int
            siam_ratings: 1d array, float

    Each of these elements corresponds to a model variable (shown in
    the accompanied Jupyter Notebook).

    Attributes can be
        scaling: the inverse square-root term
        indices: item IDs
        ratings: user's ratings to the item IDs

    While only one neighbourhood type is sufficient, all four are
    included, at the cost of I/O, for the purpose of experimenting
    different choices/combinations.

    Args:
        src_ratings, src_fdbk_rating, src_nbhd_soft_genre,
        src_nbhd_hard_genre, src_nbhd_soft_rating, src_nbhd_hard_rating
            PyTables paths.
        file_prefix
            Prefix to the TFRecords files. Useful for distinguishing
            between train and test sets, and/or different K-fold splits.
            Generated files are in
            f'{config.path_tfrecords}/{file_prefix}*.tfrecords',
            where * is replaced by a zero-based file ID up to a value
            provided in number_of_files. Existing files only with names
            to be written to will be removed.
        number_of_files: >=1
            The number of TFRecords files to generate. Provide another
            degree of shuffling of streamed data.
    '''
    # Helpers
    rng = _np.random.default_rng(_config.default_seed)

    _scale = lambda x: x and (1. / x)**.5
    _dropna = lambda x: x[~_np.isnan(x).any(axis=1)]
    _drop_item = lambda item, x: x[x[:, 0] != item]

    @_dataclasses.dataclass
    class _Data:
        loop_id: int
        sample_id: int
        features: _t.Dict[str, _tf.train.Feature] | None = None

    def _create_features(data: _Data) -> _Data:
        # Read a sample of user-item-ratings, then use the userId and
        # movieId to get other data for implicit feedback & neighbood.
        row = src_ratings[data.sample_id]
        data.features = _TFRecordFeatures(
            # Experimental way of using the Walrus expression
            u := row['userId'],
            i := row['movieId'],
            _ := row['rating'],
            siam=_Siam(
                (x := _drop_item(i, src_fdbk_rating[u]))[:, 0].astype(int),
                (x[:, 1]),
            ),
            fdbk_rating=_ImplicitFeedback(
                x[:, 0].astype(int),
                _scale(len(x)),
            ),
            nbhd_soft_genre=(s := _Neighbourhood( # Soft, Genre
                (x := _dropna(src_nbhd_soft_genre[u, i]))[:, 0].astype(int),
                (x[:, 1]),
                _scale(len(x)),
            )),
            nbhd_hard_genre=_Neighbourhood( # Hard, Genre, depends on the above
                s.indices[:(x := src_nbhd_hard_genre[u, i])],
                s.ratings[:x],
                _scale(x),
            ),
            nbhd_soft_rating=(t := _Neighbourhood( # Soft, Rating
                (x := _dropna(src_nbhd_soft_rating[u, i]))[:, 0].astype(int),
                (x[:, 1]),
                _scale(len(x)),
            )),
            nbhd_hard_rating=_Neighbourhood( # Hard, Rating, depends on the above
                t.indices[:(x := src_nbhd_hard_rating[u, i])],
                t.ratings[:x],
                _scale(x),
            ),
        ).serializer_spec()
        return data

    def _write_to_tf_records(data: _Data) -> None:
        '''To evenly write to the writers, use them circulatively.'''
        writer_id = divmod(data.loop_id, number_of_files)[1]
        writers[writer_id].write(
            _tf.train
            .Example(features=_tf.train.Features(feature=data.features))
            .SerializeToString()
        )

    # Enable multiprocessing
    TopLevelObjectRegister.add_objs(_Data, _create_features)

    # ConfiguredPipeLines
    with _ExitStack() as stack:
        writers = (
            _ConfiguredPipeLine(
                range(number_of_files)
            )
            .map(
                lambda x: f'{file_prefix}{x:03d}.tfrecords',
                doc='Construct file name',
            )
            .map(
                lambda x: _os.path.join(_config.path_tfrecords, x),
                doc='Construct file path',
            )
            .peek(
                lambda x: _os.remove(x) if _os.path.exists(x) else None,
                doc='Remove tfrecords files if exist'
            )
            .map(
                _tf.io.TFRecordWriter,
                doc='Make a TFRecordWriter',
            )
            .map(
                stack.enter_context,
                doc='Add writers to the context manager',
            )
            .call(
                list,
                doc='iterate through all elements and return as a list'
            )
            .execute()
        )
        (
            _ConfiguredPipeLine(
                enumerate(rng.permutation(src_ratings.nrows))
                # read table rows randomly
            )
            .starmap(
                _Data,
                doc='Construct dataclass',
            )
            .parallel_map(
                _create_features,
                doc='Create a feature record',
            )
            .map(
                _write_to_tf_records,
                doc='Save record to one of the TF files',
            )
            .tqdm(
                total=src_ratings.nrows,
                desc='Loading TF Records',
            )
            .consume()
            .execute()
        )


def example_parser(x: _t.TFTrainExample) -> _t.Dict:
    '''Parse one or more serialized TFRecords created by
    `create_tfrecords()`.'''
    features=_TFRecordFeatures.parser_spec()
    return _tf.io.parse_example(x, features=features)


def get_dataset(
        file_prefix: str,
        batch_size: _t.Annotated[int, _t.ValueRange(1, None, True)],
        shuffle_buffer_size: _t.Annotated[int, _t.ValueRange(0, None, True)],
    ) -> _t.TFDataset:
    '''Create TF Dataset from TFRecords created by `create_tfrecords()`.

    Args:
        file_prefix
            Same as that used in `create_tfrecords()`.
        batch_size: >= 1
            Batch size.
        shuffle_buffer_size: >= 0
            0 to disable, otherwise, both the order of reading TFRecords
            files and the order of samples for batching are shuffled.
            `config.default_seed` is used.
    '''

    dset = _tf.data.TFRecordDataset(
        _tf.data.Dataset.list_files(
            _os.path.join(_config.path_tfrecords, f'{file_prefix}*.tfrecords'),
            shuffle=shuffle_buffer_size > 0, seed=_config.default_seed,
        )
    )

    if shuffle_buffer_size > 0:
        dset = dset.shuffle(shuffle_buffer_size, _config.default_seed, True)

    return (
        dset
        .batch(batch_size, drop_remainder=True)
        .map(example_parser)
        .prefetch(_tf.data.AUTOTUNE)
    )
