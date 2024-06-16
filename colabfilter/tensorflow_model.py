from colabfilter.utils import typing as _t


import pickle as _pickle
import dataclasses as _dataclasses


import tensorflow as _tf


from colabfilter import config as _config, _configured_tqdm, utils as _utils


@_dataclasses.dataclass
class CFModelParams:
    '''CFModel's arguments.

    An experiment to separate a class's arguments out as a dataclass.

    These arguments should be understood with the model description in
    the accompanied Jupyter Notebook.

    Args:
        num_user, num_item, low_rank
            For determining the sizes of trainable variables. low_rank
            is the embedding size of each user or item vector.
        option_1, option_2, option_3
            For whether each option is turned on or off.
        lambda_0, lambda_1, lambda_2, lambda_3
            L2 regularization coefficients.
        seed
            For initializing trainable variables.
        epochs
            Number of epochs to train with.
        fdbk_prefix: one of {None, 'fdbk_rating'}
        nbhd_prefix: one of {None, 'nbhd_soft_genre', 'nbhd_hard_genre',
            'nbhd_soft_rating', 'nbhd_hard_rating'}
            If the relevant option is turned on, then the prefix is
            used. For the meaning of each option, refer to the
            accompanied Jupyter Notebook.

    Methods:
        with_callable_values(): a static method
            It creates a new instance with the dictionary of keyword
            arguments, and if any argument's value is callable, it
            calls it with the instance, and replaces it with the return
            as the new argument value. It does the following:
            `setattr(instance, keyword, callable_value(instance))`.
    '''
    _: _dataclasses.KW_ONLY
    num_user: int
    num_item: int
    low_rank: int
    option_1: bool
    option_2: bool
    option_3: bool
    lambda_0: float
    lambda_1: float
    lambda_2: float
    lambda_3: float
    seed: int
    epochs: int
    learning_rate: float
    fdbk_prefix: _t.Literal['fdbk_rating']
    nbhd_prefix: _t.Literal['nbhd_soft_genre', 'nbhd_hard_genre',
                            'nbhd_soft_rating', 'nbhd_hard_rating',
                            ]

    @staticmethod
    def with_callables(**kwargs: _t.Any) -> _t.Self:
        '''It creates a new instance with the dictionary of keyword
        arguments, and if any argument's value is callable, it calls it
        with the instance, and replaces the callable value with the
        call's return as the new argument value. It does the following:
        `setattr(instance, keyword, callable_value(instance))`.
        '''
        params = CFModelParams(**kwargs)
        for key, val in kwargs.items():
            if callable(val):
                setattr(params, key, val(params))
        return params

    def as_dict(self):
        return _dataclasses.asdict(self)

    @property
    def repr(self):
        # Representation of this param set
        d = self.as_dict()
        if d['option_1'] == 0:
            d['lambda_1'] = d['fdbk_prefix'] = None
        if d['option_2'] == 0:
            d['lambda_2'] = d['nbhd_prefix'] = None
        if d['option_3'] == 0:
            d['lambda_3'] = None
        return tuple(d.values())


@_dataclasses.dataclass
class _TrainableVariables:
    all: _t.List[_t.TFVariable] = _dataclasses.field(default_factory=list)

    def add(self, key, val):
        setattr(self, key, val)
        self.all.append(val)


class CFModel:
    '''Colaborative Filtering Model.

    It accepts a `CFModelParams` instance as its only input argument.
    It is an experiment to separate this class's argument out as a
    dataclass.

    A prediction may be made by calling an instance of this class with
    a batch of samples from a `tensorflow.data.Dataset` created with
    `tensorflow_dataset.get_dataset()`. For example,
    `pred = model(batch)`.

    Attributes:
        trainable_weights
            Return a list of trainable weight tensors.
        num_trainable_parameters
            Return the number of trainable parameters in all weight
            tensors.

    Other methods:
        save()
            To save this instance as a pickle file.
        from_pickle_file()
            A static method to load a saved pickle file. May be called
            like `model = CFModel.from_pickle_file(path)`
    '''

    def __init__(self, params: CFModelParams) -> None:
        self.params = params
        self.vars = _TrainableVariables()
        self.build()

    def build(self) -> None:
        p = self.params
        x = [
            (True, 'zeros', 'mu', (1, )),
            (True, 'zeros', 'bi', (p.num_item, )),
            (True, 'random_normal', 'bu', (p.num_user, )),
            (True, 'random_normal', 'Qi', (p.num_item, p.low_rank)),
            (True, 'random_normal', 'Pu', (p.num_user, p.low_rank)),
            (p.option_1, 'random_normal', 'Yi', (p.num_item, p.low_rank)),
            (p.option_2, 'random_normal', 'W1', (p.num_item, p.low_rank)),
            (p.option_2, 'random_normal', 'W2', (p.num_item, p.low_rank)),
            (p.option_2, 'random_normal', 'B1', (p.num_user, p.low_rank)),
            (p.option_2, 'random_normal', 'B2', (p.num_item, p.low_rank)),
            (p.option_3, 'random_normal', 'S1', (p.num_item, p.low_rank)),
            (p.option_3, 'random_normal', 'S2', (p.num_item, p.low_rank)),
        ]

        _tf.keras.utils.set_random_seed(p.seed)
        tf_float = _tf.dtypes.as_dtype(_config.default_float)
        for use, initializer_name, name, shape in x:
            if use:
                f = _tf.keras.initializers.get(initializer_name)
                self.vars.add(
                    name,
                    _tf.Variable(f(shape=shape, dtype=tf_float), name=name),
                )

    def fit(
            self,
            train_dataset: _t.TFDataset,
            test_dataset: _t.TFDataset,
            recorder_update_func: _t.Callable | None = None,
        ) -> _t.Dict:
        '''Model fitting.

        Args:
            train_dataset, test_dataset
            recorder_update_func
                Expected to be either None (not used) or a
                `ResultsRecorder.update` callable. At the end of each
                epoch, it will call recorder_update_func(...) with the
                following keyword arguments: epochs, train_MSE,
                train_MAE, test_MSE, test_MAE. The recorder is
                responsible for logging these information and print
                them (if the recorder is configured to)
        '''
        p = self.params
        self._optimizer = _tf.keras.optimizers.Adam(p.learning_rate)
        self._loss = _tf.keras.losses.MeanSquaredError()
        self._metrics = {
            'train_MSE': _tf.keras.metrics.Mean(),
            'train_MAE': _tf.keras.metrics.MeanAbsoluteError(),
            'test_MSE': _tf.keras.metrics.Mean(),
            'test_MAE': _tf.keras.metrics.MeanAbsoluteError(),
        }
        train_step = _tf.function(
            self._train_step,
            input_signature = [train_dataset.element_spec],
        )
        test_step = _tf.function(
            self._test_step,
            input_signature = [test_dataset.element_spec],
        )
        train_num_batches = None
        test_num_batches = None
        history = {k: [] for k in self._metrics.keys()}
        for i in range(p.epochs):
            with _configured_tqdm(
                    total=train_num_batches,
                    desc=f'Epoch {i + 1}/{p.epochs} train set',
                ) as pbar:
                for j, batch in train_dataset.enumerate():
                    train_step(batch)
                    pbar.update()
                else:
                    train_num_batches = j.numpy() + 1

            with _configured_tqdm(
                    total=test_num_batches,
                    desc=f'Epoch {i + 1}/{p.epochs} test set',
                ) as pbar:
                for j, batch in test_dataset.enumerate():
                    test_step(batch)
                    pbar.update()
                else:
                    test_num_batches = j.numpy() + 1

            results = {}
            for k, obj in self._metrics.items():
                result = obj.result().numpy()
                results[k] = result
                history[k].append(result)
                obj.reset_state()

            if callable(recorder_update_func):
                recorder_update_func(epochs=i + 1, **results)

        return history

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle_file(path: str) -> _t.Self:
        with open(path, 'rb') as f:
            return pickle.load(f)

    @property
    def trainable_weights(self):
        return self.vars.all

    @property
    def num_trainable_parameters(self):
        return sum(
            _tf.reduce_prod(v.shape) for v in self.trainable_weights
        ).numpy()

    def _train_step(self, batch: _t.Dict[str, _t.TFTensor]) -> None:
        with _tf.GradientTape() as tape:
            y_true = batch['rating']
            y_pred = self(batch)
            J = self._cost(y_true, y_pred)
        grads = tape.gradient(J, self.vars.all)
        self._optimizer.apply_gradients(zip(grads, self.vars.all))
        self._metrics['train_MSE'].update_state(self._loss(y_true, y_pred))
        self._metrics['train_MAE'].update_state(y_true, y_pred)

    def _test_step(self, batch: _t.TFTensor) -> None:
        y_true = batch['rating']
        y_pred = self(batch)
        self._metrics['test_MSE'].update_state(self._loss(y_true, y_pred))
        self._metrics['test_MAE'].update_state(y_true, y_pred)

    def _cost(self, y_true: _t.TFTensor, y_pred: _t.TFTensor) -> _t.TFTensor:
        v = self.vars
        p = self.params

        J = _sum_squared(y_true - y_pred)
        J = J + p.lambda_0 *  _tf.add_n([
            _sum_squared(v.bi),
            _sum_squared(v.bu),
            _sum_squared(v.Qi),
            _sum_squared(v.Pu),
        ])

        if p.option_1:
            J = J + p.lambda_1 * _sum_squared(v.Yi)

        if p.option_2:
            J = J + p.lambda_2 *  _tf.add_n([
                _sum_squared(v.B1),
                _sum_squared(v.B2),
                _sum_squared(v.W1),
                _sum_squared(v.W2),
            ])

        if p.option_3:
            J = J + p.lambda_3 *  _tf.add_n([
                _sum_squared(v.S1),
                _sum_squared(v.S2),
            ])

        return 0.5 * J


    def __call__(self, batch: _t.TFTensor) -> _t.TFTensor:
        v = self.vars
        p = self.params
        batch_size = _tf.shape(batch['user'])[0]

        mu = v.mu
        bi = _tf.gather(v.bi, batch['item'])
        bu = _tf.gather(v.bu, batch['user'])
        Qi = _tf.gather(v.Qi, batch['item'])
        Pu = _tf.gather(v.Pu, batch['user'])

        r = mu + bu + bi

        if p.option_1:
            j = batch[f'{p.fdbk_prefix}_indices']
            C = batch[f'{p.fdbk_prefix}_scaling']
            C = _tf.expand_dims(C, axis=1)
            Yi = _tf.nn.embedding_lookup_sparse(v.Yi, j, None, combiner='sum')
            Yi = _pad_rows(Yi, batch_size, ndim=2)
            r += _tf.reduce_sum(Qi * (Pu + C * Yi), axis=1)
        else:
            r += _tf.reduce_sum(Qi * Pu, axis=1)

        if p.option_2:
            j = batch[f'{p.nbhd_prefix}_indices']
            R = batch[f'{p.nbhd_prefix}_ratings']
            C = batch[f'{p.nbhd_prefix}_scaling']
            W1 = _tf.gather(v.W1, batch['item'])
            W2R = _tf.nn.embedding_lookup_sparse(v.W2, j, R, combiner='sum')
            W2R = _pad_rows(W2R, batch_size, ndim=2)
            term_1 = _tf.reduce_sum(W1 * W2R, axis=1)

            B1 = _tf.gather(v.B1, batch['user'])
            W1 = _tf.gather(v.W1, batch['item'])

            idxs = j.indices[:, 0]
            vals = j.values
            B1B2 = _tf.reduce_sum(
                _tf.gather(B1, idxs) * _tf.gather(v.B2, vals), axis=1,
            )
            W1W2 = _tf.reduce_sum(
                _tf.gather(W1, idxs) * _tf.gather(v.W2, vals), axis=1,
            )
            term_2 = _tf.math.segment_sum(B1B2 * W1W2, idxs)
            term_2 = _pad_rows(term_2, batch_size, ndim=1)
            r += C * (term_1 - term_2)

        if p.option_3:
            j = batch['siam_indices']
            R = batch['siam_ratings']
            S2 = _tf.gather(v.S2, batch['item'])
            S1R = _tf.nn.embedding_lookup_sparse(v.S1, j, R, combiner='sum')
            S1R = _pad_rows(S1R, batch_size, ndim=2)
            r += _tf.reduce_sum(S2 * S1R, axis=1)

        return r


def _sum_squared(Y_diff: _t.TFTensor) -> _t.TFTensor:
    return _tf.reduce_sum(_tf.math.square(Y_diff))


def _sum_absoluted(Y_diff: _t.TFTensor) -> _t.TFTensor:
    return _tf.reduce_sum(_tf.math.abs(Y_diff))


def _pad_rows(tensor: _t.TFTensor, batch_size: int, ndim: int) -> _t.TFTensor:
    # tf.nn.embedding_lookup_sparse and tf.math.segment_sum do not perserve the
    # size of axis=0. In particular, they will miss out the last N rows if
    # `sp_ids` or `segment_ids` do not contain the last N values. Therefore,
    # padding is needed when the size mismatches.
    diff = batch_size - _tf.shape(tensor)[0]
    if ndim == 1:
        true_fn = lambda: _tf.pad(tensor, [[0, diff],])
    elif ndim == 2:
        true_fn = lambda: _tf.pad(tensor, [[0, diff], [0, 0]])
    else:
        raise NotImplementedError
    return _tf.cond(diff > 0, true_fn, lambda: tensor)


def free_tensorflow_resources() -> None:
    with _utils.try_release_free_memory_from_heap():
        _tf.keras.backend.clear_session()

