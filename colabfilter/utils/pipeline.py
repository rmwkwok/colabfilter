'''Pipeline experiment.'''


from . import typing as _t


import inspect as _inspect
import itertools as _itertools
from functools import partial as _partial, reduce as _reduce, wraps as _wraps
from multiprocessing import Pool as _Pool, Semaphore as _Semaphore


from tqdm.std import tqdm as _tqdm


class PipeLine:
    '''Pipelining an iterable source through methods.

    The pipeline must be terminated with `execute()` to run.

    Examples:
    ```
    print(
        PipeLine(range(10), default_num_procs=1)
        .map(lambda x: x + 1)
        .peek(lambda x: time.sleep(.1)) # input is passed thru
        .parallel_map(worker, num_procs=3) # override default_num_procs
        .filter(lambda x: x > 3)
        .filter(lambda x: x < 3, chain_if=False) # this is skipped
        .peek(lambda x: print('Remaining', x))
        .groupby(lambda x: x // 2)
        .starmap(lambda key, grp: grp)
        # At this point, the stream is an iterable of iterable of values
        .flatten()
        # At this point, the stream is flatten to an iterable of values
        # All above steps are lazy, below actually evaluates
        .reduce(lambda x, y: x + y) # iterate thru all element
        .call(lambda x: (x, print('Reduced result', x))[0]) # print 49
        .call(range) # this create another lazy iterable `range(49)`
        .filter(lambda x: x < 3)
        .peek(print)
        # If proceed to execute() without comsume(), since both filter
        # and peek are lazy, they won't evaluate. We can use
        # `call(list)` or use `reduce` (as above) to trigger the
        # evaluations, or `consume()` but it only returns `None`.
        .consume()
        .execute() # this is necessary to execute the pipeline
        # return the result, which is `None`, the outcome of `consume`
    )
    ```

    Args:
        source
            An iterable to be pipelined.
        default_num_procs
            For `parallel_map()` and may be overridden by a value
            provided in the call to the method.
        default_tqdm_kwargs
            For `tqdm()` and may be overridden by kwargs provided in
            the call to the method.
        logger
            If not provided, use `print`.

    Available methods by type:
        Intermediate
            map, parallel_map, map_where, starmap, peek, starpeek,
            filter, flatten, batch, groupby, tqdm
        Terminal
            reduce, consume
        Either
            call

    Extra kwargs to all methods except `execute()`:
        chain_if: bool = True
            To dynamically decide whether to chain the method.
        doc: str = ''
            Reserved for putting down an explanation to what the
            chained action is intended for.
            It may be used in the future.
    '''
    WARNING_TYPES = (map, filter, _itertools.starmap, _t.Generator, _tqdm)

    def __init__(
            self,
            source: _t.Iterable,
            *,
            default_num_procs: int = 1,
            default_tqdm_kwargs: dict | None = None,
            logger: _t.Logger | None = None,
        ) -> None:
        self.source = source
        self.pipeline = []
        self.default_num_procs = default_num_procs or 1
        self.default_tqdm_kwargs = default_tqdm_kwargs or {}
        self.logger = logger

    def execute(self, suppress_warning: bool=False) -> _t.Any:
        '''Execute pipeline and return the output of it.'''
        self.executed = True
        res = _reduce(lambda x, f: f(x), self.pipeline, self.source)
        if not suppress_warning and isinstance(res, self.WARNING_TYPES):
            getattr(self.logger, 'warning', print)(
                f'Pipeline results a "{type(res)}". If this is expected, set '
                f'`suppress_warning=True` to suppress this warning.')
        return res

    def pipeline_method(method: _t.Callable) -> _t.Callable:

        @_wraps(method)
        def _wrapper(
                self,
                *args: _t.Any,
                chain_if: bool = True,
                doc: str = '',
                **kwargs: _t.Any,
            ) -> _t.Self:
            # doc is reserved.
            if chain_if:
                self.pipeline.append(method(self, *args, **kwargs))
            return self

        _wrapper.__signature__ = _inspect.signature(_wrapper).replace(
            return_annotation=_t.Self)
        return _wrapper

    @pipeline_method
    def map(self, func: _t.Callable) -> _t.Callable:
        return _partial(map, func)

    @pipeline_method
    def map_where(
            self, cond: bool, func1: _t.Callable, func2: _t.Callable
        ) -> _t.Callable:
        '''Use func1 if cond=True else func2.'''
        if cond:
            return _partial(map, func1)
        else:
            return _partial(map, func2)

    @pipeline_method
    def call(self, func: _t.Callable) -> _t.Callable:
        '''Apply func to each element and pass the return downstream.'''
        return func

    @pipeline_method
    def peek(self, func: _t.Callable, **kwargs: _t.Any) -> _t.Callable:
        '''Apply func(**kwargs) to each element but pass the element
        downstream.'''
        return _partial(map, lambda x: (x, func(x, **kwargs))[0])

    @pipeline_method
    def tqdm(self, **kwargs) -> _t.Callable:
        '''Add progress bar tqdm(**kwargs).'''
        _kwargs = self.default_tqdm_kwargs.copy()
        _kwargs.update(kwargs)
        return _partial(_tqdm, **dict(self.default_tqdm_kwargs, **_kwargs))

    @pipeline_method
    def batch(self, n: int) -> _t.Callable:
        # recipe from:
        # https://docs.python.org/3/library/itertools.html#itertools.batched
        def batched(it: _t.Iterable) -> _t.Generator:
            it = iter(it)
            while batch := tuple(_itertools.islice(it, n)):
                yield batch
        return batched

    @pipeline_method
    def filter(self, func: _t.Callable) -> _t.Callable:
        return _partial(filter, func)

    @pipeline_method
    def reduce(
            self, func: _t.Callable, *, initializer: _t.Any = ...
        ) -> _t.Callable:
        # No overloading in Python, but to detect whether a valid initializer
        # is provided, I need a very unlikely default or an unique identifier.
        # I can have used `initializer = {}` since a mutable like this will
        # have an unique ID, but I am quite attracted to the idea of ..., so
        # I am giving it a try. `None` is out because it is more likely to be
        # a valid initializer.
        if initializer is ...:
            return _partial(_reduce, func)
        else:
            return lambda it: _reduce(func, it, initializer)

    @pipeline_method
    def groupby(self, func: _t.Callable) -> _t.Callable:
        '''Group consecutive elements with the same output of func that
        is applied to each element.'''
        return lambda it: _itertools.groupby(it, func)

    @pipeline_method
    def flatten(self) -> _t.Callable:
        return _itertools.chain.from_iterable

    @pipeline_method
    def starmap(self, func: _t.Callable) -> _t.Callable:
        return _partial(_itertools.starmap, func)

    @pipeline_method
    def consume(self) -> _t.Callable:
        '''Iterate through all elements and return a None.'''
        return _partial(_reduce, lambda *x: None)

    @pipeline_method
    def starpeek(self, func: _t.Callable, **kwargs: _t.Any) -> _t.Callable:
        '''Apply func(**kwargs) to each element but pass the element
        downstream.'''
        return _partial(
            _itertools.starmap, lambda *x: (x, func(*x, **kwargs))[0])

    @pipeline_method
    def parallel_map(
            self, func: _t.Callable, *, num_procs: int = 0
        ) -> _t.Callable:
        '''Parallelly apply func to at most `num_procs` elements at a
        time.'''
        num_procs = num_procs or self.default_num_procs

        def _parallel_map(it: _t.Iterable) -> _t.Generator:
            counter = _Semaphore(num_procs)

            def _producer() -> _t.Any:
                for item in it:
                    counter.acquire()
                    yield item

            with _Pool(num_procs) as pool:
                for item in pool.imap(func, _producer()):
                    yield item
                    counter.release()

        return _parallel_map

    def __del__(self):
        if not getattr(self, 'executed', False):
            getattr(self.logger, 'warning', print)(
                'A pipeline was created without executed.')
