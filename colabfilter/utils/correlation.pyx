'''Computing correlations.

This code is originally based on pandas._libs.algos.nancorr[1] which
uses the Welford's method for the variance-calculation[2].

[1] https://github.com/pandas-dev/pandas/blob/d9cdd2ee5a58015ef6f4d15c7226110c9aab8140/pandas/_libs/algos.pyx#L347
[2] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
'''
from numpy import typing as _npt


cimport cython
from libc.math cimport NAN, sqrt
from libc.stdlib cimport malloc, free


cimport numpy as _np
import numpy as _np


_IntegralArray = _npt.NDArray[_np.int32 | _np.int64]
_FloatingArray = _npt.NDArray[_np.float32 | _np.float64]


def corr_via_supports(
        item_vals: _FloatingArray,
        other_item_lens: _IntegralArray,
        other_item_idxs: _IntegralArray,
        other_item_vals: _FloatingArray,
        min_support: int = 1,
        *,
        N: int | None = None,
        out_corr: _FloatingArray | None = None,
    ) -> _FloatingArray:
    '''Compute correlations between an item and other items via their
    common supports, and the k-th largest correlation value with the
    item.

    To explain the following input arrays in context, consider:

    Situation 1: if the first 3 supports (users) of the item has rated
        4, 1, and 2 other items respectively, then

        - other_item_lens[0:3] = [4, 1, 2]
        - other_item_idxs[0:7] = [item IDs, ordered by user]
        - other_item_vals[0:7] = [ratings, ordered by user and item IDs]

    Situation 2: if there are a total of 5 supports (genres), and these
        genres have 1, 3, 0, 2, and 1 other items, then

        - other_item_lens = [1, 3, 0, 2, 1]
        - other_item_idxs[0:7] = [item IDs, ordered by genre]
        - other_item_vals = [] empty

        Because genre-movie is a binary relation, the last array, if not
        empty, may only contain all ones, which is redundant. It is thus
        expected to be empty as a signal that this is a binary relation,
        so the function will build a new `other_item_vals` with 1s
        for indices listed in `other_item_idxs` and 0s otherwise. For
        example, if other_item_idxs = [0, 0, 1, 3, ...], then

        [1, 0, 0, 0, 0 | 1, 1, 0, 1, 0 | 0, 0, 0, 0, 0 | ...]

        where vertical bars are used to separate the genres for clarity.

    Floating arrays must all either be np.float32 or np.float64, and
    Integral arrays must all either be np.int32 or np.int64.

    This function assumes no NaN in any input. Otherwise, the output
    may be invalid.

    Meaning of shape variables:
        S: number of supports
        M: number of items by all supports, M = sum(other_item_lens)
        N: max(item_ids) + 1

    Args:
        item_vals: floating array, (S, )
        other_item_lens: integral array, (S, )
        other_item_idxs: integral array, (M, )
        other_item_vals: floating array, (M, )
        min_support: integral, default 1
            Minimum number of supports for a non-NaN correlation value.
        N: integral, optional, default None
            If `out_corr` is None, `N` must be a positive integer to
            create `out_corr`.
        out_corr: floating array, (N, ) optional, default None
            To place the results. If provided, it can be larger than
            the expected shape (N, ) but only that shape will
            be initialized and used. Otherwise, an `out_corr` of the
            expected shape will be created with `N`.

    Returns:
        out_corr: floating array, (N, )
    '''
    floating_type = item_vals.dtype
    integral_type = other_item_lens.dtype

    if out_corr is None:
        if N is None:
            msg = f'N must be provided when out_corr=None.'
            raise ValueError(msg)
        else:
            out_corr = _np.full(N, _np.nan, dtype=floating_type)

    args = (
        item_vals,
        other_item_lens,
        other_item_idxs,
        other_item_vals,
        min_support,
        out_corr,
    )

    if floating_type == _np.float32 and integral_type == _np.int32:
        _corr_via_supports[cython.float, cython.int](*args)
    elif floating_type == _np.float32 and integral_type == _np.int64:
        _corr_via_supports[cython.float, cython.long](*args)
    elif floating_type == _np.float64 and integral_type == _np.int32:
        _corr_via_supports[cython.double, cython.int](*args)
    elif floating_type == _np.float64 and integral_type == _np.int64:
        _corr_via_supports[cython.double, cython.long](*args)
    else:
        msg = (
            f'Floating must all be either np.float32 or np.float64, and '
            f'integral must all be either np.int32 or np.int64. '
            f'Got {tuple(x.dtype for x in args if hasattr(x, "dtype"))}.'
        )
        raise TypeError(msg)

    return out_corr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _corr_via_supports(
        const cython.floating[:] item_vals,
        const cython.integral[:] other_item_lens,
        const cython.integral[:] other_item_idxs,
        const cython.floating[:] other_item_vals,
        cython.integral min_support,
        cython.floating[:] out_corr,
    ) noexcept nogil:
    cdef:
        # by cython -a correlation.pyx
        # .shape[0] isn't hinted at Python interaction, but len() does
        bint has_other_item_vals = other_item_vals.shape[0]
        cython.integral S = item_vals.shape[0]
        cython.integral N = out_corr.shape[0]
        cython.integral f_size = sizeof(cython.floating)
        cython.integral i_size = sizeof(cython.integral)

        cython.integral _other_item_idx
        cython.integral iN, iS, iL, L, L_base

        cython.integral* nobs = <cython.integral*> malloc(N * i_size)
        cython.integral* default_other_item_idxs = (
            <cython.integral*> malloc(N * i_size)
        )

        cython.floating _item_val, _other_item_val
        cython.floating vx, vy, dx, dy, divisor

        cython.floating* meanx = <cython.floating*> malloc(N * f_size)
        cython.floating* meany = <cython.floating*> malloc(N * f_size)
        cython.floating* covxy = <cython.floating*> malloc(N * f_size)
        cython.floating* ssqdmx = <cython.floating*> malloc(N * f_size)
        cython.floating* ssqdmy = <cython.floating*> malloc(N * f_size)
        cython.floating* default_other_item_vals = (
            <cython.floating*> malloc(N * f_size)
        )

        const cython.integral* ptr_other_item_idxs = NULL
        const cython.floating* ptr_other_item_vals = NULL

    # Initialization
    for iN in range(N):
        nobs[iN] = 0
        out_corr[iN] = NAN
        meanx[iN] = meany[iN] = ssqdmx[iN] = ssqdmy[iN] = covxy[iN] = 0.
        default_other_item_idxs[iN] = iN

    L_base = 0
    for iS in range(S): # THE support iS
        _item_val = item_vals[iS]
        L = other_item_lens[iS]

        # pointers to locate the data for THE support iS
        if has_other_item_vals:
            ptr_other_item_vals = &other_item_vals[L_base]
            ptr_other_item_idxs = &other_item_idxs[L_base]
            L_base += L
        else:
            ptr_other_item_idxs = &other_item_idxs[L_base]
            L_base += L
            # All items are "other_items":
            # used_other_item_idxs = all idxs = [0 ... N)
            # used_other_item_vals = ones for those listed in other_item_idxs
            #                        zeros otherwise
            for iN in range(N):
                default_other_item_vals[iN] = 0.
            for iL in range(L):
                default_other_item_vals[ptr_other_item_idxs[iL]] = 1.
            ptr_other_item_idxs = default_other_item_idxs
            ptr_other_item_vals = default_other_item_vals
            L = N

        for iL in range(L):
            _other_item_idx = ptr_other_item_idxs[iL]
            _other_item_val = ptr_other_item_vals[iL]

            # Welford's method for covariance
            vx = _item_val
            vy = _other_item_val

            nobs[_other_item_idx] += 1

            dx = vx - meanx[_other_item_idx]
            dy = vy - meany[_other_item_idx]

            meanx[_other_item_idx] += 1. / nobs[_other_item_idx] * dx
            meany[_other_item_idx] += 1. / nobs[_other_item_idx] * dy

            ssqdmx[_other_item_idx] += (vx - meanx[_other_item_idx]) * dx
            ssqdmy[_other_item_idx] += (vy - meany[_other_item_idx]) * dy
            covxy[_other_item_idx] += (vx - meanx[_other_item_idx]) * dy

    for iN in range(N):
        if nobs[iN] >= min_support:
            divisor = sqrt(ssqdmx[iN] * ssqdmy[iN])

            if divisor != 0:
                out_corr[iN] = covxy[iN] / divisor

    free(nobs)
    free(meanx)
    free(meany)
    free(covxy)
    free(ssqdmx)
    free(ssqdmy)
    free(default_other_item_idxs)
    free(default_other_item_vals)

    return 0
