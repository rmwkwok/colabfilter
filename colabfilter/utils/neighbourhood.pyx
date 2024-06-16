# distutils: language = c++
# distutils: extra_link_args = ['-fopenmp']
# distutils: extra_compile_args = ['-fopenmp']


from numpy import typing as _npt


cimport cython
from cython.parallel cimport prange


from libc.math cimport isnan, NAN
from libc.stdlib cimport malloc, free
from libcpp.algorithm cimport sort


cimport numpy as _np
import numpy as _np


_IntegralArray = _npt.NDArray[_np.int32 | _np.int64]
_FloatingArray = _npt.NDArray[_np.float32 | _np.float64]


cdef struct I32_F32_Element:
    bint mask
    cython.int index
    cython.float value
    cython.float other_value


cdef struct I32_F64_Element:
    bint mask
    cython.int index
    cython.double value
    cython.double other_value


cdef struct I64_F32_Element:
    bint mask
    cython.long index
    cython.float value
    cython.float other_value


cdef struct I64_F64_Element:
    bint mask
    cython.long index
    cython.double value
    cython.double other_value


ctypedef fused Element:
    I32_F32_Element
    I32_F64_Element
    I64_F32_Element
    I64_F64_Element


cdef inline bint _comp_rev(Element* ptr_a, Element* ptr_b) noexcept nogil:
    cdef Element a = ptr_a[0]
    cdef Element b = ptr_b[0]
    cdef cython.float dv
    m1 = a.mask
    m2 = b.mask
    v1 = a.value
    v2 = b.value
    dv = v1 - v2
    if m1 and m2:
        return 0
    elif m1:
        return 0
    elif m2:
        return 1
    elif dv <= 0:
        return 0
    else:
        return 1


def get_neighbours(
        items_corr: _FloatingArray,
        supps_items_lens: _IntegralArray,
        supps_items_idxs: _IntegralArray,
        supps_items_vals: _FloatingArray,
        K: int,
        items_k_th_corrs: _FloatingArray,
        return_bincount: bool = False,
        out_soft: _IntegralArray | None = None,
        out_hard: _IntegralArray | None = None,
        out_bincount: _IntegralArray | None = None,
        num_threads: int = 1,
    ) -> Tuple[_FloatingArray, _IntegralArray]:
    '''Compute the top-K soft and hard neighbours for each support-item
    pair.

    Neighbours are items that are close to the item in the support-item
    pair. To explain in context, consider the case that (1) supports are
    users, (2) the user-item relation is by ratings, and (3) closeness
    is measured by the correlations of items:

        Soft: Among items rated by the user, get the K closest items
        Hard: Among the K closest items, get those rated by the user

    For example, if there are 10 items, marked as O for those that has
    been rated by the user and X otherwise, and they are sorted by
    their correlations, in descending order, with the item:

        O X O X X O O X O O

    Then, if `K = 5`, the first 2 "O"s are the hard neighbours while the
    first 5 are the soft. This also shows that the hard ones are always
    a subset of the soft ones.

    To explain the following input arrays, if the first 3 users rated
    4, 1, and 2 items respectively, then:

        supps_items_lens[0:3] = [4, 1, 2]
        supps_items_idxs[0:7] = [item IDs, ordered by user]
        supps_items_vals[0:7] = [ratings, ordered by user and item IDs]

    Floating arrays must all either be np.float32 or np.float64, and
    Integral arrays must all either be np.int32 or np.int64.

    Only `items_corrs` may have input, otherwise, the outputs may be
    invalid.

    Meaning of arguments' shape variables:

        I: number of items to form support-item pairs
        S: number of supports to form support-item pairs
        N: number of all items as neighbour candidates
        L: number of ratings = sum(supps_items_lens)

    Args:
        items_corrs: floating array, (I, N)
        supps_items_lens: integral array, (S, )
        supps_items_idxs: integral array, (L, )
        supps_items_vals: floating array, (L, )
        K: int
            Control the neighbour size.
        items_k_th_corrs: floating array, (N, )
            The K-th largest correlation values for all items.
        return_bincount: bool, default False
            Whether to return the bincounts of neighbours for all
            support-item pairs. If `True`, `out_bincount` may be
            provided or it will be created.
        out_soft: floating array, (S, I, K, 2), default None
            To place the results. If provided, it can be larger than
            the expected shape (S, I, K, 2) but only that shape will
            be initialized and used. Otherwise, an `out_soft` of the
            expected shape will be created.
        out_hard: integral array, (S, I), default None
            To place the results. Same requirement of shape as
            `out_soft`.
        out_bincount: integral array, (2, K + 1), default None
            Only useful if `return_bincount=True`. To place the
            results. Same requirement of shape as `out_soft`.
        num_threads: integral, >=1, default 1
            Control multithreading.

    Returns:
        out_soft: floating array, (S, I, K, 2)
            For example, out_soft[s, i, k, 0] is the item ID for the
            k-th neighbour of the user-item pair (s, i), and
            out_soft[s, i, k, 1] is the user's rating to that item.
            If there is no k-th neighbour, both values are NaNs.
        out_hard: integral array, (S, I)
            Since, as explained, hard neighbours are a subset of the
            soft neighbours, for example, if out_hard[s, i] = 4, then
            out_soft[s, i, :4] are the hard neighbours.
        out_bincount: integral array, (2, K + 1)
            Returned if `return_bincount=True`. For example,
            out_bincount[0, k] is the number of support-item pairs that
            have k soft neighbours, whereas out_bincount[1, k] is the
            number of pairs that have k hard neighbours.
    '''
    # TODO: support disabling multithreading, when `use_threads_if`
    # argument is avilable in cython.parallel.prange
    I = items_corr.shape[0]
    S = supps_items_lens.shape[0]

    floating_type = items_corr.dtype
    integral_type = supps_items_lens.dtype

    if out_soft is None:
        out_soft = _np.full((S, I, K, 2), _np.nan, dtype=floating_type)

    if out_hard is None:
        out_hard = _np.zeros((S, I), dtype=integral_type)

    if return_bincount:
        if out_bincount is None:
            out_bincount = _np.zeros((2, K + 1), dtype=integral_type)
    else:
        out_bincount = _np.array([], dtype=integral_type).reshape(2, 0)

    # When prange is imported, *args triggers compilation  error, so
    # copied & pasted args in below. The last arg is for specializing
    # which Element type to use.
    if floating_type == _np.float32 and integral_type == _np.int32:
        _get_neighbours[cython.float, cython.int, I32_F32_Element](
            items_corr, supps_items_lens, supps_items_idxs, supps_items_vals,
            K, items_k_th_corrs, return_bincount,
            out_soft, out_hard, out_bincount, num_threads,
            I32_F32_Element(0, 0, 0, 0),
        )
    elif floating_type == _np.float32 and integral_type == _np.int64:
        _get_neighbours[cython.float, cython.long, I64_F32_Element](
            items_corr, supps_items_lens, supps_items_idxs, supps_items_vals,
            K, items_k_th_corrs, return_bincount,
            out_soft, out_hard, out_bincount, num_threads,
            I64_F32_Element(0, 0, 0, 0),
        )
    elif floating_type == _np.float64 and integral_type == _np.int32:
        _get_neighbours[cython.double, cython.int, I32_F64_Element](
            items_corr, supps_items_lens, supps_items_idxs, supps_items_vals,
            K, items_k_th_corrs, return_bincount,
            out_soft, out_hard, out_bincount, num_threads,
            I32_F64_Element(0, 0, 0, 0),
        )
    elif floating_type == _np.float64 and integral_type == _np.int64:
        _get_neighbours[cython.double, cython.long, I64_F64_Element](
            items_corr, supps_items_lens, supps_items_idxs, supps_items_vals,
            K, items_k_th_corrs, return_bincount,
            out_soft, out_hard, out_bincount, num_threads,
            I64_F64_Element(0, 0, 0, 0),
        )
    else:
        # No intention to do type check here, but since no type check,
        # terminate the if-block with the following raise.
        msg = (
            f'Floating must all be either np.float32 or np.float64, and '
            f'integral must all be either np.int32 or np.int64. '
        )
        raise TypeError(msg)

    res = (out_soft, out_hard)

    if return_bincount:
        return res + (out_bincount, )
    else:
        return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _get_neighbours(
        const cython.floating[:,:] items_corr,
        const cython.integral[:] supps_items_lens,
        const cython.integral[:] supps_items_idxs,
        const cython.floating[:] supps_items_vals,
        cython.integral K,
        cython.floating[:] items_k_th_corrs,
        bint return_bincount,
        cython.floating[:,:,:,:] out_soft,
        cython.integral[:,:] out_hard,
        cython.integral[:,:] out_bincount,
        cython.integral num_threads,
        Element redundant, # for specializing fused type only
    ) noexcept nogil:
    cdef:
        cython.integral i_size = sizeof(cython.integral)
        cython.integral S = supps_items_lens.shape[0] # number of Supports
        cython.integral I = items_corr.shape[0] # number of Items
        cython.integral N = items_corr.shape[1] # Number of all items
        cython.integral iS, iI, iK = 0, iN, iL, L, item
        cython.floating corr

        cython.integral* L_base = <cython.integral*> malloc(S * i_size)
        const cython.integral* ptr_supps_items_idxs = NULL
        const cython.floating* ptr_supps_items_vals = NULL

        # Sorting elements. Element's size is 32, Element*'s size is 8
        # Faster to sort Element*
        Element* elements
        Element** ptr_elements

    # initialize arrays
    L_base[0] = 0
    for iS in range(1, S):
        L_base[iS] = L_base[iS - 1] + supps_items_lens[iS - 1]

    # this part may be merged into prange.
    for iS in range(S):
        for iI in range(I):
            out_hard[iS, iI] = 0
            for iK in range(K):
                out_soft[iS, iI, iK, 0] = NAN
                out_soft[iS, iI, iK, 1] = NAN

    for iI in prange(I, num_threads=num_threads, schedule='guided'):
        # THE item iI
        elements = <Element*> malloc(N * sizeof(Element))
        ptr_elements = <Element**> malloc(N * sizeof(Element*))

        for iS in range(S):
            # pointers to locate the data for THE support iS
            L = supps_items_lens[iS]
            ptr_supps_items_idxs = &supps_items_idxs[L_base[iS]]
            ptr_supps_items_vals = &supps_items_vals[L_base[iS]]

            # Sort THE support's items by their corrs to THE primary item
            for iL in range(L):
                item = ptr_supps_items_idxs[iL]
                corr = items_corr[iI, item]
                elements[iL].mask = isnan(corr)
                elements[iL].value = corr
                elements[iL].index = item
                elements[iL].other_value = ptr_supps_items_vals[iL]
                ptr_elements[iL] = &elements[iL]

            sort(ptr_elements, ptr_elements + L, _comp_rev[Element])

            # Fill the outputs with valid values
            for iL in range(min(L, K)): # Need only the first K out of L
                if ptr_elements[iL][0].mask:
                    break # NaN. By sorting, all subsequent elements are NaNs
                else:
                    item = ptr_elements[iL][0].index
                    corr = ptr_elements[iL][0].value
                    out_soft[iS, iI, iL, 0] = item
                    out_soft[iS, iI, iL, 1] = ptr_elements[iL][0].other_value
                    if corr >= items_k_th_corrs[iI]:
                        out_hard[iS, iI] = iL + 1

        free(ptr_elements)
        free(elements)

    # Do bin-count here. Can't do in prange unless lock is used.
    if return_bincount:
        for iK in range(0, K + 1):
            out_bincount[0, iK] = 0
            out_bincount[1, iK] = 0

        for iS in range(S):
            for iI in range(I):
                for iK in range(K):
                    if isnan(out_soft[iS, iI, iK, 0]):
                        break
                    else:
                        iK += 1
                out_bincount[0, iK] += 1
                out_bincount[1, out_hard[iS, iI]] += 1

    return 0


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef int _get_neighbours(
#         const cython.floating[:,:] items_corr,
#         const cython.integral[:] supps_items_lens,
#         const cython.integral[:] supps_items_idxs,
#         const cython.floating[:] supps_items_vals,
#         cython.integral K,
#         cython.floating[:] items_k_th_corrs,
#         cython.floating[:,:,:,:] out_soft,
#         cython.integral[:,:] out_hard,
#         Element redundant, # for specializing fused type only
#     ) noexcept nogil:
#     cdef:
#         cython.integral S = supps_items_lens.shape[0] # number of Supports
#         cython.integral I = items_corr.shape[0] # number of Items
#         cython.integral N = items_corr.shape[1] # Number of all items
#         cython.integral iS, iI, iK, iN, iL, L, L_base, item
#         cython.floating corr

#         const cython.integral* ptr_supps_items_idxs = NULL
#         const cython.floating* ptr_supps_items_vals = NULL

#         # Sorting elements. Element's size is 32, Element*'s size is 8
#         # Faster to sort Element*
#         Element* elements = <Element*> malloc(N * sizeof(Element))
#         Element** ptr_elements = <Element**> malloc(N * sizeof(Element*))

#     # initialize output arrays
#     for iS in range(S):
#         for iI in range(I):
#             out_hard[iS, iI] = 0
#             for iK in range(K):
#                 out_soft[iS, iI, iK, 0] = NAN
#                 out_soft[iS, iI, iK, 0] = NAN

#     for iI in range(I): # THE item iI
#         L_base = 0

#         for iS in range(S):
#             # pointers to locate the data for THE support iS
#             L = supps_items_lens[iS]
#             ptr_supps_items_idxs = &supps_items_idxs[L_base]
#             ptr_supps_items_vals = &supps_items_vals[L_base]
#             L_base += L

#             # Sort THE support's items by their corrs to THE primary item
#             for iL in range(L):
#                 item = ptr_supps_items_idxs[iL]
#                 corr = items_corr[iI, item]
#                 elements[iL].mask = isnan(corr)
#                 elements[iL].value = corr
#                 elements[iL].index = item
#                 elements[iL].other_value = ptr_supps_items_vals[iL]
#                 ptr_elements[iL] = &elements[iL]

#             sort(ptr_elements, ptr_elements + L, _comp_rev[Element])

#             # Fill the outputs with valid values
#             for iL in range(min(L, K)): # Need only the first K out of L
#                 if ptr_elements[iL][0].mask:
#                     break # NaN. By sorting, all subsequent elements are NaNs
#                 else:
#                     item = ptr_elements[iL][0].index
#                     corr = ptr_elements[iL][0].value
#                     out_soft[iS, iI, iL, 0] = item
#                     out_soft[iS, iI, iL, 1] = ptr_elements[iL][0].other_value
#                     if corr >= items_k_th_corrs[item]:
#                         out_hard[iS, iI] = iL + 1

#     free(ptr_elements)
#     free(elements)

#     return 0


# import pandas as pd
# rng = np.random.default_rng(10)

# num_user = 4
# num_item = 8
# num_ratings = 25
# nan_ratio = .3

# items_similarity = rng.random((num_item, num_item), dtype=np.float64).round(2)
# items_similarity[rng.random((num_item, num_item)) < nan_ratio] = np.nan
# user_item_ratings = rng.random((num_user, num_item), dtype=np.float64)

# table = pd.DataFrame({
#     'user_id': rng.integers(0, num_user, size=num_ratings, dtype=np.intp),
#     'item_id': rng.integers(0, num_item, size=num_ratings, dtype=np.intp),
#     'rating': rng.random(size=num_ratings, dtype=np.float64),
# }).drop_duplicates(['user_id', 'item_id']).sort_values(['user_id', 'item_id'])

# supps_item_lens = table.groupby('user_id').size().to_numpy(np.intp)
# users_item_idxs = table['item_id'].to_numpy(np.intp)
# users_item_vals = table['rating'].to_numpy(np.float64)
# K = 5

# argsort = np.argsort(items_similarity)[:, :K]
# items_k_th_corrs = np.nanmin(np.take_along_axis(items_similarity, argsort, axis=1), axis=1)

# out_soft, out_hard = get_neighbours(
#     items_similarity,
#     supps_item_lens,
#     users_item_idxs,
#     users_item_vals,
#     K,
#     items_k_th_corrs,
# )

# df = table.set_index(['user_id', 'item_id'])['rating'].unstack().to_numpy()
# for iS, (s, a) in enumerate(zip(out_soft, df)):
#     s, v = s.T
#     s = s[~np.isnan(s)]
#     v = v[~np.isnan(v)]
#     s = s.astype(int)
#     for iI, (b, c) in enumerate(zip(s, v)):
#         ans = np.array_equal(a[b][~np.isnan(c)], c[~np.isnan(c)], equal_nan=True)
#         print(iS, iI, ans)
