# from typing import Callable, Tuple
# from numpy import typing as npt


import sys
sys.path.append('../')


import os
import unittest
import tempfile
import contextlib
from itertools import product


import numpy as np
import pandas as pd
import tables as tb


from colabfilter import config, data_processing as dp


class DataProcessTestCaseBase:

    @contextlib.contextmanager
    def env(self, *, seed: int) -> None:
        # synthetic data
        rng = np.random.default_rng(seed)

        source_df = (
            pd
            .DataFrame({
                'userId': rng.integers(0, 10, size=50),
                'itemId': rng.integers(0, 10, size=50),
                'rating': rng.random(size=50).astype(config.default_float),
            })
            .drop_duplicates(['userId', 'itemId'])
        )

        empty_array = np.array(
            [], dtype=config.default_float).reshape(0, 2)
        interaction_user_item = (
            source_df
            .groupby('userId')[['itemId', 'rating']]
            .apply(np.array)
            .reindex(np.arange(10))
            .apply(lambda x: x if isinstance(x, np.ndarray) else empty_array)
            .to_list()
        )
        interaction_item_user = (
            source_df
            .groupby('itemId')[['userId', 'rating']]
            .apply(np.array)
            .reindex(np.arange(10))
            .apply(lambda x: x if isinstance(x, np.ndarray) else empty_array)
            .to_list()
        )

        with tempfile.TemporaryDirectory() as data_folder:
            # Config
            config.path_data_folder = f'{data_folder}'
            config.show_pbar = False

            with tb.File(config.path_pytables, 'w') as t:
                table = t.create_table('/', 'table', obj=source_df.to_records())
                table.colinstances['userId'].create_index(optlevel=9, kind='full')

                atom = tb.Atom.from_type(
                    np.dtype(config.default_float).name,
                    shape=(2, )
                )
                inter = t.create_vlarray('/', 'user_item', atom=atom)
                for x in interaction_user_item:
                    inter.append(x)

                inter = t.create_vlarray('/', 'item_user', atom=atom)
                for x in interaction_item_user:
                    inter.append(x)

            try:
                yield source_df
            finally:
                pass

class InteractionFromTableTestCase(DataProcessTestCaseBase, unittest.TestCase):

    def test_with_field_2(self) -> None:
        with self.env(seed=10) as source_df:
            sort_by_max = 19
            empty_array = np.array(
                [], dtype=config.default_float).reshape(0, 2)

            # Answers
            ans_interaction = (
                source_df
                .groupby('userId')[['itemId', 'rating']]
                .apply(np.array)
                .reindex(np.arange(sort_by_max + 1))
                .apply(lambda x: x if isinstance(x, np.ndarray) else empty_array)
                .to_list()
            )
            ans_bincount = np.bincount(list(map(len, ans_interaction)))

            # Outputs
            out_bincount = dp.interaction_from_table(
                config.path_pytables,
                source_table='/table',
                target_array='/target_array',
                sort_by='userId',
                field_1='itemId',
                field_2='rating',
                sort_by_max=sort_by_max,
            )
            with tb.File(config.path_pytables, 'r') as t:
                out_interaction = t.root['/target_array'][:]

            # Tests
            np.testing.assert_array_equal(ans_bincount, out_bincount)
            self.assertEqual(len(ans_interaction), len(out_interaction))
            for ans_inter, out_inter in zip(ans_interaction, out_interaction):
                # sort them in the same way
                out_inter = out_inter[np.argsort(out_inter[:, 0])]
                ans_inter = ans_inter[np.argsort(ans_inter[:, 0])]
                np.testing.assert_array_equal(ans_inter, out_inter)

    def test_without_field_2(self) -> None:
        with self.env(seed=10) as source_df:
            sort_by_max = 19
            empty_array = np.array(
                [], dtype=config.default_float).reshape(0, 1)

            # Answers
            ans_interaction = (
                source_df
                .groupby('userId')[['itemId', ]]
                .apply(np.array)
                .reindex(np.arange(sort_by_max + 1))
                .apply(lambda x: x if isinstance(x, np.ndarray) else empty_array)
                .to_list()
            )
            ans_bincount = np.bincount(list(map(len, ans_interaction)))

            # Outputs
            out_bincount = dp.interaction_from_table(
                config.path_pytables,
                source_table='/table',
                target_array='/target_array',
                sort_by='userId',
                field_1='itemId',
                field_2=None,
                sort_by_max=sort_by_max,
            )
            with tb.File(config.path_pytables, 'r') as t:
                out_interaction = t.root['/target_array'][:]

            # Tests
            np.testing.assert_array_equal(ans_bincount, out_bincount)
            self.assertEqual(len(ans_interaction), len(out_interaction))
            for ans_inter, out_inter in zip(ans_interaction, out_interaction):
                # sort them in the same way
                out_inter = out_inter[np.argsort(out_inter[:, 0])]
                ans_inter = ans_inter[np.argsort(ans_inter[:, 0])]
                np.testing.assert_array_equal(ans_inter, out_inter)


class CorrFromInteractionTestCase(DataProcessTestCaseBase, unittest.TestCase):

    def test_parameters(self) -> None:
        with (
                # pd.DataFrame.Corr only supports np.float64
                config(default_float=np.float64),
                self.env(seed=10) as source_df
            ):
            for min_support, K_th, self_correlation in product(
                    [3, 4, 5], # min_support
                    [1, 2, 3], # K_th
                    [None, np.nan, 1.] # self_correlation
                ):
                # Answer
                ans_corr = (
                    source_df
                    .set_index(['userId', 'itemId'])
                    .reindex(
                        pd.MultiIndex.from_product([np.arange(10), np.arange(10)])
                    )
                    ['rating']
                    .unstack()
                    .corr(min_periods=min_support)
                    .to_numpy()
                    .astype(config.default_float)
                )
                if self_correlation is not None:
                    for i in range(len(ans_corr)):
                        ans_corr[i, i] = self_correlation

                ans_k_th = np.fmin.reduce(
                    np.take_along_axis(
                        ans_corr,
                        np.argsort(-ans_corr, axis=1)[:,:K_th],
                        axis=1,
                    ),
                    axis=1,
                )
                ans_bincount = np.bincount((~np.isnan(ans_corr)).sum(axis=1))

                # Outputs
                out_bincount = dp.corr_from_interaction(
                    config.path_pytables,
                    source_item_supp='/item_user',
                    source_supp_item='/user_item',
                    target_corr_array='/corr',
                    target_k_th_corr_array='/corr_kth',
                    min_support=min_support,
                    K_th=K_th,
                    self_correlation=self_correlation,
                )
                with tb.File(config.path_pytables, 'r') as t:
                    out_corr = t.root['/corr'][:]
                    out_k_th = t.root['/corr_kth'][:]

                err_msg = f'{min_support}, {K_th}, {self_correlation}'
                np.testing.assert_allclose(ans_k_th, out_k_th, err_msg=err_msg)
                np.testing.assert_allclose(ans_corr, out_corr, err_msg=err_msg)
                np.testing.assert_array_equal(ans_bincount, out_bincount, err_msg=err_msg)


class NeighboursTestCase(DataProcessTestCaseBase, unittest.TestCase):

    def test1(self) -> None:
        with self.env(seed=10) as source_df:
            from numpy import nan
            K_th = 3
            supp_item = [
                np.array([[ 0, 10.], [ 1, 11.], [ 4, 14.], [ 5, 15.]]),
                np.array([[ 0, 16.], [ 2, 18.], [ 3, 19.]]),
                np.array([[ 1, 23.], [ 3, 25.]]),
                np.array([[ 0, 28.], [ 1, 29.], [ 3, 31.]]),
            ]
            item_corr = np.array([
                [nan, nan, 0.5, 0.2, 0.1, 0.3],
                [nan, nan, 0.3, 0.2, 0.2, 0.1],
                [0.5, 0.3, nan, 0.9, 0.4, nan],
                [0.2, 0.2, 0.9, nan, nan, 0.4],
                [0.1, 0.2, 0.4, nan, nan, 0.6],
                [0.3, 0.1, nan, 0.4, 0.6, nan],
            ], dtype=config.default_float)
            item_k_th_corr = np.array(
                [0.2, 0.2, 0.4, 0.2, 0.2, 0.3], dtype=config.default_float)

            # Answers
            ans_soft = np.array([
               [[[ 5., 15.], [ 4., 14.], [nan, nan]],
                [[ 4., 14.], [ 5., 15.], [nan, nan]],
                [[ 0., 10.], [ 4., 14.], [ 1., 11.]],
                [[ 5., 15.], [ 0., 10.], [ 1., 11.]],
                [[ 5., 15.], [ 1., 11.], [ 0., 10.]],
                [[ 4., 14.], [ 0., 10.], [ 1., 11.]]],
               [[[ 2., 18.], [ 3., 19.], [nan, nan]],
                [[ 2., 18.], [ 3., 19.], [nan, nan]],
                [[ 3., 19.], [ 0., 16.], [nan, nan]],
                [[ 2., 18.], [ 0., 16.], [nan, nan]],
                [[ 2., 18.], [ 0., 16.], [nan, nan]],
                [[ 3., 19.], [ 0., 16.], [nan, nan]]],
               [[[ 3., 25.], [nan, nan], [nan, nan]],
                [[ 3., 25.], [nan, nan], [nan, nan]],
                [[ 3., 25.], [ 1., 23.], [nan, nan]],
                [[ 1., 23.], [nan, nan], [nan, nan]],
                [[ 1., 23.], [nan, nan], [nan, nan]],
                [[ 3., 25.], [ 1., 23.], [nan, nan]]],
               [[[ 3., 31.], [nan, nan], [nan, nan]],
                [[ 3., 31.], [nan, nan], [nan, nan]],
                [[ 3., 31.], [ 0., 28.], [ 1., 29.]],
                [[ 0., 28.], [ 1., 29.], [nan, nan]],
                [[ 1., 29.], [ 0., 28.], [nan, nan]],
                [[ 3., 31.], [ 0., 28.], [ 1., 29.]]]],
            dtype=config.default_float)

            ans_hard = np.array([
                [1., 1., 2., 3., 2., 2.],
                [2., 2., 2., 2., 1., 2.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 2., 2., 1., 2.],
            ], dtype=config.default_int)

            ans_soft_counts = (~np.isnan(ans_soft[:, :, :, 0])).sum(axis=-1)
            ans_soft_counts = np.bincount(ans_soft_counts.flatten())
            ans_hard_counts = np.bincount(ans_hard.flatten().astype(int))

            # Prepare input data
            with tb.File(config.path_pytables, 'w') as f:
                atom = tb.Atom.from_type(
                    np.dtype(config.default_float).name,
                    shape=(2, )
                )
                inter = f.create_vlarray('/', 'supp_item', atom=atom)
                for x in supp_item:
                    inter.append(x)

                array = f.create_carray('/', 'item_corr', obj=item_corr)
                array = f.create_carray('/', 'item_k_th_corr', obj=item_k_th_corr)
                array._f_setattr('K_th', K_th)

            # Outpus
            out_soft_counts, out_hard_counts = dp.neighbours(
                config.path_pytables,
                source_item_corr='/item_corr',
                source_item_k_th_corr='/item_k_th_corr',
                source_supp_item='/supp_item',
                target_soft_array='/soft',
                target_hard_array='/hard',
            )
            with tb.File(config.path_pytables) as f:
                out_soft = f.root['/soft'][:]
                out_hard = f.root['/hard'][:]

            np.testing.assert_array_equal(ans_soft, out_soft)
            np.testing.assert_array_equal(ans_hard, out_hard)
            np.testing.assert_array_equal(ans_soft_counts, out_soft_counts)
            np.testing.assert_array_equal(ans_hard_counts, out_hard_counts)
