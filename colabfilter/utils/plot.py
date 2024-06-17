from . import typing as _t


import contextlib as _contextlib


from itertools import groupby as _groupby
from functools import partial as _partial, wraps as _wraps


import numpy as _np
import pandas as _pd


from matplotlib import pyplot as _plt
from matplotlib.font_manager import (
    findfont as _findfont,
    FontProperties as _FontProperties,
)
from PIL import ImageFont as _ImageFont


@_contextlib.contextmanager
@_wraps(_plt.subplot)
def plt_subplots(*args, **kwargs):
    '''Just matplotlib.pyplot.subplots as a context manager that yields
    the created matplotlib.figure.Figure and the flattened
    matplotlib.axes.Axes.

    It will also call matplotlib.pyplot.tight_layout() and
    matplotlib.pyplot.show() at exit. Any args and kwargs will be passed
    on to matplotlib.pyplot.subplots().
    '''
    try:
        fig, axes = _plt.subplots(*args, **kwargs)
        yield fig, iter(axes.flatten())
    finally:
        _plt.tight_layout()
        _plt.show()


@_contextlib.contextmanager
def plt_pivot_table(
        df: _t.DataFrame,
        subplot_size_in_pixel: _t.Tuple[float, float],
        suptitle_text: str | None = None,
        suptitle_font_size: float | None = None,
        rc_params: _t.Dict[str, _t.Any] | None = None,
        facecolor: _t.FloatingArray | None = None,
    ) -> _t.Generator:
    '''Plot a pivot table of curves. This function is called as a
    context manager, and on exit, it will call matplotlib.pyplot.show().

    This is experimental. For example, it has only been tested with
    cases where all text (including suptitle) are single-lined.

    Args:
        df
            A pivot table DataFrame. This function is responsible for
            drawing the headers and the indices.
        subplot_size_in_pixel
            Width and height in pixels for each cell. Unlike usually
            specifying the size of the whole figure, this function
            requires the size of one cell (i.e. one subplot) in pixels,
            then based on which, font type and font size, calculates the
            figure size.
        suptitle_text, suptitle_font_size
        rc_params
            This function, after called and before exit, is under the
            context matplotlib.pyplot.rc_context(rc_params) that allows
            temporary change of rcParams, such as font.size. See [1]
            for more on rcParams.
        facecolor
            An array of RGBA color values that will be used circularly
            to color the headers and indices. If None, use
            `plt.get_cmap('tab20c')([11, 10])` as default.

    Returns:
        fig
            The matploblib.figure.Figure object for the whole figure
        (fig_w, fig_h)
            The size of the created figure in inches.
        get_subplots_func(**kwargs) -> Iterable[matplotlib.axes.Axes]
            A function that gets the Axes objects.

    [1] https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams
    '''

    # Constants
    subplot_width, subplot_height = subplot_size_in_pixel
    spec_kw = dict(hspace=0., wspace=0., top=1, bottom=0, left=0, right=1)
    num_rows, num_cols = df.shape

    # Texts in header, index, and corner
    text_index = df.index.to_frame().astype(str).to_numpy()
    text_header = df.columns.to_frame().astype(str).to_numpy().T
    text_corner = _np.full(
        (df.columns.nlevels + 1, df.index.nlevels), '', dtype='object'
    )
    text_corner[ -1, :] = df.index.names
    text_corner[:-1, :] = _np.expand_dims(df.columns.names, axis=1)
    text_corner = text_corner.astype(str)

    # Dummy index_corner and header_body for calculating their width-ratio
    body_length = _np.full(df.shape, subplot_width * df.shape[1])
    text_index_corner = _np.concatenate([text_corner, text_index]).astype(str)
    header_body = _np.concatenate([text_header, body_length])

    with _plt.rc_context(rc_params or {}):

        # Convert texts to tuples of id, pixels, spans and text
        index = list(_compute_cell_span(text_index.T, True))
        header = list(_compute_cell_span(text_header, False))
        corner = list(_compute_cell_span(text_corner, False))
        header_body = list(_compute_cell_span(header_body, False))
        index_corner = list(_compute_cell_span(text_index_corner, False))
        index_corner_2 = list(_compute_cell_span(text_index_corner.T, False))

        # Calculate width ratio between 1: index_corner and 2: header
        max_width_1 = _pd.DataFrame(index_corner).groupby(0)[1].sum().max()
        max_width_2 = _pd.DataFrame(header_body).groupby(0)[1].sum().max()
        left_right_ratio = (max_width_1, max_width_2)

        # Calculate width ratio among index levels
        index_corner_ratio = (
            _pd.DataFrame(index_corner_2).groupby(0)[1].max().to_list()
        )

        # top_down_ratio: suptitle, header, body
        font_size = _plt.rcParams['font.size']
        top_down_ratio = (
            suptitle_font_size or font_size, # suptitle
            (df.columns.nlevels + 1) * font_size, # header
            df.shape[0] * subplot_height, # body
        )

        # Starting drawing
        # Create a figure that will be sub-divided into different parts
        # 1.8 is heuristic.
        # TODO: change 1.8 to dynamic value
        fig_w = sum(left_right_ratio) / _plt.rcParams['figure.dpi'] * 1.8
        fig_h = sum(top_down_ratio) / _plt.rcParams['figure.dpi'] * 1.8
        fig = _plt.figure(figsize=(fig_w, fig_h)) # size in inches

        # Divide the upper part for suptitle
        fig_suptitle, fig_table = fig.subfigures(
            2, 1, wspace=0, hspace=0,
            height_ratios=[top_down_ratio[0], sum(top_down_ratio[1:3])],
        )

        # Divide the lower part into different parts of a pivot table
        fig_corner, fig_header, fig_index, fig_body = fig_table.subfigures(
            2, 2, #wspace=0, hspace=0,
            height_ratios=top_down_ratio[1:3], width_ratios=left_right_ratio,
        ).flatten()

        # Draw suptitle:
        if suptitle_text:
            ax = fig_suptitle.subplots(1, 1, gridspec_kw=spec_kw)
            ax.text(0.5, 0.5, suptitle_text, fontsize=suptitle_font_size,
                    transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')

        # Draw table index, header and corner
        shape_index = (num_rows, df.index.nlevels)
        shape_header = (df.columns.nlevels + 1, num_cols)
        shape_corner = (df.columns.nlevels + 1, df.index.nlevels)

        if facecolor is None:
            facecolor = _plt.get_cmap('tab20c')([11, 10])

        index_header_corner = [
            (fig_index, index, shape_index, index_corner_ratio, True),
            (fig_header, header, shape_header, None, True),
            (fig_corner, corner, shape_corner, index_corner_ratio, False),
        ]
        for _fig, _texts, _shape, _ratios, _color in index_header_corner:
            _spec = _fig.add_gridspec(*_shape, width_ratios=_ratios, **spec_kw)
            for i, (_, _, span, text) in enumerate(_texts):
                ax = _fig.add_subplot(_spec[span])
                ax.text(0, 0.5, str(text),
                        transform=ax.transAxes, ha='left', va='center')
                ax.axis('off')
                if _color:
                    # As axis is turned off, need to add a new patch for bkg.
                    ax.add_patch(_plt.Rectangle(
                        (0,0), 1, 1,
                        facecolor=facecolor[int(i % len(facecolor))],
                        transform=ax.transAxes,
                        zorder=-1
                    ))

        def get_subplots_func(**kwargs) -> _t.Iterable[_t.MPLAxes]:
            '''Call this to make matplotlib.axes.Axes objects for
            manually programming of plotting of curves in `df`. The
            returned Axes is an array with a shape of df.shape. Any
            kwargs are passed on to matplotlib.figure.Figure.subplots().
            '''
            axes = fig_body.subplots(*df.shape, gridspec_kw=spec_kw, **kwargs)
            for ax in axes.flatten():
                ax.set_xticks([], minor=True)
                ax.set_xticks([], minor=False)
                ax.set_yticks([], minor=True)
                ax.set_yticks([], minor=False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
            return axes

        try:
            yield fig, (fig_w, fig_h), get_subplots_func

        finally:
            _plt.show()


def _compute_cell_span(array, transpose: bool):
    for i, row in enumerate(array):
        j = 0
        for val, group in _groupby(row):
            span = slice(j, j := j + len(list(group)))
            if transpose:
                yield i, _length_in_pixel(val), (span, i), val
            else:
                yield i, _length_in_pixel(val), (i, span), val


def _length_in_pixel(val):
    return (
        val if isinstance(val, int) else
        _ImageFont.truetype(
            _findfont(_FontProperties(_plt.rcParams['font.family'])),
            _plt.rcParams['font.size'],
        ).getlength(val)
    )
