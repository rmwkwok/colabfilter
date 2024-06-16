from . import typing as _t


import sys as _sys
import time as _time


from .misc import simple_tabulate as _simple_tabulate


class ResultsRecorder:
    '''A recorder and printer of results.

    Examples:
    ```
    import time
    field_fmts = {
        'elapsed_time': '12s',
        'params_id': ' 9d',
        'epochs': ' 6d',
        'train_loss': ' 10.4f',
        'train_metric': ' 12.4f',
    }
    recorder = ResultsRecorder(field_fmts)
    time.sleep(1)
    recorder.update(
        # elapsed_time is always provided by the ResultsRecorder.
        params_id=1,
        epochs=20,
        train_loss=3.2,
        train_metric=1.2,
    )

    # printed:
    # elapsed_time | params_id | epochs | train_loss | train_metric
    # 00:00:01.0   |         1 |     20 |     3.2000 |       1.2000
    ```

    The header's and the first record's column widths are adjusted to
    each other, and the column widths of records can be controlled by
    field_fmts. It is therefore good to use the format-specs to set the
    widths of each column to at least the column header's width. In the
    above example, params_id is 9-character long, so its format-spec is
    ' 9d'. If set to ' 2d' instead, subsequent records' widths will
    not be matched.

    Args:
        field_fmts
            A dictionary of field names and field format-spec.
            Format-spec should comply with
            https://docs.python.org/3/library/string.html#formatspec
            The field "elapsed_time" is always provided, which is
            a string with format "01:23:45.6" indicating elapsed time
            in hours:minutes:seconds.
        print_header
            Whether to print header on the first update.
        skip_repeated
            Whether to skip printing value that is the same as the last
            one in the same field.

    Attributes:
        field_fmts
        field_vals
            The recorded values. It is a list of dictionary, which
            can be easily tabulated with pandas.DataFrame(field_vals).
    '''
    def __init__(
            self,
            field_fmts: _t.Dict[str, str],
            print_header: bool = True,
            skip_repeated: bool = True,
        ) -> None:
        self.field_fmts = field_fmts
        self.field_vals = []
        self._time = _time.time()
        self._print_header = print_header
        self._skip_repeated = skip_repeated
        self._last_vals = [''] * len(field_fmts)

    def update(self, *, do_print: bool = True, **field_vals: _t.Any) -> None:
        '''Create a new record with all keyword arguments as fields,
        except that do_print is reserved for the control of whether
        to print out the new record as specified in field_fmts. Only
        those arguments enlisted in field_fmts will be printed.

        Args:
            do_print
                Whether to print the record.
            **field_vals
                It must contain at least keyword arguments that are
                specified in field_fmts.
        '''
        # Add elapsed time and store to field_vals
        vals = dict(field_vals, elapsed_time=self._elapsed_time())
        self.field_vals.append(vals)

        # Formatting
        vals = [format(vals[key], fmt) for key, fmt in self.field_fmts.items()]

        # Skip repeated
        if self._skip_repeated:
            for i in range(len(vals)):
                if vals[i] == self._last_vals[i]:
                    vals[i] = ' ' * len(vals[i])
                else:
                    self._last_vals[i] = vals[i]
        else:
            self._last_vals = vals

        # Add header
        if self._print_header:
            vals = [
                self.field_fmts.keys(),
                ['-' * len(s) for s in vals],
                vals,
            ]
            self._print_header = False
        else:
            vals = [vals]

        if do_print:
            self._print(vals)

    def print_separator_line(self):
        '''Print a separator line with widths that reference to the
        last printed line.'''
        vals = [['-' * len(s) for s in self._last_vals]]
        self._print(vals)

    def _print(self, vals: _t.List[_t.List[str]]) -> None:
        msg = _simple_tabulate(vals)
        print(' ' * (len(msg) * 2), end='\r')
        print(msg)

    def _elapsed_time(self) -> str:
        s = _time.time() - self._time
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f'{int(h):02d}:{int(m):02d}:{s:04.1f}'

