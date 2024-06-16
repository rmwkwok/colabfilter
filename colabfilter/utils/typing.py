'''An experiment to put all typing-related things in one file.
Non-Python-native, except for typing-specific (e.g. numpy.typing),
modules are not imported here.
'''


from typing import *
from typing import types
from typing_extensions import *


import dataclasses as _dataclasses


from enum import Enum
from inspect import Signature
from logging import Logger
from collections.abc import KeysView


from numpy import typing as npt
from numpy import (
    int32 as np_int32, int64 as np_int64,
    float32 as np_float32, float64 as np_float64,
)


Str = str
Int = int
Bool = bool
Float = float


Array = npt.NDArray
IntegralArray = npt.NDArray[np_int32 | np_int64]
FloatingArray = npt.NDArray[np_float32 | np_float64]


DataFrame = 'pandas.DataFrame'
TFTensor = 'tensorflow.Tensor'
TFDataset = 'tensorflow.data.Dataset'
TFVariable = 'tensorflow.Variable'
TFTrainExample = 'tensorflow.train.Example'
TFTensorSpec = 'tensorflow.TensorSpec'
MPLAxes = 'matplotlib.axes.Axes'


@_dataclasses.dataclass
class ValueRange:
    lo: float | str | None
    hi: float | str | None
    lo_inclusive: bool = False
    hi_inclusive: bool = False

    def __repr__(self):
        left1 = ('>', '>=')[self.lo_inclusive]
        left2 = (f'{left1} {self.lo}', '')[self.lo is None]

        right1 = ('<', '<=')[self.hi_inclusive]
        right2 = (f'{right1} {self.hi}', '')[self.hi is None]

        middle = ('', ' and ')[self.lo is not None and self.hi is not None]
        return f'range: {left2}{middle}{right2}'

    def __hash__(self):
        return hash((self.lo, self.hi, self.lo_inclusive, self.hi_inclusive))
