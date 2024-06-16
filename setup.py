from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension


import numpy



if __name__ == '__main__':
    extensions = [
        Extension(
            'colabfilter.utils.neighbourhood',
            ['./colabfilter/utils/neighbourhood.pyx'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        ),
        Extension(
            'colabfilter.utils.correlation',
            ['./colabfilter/utils/correlation.pyx'],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        ),
    ]

    setup(
        name='colabfilter',
        ext_modules = cythonize(
            extensions,
            language_level='3',
        ),
        include_dirs=[numpy.get_include()],
    )