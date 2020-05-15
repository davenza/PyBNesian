# Compile manually
# c++ -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -shared -std=c++11 -fPIC
# `python3 -m pybind11 --includes`
# -I/home/david/cpp/pgm_dataset/venv/lib/python3.6/site-packages/pyarrow/include cpp/data.cpp
# -I/home/david/cpp/pgm_dataset/venv/lib/python3.6/site-packages/numpy/core/include
# -L/home/david/cpp/pgm_dataset/venv/lib/python3.6/site-packages/pyarrow/
# -larrow
# -larrow_python
# -Wl,-rpath,$RPATH -o data`python3-config --extension-suffix`
# Set LD flag
# LD_LIBRARY_PATH=/home/david/cpp/pgm_dataset/venv/lib/python3.6/site-packages/pyarrow
# export LD_LIBRARY_PATH

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

__version__ = '0.0.1'

os.environ['CC'] = "ccache gcc"

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        print("Include pybind11: " + pybind11.get_include(self.user))
        return pybind11.get_include(self.user)

import pyarrow as pa

ext_modules = [
    Extension(
        'pgm_dataset',
        [
         'src/lib.cpp',
         'src/factors/continuous/LinearGaussianCPD.cpp',
         'src/dataset/dataset.cpp',
         'src/util/bit_util.cpp',
         'src/util/validate_dtype.cpp',
         'src/graph/dag.cpp',
         'src/learning/parameter/mle_LinearGaussianCPD.cpp',
         'src/learning/scores/bic.cpp',
         'src/learning/algorithms/hillclimbing.cpp',
         'src/models/GaussianNetwork.cpp',
         ],
        include_dirs=[
        #     # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            # pa.get_include(),
            "src",
            "lib/eigen-3.3.7",
            "lib/graph"
        ],
        libraries=pa.get_libraries(),
        library_dirs=pa.get_library_dirs(),
        language='c++',
        # Included as isystem to avoid errors in arrow headers.
        extra_compile_args=['-isystem' + d for d in
                                [pa.get_include()]
                            ],
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++17 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        # Include this because the name mangling affects to find the pyarrow functions.
        print("extra " + str(opts))
        opts.append("-D_GLIBCXX_USE_CXX11_ABI=0")
        # opts.append("-Werror")
        opts.append("-Wall")
        opts.append("-Wextra")
        opts.append("-Wno-error=unused-variable")
        opts.append("-march=native")
        opts.append("-fdiagnostics-color=always")
        # opts.append("-S")
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            print("Other extra " + str(ext.extra_compile_args))
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(link_opts)
        build_ext.build_extensions(self)

setup(
    name='pgm_dataset',
    version=__version__,
    author='David Atienza',
    author_email='datienza@fi.upm.es',
    # url='https://github.com/pybind/python_example',
    description='A test project using pybind11',
    long_description='',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.4', 'pyarrow'],
    install_requires=['pybind11>=2.4', 'pyarrow'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
