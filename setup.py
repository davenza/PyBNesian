from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import find_opencl

__version__ = '0.1.0'

if sys.platform == 'darwin':
    darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
else:
    darwin_opts = []

# https://stackoverflow.com/questions/49266003/setuptools-build-shared-libary-from-c-code-then-build-cython-wrapper-linked
ext_lib_path = 'lib/libfort'
sources = ['fort.c']
ext_libraries = [('fort', {
               'sources': [os.path.join(ext_lib_path, src) for src in sources],
               'include_dirs': [ext_lib_path],
               'cflags': ['-D_GLIBCXX_USE_CXX11_ABI=0'] + darwin_opts
               })
]

ext_modules = [
    Extension(
        'pybnesian.__init__',
        [
         'pybnesian/lib.cpp',
         'pybnesian/pybindings/pybindings_dataset.cpp',
         'pybnesian/pybindings/pybindings_factors.cpp',
         'pybnesian/pybindings/pybindings_graph.cpp',
         'pybnesian/pybindings/pybindings_models.cpp',
         'pybnesian/pybindings/pybindings_learning/pybindings_learning.cpp',
         'pybnesian/pybindings/pybindings_learning/pybindings_scores.cpp',
         'pybnesian/pybindings/pybindings_learning/pybindings_independences.cpp',
         'pybnesian/pybindings/pybindings_learning/pybindings_parameters.cpp',
         'pybnesian/pybindings/pybindings_learning/pybindings_mle.cpp',
         'pybnesian/pybindings/pybindings_learning/pybindings_operators.cpp',
         'pybnesian/pybindings/pybindings_learning/pybindings_algorithms.cpp',
         'pybnesian/factors/continuous/LinearGaussianCPD.cpp',
         'pybnesian/factors/continuous/CKDE.cpp',
         'pybnesian/factors/discrete/DiscreteFactor.cpp',
         'pybnesian/dataset/dataset.cpp',
         'pybnesian/dataset/dynamic_dataset.cpp',
         'pybnesian/dataset/crossvalidation_adaptator.cpp',
         'pybnesian/dataset/holdout_adaptator.cpp',
         'pybnesian/util/bit_util.cpp',
         'pybnesian/util/validate_options.cpp',
         'pybnesian/util/validate_whitelists.cpp',
         'pybnesian/util/temporal.cpp',
         'pybnesian/util/rpoly.cpp',
         'pybnesian/kdtree/kdtree.cpp',
         'pybnesian/learning/operators/operators.cpp',
         'pybnesian/learning/algorithms/hillclimbing.cpp',
         'pybnesian/learning/algorithms/pc.cpp',
         'pybnesian/learning/algorithms/mmpc.cpp',
         'pybnesian/learning/algorithms/mmhc.cpp',
         'pybnesian/learning/algorithms/dmmhc.cpp',
         'pybnesian/learning/independences/continuous/linearcorrelation.cpp',
         'pybnesian/learning/independences/continuous/mutual_information.cpp',
         'pybnesian/learning/independences/continuous/RCoT.cpp',
         'pybnesian/learning/parameters/mle_LinearGaussianCPD.cpp',
         'pybnesian/learning/parameters/mle_DiscreteFactor.cpp',
         'pybnesian/learning/scores/bic.cpp',
         'pybnesian/learning/scores/bge.cpp',
         'pybnesian/learning/scores/cv_likelihood.cpp',
         'pybnesian/learning/scores/holdout_likelihood.cpp',
         'pybnesian/graph/generic_graph.cpp',
         'pybnesian/models/BayesianNetwork.cpp',
         'pybnesian/models/GaussianNetwork.cpp',
         'pybnesian/models/SemiparametricBN.cpp',
         'pybnesian/models/KDENetwork.cpp',
         'pybnesian/models/DiscreteBN.cpp',
         'pybnesian/models/HomogeneousBN.cpp',
         'pybnesian/models/HeterogeneousBN.cpp',
         'pybnesian/models/DynamicBayesianNetwork.cpp',
         'pybnesian/opencl/opencl_config.cpp'
         ],
        language='c++',
        define_macros=[("VERSION_INFO", __version__)]
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

def path_to_build_folder():
    """Returns the name of a distutils build directory"""
    import sysconfig
    import sys
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    dir_name = f.format(dirname='lib',
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)
    return os.path.join('build', dir_name, 'pybnesian')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def create_options(self):
        import pyarrow as pa
        import numpy as np

        c_opts = {
            'msvc': ['/EHsc', "/std:c++17", "/experimental:external", "/external:W0",
                    "/external:I" + pa.get_include(),
                    "/external:I" + np.get_include(),
                    "/external:Ilib\\eigen-3.3.7",
                    "/external:Ilib\\OpenCL",
                    "/external:Ilib\\boost",
                    "/external:Ilib\\indicators",
                    "-DNOGDI"],
            'unix': ["-std=c++17",
                    "-isystem" + pa.get_include(),
                    "-isystem" + np.get_include(),
                    "-isystemlib/eigen-3.3.7",
                    "-isystemlib/OpenCL",
                    "-isystemlib/boost",
                    "-isystemlib/indicators"]
        }

        l_opts = {
            'msvc': [],
            'unix': [],
        }

        if sys.platform == 'darwin':
            opencl_opts = ["-framework", "OpenCL"]
            c_opts['unix'].extend(darwin_opts)

            l_opts['unix'].extend(darwin_opts)
            l_opts['unix'].extend(opencl_opts)

        return (c_opts, l_opts)

    # Include libraries from https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
    def finalize_options(self):
        import pybind11
        import pyarrow as pa
        build_ext.finalize_options(self)

        if not hasattr(self, 'include_dirs'):
            self.include_dirs = []
        self.include_dirs.append("pybnesian/")
        self.include_dirs.append("lib/libfort")
        self.include_dirs.append(pybind11.get_include())

        if not hasattr(self, 'libraries'):
            self.libraries = []
        if sys.platform != 'darwin':
            self.libraries.append("OpenCL")
        self.libraries.extend(pa.get_libraries())
        
        if not hasattr(self, 'library_dirs'):
            self.library_dirs = []
        self.library_dirs.extend(pa.get_library_dirs())

        if sys.platform == "win32":
            if "CL_LIBRARY_PATH" in os.environ:
                cl_library_path = os.environ["CL_LIBRARY_PATH"]
            else:
                cl_library_path = find_opencl.find_opencl_library_dir()
                if cl_library_path is None:
                    raise RuntimeError("OpenCL library path not found. Set \"CL_LIBRARY_PATH\" environment variable to provide the OpenCL library folder.")

            self.library_dirs.append(cl_library_path)

        if not hasattr(self, 'rpath'):
            self.rpath = []

        if sys.platform == "linux":
            # Use relative RPATH to support out-of-source builds, i.e. pip install .
            # Check https://man7.org/linux/man-pages/man8/ld.so.8.html for the $ORIGIN syntax
            self.rpath.append("$ORIGIN/../pyarrow")

            # Use absolute path so auditwheel and develop builds can find pyarrow.
            self.rpath.extend(pa.get_library_dirs())

    def create_symlinks(self):
        import pyarrow as pa
        pa.create_library_symlinks()

    def expand_sources(self):
        import conv_template

        sources = ['pybnesian/factors/continuous/opencl/CKDE.cl.src']
        
        for source in sources:
            (base, _) = os.path.splitext(source)
            outstr = conv_template.process_file(source)
            with open(base, 'w') as fid:
                fid.write(outstr)

    def copy_opencl_code(self):
        sources = ['pybnesian/factors/continuous/opencl/CKDE.cl']

        # Split the CPP code because the MSVC only allow strings of a max size.
        # Error C2026: https://docs.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026?view=msvc-160
        MAX_LENGTH=16378
        code_str = ""
        for source in sources:
            code_str += '\n'
            with open(source) as f:
                source_code = f.read()
                code_str += source_code

        fragments = [code_str[i:(i + MAX_LENGTH)] for i in range(0, len(code_str), MAX_LENGTH)]

        cpp_code = \
        """#ifndef PYBNESIAN_OPENCL_OPENCL_CODE_HPP
#define PYBNESIAN_OPENCL_OPENCL_CODE_HPP

namespace opencl {
    const std::string OPENCL_CODE = """
    
        for f in fragments:
            cpp_code += 'R"foo({})foo"'.format(f) + "\n"
    
        cpp_code += """;
}
#endif //PYBNESIAN_OPENCL_OPENCL_CODE_HPP"""

        with open('pybnesian/opencl/opencl_code.hpp', 'w') as f:
            f.write(cpp_code)

    def create_clang_tidy_compilation_db(self, extensions):
        db = "[{}\n]"
        template = """
        {{
            "directory": "{0}",
            "file": "{1}",
            "output": "{2}",
            "arguments": ["/usr/lib/llvm-11/bin/clang", "-xc++", "{1}", "-Wno-unused-result", "-Wsign-compare", "-D", "NDEBUG", "-g", "-fwrapv", "-O2", "-Wall", "-g", "-fstack-protector-strong", "-Wformat", "-Werror=format-security", "-g", "-fwrapv", "-O2", "-g", "-fstack-protector-strong", "-Wformat", "-Werror=format-security", "-Wdate-time", "-D", "_FORTIFY_SOURCE=2", "-fPIC", "-D", "VERSION_INFO=0.1.0dev", "-I", "/home/david/cpp/PyBNesian/venv/include", "-I", "/usr/include/python3.8", "-I", "pybnesian/", "-I", "lib/libfort", "-I", "/home/david/cpp/PyBNesian/venv/lib/python3.8/site-packages/pybind11/include", "-c", "-o", "{2}", "-std=c++17", "-isystem", "/home/david/cpp/PyBNesian/venv/lib/python3.8/site-packages/pyarrow/include", "-isystem", "/home/david/cpp/PyBNesian/venv/lib/python3.8/site-packages/numpy/core/include", "-isystem", "lib/eigen-3.3.7", "-isystem", "lib/OpenCL", "-isystem", "lib/boost", "-isystem", "lib/indicators", "-D", "_GLIBCXX_USE_CXX11_ABI=0", "-march=native", "-fdiagnostics-color=always", "-Wall", "-Wextra", "-fvisibility=hidden", "--target=x86_64-pc-linux-gnu"]
        }}"""
        conf_files = []

        import pathlib

        for ext in extensions:
            for s in ext.sources:
                p = pathlib.Path(s)
                relative_path = pathlib.Path(*p.parts[1:-1])

                new_file = pathlib.Path(os.path.splitext(p.parts[-1])[0] + ".o")

                output = pathlib.Path(path_to_build_folder(), relative_path, new_file)
                conf_files.append(
                    template.format(os.getcwd(), s, str(output))
                )

        json = db.format(','.join(conf_files))

        with open('compile_commands.json', 'w') as f:
            f.write(json)

    def build_extensions(self):
        import pyarrow as pa

        self.create_symlinks()
        self.expand_sources()
        self.copy_opencl_code()
        self.create_clang_tidy_compilation_db(self.extensions)

        ct = self.compiler.compiler_type

        c_opts, l_opts = self.create_options()

        opts = c_opts.get(ct, [])
        link_opts = l_opts.get(ct, [])

        if sys.platform == "win32":
            if "CL_INCLUDE_PATH" in os.environ:
                cl_include_path = os.environ["CL_INCLUDE_PATH"]
            else:
                cl_include_path = find_opencl.find_opencl_include_dir()
                if cl_include_path is None:
                    raise RuntimeError("OpenCL include path not found. Set \"CL_INCLUDE_PATH\" environment variable to provide the OpenCL headers folder.")

            opts.append("/external:I" + cl_include_path)

        # Include this because the name mangling affects to find the pyarrow functions.
        opts.append("-D_GLIBCXX_USE_CXX11_ABI=0")

        for ext in self.extensions:
            ext.define_macros.append(("PYARROW_VERSION_INFO", pa.__version__))

        # The compiled extension depends on a specific version of pyarrow.
        self.distribution.install_requires = ['pybind11>=2.6', 'pyarrow=='+pa.__version__, "numpy"],
        self.distribution.setup_requires = ['pybind11>=2.6', 'pyarrow=='+pa.__version__, "numpy"],

        # opts.append("-g")
        # opts.append("-O0")
        # opts.append("-libstd=libc++")
        # opts.append("-ferror-limit=1")

        # opts.append("-Wno-unused-variable")
        # opts.append("-Wno-unused-parameter")
        # opts.append("-Wno-return-type")
        # opts.append("-Wno-sign-compare")
        
        # opts.append("-fsyntax-only")

        # Activate debug mode.
        # opts.append("-UNDEBUG")
        # opts.append("-DDEBUG")

        # This reduces the binary size because it removes the debug symbols. Check strip command to create release builds.
        opts.append("-g0")
        if ct == 'unix':
            opts.append("-march=native")
            opts.append("-fdiagnostics-color=always")
            opts.append("-Wall")
            opts.append("-Wextra")
            # opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(link_opts)

        # https://stackoverflow.com/questions/37752901/dylib-built-on-ci-cant-be-loaded
        # Create the RPATH for MacOSX
        if sys.platform == "darwin":
            for ext in self.extensions:
                ext.extra_link_args.append("-Wl,-rpath,@loader_path/../pyarrow")
                ext.extra_link_args.append("-Wl,-rpath," + pa.get_library_dirs()[0])

        build_ext.build_extensions(self)

        # Copy the pyarrow dlls because Windows do not have the concept of RPATH.
        if sys.platform == "win32":
            for lib in pa.get_libraries():
                import shutil
                shutil.copyfile(pa.get_library_dirs()[0] + '/' + lib + '.dll',
                                path_to_build_folder() + '/' + lib + '.dll')


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pybnesian',
    version=__version__,
    author='David Atienza',
    author_email='datienza@fi.upm.es',
    url='https://github.com/davenza/PyBNesian',
    description='PyBNesian is a Python package that implements Bayesian networks. PyBNesian allows extending its'
                'functionality using Python code, so new research can be easily developed.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    packages=['pybnesian'],
    ext_modules=ext_modules,
    libraries=ext_libraries,
    setup_requires=['pybind11>=2.6', 'pyarrow>=3.0', "numpy"],
    install_requires=['pybind11>=2.6', 'pyarrow>=3.0', "numpy"],
    cmdclass={'build_ext': BuildExt},
    license="MIT",
    zip_safe=False,
)