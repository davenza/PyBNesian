from distutils import log
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib
import subprocess
import sys
import setuptools
import os
import find_opencl

__version__ = '0.3.0'

if sys.platform == 'darwin':
    darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.14']
else:
    darwin_opts = []

class CMakeExternalLibrary:
    def __init__(self, cmake_folder, cmake_flags = []):
        self.cmake_folder = cmake_folder

        if not isinstance(cmake_flags, list):
            raise ValueError("cmake_flags must be list of flags.")

        self.cmake_flags = cmake_flags

class Build_CMakeExternalLibrary(build_clib):

    def build_libraries(self, libraries):
        ordinary_libs = []
        cmake_libs = []
        for lib in libraries:
            name, build_info = lib
            cmake_config = build_info.get('cmake_config')
            if cmake_config is None:
                ordinary_libs.append(lib)
            else:
                cmake_libs.append(lib)

        if ordinary_libs:
            super().build_libraries(ordinary_libs)

        for lib in cmake_libs:
            name = lib[0]
            cmake_config = lib[1].get('cmake_config')

            build_directory = os.path.join(self.build_temp, 'build', name)
            install_directory = os.path.join(self.build_temp, name)

            if not os.path.exists(build_directory):
                os.makedirs(build_directory)

            if not os.path.exists(install_directory):
                os.makedirs(install_directory)

            log.info("building CMake '%s' library", name)

            # Run CMake
            subprocess.check_call(
                ["cmake", "-B" + build_directory, "-DCMAKE_INSTALL_PREFIX=" + install_directory] +
                cmake_config.cmake_flags + [cmake_config.cmake_folder]
            )

            # Run make && make install (this command should be multi-platform (Unix, Windows).
            subprocess.check_call(
                ["cmake", "--build", build_directory, "--target", "install", "--config", "Release"]
            )

            # Copy the libraries to self.build_clib
            for name in os.listdir(install_directory):
                # The lib folder can be "lib" or "lib64"
                if "lib" in name:
                    lib_folder = os.path.join(install_directory, name)
                    break

            libraries = [f for f in os.listdir(lib_folder) if os.path.isfile(os.path.join(lib_folder, f))]

            import shutil
            for lf in libraries:
                # Copy all the files to self.build_clib
                shutil.copy(os.path.join(lib_folder, lf), os.path.join(self.build_clib, lf))

# https://stackoverflow.com/questions/49266003/setuptools-build-shared-libary-from-c-code-then-build-cython-wrapper-linked
ext_lib_path = 'lib/libfort'
sources = ['fort.c']
ext_libraries = [('fort', {
               'sources': [os.path.join(ext_lib_path, src) for src in sources],
               'include_dirs': [ext_lib_path],
               'cflags': ['-D_GLIBCXX_USE_CXX11_ABI=0'] + darwin_opts
               }),
               ('nlopt', {
                   'sources': [],
                #    Static linking using -DBUILD_SHARED_LIBS=OFF
                   'cmake_config': CMakeExternalLibrary(os.path.join('lib', 'nlopt-2.7.0'), ["-DBUILD_SHARED_LIBS=OFF"])
                })
]

ext_modules = [
    Extension(
        'pybnesian.__init__',
        [
         'pybnesian/lib.cpp',
         'pybnesian/pybindings/pybindings_dataset.cpp',
         'pybnesian/pybindings/pybindings_kde.cpp',
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
         'pybnesian/kde/KDE.cpp',
         'pybnesian/kde/ProductKDE.cpp',
         'pybnesian/kde/UCV.cpp',
         'pybnesian/factors/continuous/LinearGaussianCPD.cpp',
         'pybnesian/factors/continuous/CKDE.cpp',
         'pybnesian/factors/discrete/DiscreteFactor.cpp',
         'pybnesian/factors/discrete/discrete_indices.cpp',
         'pybnesian/dataset/dataset.cpp',
         'pybnesian/dataset/dynamic_dataset.cpp',
         'pybnesian/dataset/crossvalidation_adaptator.cpp',
         'pybnesian/dataset/holdout_adaptator.cpp',
         'pybnesian/util/bit_util.cpp',
         'pybnesian/util/validate_options.cpp',
         'pybnesian/util/validate_whitelists.cpp',
         'pybnesian/util/temporal.cpp',
         'pybnesian/util/rpoly.cpp',
         'pybnesian/util/vech_ops.cpp',
         'pybnesian/util/pickle.cpp',
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
         'pybnesian/learning/independences/discrete/chi_square.cpp',
         'pybnesian/learning/independences/hybrid/mutual_information.cpp',
         'pybnesian/learning/parameters/mle_LinearGaussianCPD.cpp',
         'pybnesian/learning/parameters/mle_DiscreteFactor.cpp',
         'pybnesian/learning/scores/bic.cpp',
         'pybnesian/learning/scores/bge.cpp',
         'pybnesian/learning/scores/bde.cpp',
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
         'pybnesian/models/CLGNetwork.cpp',
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
                    # Windows creates a build_temp/Release/pybnesian folder structure, so apply a dirname
                    "/external:I" + os.path.join(os.path.dirname(self.build_temp), 'nlopt', 'include'),
                    "-DNOGDI"],
            'unix': ["-std=c++17",
                    "-isystem" + pa.get_include(),
                    "-isystem" + np.get_include(),
                    "-isystemlib/eigen-3.3.7",
                    "-isystemlib/OpenCL",
                    "-isystemlib/boost",
                    "-isystemlib/indicators",
                    # Unix creates a build_temp/pybnesian folder structure.
                    "-isystem" + os.path.join(self.build_temp, 'nlopt', 'include')
                    ]
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
        self.libraries.append("nlopt")
        
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

        sources = ['pybnesian/kde/opencl_kernels/KDE.cl.src']
        
        for source in sources:
            (base, _) = os.path.splitext(source)
            outstr = conv_template.process_file(source)
            with open(base, 'w') as fid:
                fid.write(outstr)

    def copy_opencl_code(self):
        sources = ['pybnesian/kde/opencl_kernels/KDE.cl']

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
            "arguments": ["/usr/lib/llvm-11/bin/clang", "-xc++", "{1}", "-Wno-unused-result", "-Wsign-compare", "-D", "NDEBUG", "-g", "-fwrapv", "-O2", "-Wall", "-g", "-fstack-protector-strong", "-Wformat", "-Werror=format-security", "-g", "-fwrapv", "-O2", "-g", "-fstack-protector-strong", "-Wformat", "-Werror=format-security", "-Wdate-time", "-D", "_FORTIFY_SOURCE=2", "-fPIC", "-D", "VERSION_INFO={3}", "-I", "{4}", "-I", "pybnesian/", "-I", "lib/libfort", "-I", "{5}", "-c", "-o", "{6}", "-std=c++17", "-isystem", "{6}", "-isystem", "{7}", "-isystem", "lib/eigen-3.3.7", "-isystem", "lib/OpenCL", "-isystem", "lib/boost", "-isystem", "lib/indicators", "-D", "_GLIBCXX_USE_CXX11_ABI=0", "-fdiagnostics-color=always", "-Wall", "-Wextra", "-fvisibility=hidden", "--target=x86_64-pc-linux-gnu"]
        }}"""
        conf_files = []

        import pathlib
        import sysconfig
        import pybind11
        import pyarrow as pa
        import numpy as np

        py_include = sysconfig.get_path('include')
        pybind_include = pybind11.get_include()
        pyarrow_include = pa.get_include()
        numpy_include = np.get_include()

        for ext in extensions:
            for s in ext.sources:
                p = pathlib.Path(s)
                relative_path = pathlib.Path(*p.parts[1:-1])

                new_file = pathlib.Path(os.path.splitext(p.parts[-1])[0] + ".o")

                output = pathlib.Path(path_to_build_folder(), relative_path, new_file)
                conf_files.append(
                    template.format(os.getcwd(), s, str(output), __version__, py_include, pybind_include,
                                    pyarrow_include, numpy_include)
                )

        json = db.format(','.join(conf_files))

        with open('compile_commands.json', 'w') as f:
            f.write(json)

    def build_extensions(self):
        import pyarrow as pa

        self.create_symlinks()
        self.expand_sources()
        self.copy_opencl_code()
        # self.create_clang_tidy_compilation_db(self.extensions)

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
            # opts.append("-march=native")
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
    cmdclass={'build_clib': Build_CMakeExternalLibrary, 'build_ext': BuildExt},
    license="MIT",
    zip_safe=False,
)