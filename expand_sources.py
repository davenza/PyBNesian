import os
import conv_template


def expand_sources():

    sources = ['pybnesian/kde/opencl_kernels/KDE.cl.src']
    
    for source in sources:
        (base, _) = os.path.splitext(source)
        outstr = conv_template.process_file(source)
        with open(base, 'w') as fid:
            fid.write(outstr)


def copy_opencl_code():
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


if __name__ == "__main__":
    expand_sources()
    copy_opencl_code()