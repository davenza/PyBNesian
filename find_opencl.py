import os
import platform
# This code is based on FindOpenCL.cmake in https://github.com/Kitware/CMake/blob/master/Modules/FindOpenCL.cmake

opencl_env = ["PROGRAMFILES(X86)", "AMDAPPSDKROOT", "INTELOCLSDKROOT", "NVSDKCOMPUTE_ROOT", "CUDA_PATH", "ATISTREAMSDKROOT", "OCL_ROOT"]

def find_files(folder, search_files):
    for s in search_files:
        include_file = os.path.join(folder, s)
        if os.path.isfile(include_file):
            return True
    return False


def find_opencl_include_dir():

    search_files = ["CL/cl.h", "OpenCL/cl.h"]
    suffix = ["include", "OpenCL/common/inc", "AMD APP/include"]

    for env in opencl_env:
        if env in os.environ:
            path = os.environ[env]

            if find_files(path, search_files):
                return path

            for suf in suffix:
                suf_path = os.path.join(path, suf)
                if find_files(suf_path, search_files):
                    return suf_path

    return None

def find_opencl_library_dir():
    search_files = ["OpenCL.lib"]

    if platform.architecture()[0] == "32bit":
        suffix = ["AMD APP/lib/x86", "lib/x86", "lib/Win32", "OpenCL/common/lib/Win32"]
    elif platform.architecture()[0] == "64bit":
        suffix = ["AMD APP/lib/x86_64", "lib/x86_64", "lib/x64", "OpenCL/common/lib/x64"]


    for env in opencl_env:
        if env in os.environ:
            path = os.environ[env]

            if find_files(path, search_files):
                return path

            for suf in suffix:
                suf_path = os.path.join(path, suf)
                if find_files(suf_path, search_files):
                    return suf_path

    return None