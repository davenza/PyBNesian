# Installing PyBNesian

Here you can find a detailed installation guide to use PyBNesian including the installation of C++ and GPU tools.

We acknowledge all the members from Computational Intelligence Group (UPM) for
further discussions related to the installation procedure.

### Contents

- [Installing PyBNesian](#installing-pybnesian)
    - [Contents](#contents)
  - [Ubuntu and Linux sub-systems](#ubuntu-and-linux-sub-systems)
    - [Requirements](#requirements)
    - [Installing directly from PyPi](#installing-directly-from-pypi)
    - [Installing from source](#installing-from-source)
  - [Windows](#windows)
  - [Installation issues](#installation-issues)

## Ubuntu and Linux sub-systems
### Requirements
PyBNesian uses C++ and OpenCL in the backend to speed up certain computations.
Thus, some software is required to ensure everything works.
Note that, although setting up a Conda environment is usually recommended, it is not mandatory.
The following commands ensure that the C++ and OpenCL requirements are satisfied.

```bash
sudo apt update
sudo apt install -y --no-install-recommends \
    bison \
    build-essential \
    cmake \
    curl \
    flex \
    g++ \
    git \
    ninja-build \
    ocl-icd-opencl-dev \
    opencl-headers \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    tar \
    unzip \
    zip
```

After the previous steps you should be able to install PyBNesian and its dependencies.

### Installing directly from PyPi
The easiest way to install PyBNesian is using pip. This will install the latest version of the software from PyPi, which is the recommended way to get started.

```bash
pip install PyBNesian
```

### Installing from source
You can also install PyBNesian from source. 
This is recommended if you want to contribute to the development of the software, or if you want to use the latest version of the code.

If you want to pre-compile the C++ code, you can use the following command.
This will create a wheel file in the `dist` folder, which can be used for installation or distribution.

```bash
git clone https://github.com/carloslihu/PyBNesian.git
cd PyBNesian
# pip install . --verbose
pip wheel . -w dist --verbose
pip install dist/pybnesian*.whl
```

If no errors were raised, then the software is ready to be used. Otherwise, please restart the process or raise an issue in the repository.

## Windows

Sometimes, in order to reduce possible inconvenient regarding Windows OS,
a Linux sub-system is installed (https://learn.microsoft.com/es-es/windows/wsl/install).
If this was the case, please go to [Ubuntu and Linux sub-systems](#ubuntu-and-linux-sub-systems) section.
Otherwise, please follow the next steps.

1. Download Visual Studio 2022 from https://visualstudio.microsoft.com/es/vs/

   1.1. Download the requirements for C++

2. Download Visual Studio Build Tools 2022.

```bash
winget install "Visual Studio Build Tools 2022"
```

3. Download developer tools for GPU.

   3.1. For Nvidia, download Nvidia Toolkit (https://developer.nvidia.com/cuda-downloads)

   3.2. For Intel, download OneApi (https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)

4. Download OpenCL for windows. This guide explains the installation process: https://windowsreport.com/opencl-install-windows-11/

5. Install PyBNesian


## Installation issues

1. If default [Ubuntu and Linux sub-systems](#ubuntu-and-linux-sub-systems) installation
   fails, there might be necessary to install GPU toolkits for Linux.
   Please, visit https://developer.nvidia.com/cuda-downloads for Nvidia, and
   https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html for Intel.
