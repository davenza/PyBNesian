#!/usr/bin/env
# workon aiTanium-master # This doesn't work
export CC="ccache gcc"
pip uninstall pybnesian -y
# python setup.py clean --all
# rm -rf build/temp.linux-x86_64-cpython-310/pybnesian/
rm -rf build/lib.linux-x86_64-cpython-310


time python setup.py install
# python setup.py develop # For verbose output
# export CC="ccache clang-14"

# export LDSHARED="clang-14 -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2" # NOTE: Ignored?
# source venv/bin/activate