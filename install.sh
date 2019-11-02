#!/usr/bin/env bash

git clone -b ncsdk2 http://github.com/Movidius/ncsdk
cd ncsdk
# modificar variables de compilacion
./install.sh
cd ..