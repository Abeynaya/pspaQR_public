#!/bin/bash

###
# CMake requires a clean reconfigure anytime these configs change
###

BUILD_TYPE=Release

SOFTROOT="${HOME}/Softwares"
SPAQR_USE_MKL=ON
SPAQR_USE_METIS=ON
SPAQR_USE_HSL=OFF
###

# Find generator to use
GENERATOR="Unix Makefiles"
if ! [ -x "$(command -v ninja)" ]; then
    echo 'Info: ninja is not installed.'
    GENERATOR="Unix Makefiles"
fi

mkdir -p build
mkdir -p build/ttor
cd build

cmake .. \
    -G "$GENERATOR" \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -DTASKTORRENT_ROOT_DIR=$SOFTROOT/tasktorrent/ \
    -DSOFTROOT=$SOFTROOT \
    -DSPAQR_USE_MKL=$SPAQR_USE_MKL \
    -DSPAQR_USE_METIS=$SPAQR_USE_METIS \
    -DSPAQR_USE_HSL=$SPAQR_USE_HSL \
    -DTTOR_SHARED=OFF \
    -DTTOR_USE_HWLOC=OFF \
    -DMETIS_INC_DIR=$SOFTROOT/metis/include \
    -DMETIS_LIB=$SOFTROOT/metis/build/Darwin-x86_64/libmetis/libmetis.a \
    # Change Darwin-x86_64 to system name
    

if [[ $GENERATOR == "Ninja" ]]
then
    ninja
else
    make -j 2
fi