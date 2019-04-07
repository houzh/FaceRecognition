#!/bin/bash

cd out-linux-armv8 

cmake  -DCMAKE_TOOLCHAIN_FILE=/home/houzh/AI/linux-armv8.cmake \
    -DCMAKE_INSTALL_PREFIX=/home/houzh/AI/deplibs/linux-armv8/facerecognition \
    -DTENGINE_DIR=/home/houzh/AI/deplibs/linux-armv8/tengine \
    -DOpenCV_DIR=/home/houzh/AI/deplibs/linux-armv8/opencv/lib/cmake/opencv4 \
    -DPROTOBUF_DIR=/home/houzh/AI/deplibs/linux-armv8/protobuf \
    -DBLAS_DIR=/home/houzh/AI/deplibs/linux-armv8/openblas \
    -DCMAKE_BUILD_TYPE="Debug" \
    .. 
