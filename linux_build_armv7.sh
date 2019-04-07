#!/bin/bash

cd out-linux-armv7 

cmake  -DCMAKE_TOOLCHAIN_FILE=/home/houzh/AI/linux-armv7.cmake \
    -DCMAKE_INSTALL_PREFIX=/home/houzh/AI/deplibs/linux-armv7/facerecognition \
    -DTENGINE_DIR=/home/houzh/AI/deplibs/linux-armv7/tengine \
    -DOpenCV_DIR=/home/houzh/AI/deplibs/linux-armv7/opencv/share/OpenCV \
    -DPROTOBUF_DIR=/home/houzh/AI/deplibs/linux-armv7/protobuf \
    -DBLAS_DIR=/home/houzh/AI/deplibs/linux-armv7/openblas \
    .. 
