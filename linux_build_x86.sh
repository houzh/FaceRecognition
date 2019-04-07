#!/bin/bash

cd out-linux-x86 

cmake \
    -DCMAKE_INSTALL_PREFIX=/home/houzh/AI/deplibs/linux-x86/facerecognition \
    -DTENGINE_DIR=/home/houzh/AI/deplibs/linux-x86/tengine \
    -DOpenCV_DIR=/home/houzh/AI/deplibs/linux-x86/opencv/lib/cmake/opencv4 \
    -DPROTOBUF_DIR=/home/houzh/AI/deplibs/linux-x86/protobuf \
    -DBLAS_DIR=/home/houzh/AI/deplibs/linux-x86/openblas \
    -DCMAKE_BUILD_TYPE="Debug" \
    .. 

#-DOpenCV_DIR=/home/houzh/AI/deplibs/linux-x86/opencv/share/OpenCV \
