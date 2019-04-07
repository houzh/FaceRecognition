#!/bin/bash

cd out-android-armv8 
for i in $*      #在$*中遍历参数，此时每个参数都是独立的，会遍历$#次
do
   if [ $i == 'acl' ]; then
           ACL_CONFIG+="-DCONFIG_ACL_GPU=ON "
   fi

   if [ $i == 'blas' ]; then
	   ACL_CONFIG+="-DCONFIG_ARCH_BLAS=ON "
	   echo "BLAS if only for arm32"
   fi

   if [ $i == 'Release' ]; then
           ACL_CONFIG+='-DCMAKE_BUILD_TYPE="Release"'
   fi
done

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_INSTALL_PREFIX=/home/houzh/AI/deplibs/android-armv8/facerecognition \
    -DTENGINE_DIR=/home/houzh/AI/deplibs/android-armv8/tengine \
    -DOpenCV_DIR=/home/houzh/AI/deplibs/android-armv8/opencv/sdk/native/jni \
    -DPROTOBUF_DIR=/home/houzh/AI/deplibs/android-armv8/protobuf \
    -DBLAS_DIR=/home/houzh/AI/deplibs/android-armv8/openblas \
    -DACL_ROOT=/home/houzh/AI/deplibs/android-armv8/computelibrary \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_STL=c++_shared \
    -DCMAKE_SYSTEM_NAME="Android" \
    -DANDROID_PLATFORM=android-26 \
    ${ACL_CONFIG} \
    .. 
  
#    -DANDROID_STL=c++_shared \
#  -DANDROID_ARM_NEON=ON \
