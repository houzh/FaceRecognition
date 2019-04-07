
cmake_minimum_required (VERSION 3.10)
project(FaceRecognition2 C CXX)

set(CMAKE_CXX_STANDARD 11)

set(TENGINE_LIBS tengine protobuf )
#openblas )

aux_source_directory (facecaffe SRCS_FACECAFFE)

#opencv

#set(CMAKE_MODULE_PATH ${OpenCV_DIR})
find_package(OpenCV  REQUIRED)
set(CMAKE_CXX_FLAGS "-std=c++11  -Wall")
#include
include_directories(include 
	${TENGINE_DIR}/include
	${TENGINE_DIR}/wrapper_include
	${PROTOBUF_DIR}/include
	${OpenCV_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}/opencv)

link_directories(
	${TENGINE_DIR}/lib
	${OpenCV_DIR}/lib
        ${BLAS_DIR}/lib
	${PROTOBUF_DIR}/lib
        ${INSTALL_DIR}/lib
	${ACL_ROOT}/lib
	${CMAKE_CURRENT_BINARY_DIR})

if(ANDROID)
  if(("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
   include_directories(${ANDROID_NDK}/sysroot/usr/include/arm-linux-androideabi)
  elseif(("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
   include_directories(${ANDROID_NDK}/sysroot/usr/include/aarch64-linux-android)
  endif()
  add_subdirectory(java)
  set(OpenCV_LIBRARIES opencv_java4)
  set(TENGINE_LIBS ${TENGINE_LIBS} log)
  link_directories(${OpenCV_DIR}/../libs/${ANDROID_ABI})
endif()

message("=====ACL=${CONFIG_ACL_GPU} blas=${CONFIG_ARCH_BLAS}")
if(CONFIG_ACL_GPU)
  set(ACL_LIBS arm_compute_core arm_compute_graph arm_compute)
  link_libraries(${ACL_LIBS})
  link_directories(${ACL_ROOT}/lib)
endif()
if(CONFIG_ARCH_BLAS)
	link_libraries(openblas)
endif()

add_library(recognize STATIC ${SRCS_FACECAFFE})
target_link_libraries(recognize ${TENGINE_LIBS})

add_executable(FaceRecognition2 main.cpp )

if (NOT ANDROID)
   target_link_libraries(FaceRecognition2 pthread ) 
endif()

target_link_libraries(FaceRecognition2 recognize wrapper 
	${TENGINE_LIBS} ${OpenCV_LIBRARIES} ${ACL_LIBS} )

add_executable(ftest face_test.cpp )

#add_executable(ftest facecaffe/age_gender.cpp )
#target_compile_definitions(ftest PUBLIC -D_CAFFE_TEST_)
target_link_libraries(ftest recognize wrapper  ${TENGINE_LIBS} ${OpenCV_LIBRARIES} )

install(TARGETS FaceRecognition2  DESTINATION ./)
install(TARGETS ftest   DESTINATION ./)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/demo.conf DESTINATION ./)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/models DESTINATION ./)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/copy_deps.sh DESTINATION ./)
