cmake_minimum_required(VERSION 3.4.1) #指定cmake版本

project(tengine-face C CXX)

set(command_javah "javah")
message("javah=${command_javah}")
set(class_path ${CMAKE_CURRENT_SOURCE_DIR}:${OpenCV_DIR}/../../java/src:)

message("===>> class_path=${class_path}")
add_custom_command(OUTPUT FaceRecognize.h
	PRE_BUILD 
	COMMAND javah -o FaceRecognize.h -classpath "${class_path}" com.facerecognize.FaceRecognize
                   )
aux_source_directory(jni SRCS_JNI)


find_package(OpenCV REQUIRED)
set(OPENCV_JAVA ${OpenCV_DIR}/../libs/${ANDROID_ABI})
include_directories(
	${CMAKE_CURRENT_BINARY_DIR}
	${OpenCV_INCLUDE_DIRS}
	../include
	     )

link_directories(
	${PROTOBUF_DIR}/lib
	${TENGINE_DIR}/lib
	)

if(ANDROID)
   link_directories(${OpenCV_DIR}/../libs)
   link_directories(${OPENCV_JAVA})
   set(OpenCV_LIBS opencv_java4)
endif()

add_library(tengine-face  
             SHARED
	     ${SRCS_JNI} 
	     FaceRecognize.h)

target_link_libraries(
            tengine-face
            android log 
	    recognize
	    wrapper
	    tengine
	    protobuf
	    ${OpenCV_LIBS} 
	    )

install(TARGETS tengine-face  DESTINATION lib)


