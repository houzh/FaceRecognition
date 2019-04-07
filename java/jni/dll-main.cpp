#include "jni.h"
#include "FaceRecognize.h"
#include"android_buf.hpp"
#include "caffe_mtcnn.hpp"
#include<android/log.h>

#define LOG "FaceRecognize"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG,__VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG,__VA_ARGS__)

int registerFaceRecognizeNatives(JNIEnv* env);

JNIEXPORT int JNI_OnLoad(JavaVM* vm, void* reserved){
    for(int i=0;i<3;i++)
	 LOGE("%s:%d",__FUNCTION__,__LINE__);
    JNIEnv* env = NULL;
    std::cout.rdbuf(new AndroidBuf);
    std::cerr.rdbuf(new AndroidBuf);
    jint result = -1;
    LOGI(__FUNCTION__);
    init_tengine_library();
    LOGE("%s:%d",__FUNCTION__,__LINE__);
    if (request_tengine_version("0.1") < 0)
        return -1;
    LOGE("%s:%d",__FUNCTION__,__LINE__);

    if (vm->GetEnv((void**) &env, JNI_VERSION_1_4) != JNI_OK) {
        LOGE("%s:%d",__FUNCTION__,__LINE__);
        return -1;
    }
    LOGE("%s:%d",__FUNCTION__,__LINE__);
    if(registerFaceRecognizeNatives(env)!=JNI_OK){
	LOGE("registerFaceRecognizeNatives error");
        return -1;
    }
    result = JNI_VERSION_1_4;
    return result;
}
