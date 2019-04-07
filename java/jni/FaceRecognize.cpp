#include"jni.h"
#include"FaceRecognize.h"
#include <opencv2/opencv.hpp>
#include "facerecognize.hpp"
#include "string.h"
#include<android/log.h>
#define LOG "FaceRecognize"

#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG,__VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG,__VA_ARGS__)

/*
 * Class:     FaceRecognize
 * Method:    nativeCreate
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_facerecognize_FaceRecognize_nativeCreate
  (JNIEnv *env, jclass _clazz, jstring moddir)
{
    FaceRecognize*mFaceRecognize;
    const char *pathStr =env->GetStringUTFChars(moddir, NULL);
    LOGE("===>>DIR=%s modelPath:",__FUNCTION__,pathStr);
    float threshold_p=.6f;//Util::String2Double(Config::Instance()->GetValue("Threshold_P"));
    float threshold_r=.6f;//Util::String2Double(Config::Instance()->GetValue("Threshold_R"));
    float threshold_o=.6f;//Util::String2Double(Config::Instance()->GetValue("Threshold_O"));
    int  min_size=80;//Util::String2Int(Config::Instance()->GetValue("MinFaceSize"));

    mFaceRecognize=new FaceRecognize(pathStr,0);
    int ret=mFaceRecognize->Init(threshold_p,threshold_r,threshold_o, 0.5, 0.7,min_size);
    LOGE("FaceRecognize.Init()=%d",ret);
    return (jlong)mFaceRecognize;
}

static FaceRecognize*GET_OBJECT(JNIEnv*env,jobject thiz){
  jclass clazz=env->GetObjectClass(thiz);
  jfieldID fieldID=env->GetFieldID(clazz,"nativeObj","J");
  return (FaceRecognize*)env->GetLongField(thiz,fieldID);
}

static cv::Mat*GET_MAT(JNIEnv*env,jobject thiz){
  jclass clazz=env->GetObjectClass(thiz);
  jfieldID fieldID=env->GetFieldID(clazz,"nativeObj","J");
  return (cv::Mat*)env->GetLongField(thiz,fieldID);
}

static void FaceBox2Object(JNIEnv*env,FaceBox &box,jobject boxobj){
  jfieldID fID;
  jobject arrayObj;
  jfloatArray*array;
  jclass clazz=env->GetObjectClass(boxobj);

#define SET_FLOAT_ARRAY(name,data,size)\
  fID=env->GetFieldID(clazz,name,"[F");\
  arrayObj=env->GetObjectField(boxobj,fID);\
  array=reinterpret_cast<jfloatArray*>(&arrayObj);\
  env->SetFloatArrayRegion(*array,0,size,data);

  float xy[]={box.x0,box.y0,box.x1,box.y1};
  SET_FLOAT_ARRAY("bounds",xy,4);
  SET_FLOAT_ARRAY("regress",box.regress,4); 

  float pads[]={box.px0,box.py0,box.px1,box.py1};
  SET_FLOAT_ARRAY("paddings",pads,4);

  fID=env->GetFieldID(clazz,"score","F");
  env->SetFloatField(boxobj,fID,box.score);

  SET_FLOAT_ARRAY("landmark5",box.landmark,10);
  SET_FLOAT_ARRAY("landmark68",box.landmark68,136);

#undef SET_FLOAT_ARRAY
}
static void Object2FaceBox(JNIEnv*env,jobject boxobj,FaceBox &box){
    jfieldID fID;
    jobject arrayObj;
    jfloatArray*array;
    float data[8];
    jclass clazz=env->GetObjectClass(boxobj);
    
#define GET_FLOAT_ARRAY(name,buf,size)\
    fID=env->GetFieldID(clazz,name,"[F");\
    arrayObj=env->GetObjectField(boxobj,fID);\
    array=reinterpret_cast<jfloatArray*>(&arrayObj);\
    env->GetFloatArrayRegion(*array,0,size,buf);
    
    GET_FLOAT_ARRAY("bounds",data,4);
    box.x0=data[0];  box.y0=data[1];
    box.x1=data[1];  box.y1=data[3];
    
    fID=env->GetFieldID(clazz,"score","F");
    box.score=env->GetFloatField(boxobj,fID);
    GET_FLOAT_ARRAY("regress",box.regress,4);
    
    GET_FLOAT_ARRAY("paddings",data,4);
    box.px0=data[0];  box.py0=data[1];
    box.px1=data[2];  box.py1=data[3];

    GET_FLOAT_ARRAY("landmark5",box.landmark,10);
    GET_FLOAT_ARRAY("landmark68",box.landmark,136);
#undef GET_FLOAT_ARRAY
}
/*
 * Class:     FaceRecognize
 * Method:    detect
 * Signature: (Lorg/opencv/core/Mat;[Lcom/facerecognize/FaceBox;)I
 */
JNIEXPORT jint JNICALL Java_com_facerecognize_FaceRecognize_detect
  (JNIEnv *env, jobject thiz, jobject _frame,jobjectArray boxArray)
{
   FaceRecognize*fc=GET_OBJECT(env,thiz);
   cv::Mat *frame=GET_MAT(env,_frame);
   std::vector<FaceBox>boxes;
   std::cout<<__FUNCTION__<<" fc="<<fc<<std::endl;
   int ret=fc->Detect(*frame,boxes);
   std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
   jclass boxclass=env->FindClass("com/facerecognize/FaceBox");
   jmethodID mid =env->GetMethodID(boxclass, "<init>","()V");
   std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
   for(int i=0;i<ret;i++){
      jobject boxobj;//=env->GetObjectArrayElement(boxArray,i);
      boxobj=env->NewObject(boxclass, mid);
      std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
      FaceBox2Object(env,boxes[i],boxobj);
      std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
      env->SetObjectArrayElement(boxArray,i,boxobj);
      env->DeleteLocalRef(boxobj);
   }
   std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
   return ret;
}
/*
 * Class:     FaceRecognize
 * Method:    getFeature
 * Signature: (Lorg/opencv/core/Mat;Lcom/facerecognize/FaceBox;[F)I
 */
JNIEXPORT jint JNICALL Java_com_facerecognize_FaceRecognize_getFeature
  (JNIEnv *env, jobject thiz, jobject _frame, jobject facebox, jfloatArray feature_)
{
   FaceRecognize*fc=GET_OBJECT(env,thiz);
   cv::Mat *frame=GET_MAT(env,_frame);
   std::vector<float>feature;
   FaceBox cbox;
   Object2FaceBox(env,facebox,cbox);
   int ret=fc->GetFeature(*frame,cbox,feature);
   if(ret>0)
	env->SetFloatArrayRegion(feature_,0,feature.size(),feature.data());
   return feature.size();
}

/*
 * Class:     com_facerecognize_FaceRecognize
 * Method:    setTheshold
 * Signature: (FFF)V
 */
JNIEXPORT void JNICALL Java_com_facerecognize_FaceRecognize_setThreshold
  (JNIEnv *env, jobject thiz, jfloat threshold_p, jfloat threshold_r, jfloat threshold_o)
{
   FaceRecognize*fc=GET_OBJECT(env,thiz);
   fc->SetThreshold(threshold_p,threshold_r,threshold_o);
}
/*
 * Class:     com_facerecognize_FaceRecognize
 * Method:    setFactorMinFace
 * Signature: (FI)V
 */
JNIEXPORT void JNICALL Java_com_facerecognize_FaceRecognize_setFactorMinFace
  (JNIEnv *env, jobject thiz, jfloat factor, jint min_face)
{
  FaceRecognize*fc=GET_OBJECT(env,thiz);
  fc->SetFactorMinFace(factor,min_face);
}
/*
 * Class:     FaceRecognize
 * Method:    matchFeature
 * Signature: ([F[F)F
 */
JNIEXPORT jfloat JNICALL Java_com_facerecognize_FaceRecognize_matchFeature
  (JNIEnv *env, jobject thiz, jfloatArray feature1, jfloatArray feature2)
{
   FaceRecognize*fc=GET_OBJECT(env,thiz);
   float score;
   float f1[256];
   float f2[256];
   int size=std::min(env->GetArrayLength(feature1),fc->GetFeatureLength());
   env->GetFloatArrayRegion(feature1,0,size,f1);
   env->GetFloatArrayRegion(feature2,0,size,f2);
   fc->FaceMatch(f1,f2,&score);
   return score;
}
/*
 * Class:     FaceRecognize
 * Method:    getFeatureSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_facerecognize_FaceRecognize_getFeatureSize
  (JNIEnv *env, jobject thiz)
{
   FaceRecognize*fc=GET_OBJECT(env,thiz);
   return fc->GetFeatureLength();
}

static JNINativeMethod methods[] = {
	{"nativeCreate","(Ljava/lang/String;)J",(void*)Java_com_facerecognize_FaceRecognize_nativeCreate },
	{"detect","(Lorg/opencv/core/Mat;[Lcom/facerecognize/FaceBox;)I",
		(void*)Java_com_facerecognize_FaceRecognize_detect},
	{"getFeature","(Lorg/opencv/core/Mat;Lcom/facerecognize/FaceBox;[F)I",
		(void*)Java_com_facerecognize_FaceRecognize_getFeature },
	{"matchFeature","([F[F)F",(void*)Java_com_facerecognize_FaceRecognize_matchFeature},
	{"getFeatureSize","()I",(void*)Java_com_facerecognize_FaceRecognize_getFeatureSize},
	{"setThreshold","(FFF)V",(void*)Java_com_facerecognize_FaceRecognize_setThreshold},
	{"setFactorMinFace","(FI)V",(void*)Java_com_facerecognize_FaceRecognize_setFactorMinFace}
};


static const char*classPathName="com/facerecognize/FaceRecognize";

int registerFaceRecognizeNatives(JNIEnv* env)
{
	if (!env->RegisterNatives(env->FindClass(classPathName),
	   methods, sizeof(methods) / sizeof(methods[0]))) {
		return JNI_FALSE;
	}
	return JNI_TRUE;
}


