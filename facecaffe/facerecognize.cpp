
#include "facerecognize.hpp"
#include "caffe_mtcnn.hpp"

#include "preprocess.h"
#include "face_align.hpp"
#include "scale_angle.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/types_c.h"
#include <chrono>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include<sys/time.h>

#define UNKNOWN_FACE_ID_MAX 1000

#ifndef CV_BGR2GRAY
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define cvPoint CvPoint
#endif


static bool GreaterSort (FaceBox a, FaceBox b)
{
    return (abs(a.x1 - a.x0) * (a.y1 - a.y0) > abs((b.x1 - b.x0) * (b.y1 - b.y0)));
}

/*****************************************************************************************************/
FaceRecognize::FaceRecognize(const std::string& _dir,int mode){
    pmtcnn = new caffe_mtcnn();
    pmtcnn->load_3model(_dir);
    mfnet=new MobileFacenet(_dir);
    vanface=new VanFace(_dir);
    feature_len=0;
    std::cout<<"mtcnn="<<pmtcnn<<" mfnet="<<mfnet<<" vanface="<<vanface<<std::endl;
#ifndef NDEBUG
    verbose=1;
#endif

}

/*
 *P-net R-net O-net
 *P-net :Proposal Network select face's bound,NMS 
 *R-net :Refine Network
 *O-net :Output Network last step get landmark positions
 *threshold_score:the score used by (face verifier)feature compare
 * */
int FaceRecognize::Init(double theshold_p, double theshold_r, double theshold_o, double threshold_score, double factor, int min_size)
{
    
    pmtcnn->set_threshold(theshold_p, theshold_r, theshold_o);
    pmtcnn->set_factor_min_size(factor, min_size);

    //agc.Init();
    return 0;
}
void FaceRecognize::SetThreshold(float threshold_p,float threshold_r,float threshold_o)
{
    pmtcnn->set_threshold(threshold_p, threshold_r, threshold_o);
}
void FaceRecognize::SetFactorMinFace(float factor,int min_size){
    pmtcnn->set_factor_min_size(factor, min_size);
}

bool FaceRecognize::AlignedFace(Mat&frame,FaceBox &fb,int width,int height,Mat&aligned){
    return get_aligned_face(frame,(float*)&fb.landmark,5,width,height,aligned);
}

void FaceRecognize::GetAgeGender(cv::Mat&frame, FaceBox &b,int*age,int*gender)
{
     float pad=(b.landmark[6]-b.landmark[5]);
     float padlr=pad*1.3f;
     float padtop=pad*2.5f;
     float padbt=pad*1.5f;
     cv::Rect r(b.x0-padlr,b.y0-padtop,b.x1-b.x0+padlr+padlr,b.y1-b.y0+padtop+padbt);
     if( (r.x<0) || (r.x+r.width>frame.cols) || (r.y<0) || (r.y+r.height>frame.rows) )
	  return;
     struct timeval tv_start;
     cv::Mat face(frame,r);
     gettimeofday(&tv_start,NULL);
     //agc.GetAgeGender(face,age,gender); 
     if(verbose){
          struct timeval tv_end;
          gettimeofday(&tv_end, NULL);
	  std::cout<<"GetAgeGender's time:"<<tv_end.tv_sec * 1000 + tv_end.tv_usec / 1000 
			  - tv_start.tv_sec * 1000 - tv_start.tv_usec / 1000<<std::endl;
     }
}

int FaceRecognize::Detect(cv::Mat &frame,std::vector<FaceBox>&boxes,bool landmark68){
    struct timeval tv_start;
    gettimeofday(&tv_start,NULL);
    pmtcnn->detect(frame,boxes);
    if(landmark68 &&boxes.size()>0)
        vanface->GetLandmark(frame,boxes);
    if(verbose){
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);
        std::cout<<"Detect's time:"<<tv_end.tv_sec*1000 + tv_end.tv_usec/1000 - tv_start.tv_sec*1000
	    - tv_start.tv_usec/1000<<std::endl;
    }
    return boxes.size();
}

int FaceRecognize::GetFeature(cv::Mat& frame,FaceBox& fb,std::vector<float>& feature){
    float fbuf[512];
    int rc=mfnet->GetFeature(frame,fb,fbuf);
    feature.resize(rc);
    for(int i=0;i<rc;i++)feature[i]=fbuf[i];
    return rc;
}

int FaceRecognize::GetFeature(cv::Mat& frame,FaceBox &box,float* feature,int fsize){
    return get_feature(frame,box,feature,fsize);
}

int FaceRecognize::get_feature(cv::Mat& frame,FaceBox &box,float* feature,int fsize){
    struct timeval tv_start;
    gettimeofday(&tv_start,NULL);
    int ret=mfnet->GetFeature(frame,box,feature);
    if(ret>0)
	  feature_len=ret;
    if(verbose){
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);
        std::cout<<"GetFeature's time:"<<tv_end.tv_sec*1000 + tv_end.tv_usec/1000 
		- tv_start.tv_sec*1000 - tv_start.tv_usec/1000<<std::endl;
    }
    return ret;
}

void FaceRecognize::FaceMatch(const float*feature1,const float*feature2, float*match_score)
{
     cv::Mat m1(feature_len, 1, CV_32FC1, (void*)feature1), m2(feature_len, 1, CV_32FC1, (void*)feature2);
     *match_score= m1.dot(m2) / cv::norm(m1, CV_L2) / cv::norm(m2, CV_L2);
}


void drawLandmark(cv::Mat frame,FaceBox fb,int start,int num,bool closed){
     std::vector<cv::Point> points;
     for(int i=start,j=0;j<num;i++,j++){
         points.push_back(cv::Point(fb.landmark68[i],fb.landmark68[i+68]));
     }
     cv::polylines(frame,points,closed,Scalar(0,0,255));//,1,8,0);
}

void FaceRecognize::LableFace(Mat&frame,FaceBox& box,Scalar color){
    cv::rectangle(frame,cvPoint(box.x0,box.y0),cvPoint(box.x1,box.y1),color,1);
    for(int i=0;i<5;i++){
      cv::circle(frame,cvPoint(box.landmark[i],box.landmark[i+5]),2,color,-1);
    }
    if(box.landmark68[0]!=.0f&&box.landmark68[1]!=.0f){
        drawLandmark(frame, box, 0, 17, false);
        drawLandmark(frame, box, 17, 5, false);//eyeblow_right
        drawLandmark(frame, box, 22, 5, false);//eyebrow_left
        drawLandmark(frame, box, 27, 4, false);//nose bridge
        drawLandmark(frame, box, 30, 6, true);//nose
        drawLandmark(frame, box, 36, 6, true);//eye_right
        drawLandmark(frame, box, 42, 6, true);//eye_left
        drawLandmark(frame, box, 48, 12,true);//mouse outter
        drawLandmark(frame, box, 60, 8, true);
    }
}

void FaceRecognize::LableFaces(cv::Mat &frame,std::vector<FaceBox>&boxes,Scalar color)
{
   for(size_t i=0;i<boxes.size();i++)
      LableFace(frame,boxes[i],color);
}

