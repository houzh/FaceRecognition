#ifndef __FACE_RECOGNIZE__HPP__
#define __FACE_RECOGNIZE__HPP__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <unistd.h>
#include <signal.h>

#include "mtcnn.hpp"
#include "face_align.hpp"
#include "mobilefacenet.hpp"
#include "vanface.hpp"
#include "tengine_c_api.h"

#define DEBUG 0

using namespace cv;

class FaceRecognize{
    public:
        FaceRecognize(const std::string&_model_dir,int mode=0);
        int Init(double threshold_p, double threshold_r, double threshold_o, double threshold_score, double factor, int mim_size);
	int Detect(cv::Mat &frame,std::vector<FaceBox> &face_info,bool landmark68=false);
	int GetFeature(cv::Mat& frame,FaceBox &box,std::vector<float>& feature);
	int GetFeature(cv::Mat& frame,FaceBox &box,float* feature,int fsize);
        
	void GetAgeGender(cv::Mat& frame,FaceBox& box,int *age,int*gender);
	void FaceMatch(const float*feature1,const float*feature2, float*matchscore);
	void LableFace(Mat&frame,FaceBox &box,Scalar color=Scalar(255,255,255));
	void LableFaces(cv::Mat &frame,std::vector<FaceBox>&boxes,Scalar color=Scalar(255,255,255));
	int GetFeatureLength(){return feature_len;}
	void SetVerbose(int v=1){verbose=v;}
	
	void SetThreshold(float p_threshold,float r_threshold,float o_threshold);
	void SetFactorMinFace(float factor,int min_size);
	bool AlignedFace(Mat&frame,FaceBox &fb,int width,int height,Mat&aligned);
    protected:
        void get_data(float* input_data, Mat &gray, int img_h, int img_w);
	int get_feature(cv::Mat& frame,FaceBox &box,float* feature,int fsize);

        mtcnn * pmtcnn = nullptr;
	MobileFacenet*mfnet=nullptr;
	VanFace*vanface=nullptr;
	int feature_len;
	int verbose;
};

#endif
