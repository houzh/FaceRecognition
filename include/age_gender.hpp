#ifndef __AGE_GENDER_H__
#define __AGE_GENDER_H__
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
typedef std::pair<string, float> Prediction;
enum Gender{Male,Female};
class AgeGenderClassifier{
private:
	std::string model_dir;
	cv::Mat mean_;
	int num_channels_;
	cv::Size input_geometry_;
	std::vector<string> labels_[2];
	graph_t graph[2];
	float*input_data[2];
	tensor_t input_tensor[2];
	int enable_age,enable_gender;
        static std::vector<int>Argmax(const std::vector<float>& v, int N);	
	static void WrapInputLayer(float*input_data,int channels,std::vector<cv::Mat>* input_channels);
	void SetMean(const std::string& mean_file);
	void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
	std::vector<float>Predict(int mode,const cv::Mat& Image);
public:
	AgeGenderClassifier();
	int Init(std::string model_dir);
	int GetAgeGender(cv::Mat&frame,int*age,Gender*gender);
	int GetAge(cv::Mat&frame,int&age);
	void enable(int age,int gender);
	bool age_enabled(){return enable_age;}
	bool gender_enabled(){return enable_gender;}
	int GetGender(cv::Mat&frame,Gender&gender);
	std::vector<Prediction>Classify(int mode,const cv::Mat& img, int N=5);
};
#endif
