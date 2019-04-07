#ifndef __MOBILE_FACENET_H__
#define __MOBILE_FACENET_H__
#include "tengine_c_api.h"
#include "mtcnn.hpp"
#include <opencv2/opencv.hpp>
class MobileFacenet{
private:
    graph_t graph;
    tensor_t input_tensor;
    tensor_t out_tensor;
    float*input_data;
public:
    MobileFacenet(const std::string&dir);
    ~MobileFacenet();
    int GetFeature(const cv::Mat&frame,FaceBox&box,float*feature);
};
#endif

