#ifndef __VAN_FACE_H__
#define __VAN_FACE_H__
#include "tengine_c_api.h"
#include "mtcnn.hpp"
#include <opencv2/opencv.hpp>
class VanFace{
private:
    graph_t graph;
    tensor_t input_tensor;
    tensor_t out_tensor;
    float*input_data;
public:
    VanFace(const std::string&dir);
    ~VanFace();
    int GetLandmark(const cv::Mat&frame,FaceBox&box);
    int GetLandmark(const cv::Mat&frame,std::vector<FaceBox>&boxes);
};
#endif

