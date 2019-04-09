#ifndef __CAFFE_MTCNN_HPP__
#define __CAFFE_MTCNN_HPP__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "mtcnn.hpp"
#include "comm_lib.hpp"
#include "tengine_c_api.h"

class caffe_mtcnn: public mtcnn {

    public:
        caffe_mtcnn()=default;

        int load_3model(const std::string& model_dir);

        void detect(cv::Mat& img, std::vector<FaceBox>& face_list);

        ~caffe_mtcnn();

    protected:
        void copy_one_patch(const cv::Mat& img,FaceBox&input_box,float * data_to, int width, int height);

        int run_PNet(const cv::Mat& img, scale_window& win, std::vector<FaceBox>& box_list);
        void run_RNet(const cv::Mat& img,std::vector<FaceBox>& pnet_boxes, std::vector<FaceBox>& output_boxes);
        void run_ONet(const cv::Mat& img,std::vector<FaceBox>& rnet_boxes, std::vector<FaceBox>& output_boxes);

    private:
        graph_t graph[3];
        tensor_t input_tensor[3];
	float*input_data[3];
	int input_size[3];
};


#endif
