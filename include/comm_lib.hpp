#ifndef __COMMON_LIB_HPP__
#define __COMMON_LIB_HPP__

#define NMS_UNION 1
#define NMS_MIN  2



struct scale_window
{
	int h;
	int w;
	float scale;
};

int numpy_round(float f);

void nms_boxes(std::vector<FaceBox>& input, float threshold, int type, std::vector<FaceBox>&output);

void regress_boxes(std::vector<FaceBox>& rects);

void square_boxes(std::vector<FaceBox>& rects);

void padding(int img_h, int img_w, std::vector<FaceBox>& rects);

void process_boxes(std::vector<FaceBox>& input, int img_h, int img_w, std::vector<FaceBox>& rects,float nms_threshold=.7f);

void generate_bounding_box(const float * confidence_data,
		const float * reg_data, float scale, float threshold, 
		int feature_h, int feature_w, std::vector<FaceBox>&  output, bool transposed);


void set_input_buffer(std::vector<cv::Mat>& input_channels,
		float* input_data, const int height, const int width);


void  cal_pyramid_list(int height, int width, int min_size, float factor,std::vector<scale_window>& list);

void cal_landmark(std::vector<FaceBox>& box_list);

void set_box_bound(std::vector<FaceBox>& box_list, int img_h, int img_w);

#endif
