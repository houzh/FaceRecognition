//#include <string>
#include <vector>

#include "mtcnn.hpp"

#include "caffe_mtcnn.hpp"
using namespace std;

enum{ PNET=0, RNET=1,ONET=2 };

caffe_mtcnn::~caffe_mtcnn(void)
{
    for(int i=0;i<3;i++){
        destroy_graph(graph[i]);
	release_graph_tensor(input_tensor[i]);
    }
}

int caffe_mtcnn::load_3model(const std::string &proto_model_dir)
{
    const char*mdname[]={"/det1","/det2","/det3"};
    std::string proto_name, mdl_name;
    const char * proto_name_, *mdl_name_;
    const char * input_node_name = "input";
    for(int i=0;i<3;i++){
        proto_name = proto_model_dir + mdname[i]+".prototxt";
        mdl_name = proto_model_dir + mdname[i]+".caffemodel";
        proto_name_ = proto_name.c_str();
        mdl_name_ = mdl_name.c_str();
        graph[i]=create_graph(nullptr, "caffe", proto_name_, mdl_name_);
        std::cout << "load PNet model="<<graph[i]<<endl;
        set_graph_input_node(graph[i], &input_node_name, 1);
	if(prerun_graph(graph[i])!=0)
        std::cout<<__FUNCTION__<<" prerun_graph error"<<std::endl;
	input_tensor[i]=get_graph_tensor(graph[i],"data");
	input_data[i]=nullptr;
	input_size[i]=0;
    }

    return 0;
}

void caffe_mtcnn::detect(cv::Mat& img, std::vector<FaceBox>& face_list)
{
    cv::Mat working_img;
    float alpha = 0.0078125;
    float mean = 127.5;
    img.convertTo(working_img, CV_32FC3);
    working_img = (working_img - mean) * alpha;
    working_img = working_img.t();
    cv::cvtColor(working_img, working_img, cv::COLOR_BGR2RGB);

    int img_h = working_img.rows;
    int img_w = working_img.cols;

    std::vector<scale_window> win_list;

    std::vector<FaceBox> total_pnet_boxes;
    std::vector<FaceBox> total_rnet_boxes;
    std::vector<FaceBox> total_onet_boxes;

    cal_pyramid_list(img_h, img_w, min_size_, factor_, win_list);
    for(size_t i = 0; i < win_list.size(); i++) {
        std::vector<FaceBox>boxes;
        run_PNet(working_img, win_list[i], boxes);
        total_pnet_boxes.insert(total_pnet_boxes.end(), boxes.begin(), boxes.end());
    }
    std::vector<FaceBox> pnet_boxes;
    process_boxes(total_pnet_boxes, img_h, img_w, pnet_boxes,.7f);

    if(!pnet_boxes.size())
        return;


    run_RNet(working_img, pnet_boxes, total_rnet_boxes);

    std::vector<FaceBox> rnet_boxes;
    process_boxes(total_rnet_boxes, img_h, img_w, rnet_boxes);

    if(!rnet_boxes.size())
	 return;

    run_ONet(working_img, rnet_boxes, total_onet_boxes);

    //calculate the landmark
    for(unsigned int i = 0; i < total_onet_boxes.size(); i++) {
        FaceBox& box = total_onet_boxes[i];
        float h = box.x1 - box.x0 + 1;
        float w = box.y1 - box.y0 + 1;
        for(int j = 0; j < 5; j++) {
            box.landmark[j]  = box.x0 + w * box.landmark[j] - 1;//x
            box.landmark[j+5]= box.y0 + h * box.landmark[j+5] - 1;//y
        }
    }

    //Get Final Result
    regress_boxes(total_onet_boxes);
    nms_boxes(total_onet_boxes, 0.7, NMS_MIN, face_list);

    //set_box_bound(face_list,img_h,img_w);

    //switch x and y, since working_img is transposed

    for(unsigned int i = 0; i < face_list.size(); i++) {
        FaceBox& box = face_list[i];
        std::swap(box.x0, box.y0);
        std::swap(box.x1, box.y1);
        for(int l = 0; l < 5; l++) {
            std::swap(box.landmark[l], box.landmark[l+5]);
        }
	memset(box.landmark68,0,sizeof(box.landmark68));
    }
}

void caffe_mtcnn::copy_one_patch(const cv::Mat& img, FaceBox&input_box, float * data_to, int width, int height)
{
    std::vector<cv::Mat> channels;

    set_input_buffer(channels, data_to, height, width);

    cv::Mat chop_img = img(cv::Range(input_box.py0, input_box.py1),
                           cv::Range(input_box.px0, input_box.px1));

    int pad_top = std::abs(input_box.py0 - input_box.y0);
    int pad_left = std::abs(input_box.px0 - input_box.x0);
    int pad_bottom = std::abs(input_box.py1 - input_box.y1);
    int pad_right = std::abs(input_box.px1 - input_box.x1);

    cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom, pad_left, pad_right,  cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::resize(chop_img, chop_img, cv::Size(width, height), 0, 0);
    cv::split(chop_img, channels);

}


int caffe_mtcnn::run_PNet(const cv::Mat& img, scale_window& win, std::vector<FaceBox>& box_list)
{
    cv::Mat  resized;
    int scale_h=win.h;
    int scale_w=win.w;
    float scale=win.scale;
    cv::resize(img, resized, cv::Size(scale_w, scale_h), 0, 0);
    /* input */

    int dims[]={1,3,scale_h,scale_w};
    set_tensor_shape(input_tensor[PNET],dims,4);
    int in_mem=sizeof(float)*scale_h*scale_w*3;
   
    //std::cout<<"mem "<<in_mem<<"\n";
    if(input_size[PNET]<in_mem){
	 input_data[PNET]=(float*)realloc(input_data[PNET],in_mem);
	 input_size[PNET]=in_mem;
    }
    std::vector<cv::Mat> input_channels;
    set_input_buffer(input_channels, input_data[PNET], scale_h, scale_w);
    cv::split(resized, input_channels);

    set_tensor_buffer(input_tensor[PNET],input_data[PNET],in_mem);
    
    if(run_graph(graph[PNET],1)!=0)cout<<__FUNCTION__<<" rungraph error\n";
    /* output */
    tensor_t tensor=get_graph_tensor(graph[PNET],"conv4-2");
    get_tensor_shape(tensor,dims,4);
    float *  reg_data=(float *)get_tensor_buffer(tensor);
    int feature_h=dims[2];
    int feature_w=dims[3];
    //std::cout<<"Pnet scale h,w= "<<feature_h<<","<<feature_w<<"\n";

    tensor=get_graph_tensor(graph[PNET],"prob1");
    float *  prob_data=(float *)get_tensor_buffer(tensor);
    std::vector<FaceBox> candidate_boxes;
    generate_bounding_box(prob_data,
	reg_data, scale,pnet_threshold_,feature_h,feature_w,candidate_boxes,true);


    nms_boxes(candidate_boxes, 0.5, NMS_UNION,box_list);
    //std::cout<<"condidate boxes size :"<<candidate_boxes.size()<<"\n";
    return 0;
}


void caffe_mtcnn::run_RNet(const cv::Mat& img, std::vector<FaceBox>& pnet_boxes, std::vector<FaceBox>& output_boxes)
{
    const int channel = 3, height = 24, width = 24;
    const int img_size=channel*height*width;
    int batch=pnet_boxes.size();

    int dims[]={batch,channel,height,width};
    set_tensor_shape(input_tensor[RNET],dims,4);
    int in_mem=sizeof(float)*batch*img_size;
    
    if(input_size[RNET]<in_mem){
        input_data[RNET]=(float*)realloc(input_data[RNET],in_mem);
	input_size[RNET]=in_mem;
    }
    
    float *input_ptr=input_data[RNET];
    set_tensor_buffer(input_tensor[RNET],input_ptr,in_mem);

    for(int i=0;i<batch;i++){
        copy_one_patch(img,pnet_boxes[i],input_ptr,height,width);
        input_ptr+=img_size;
    }
    
    if(run_graph(graph[RNET],1)!=0)std::cout<<__FUNCTION__<<" rungraph error\n";
    //std::cout<<"run done ------\n";
    /* output */
    tensor_t tensor=get_graph_tensor(graph[RNET],"conv5-2");
    float *  reg_data=(float *)get_tensor_buffer(tensor);

    tensor=get_graph_tensor(graph[RNET],"prob1");
    float *  confidence_data=(float *)get_tensor_buffer(tensor);

    int conf_page_size=2;
    int reg_page_size=4;

    for(int i=0;i<batch;i++)
    {

        if (*(confidence_data+1) > rnet_threshold_){

            FaceBox output_box;
            FaceBox& input_box=pnet_boxes[i];

            output_box.x0=input_box.x0;
            output_box.y0=input_box.y0;
            output_box.x1=input_box.x1;
            output_box.y1=input_box.y1;

            output_box.score = *(confidence_data+1);

            /*Note: regress's value is swaped here!!!*/

            output_box.regress[0]=reg_data[1];
            output_box.regress[1]=reg_data[0];
            output_box.regress[2]=reg_data[3];
            output_box.regress[3]=reg_data[2];

            output_boxes.push_back(output_box);
        }

        confidence_data+=conf_page_size;
        reg_data+=reg_page_size;
    }
}	


void caffe_mtcnn::run_ONet(const cv::Mat& img, std::vector<FaceBox>& rnet_boxes, std::vector<FaceBox>& output_boxes) 
{
    const int channel = 3,height = 48,width = 48;
    const int img_size=channel*height*width;
    int batch=rnet_boxes.size();

    int dims[]={batch,channel,height,width};
    set_tensor_shape(input_tensor[ONET],dims,4);
    int in_mem=sizeof(float)*batch*img_size;
    
    if(input_size[ONET]<in_mem){
        input_data[ONET]=(float*)realloc(input_data[ONET],in_mem);
	input_size[ONET]=in_mem;
    }
    
    float*input_ptr=input_data[ONET];
    set_tensor_buffer(input_tensor[ONET],input_ptr,in_mem);
    for(int i=0;i<batch;i++){
        copy_one_patch(img,rnet_boxes[i],input_ptr,height,width);
        input_ptr+=img_size;
    }
    if(run_graph(graph[ONET],1)!=0)std::cout<<__FUNCTION__<<" rungraph error\n";
    /* output */
    tensor_t tensor=get_graph_tensor(graph[ONET],"conv6-3");
    float *  points_data=(float *)get_tensor_buffer(tensor);

    tensor=get_graph_tensor(graph[ONET],"prob1");
    float *  confidence_data=(float *)get_tensor_buffer(tensor);

    tensor=get_graph_tensor(graph[ONET],"conv6-2");
    float *  reg_data=(float *)get_tensor_buffer(tensor);

    int conf_page_size=2;
    int reg_page_size=4;
    int points_page_size=10;
    for(int i=0;i<batch;i++){
        if (*(confidence_data+1) > rnet_threshold_){
            FaceBox output_box;
            FaceBox& input_box=rnet_boxes[i];

            output_box.x0=input_box.x0;
            output_box.y0=input_box.y0;
            output_box.x1=input_box.x1;
            output_box.y1=input_box.y1;

            output_box.score=*(confidence_data+1);

            output_box.regress[0]=reg_data[1];
            output_box.regress[1]=reg_data[0];
            output_box.regress[2]=reg_data[3];
            output_box.regress[3]=reg_data[2];

            /*Note: switched x,y points value too..*/

            for (int j = 0; j<5; j++){
                output_box.landmark[j]  = *(points_data + j+5);
                output_box.landmark[j+5]= *(points_data + j);
            }
            output_boxes.push_back(output_box);

        }

        confidence_data+=conf_page_size;
        reg_data+=reg_page_size;
        points_data+=points_page_size; 
    }
}	


