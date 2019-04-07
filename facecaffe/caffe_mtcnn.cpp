//#include <string>
#include <vector>

#include "mtcnn.hpp"

#include "caffe_mtcnn.hpp"
using namespace std;

caffe_mtcnn::~caffe_mtcnn(void)
{

    if(PNet_graph)
    {
        destroy_graph(PNet_graph);
    }

    if(RNet_graph)
    {
        destroy_graph(RNet_graph);
    }

    if(ONet_graph)
    {
        destroy_graph(ONet_graph);
    }



}

int caffe_mtcnn::load_3model(const std::string &proto_model_dir)
{
        std::string proto_name, mdl_name;
        const char * proto_name_, *mdl_name_;
        const char * input_node_name = "input";

        // Pnet
        proto_name = proto_model_dir + "/det1.prototxt";
        mdl_name = proto_model_dir + "/det1.caffemodel";
        proto_name_ = proto_name.c_str();
        mdl_name_ = mdl_name.c_str();
        PNet_graph=create_graph(nullptr, "caffe", proto_name_, mdl_name_);
        std::cout << "load PNet model="<<PNet_graph<<endl;
        set_graph_input_node(PNet_graph, &input_node_name, 1);
	if(prerun_graph(PNet_graph)!=0)std::cout<<__FUNCTION__<<" prerun_graph PNET error"<<std::endl;

        //Rnet
        proto_name = proto_model_dir + "/det2.prototxt";
        mdl_name = proto_model_dir + "/det2.caffemodel";
        proto_name_ = proto_name.c_str();
        mdl_name_ = mdl_name.c_str();
        RNet_graph=create_graph(nullptr, "caffe", proto_name_, mdl_name_);
        std::cout << "load RNet model ="<<RNet_graph<<endl;
        set_graph_input_node(RNet_graph, &input_node_name, 1);
	if(prerun_graph(RNet_graph)!=0)std::cout<<__FUNCTION__<<" prerun_graph RNET error"<<std::endl;

        //Onet
        proto_name = proto_model_dir + "/det3.prototxt";
        mdl_name = proto_model_dir + "/det3.caffemodel";
        proto_name_ = proto_name.c_str();
        mdl_name_ = mdl_name.c_str();
        ONet_graph=create_graph(nullptr, "caffe", proto_name_, mdl_name_);
        std::cout << "load ONet model="<<ONet_graph<<endl;
        set_graph_input_node(ONet_graph, &input_node_name, 1);
	if(prerun_graph(ONet_graph)!=0)std::cout<<__FUNCTION__<<" prerun_graph ONET error"<<std::endl;
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
    for(int i = 0; i < win_list.size(); i++) {
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

    tensor_t input_tensor=get_graph_tensor(PNet_graph,"data"); 
    int dims[]={1,3,scale_h,scale_w};
    set_tensor_shape(input_tensor,dims,4);
    int in_mem=sizeof(float)*scale_h*scale_w*3;
    //std::cout<<"mem "<<in_mem<<"\n";
    float* input_data=(float*)malloc(in_mem);
   
    std::vector<cv::Mat> input_channels;
    set_input_buffer(input_channels, input_data, scale_h, scale_w);
    cv::split(resized, input_channels);

    set_tensor_buffer(input_tensor,input_data,in_mem);
    
    if(run_graph(PNet_graph,1)!=0)cout<<__FUNCTION__<<" rungraph error\n";
    free(input_data);
    /* output */
    tensor_t tensor=get_graph_tensor(PNet_graph,"conv4-2");
    get_tensor_shape(tensor,dims,4);
    float *  reg_data=(float *)get_tensor_buffer(tensor);
    int feature_h=dims[2];
	int feature_w=dims[3];
    //std::cout<<"Pnet scale h,w= "<<feature_h<<","<<feature_w<<"\n";

    tensor=get_graph_tensor(PNet_graph,"prob1");
    float *  prob_data=(float *)get_tensor_buffer(tensor);
    std::vector<FaceBox> candidate_boxes;
    generate_bounding_box(prob_data,
			reg_data, scale,pnet_threshold_,feature_h,feature_w,candidate_boxes,true);


    nms_boxes(candidate_boxes, 0.5, NMS_UNION,box_list);
#ifdef TENGINE_API_2
    release_graph_tensor(input_tensor); 
    release_graph_tensor(tensor);
#endif
    //std::cout<<"condidate boxes size :"<<candidate_boxes.size()<<"\n";
    return 0;
}


void caffe_mtcnn::run_RNet(const cv::Mat& img, std::vector<FaceBox>& pnet_boxes, std::vector<FaceBox>& output_boxes)
{
    int batch=pnet_boxes.size();
    int channel = 3;
    int height = 24;
    int width = 24;

    tensor_t input_tensor=get_graph_tensor(RNet_graph,"data"); 
    int dims[]={batch,channel,height,width};
    set_tensor_shape(input_tensor,dims,4);
    int img_size=channel*height*width;
    int in_mem=sizeof(float)*batch*img_size;
    float* input_data=(float*)malloc(in_mem);
    float* input_ptr=input_data;
    set_tensor_buffer(input_tensor,input_ptr,in_mem);

    for(int i=0;i<batch;i++)
    {
        copy_one_patch(img,pnet_boxes[i],input_ptr,height,width);
        input_ptr+=img_size;
    }
    
    if(run_graph(RNet_graph,1)!=0)std::cout<<__FUNCTION__<<" rungraph error\n";
    free(input_data);
    //std::cout<<"run done ------\n";
    //
    /* output */
    tensor_t tensor=get_graph_tensor(RNet_graph,"conv5-2");
    float *  reg_data=(float *)get_tensor_buffer(tensor);

    tensor=get_graph_tensor(RNet_graph,"prob1");
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
#ifdef TENGINE_API_2
    release_graph_tensor(input_tensor);
    release_graph_tensor(tensor);
#endif
}	


void caffe_mtcnn::run_ONet(const cv::Mat& img, std::vector<FaceBox>& rnet_boxes, std::vector<FaceBox>& output_boxes) 
{
    int batch=rnet_boxes.size();

    int channel = 3;
    int height = 48;
    int width = 48;
    tensor_t input_tensor=get_graph_tensor(ONet_graph,"data"); 
    int dims[]={batch,channel,height,width};
    set_tensor_shape(input_tensor,dims,4);
    int img_size=channel*height*width;
    int in_mem=sizeof(float)*batch*img_size;
    float* input_data=(float*)malloc(in_mem);
    float*  input_ptr=input_data;
    set_tensor_buffer(input_tensor,input_ptr,in_mem);
    for(int i=0;i<batch;i++)
    {
        copy_one_patch(img,rnet_boxes[i],input_ptr,height,width);
        input_ptr+=img_size;
    }
    if(run_graph(ONet_graph,1)!=0)std::cout<<__FUNCTION__<<" rungraph error\n";
    free(input_data);
    /* output */
    tensor_t tensor=get_graph_tensor(ONet_graph,"conv6-3");
    float *  points_data=(float *)get_tensor_buffer(tensor);

    tensor=get_graph_tensor(ONet_graph,"prob1");
    float *  confidence_data=(float *)get_tensor_buffer(tensor);

    tensor=get_graph_tensor(ONet_graph,"conv6-2");
    float *  reg_data=(float *)get_tensor_buffer(tensor);

    int conf_page_size=2;
    int reg_page_size=4;
    int points_page_size=10;
    for(int i=0;i<batch;i++)
    {
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
#ifdef TENGINE_API_2 
    release_graph_tensor(input_tensor); 
    release_graph_tensor(tensor);
#endif
}	

static mtcnn * caffe_creator(void)
{
    return new caffe_mtcnn();
}

//REGISTER_MTCNN_CREATOR(caffe,caffe_creator);

