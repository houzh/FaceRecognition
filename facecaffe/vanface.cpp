#include "vanface.hpp"
using namespace std;
using namespace cv;

VanFace::VanFace(const std::string&dir){
    int dims[8];
    std::string proto=dir+"/vanface.prototxt";
    std::string model=dir+"/vanface.caffemodel";
    graph=create_graph(nullptr,"caffe",proto.c_str(),model.c_str());
    input_tensor =get_graph_input_tensor(graph, 0, 0);
    get_tensor_shape(input_tensor,dims,4);
    cout<<"==== VanFace.dims="<<dims[0]<<","<<dims[1]<<","<<dims[2]<<","<<dims[3]<<endl;
    input_data = (float *)malloc(sizeof(float) * dims[1]*dims[2]*dims[3]);
    int rc=prerun_graph(graph);
    cout<<__func__<<" prerun_graph="<<rc<<" graph="<<graph<<endl<<endl;
    out_tensor=get_graph_output_tensor(graph,0,0);
}
VanFace::~VanFace(){
    release_graph_tensor(input_tensor);
    release_graph_tensor(out_tensor);
    destroy_graph(graph);
}

static void get_data(float* input_data, Mat &frame, int img_h, int img_w)
{
    cv::Mat gray;
    std::vector<cv::Mat>channels;
    cv::Mat channel(img_w,img_h, CV_32FC1, input_data);
    if(frame.channels()>1){
	cvtColor(frame, gray, COLOR_RGB2GRAY);
    }else{
	gray=frame;
    }
    cv::Mat sample =cv::Mat(cv::Size(img_w,img_h), CV_32FC1);
    gray.convertTo(gray,CV_32FC1);
    resize(gray,sample,cv::Size(img_w,img_h), 0, 0, INTER_CUBIC);
    
    Mat tmp_m, tmp_sd;
    double m = 0, sd = 0;
    meanStdDev(sample, tmp_m, tmp_sd); //Calculate the mean value and variance value
    m = tmp_m.at<double>(0, 0);
    sd = tmp_sd.at<double>(0, 0);
    sample = (sample - m) / (0.000001 + sd);

    sample.convertTo(channel,CV_32FC1);
    //channel/=256.f;
}

void GetSenseBbox(FaceBox& faceBbox, int imgHeight, int imgWidth, cv::Rect& senseBbox)
{
    int faceH, faceW, faceX1, faceY1, faceX2, faceY2;
    faceH = faceBbox.y1 - faceBbox.y0 + 1;
    faceW = faceBbox.x1 - faceBbox.x0 + 1;

    faceX1 = (std::min)(faceBbox.landmark[0], faceBbox.landmark[3]) - 0.2 * faceW;
    faceY1 = (std::min)(faceBbox.landmark[5], faceBbox.landmark[6]) - 0.1 * faceH;
    faceX2 = (std::max)(faceBbox.landmark[1], faceBbox.landmark[4]) + 0.2 * faceW;
    faceY2 = (std::max)(faceBbox.landmark[8], faceBbox.landmark[9]) + 0.2 * faceH;

    faceX1 = (std::max)(0, faceX1);
    faceY1 = (std::max)(0, faceY1);
    faceX2 = (std::min)(imgWidth - 1, faceX2);
    faceY2 = (std::min)(imgHeight - 1, faceY2);

    senseBbox = cv::Rect(faceX1, faceY1, faceX2 - faceX1 + 1, faceY2 - faceY1 + 1);
}

int VanFace::GetLandmark(const cv::Mat&frame,FaceBox&box)
{
    cv::Rect senseBbox;
    int outsize=0,dims[4];
    get_tensor_shape(input_tensor,dims,4);
    GetSenseBbox(box,frame.rows,frame.cols,senseBbox);
    cv::Mat mface(frame,senseBbox);
    get_data(input_data,mface,dims[3]/*img_h*/,dims[2]/*img_w*/);
    if (set_tensor_buffer(input_tensor, input_data, dims[2]*dims[3] * sizeof(float)*dims[1]) < 0){
        std::printf("set buffer for tensor: %s failed\n", get_tensor_name(input_tensor));
        return 0;
    }

    int rc=run_graph(graph, 1);//!=0)
    if(rc!=0)cout<<__FUNCTION__<<" run_graph ="<<rc<<endl;
    float *data = (float *)get_tensor_buffer(out_tensor);
    outsize=get_tensor_buffer_size(out_tensor)/sizeof(float);
    outsize>>=1;
    for(int i=0;i<outsize;i++,data+=2){
        box.landmark68 [i]  =senseBbox.x+senseBbox.width*data[0];
	box.landmark68[68+i]=senseBbox.y+senseBbox.height*data[1];
    }
    return outsize;
}
int VanFace::GetLandmark(const cv::Mat&frame,std::vector<FaceBox>&boxes){
    for(int i=0;i<boxes.size();i++)
	 GetLandmark(frame,boxes[i]);
}
