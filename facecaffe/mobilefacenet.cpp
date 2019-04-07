#include "mobilefacenet.hpp"
#include "face_align.hpp"
using namespace std;
using namespace cv;

MobileFacenet::MobileFacenet(const std::string&dir){
    int dims[8];
    std::string proto=dir+"/MobileFaceNet.prototxt";
    std::string model=dir+"/MobileFaceNet.caffemodel";
    graph=create_graph(nullptr,"caffe",proto.c_str(),model.c_str());
    input_tensor =get_graph_input_tensor(graph, 0, 0);
    get_tensor_shape(input_tensor,dims,4);
    cout<<"==== MobileFacenet.dims="<<dims[0]<<","<<dims[1]<<","<<dims[2]<<","<<dims[3]<<endl;
    input_data = (float *)malloc(sizeof(float) * dims[1]*dims[2]*dims[3]);
    set_tensor_shape(input_tensor, dims, 4);
    int rc=infer_shape(graph);
    std::cout<<" MobileFacenet.InferShape()="<<rc<<std::endl;
    rc=prerun_graph(graph);
    //if(rc!=0)dump_graph(graph);
    cout<<__func__<<" prerun_graph="<<rc<<" graph="<<graph<<endl<<endl;
    out_tensor=get_graph_output_tensor(graph,0,0);
}
MobileFacenet::~MobileFacenet(){
    release_graph_tensor(input_tensor);
    release_graph_tensor(out_tensor);
    destroy_graph(graph);
}

static void get_data(float* input_data, Mat &gray, int img_h, int img_w)
{
    cv::Mat sample;
    std::cout<<"get_data channels="<<gray.channels()<<std::endl;
    std::vector<cv::Mat>channels;
    if(gray.channels()>1){
      gray.convertTo(sample, CV_32FC3);
      for(int i=0;i<gray.channels();i++){
          cv::Mat channel(img_w,img_h, CV_32FC1, input_data);
          input_data+=img_w*img_h;
          channels.push_back(channel);
      }
      sample/=256.f;
      cv::split(sample,channels);
      return;
    }else{
      cv::Mat channel(img_w,img_h,CV_32FC1,input_data);
      gray.convertTo(channel,CV_32FC1);
      channel/=256.f;
    }
}
int MobileFacenet::GetFeature(const cv::Mat&frame,FaceBox&box,float*feature)
{
    cv::Mat aligned;
    int alignflag,outsize=0,dims[4];
    get_tensor_shape(input_tensor,dims,4);
    alignflag = get_aligned_face(frame, (float *)&box.landmark, 5, dims[2],dims[3], aligned);
    if(!alignflag){
       std::cout<<"MobileFacenet::GetFeature aligned failed"<<std::endl;
       return 0;
    }
    //get face feature

    get_data(input_data,aligned,dims[3]/*img_h*/,dims[2]/*img_w*/);
    if (set_tensor_buffer(input_tensor, input_data, dims[2]*dims[3] * sizeof(float)*dims[1]) < 0){
        std::printf("set buffer for tensor: %s failed\n", get_tensor_name(input_tensor));
        return 0;
    }

    int rc=run_graph(graph, 1);//!=0)
    if(rc!=0)cout<<__FUNCTION__<<" run_graph ="<<rc<<endl;
    float *data = (float *)get_tensor_buffer(out_tensor);
    outsize=get_tensor_buffer_size(out_tensor);
    for(int i=0;i<outsize;i++)
	    feature[i]=data[i];
    return outsize;
}
