#include <string>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/types_c.h>
#include "facerecognize.hpp"

#define CONFIG_FILE_PATH "demo.conf"
#ifndef CV_CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FRAME_HEIGHT CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FRAME_WIDTH  CAP_PROP_FRAME_WIDTH
#define CV_LOAD_IMAGE_UNCHANGED -1
#endif
using namespace std;
using namespace cv;


void TestFaceDemo(const char*path){
    FaceRecognize fd("./models/");
    DIR *dir;
    char full_path[512];
    struct dirent*ptr;
    if((dir=opendir(path))==NULL)
	return;
    fd.Init(0.9, 0.9, 0.9, 0.6, 0.7,100);
    cout<<"============FaceDemo=======test from file:"<<path<<endl;
    //fd.LocalLoad("faces.data");
    while((ptr=readdir(dir))){
       std::vector<FaceBox>faces;
       struct stat file_stat;
       strcpy(full_path,path);
       strcat(full_path,"/");
       strcat(full_path,ptr->d_name);
       stat(full_path,&file_stat);
       cout<<"checking face_image:"<<full_path<<endl;
       if(S_ISDIR(file_stat.st_mode))continue;
       Mat facepic=imread(full_path);
       cout<<"Facepic Size:"<<facepic.cols<<"x"<<facepic.rows<<endl;
       int count=fd.Detect(facepic,faces);
       //if( (count==1) && (!faces[0].registed))
	//     fd.Register(faces[0].face_id,ptr->d_name);
       fd.LableFaces(facepic,faces,Scalar(255,255,255));
       cout<<"Recognized "<<count<<endl;
       //for(int i=0;i<faces.size();i++){if(faces[i].registed)cout<<"    face:"<<faces[i].face_id<<" name:"<<faces[i].name<<endl;}
    }
    //fd.LocalSave("faces.data");
}
void TestModel(const string& path,const string&pname,const string&mname)
{
    int rc,rc2;
    string proto=path+"/"+pname;
    string model=path+"/"+mname;
    graph_t graph=create_graph(NULL,"caffe",proto.c_str(),model.c_str());
    if(graph==nullptr){
	   std::cout<<proto<<" load failed"<<std::endl;
	   return;
    }
    rc=infer_shape(graph);
    rc2=prerun_graph(graph);
    std::cout<<"infer="<<rc<<" prerun="<<rc2<<std::endl;
}
int main(int argc, char * argv[])
{
    init_tengine_library();
    if(argc == 4){
	//TestFaceDemo(argv[1]);
	TestModel(argv[1],argv[2],argv[3]);
	return 0;
    }

#if 1 
    namedWindow("AID-SHOW", 0);
    moveWindow("AID-SHOW", 640, 480);
#endif
    FaceRecognize fd(argv[1],0);
    std::vector<FaceBox>boxes;
    if(argc>2){
        cv::Mat frame=imread(argv[2]);
        std::cout<<"frame.size="<<frame.cols<<"x"<<frame.rows<<std::endl;
        fd.Detect(frame,boxes);
        std::cout<<"detected faces:"<<boxes.size()<<std::endl;
        if(boxes.size()>0){
            //std::vector<float>feature;
            //fd.GetFeature(frame,boxes[0],feature);
        }
        fd.LableFaces(frame,boxes);
        imwrite("./result.jpg",frame);
        imshow("AID_SHOW",frame);
        waitKey(2000);
        return 0;
    }
    while(true){
	cv::Mat frame;
        VideoCapture capture;
	boxes.clear();
        capture.open(0);
	capture.read(frame);
	fd.Detect(frame,boxes);
	fd.LableFaces(frame,boxes);
	imshow("AID-SHOW",frame);
	waitKey(10);
    }
    return 0;
}
