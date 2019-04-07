#include <string>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <time.h>
#include <sys/time.h>
#include "opencv2/features2d.hpp"	//SurfFeatureDetector实际在该头文件中
//#include "opencv2/legacy/legacy.hpp"	//BruteForceMatcher实际在该头文件中
//#include "opencv4/opencv2/features2d.hpp"	//FlannBasedMatcher实际在该头文件中

//#include <opencv2/videoio/legacy/constants_c.h>
#include "facerecognize.hpp"
//#include "age_gender.hpp"
#define CONFIG_FILE_PATH "demo.conf"

#ifndef CV_CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_WIDTH  CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_HEIGHT CAP_PROP_FRAME_HEIGHT
#define CV_WINDOW_NORMAL WINDOW_NORMAL
#define CV_CAP_PROP_FPS CAP_PROP_FPS
#endif
using namespace std;
using namespace cv;

void  bindToCpu(int cpu1, int cpu2)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    //CPU_SET(cpu,&mask);
    CPU_SET(cpu1, &mask);
    CPU_SET(cpu2, &mask);
#if 0//__GNUC__ >3
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
        fprintf(stderr, "set thread affinity failed\n");
    }
#endif
}

int FileFilter(const struct dirent *d){
   return strlen(d->d_name)>2;
}
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define WHITE   "\033[37m"     /*White*/
#if 1 
#define DOUI(x) x
#else
#define DOUI(x)
#endif
void LFW_Test(FaceRecognize& fd,const char*lfwpath,ofstream& outfile){
   struct dirent**namelist;
   char full_path[512],file_path[512];
   int total_error=0;
   int total_person=0;
   int namesize=scandir(lfwpath,&namelist,FileFilter,alphasort);
   int person_dir_count=0;
   DOUI(namedWindow("REGISTED",0));
   DOUI(namedWindow("COMPARED"));
   DOUI(moveWindow("REGISTED",100,100));
   DOUI(moveWindow("COMPARED",600,100));
   while(namesize--){
      struct stat file_stat;
      char person_name[512];
      strcpy(full_path,lfwpath);
      strcat(full_path,"/");
      strcat(full_path,namelist[namesize]->d_name);
      stat(full_path,&file_stat);
      bool isdir=S_ISDIR(file_stat.st_mode);
      strcpy(person_name,namelist[namesize]->d_name);
      free(namelist[namesize]);
      
      struct dirent**filelist=NULL;
      int file_count=scandir(full_path,&filelist,FileFilter,alphasort);
      int registed=0,person_error=0;
      if(file_count>0){
          person_dir_count++;
	  total_person+=file_count;
      }
      for(int findex=0;isdir&&(findex<file_count);free(filelist[findex]),findex++){
	 std::vector<FaceBox>face_boxes;
         strcpy(file_path,full_path);
	 strcat(file_path,"/");
	 strcat(file_path,filelist[findex]->d_name);
	 cv::Mat faceimg=cv::imread(file_path,1);
	 if(faceimg.empty()){
	     cout<<RED<<file_path<<" read error"<<endl;
	     continue;
	 }
	 int rc=fd.Detect(faceimg,face_boxes);
	 //fd.Recognize(faceimg,faces);
	 if(rc==0){
	    outfile  <<"FaceDetect Error:"<<file_path<<endl;
            cout<<RED<<"FaceDetect Error:"<<file_path<<endl;
	    total_error++;
	    continue;
	 }
	 long faceid;
	 cv::Mat aligned;
	 std::vector<float>feature;
	 if(0>=fd.GetFeature(faceimg,face_boxes[0],feature)){
	     outfile<<"GetFeature Error:"<<file_path<<endl;
	     cout<<RED<<"GetFeature Error:"<<file_path<<WHITE<<endl;
	 }
	 if(registed){
	      std::string name;
	      float score;
	      /*if(fd.Recognize(feature.data(),name,faceid,&score)<0){
		   person_error++;
		   total_error++;
		   outfile<<filelist[findex]->d_name<<" Recognize Error:"<<person_error<<"/"<<file_count<<" score:"<<score<<endl;
		   cout<<RED<<filelist[findex]->d_name<<" Recognize Error:"<<person_error<<"/"<<file_count<<" score:"<<score<<WHITE<<endl;
	      }else{
		  char id[32];
		  sprintf(id,"ID:%ld PICNUM:%d %.3f",faceid,file_count,score);
	          cv::putText(faceimg,name.c_str(),cvPoint(10,20),1,1,Scalar(255,255,200),2);
	          cv::putText(faceimg,id,cvPoint(10,40),1,1,Scalar(255,255,200),2);
	      }*/
	      fd.LableFaces(faceimg,face_boxes,Scalar(0,0,255));
	      DOUI(imshow("COMPARED",faceimg));	 
	 }else{
	    char id[64];
	    /*fd.Register(feature.data(),person_name,&faceid);//filelist[findex]->d_name);
	    sprintf(id,"ID:%ld PICNUM:%d",faceid,file_count);
	    cv::putText(faceimg,person_name,cvPoint(10,20),1,1,Scalar(255,255,200),2);
	    cv::putText(faceimg,id,cvPoint(10,40),1,1,Scalar(255,255,200),2);
	    //faces[0].name=person_name;
	    fd.LableFaces(faceimg,face_boxes,Scalar(0,0,255));*/
	    DOUI(imshow("REGISTED",faceimg));
	    DOUI(imshow("COMPARED",faceimg));
            registed++;
	 }
	 DOUI(waitKey(10));
      }//endof for
      if(filelist)free(filelist);
      //outfile<<person_name<<":"<<person_error<<"/"<<file_count<<" Registed:"<<fd.GetRegistedCount()<<endl;
      //cout<<WHITE<<person_name<<":"<<person_error<<"/"<<file_count<<" Registed:"<<fd.GetRegistedCount()<<endl;
      DOUI(waitKey(10));
      if((namesize%5==0)&&total_person)
	  cout<<RED<<"Total Error:"<<total_error<<" Total Person:"<<total_person<<" Rate="<<(100.f*total_error)/total_person<<endl;
   }
   cout<<RED<<"Total Error:"<<total_error<<" Total Person:"<<total_person<<" Rate="<<(100.f*total_error)/total_person<<endl;
   outfile<<"Total Error:"<<total_error<<" Total Person:"<<total_person<<" Rate="<<(100.f*total_error)/total_person<<endl;
}

void TestFaceDemo(FaceRecognize&fd,const char*path){
    char full_path[512];
    struct dirent**filelist;
    int file_count=scandir(path,&filelist,FileFilter,alphasort);
    
    cout<<"============FaceDemo=======test from file:"<<path<<endl;
    namedWindow("FACE",CV_WINDOW_NORMAL);
    resizeWindow("FACE",800,600);
    int findex=0;
    do{
       std::vector<FaceBox>faces;
       std::vector<float>feature;
       struct stat file_stat;
       strcpy(full_path,path);
       if(file_count>0){
         strcat(full_path,"/");
         strcat(full_path,filelist[findex]->d_name);
       }
       stat(full_path,&file_stat);
       cout<<"\n\nChecking Image:"<<full_path<<endl;
       if(S_ISDIR(file_stat.st_mode))continue;
       Mat facepic=imread(full_path,1);
       cout<<"Facepic Size:"<<facepic.cols<<"x"<<facepic.rows<<endl;
       if(facepic.empty())continue;
       fd.Detect(facepic,faces);
       fd.GetFeature(facepic,faces[0],feature);
       fd.LableFaces(facepic,faces,Scalar(0,0,255));
       //for(int i=0;i<feature.size();i+=2)
         //cv::circle(facepic,cvPoint(feature[i]*256.f,feature[i+1]*256.f),3,Scalar(0,0,255),-1);
       imshow("FACE",facepic);
       waitKey(faces.size()>1?1000:100);
       if(file_count>0)free(filelist[findex]);
    }while(++findex<file_count);
    //fd.LocalSave("faces.data");
    if(file_count<=0)waitKey(-1);
}
void TestModel(std::string proto,std::string model){
    const char*model_name="test_model";
    int rc=load_model(model_name, "caffe", proto.c_str(), model.c_str());
    cout<<"load_model:"<<model<<" result:"<<rc<<endl;
    graph_t graph = create_runtime_graph("graph", model_name, NULL);
    cout<<"graph="<<graph<<endl;
    if(graph)
	  dump_graph(graph);    
}
static cv::Mat frame2Parse;
static FaceBox facebox2Parse;
static int next_face_id=0;
void OnCaptureClick(int state,void*userdata){
    float feature[256];
    char name[16];
    FaceRecognize*fd=(FaceRecognize*)userdata;
    long id;
    int ret=fd->GetFeature(frame2Parse,facebox2Parse,feature,256);
    if(ret>0){
	sprintf(name,"%d",next_face_id++);
        //fd->Register(feature,name,&id);
    }
    cout<<__FUNCTION__<<":"<<__LINE__<<" getfeature="<<ret<<endl;
}
extern int diffimg( int argc, char** argv );
int main(int argc, char * argv[])
{
    init_tengine_library();
    set_log_level((log_level)7);
    if(argc==3)return diffimg(argc,argv);
    if (request_tengine_version("0.9") < 0)
        return -1;
    int mode=0,min_size=100;
    int videoWidth,videoHeight;
    float threshold_p=.8f,threshold_r=.8f,threshold_o=.6f;
    //fd.LocalLoad("faces.data");
   
    int ch,verbose=0,DeviceID=0,enable_age=0,enable_gender=0; 
    std::string facePicturePATH,videoStreamAddress;
    std::string config_file=CONFIG_FILE_PATH;
    int have_test=0;
    std::string imgdir; 
    std::string proto,model;
    std::string facedb;
              
    while((ch=getopt(argc,argv,"c:i:m:l::n:p:vt"))!=-1){
      switch(ch){
      case 'c':
	      break;
      case 'i':
	      imgdir=optarg;
	      cout<<"Face Test Suite:"<<optarg<<endl;
	      break;
      case 't':
	      have_test=1;
	      break;
      case 'n':mode=atoi(optarg);
	      break;
      case 'v':verbose=1;break;
      case 'm':model=optarg;break;
      case 'l':facedb=optarg;break;
      case 'p':proto=optarg;break;
      }
    }
    const char*modes[]={"LightenCNN-B","LightenCNN-C","LightenCNN-S","MobileNet"};
    std::string fname;
    if(!imgdir.empty())fname=imgdir+"/";
    fname+=modes[mode];
    std::ofstream outfile(fname+".txt");


    cout<<"EnableAge="<<enable_age<<" EnableGender="<<enable_gender<<endl;
    outfile<<"Threshold_P:"<<threshold_p<<" Threshold_R:"<<threshold_r<<" Threshold_O:"<<threshold_o
	    <<" MinFaceSize="<<min_size<<endl;
    FaceRecognize fd("./models/",mode);
    cout<<RED<<"Threshold_P:"<<threshold_p<<" Threshold_R:"<<threshold_r<<" Threshold_O:"<<threshold_o
	    <<" MinFaceSize="<<min_size<<endl;
    fd.Init(threshold_p/*0.9*/,threshold_r /*0.9*/,threshold_o /*0.68*/, 0.5, 0.7,min_size);
    fd.SetVerbose(verbose);
    if(!imgdir.empty()){
       if(have_test)
	  LFW_Test(fd,imgdir.c_str(),outfile);
       else
	  TestFaceDemo(fd,imgdir.c_str());
       return 0;
    }else if(!proto.empty() && !model.empty()){
        TestModel(proto,model);
	return 0;
    }
    VideoCapture capture;
    capture.open(DeviceID);
    if (!capture.isOpened()) { //判断能够打开摄像头
        cout << "can not open the camera" << endl;
	cout<<"try IPCamera from URL:"<<videoStreamAddress<<endl;
	capture.open(videoStreamAddress);
    }
    cout<<"setfps="<<capture.set(CV_CAP_PROP_FPS,10)<<endl;
    capture.set(CV_CAP_PROP_FRAME_WIDTH, videoWidth);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, videoHeight);

    Mat frame;

    namedWindow("AID-SHOW", 0);
    moveWindow("AID-SHOW",100 ,100);
    resizeWindow("AID-SHOW",800,600);
    createTrackbar("Capture","AID-SHOW",&verbose,10,OnCaptureClick,&fd);
    Mat ShowImg = Mat(cvSize(videoWidth,videoHeight),frame.type());
    frame2Parse=Mat(cvSize(videoWidth,videoHeight),frame.type());
    while(1) {
	struct timeval tv_start,tv_end;
	std::vector<FaceBox>faces;
	std::vector<float>feature;
        capture.read(frame);
	if(!frame.empty()) {
	   gettimeofday(&tv_start,NULL);
	   fd.Detect(frame,faces);
           for(int i=0;i<faces.size();i++){
	      int ret=fd.GetFeature(frame,faces[i],feature);
	      if (ret<=0)continue;
	      long id;
	      char ss[16];
	      float score=.0f;
	      std::string dbname;
	      frame.copyTo(frame2Parse);
	      facebox2Parse=faces[i];
	      //fd.Recognize(feature.data(),dbname,id,&score);
	      if(score>0.55f){
	      fd.LableFace(frame,faces[i]);
	      sprintf(ss," : %.3f",score);dbname+=ss;
	      cv::putText(frame,dbname.c_str(),cvPoint(faces[i].x0+20,faces[i].y0+20),1,1,Scalar(255,255,200),2);
	      }
	   }
	   gettimeofday(&tv_end,NULL);
	   printf("time of face check && getfeature : %ld\n", tv_end.tv_sec * 1000 + tv_end.tv_usec / 1000 
			   - tv_start.tv_sec * 1000 - tv_start.tv_usec / 1000); 
	   cout<<"recognize:"<<faces.size()<<endl;
	   fd.LableFaces(frame,faces,Scalar(255,255,0));
        }
	frame.copyTo(ShowImg);
	if(frame.cols<=0||frame.rows<=0)
	{
		cout<<__FUNCTION__<<":"<<__LINE__<<"frame empty"<<endl;
	}else{
          imshow("AID-SHOW", ShowImg);
          cv::imwrite("result.jpg",ShowImg); 
	}
        waitKey(10);
    }
    return 0;
}

/**
* @概述：采用FAST算子检测特征点，采用SIFT算子对特征点进行特征提取，并使用BruteForce匹配法进行特征点的匹配
* @类和函数：FastFeatureDetector + SiftDescriptorExtractor + BruteForceMatcher
* @author：holybin
*/
 
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
//#include "opencv2/nonfree/features2d.hpp"	//SurfFeatureDetector实际在该头文件中
//#include "opencv2/legacy/legacy.hpp"	//BruteForceMatcher实际在该头文件中
#include "opencv2/features2d.hpp"	//FlannBasedMatcher实际在该头文件中
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;
//#define  BruteForceMatcher BFMatcher 
int diffimg( int argc, char** argv )
{
    Mat rgbd1 = imread(argv[1]);
    Mat rgbd2 = imread(argv[2]);
    //imshow("rgbd1", depth2);
    //waitKey(0);
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> Keypoints1,Keypoints2;
    Mat descriptors1,descriptors2;
    orb->detectAndCompute(rgbd1, Mat(), Keypoints1, descriptors1);
    orb->detectAndCompute(rgbd1, Mat(), Keypoints2, descriptors2);

    cout << "Key points of image1:" << Keypoints1.size() << endl;
    cout << "Key points of image2:" << Keypoints2.size() << endl;
    for(int i=0;i<Keypoints1.size();i++){
      if(Keypoints1[i].pt.x>240&&Keypoints1[i].pt.x<280)
      cout<<"Point("<<Keypoints1[i].pt.x<<","<<Keypoints1[i].pt.y<<","<<Keypoints1[i].size<<")"<<endl;
    }
    //可视化，显示关键点
    Mat ShowKeypoints1, ShowKeypoints2;
    drawKeypoints(rgbd1,Keypoints1,ShowKeypoints1);
    drawKeypoints(rgbd2, Keypoints2, ShowKeypoints2);
    imshow("Keypoints1", ShowKeypoints1);
    imshow("Keypoints2", ShowKeypoints2);
    waitKey(0);

    //Matching
    vector<DMatch> matches;
    //Ptr<DescriptorMatcher> matcher =new cv::BFMatcher(cv::NORM_HAMMING,true);
    Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches);
    cout << "find out total " << matches.size() << " matches" << endl;

    float min_dist=100.f,max_dist=0;
    vector<DMatch>badmatches;
    for (int i = 0; i < matches.size(); i++){
	double dist = matches[i].distance;
	if (dist < min_dist) min_dist = dist;
	if (dist > max_dist) max_dist = dist;
	if(i%10==0||dist!=.0f)cout<<dist<<" ,";
    }
    cout<<"\nmin_dist="<<min_dist<<" max_dist="<<max_dist<<endl;
    for(int i=0;i<matches.size();i++){
      if(matches[i].distance<0.5*max_dist)continue;
         badmatches.push_back(matches[i]); 
    }
    //可视化
    Mat ShowMatches;
    drawMatches(rgbd1,Keypoints1,rgbd2,Keypoints2,badmatches,ShowMatches);
    imshow("matches", ShowMatches);
    waitKey(0);
}

