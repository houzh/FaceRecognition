package com.facerecognize;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import com.facerecognize.FaceBox;

public class FaceRecognize{
  private final long nativeObj; //used for native JNI,DO NOT modify this field
  private static native long nativeCreate(String model_dir);
  public int detect(Mat frame,FaceBox[] boxes){
      return detect(frame,boxes,false);
  }
  public native int detect(Mat frame,FaceBox[] boxes,boolean landmark68);
  public native int getFeature(Mat frame,FaceBox box,float[] featuer);
  public native float matchFeature(float[]feature1,float[] feature2);
  public native int getFeatureSize();
  public native void setThreshold(float threshold_p,float threshold_r,float threshold_o);
  public native void setFactorMinFace(float factor,int min_face);
  public FaceRecognize(String dir){
     nativeObj=nativeCreate(dir);
  } 
  static{  
        System.loadLibrary("tengine-face");  
  }  
};
