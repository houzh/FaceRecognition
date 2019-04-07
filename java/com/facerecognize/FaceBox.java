package com.facerecognize;
//import android.graphics.PointF;

public class FaceBox{
    public final static int MAX_LANDMARK=5;
    public float []bounds;
    public float score;
    public float [] regress;
    public float []paddings;
    public float[] landmark5;
    public float[] landmark68;
public FaceBox(){
     regress=new float[4];
     bounds=new float[4];//x0,y0,x1,y1
     paddings=new float[4];
     landmark5=new float[10];
     landmark68=new float[136];
}

};
