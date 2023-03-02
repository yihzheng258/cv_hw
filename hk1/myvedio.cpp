#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/imgproc/types_c.h>
#include<opencv2/imgproc/imgproc_c.h>
#include <ctime>
using namespace cv;
using namespace std;
#define FRAME_WIDTH  360
#define FRAME_HEIGHT  64
#define FRAME_FPS  30
#define TRASITION_TIME 3
#define IMAGE_TIME 1



void RollIn(VideoWriter writer,Mat scene1,Mat scene2) {
	int time = TRASITION_TIME * FRAME_FPS*100;
	for (int i = 1; i < time + 1; i++) {
		if (i % 100 == 0) {
			double weight = (i * 1.0) / (time * 1.0);
			Mat res = scene1;
            Mat src2Part(scene2, Rect(0, 0, (int)(scene1.cols * weight), scene1.rows));
            Mat imageROI = res(Rect(0, 0, (int)(scene1.cols * weight), scene1.rows));
            Mat mask;
            cvtColor(src2Part, mask, COLOR_BGR2GRAY);
            src2Part.copyTo(imageROI, mask);
            writer.write(res);
		}
	}
}


void Gradual(VideoWriter writer,Mat scene1,Mat scene2) {
	int time = TRASITION_TIME * FRAME_FPS*100;
	Mat res;
	for (int i = 0; i < time + 1; i++) {
		if (i % 100 == 0) {
			double weight = (i*1.0) / (time*1.0);
			addWeighted(scene1, 1.0 - weight, scene2, weight, 1, res);
			writer.write(res);
		}
	}
}

void draw_a_ractangle(VideoWriter writer,Mat frame,Point center,float size,Scalar scalar){
    //draw a ractangle
    int len = size*FRAME_FPS;
    for(int i=0;i<len;i++)
    {   
        line(frame, Point(center.x-len/2,center.y-len/2), Point(center.x-len/2+i,center.y-len/2), scalar, 2);
        writer.write(frame);
    }
    for(int i=0;i<len;i++)
    {   
        line(frame, Point(center.x+len/2,center.y-len/2), Point(center.x+len/2,center.y-len/2+i), scalar, 2);
        writer.write(frame);
    }
    for(int i=0;i<len;i++)
    {   
        line(frame, Point(center.x+len/2,center.y+len/2), Point(center.x+len/2-i,center.y+len/2), scalar, 2);
        writer.write(frame);
    }
    for(int i=0;i<len;i++)
    {   
        line(frame, Point(center.x-len/2,center.y+len/2), Point(center.x-len/2,center.y+len/2-i), scalar, 2);
        writer.write(frame);
    }
}
void draw_a_line(VideoWriter writer,Mat frame,Point start,Point end,Scalar scalar,int thickness = 2){
    

    double distance;  
    distance = powf((start.x - end.x),2) + powf((start.y - end.y),2);  
    distance = sqrtf(distance);

    double sin = (end.y - start.y )/distance;
    double cos = (end.x-  start.x )/distance;

    for(int i=0;i<distance;i++)
    {
        line(frame, Point(start.x,start.y), Point(start.x+i*cos,start.y+i*sin), scalar, thickness);
        writer.write(frame);
    }
}

void draw_a_triangle(VideoWriter writer,Mat frame, Point center,int size,Scalar scalar ){
    draw_a_line(writer,frame,Point(center.x-size,center.y),Point(center.x+size,center.y),scalar,2);
    draw_a_line(writer,frame,Point(center.x-size,center.y),Point(center.x,center.y+size),scalar,2);
    draw_a_line(writer,frame,Point(center.x+size,center.y),Point(center.x,center.y+size),scalar,2);
}

int main()
{
    Size frame_size = Size(FRAME_WIDTH, FRAME_HEIGHT);
    String videoName = "/home/zhengyihao/course/cv/hk1/file/out.avi";
    bool isColor = true;
    VideoWriter writer = cv::VideoWriter(videoName, VideoWriter::fourcc('M', 'J', 'P', 'G'), FRAME_FPS, frame_size, isColor);;
    //frame 0 
    Mat begin = Mat(FRAME_HEIGHT,FRAME_WIDTH,CV_8UC3,cvScalar(255,255,255,255));
    Mat imgResized2;
    resize(begin,imgResized2, frame_size);
    cv::putText(imgResized2, "Titles : a story of a tree  ", cvPoint(20,40),FONT_HERSHEY_TRIPLEX, 0.5, cvScalar(0, 0, 0), 1.5, CV_AA);
    cv::putText(imgResized2, "Director :zhengyihao", cvPoint(20,80),FONT_HERSHEY_TRIPLEX, 0.5, cvScalar(0, 0, 0), 1.5, CV_AA);
    cv::putText(imgResized2, "date :2022.11.13", cvPoint(20,120),FONT_HERSHEY_TRIPLEX, 0.5, cvScalar(0, 0, 0), 1.5, CV_AA);
    for (int j = 0; j < IMAGE_TIME * FRAME_FPS; j++) {
			writer.write(imgResized2);
	}

    // a photo of the author
    Mat person_pho=imread("/home/zhengyihao/course/cv/hk1/file/person.png");
    Mat imgResized1;
    resize(person_pho, imgResized1, frame_size);
    cv::putText(imgResized1, "3200103423 zhengyihao", cvPoint(20,600),FONT_HERSHEY_TRIPLEX, 0.5, cvScalar(200, 200, 250), 1.5, CV_AA);

    
    RollIn(writer, imgResized2, imgResized1);
    // writer.release();

    // scene1 in a world, there exits  a dry tree
    Mat frame = Mat(FRAME_HEIGHT,FRAME_WIDTH,CV_8UC3,cvScalar(255,255,255,255));
    Mat scene_1;
    resize(frame, scene_1, frame_size);
    RollIn(writer, imgResized1, scene_1);
    
    
    // scene2 :drt tree
   
    draw_a_line(writer,scene_1,Point(180,640),Point(180,320),Scalar(0,128,128));
    draw_a_line(writer,scene_1,Point(180,400),Point(140,300),Scalar(0,128,128));
    draw_a_line(writer,scene_1,Point(180,400),Point(220,300),Scalar(0,128,128));
    draw_a_line(writer,scene_1,Point(180,500),Point(140,400),Scalar(0,128,128));
    draw_a_line(writer,scene_1,Point(180,500),Point(220,400),Scalar(0,128,128));


    Mat scene_2 = scene_1.clone();  
    Mat scene_5 = scene_1.clone();  
    cv::putText(scene_1, "A long time ago, there was a drought", cvPoint(20,600),FONT_HERSHEY_TRIPLEX, 0.4, cvScalar(0,0,0), 1.5, CV_AA);
    // scene3 : pray
    // The tree can only produce black flowers

    Gradual(writer, scene_1, scene_2);
    draw_a_ractangle( writer, scene_2,Point(180,320),0.5,Scalar(0,0,0));
    draw_a_ractangle( writer, scene_2,Point(140,300),0.5,Scalar(0,0,0));
    draw_a_ractangle( writer, scene_2,Point(220,300),0.5,Scalar(0,0,0));
    draw_a_ractangle( writer, scene_2,Point(140,400),0.5,Scalar(0,0,0));
    draw_a_ractangle( writer, scene_2,Point(220,400),0.5,Scalar(0,0,0));

    Mat scene_3 = scene_2.clone();
    cv::putText(scene_2, "The tree can only produce black flowers", cvPoint(20,600),FONT_HERSHEY_TRIPLEX, 0.4, cvScalar(0,0,0), 1.5, CV_AA);


    Gradual(writer, scene_2, scene_3);
    Mat scene_4 = scene_3.clone();
    cv::putText(scene_3, "it prayed to God to rain some water ", cvPoint(20,600),FONT_HERSHEY_TRIPLEX, 0.4, cvScalar(0,0,0), 1.5, CV_AA);
    // scene4 : rain
    Gradual(writer, scene_3, scene_4);
    draw_a_triangle( writer,scene_4, Point(180,80),3,Scalar(255,0,0) );
    draw_a_triangle( writer,scene_4, Point(140,100),3,Scalar(255,0,0) );
    draw_a_triangle( writer,scene_4, Point(220,100),3,Scalar(255,0,0) );
    draw_a_triangle( writer,scene_4, Point(140,150),3,Scalar(255,0,0) );
    draw_a_triangle( writer,scene_4, Point(220,150),3,Scalar(255,0,0) );
    draw_a_triangle( writer,scene_4, Point(140,200),3,Scalar(255,0,0) );
    draw_a_triangle( writer,scene_4, Point(220,200),3,Scalar(255,0,0) );
    cv::putText(scene_4, "Luckily,it suddenly rained ", cvPoint(20,600),FONT_HERSHEY_TRIPLEX, 0.4, cvScalar(0,0,0), 1.5, CV_AA);
    

    // scene5 : red flower
    Gradual(writer, scene_4, scene_5);
    draw_a_ractangle( writer, scene_5,Point(180,320),0.5,Scalar(0,0,255));
    draw_a_ractangle( writer, scene_5,Point(140,300),0.5,Scalar(0,0,255));
    draw_a_ractangle( writer, scene_5,Point(220,300),0.5,Scalar(0,0,255));
    draw_a_ractangle( writer, scene_5,Point(140,400),0.5,Scalar(0,0,255));
    draw_a_ractangle( writer, scene_5,Point(220,400),0.5,Scalar(0,0,255));
    cv::putText(scene_5, "The tree finally  produced red flowers", cvPoint(20,600),FONT_HERSHEY_TRIPLEX, 0.4, cvScalar(0,0,0), 1.5, CV_AA);


    // end, thank you for watching
    Mat end = Mat(FRAME_HEIGHT,FRAME_WIDTH,CV_8UC3,cvScalar(255,255,255,255));
    Mat imgResized3;
    resize(end,imgResized3, frame_size);
    RollIn(writer, scene_5, imgResized3);
    cv::putText(imgResized3, "END ", cvPoint(20,40),FONT_HERSHEY_TRIPLEX, 0.5, cvScalar(0, 0, 0), 1.5, CV_AA);
    cv::putText(imgResized3, "Thank you for watching", cvPoint(20,80),FONT_HERSHEY_TRIPLEX, 0.5, cvScalar(0, 0, 0), 1.5, CV_AA);
    
    for (int j = 0; j < IMAGE_TIME * FRAME_FPS; j++) {
			writer.write(imgResized3);
	}


    writer.release();

}

