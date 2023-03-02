#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/imgproc/types_c.h>
#include<opencv2/imgproc/imgproc_c.h>
#include "opencv2/videoio.hpp"
#include <ctime>
using namespace cv;
using namespace std;
#define w_size 5
#define threshold 1500
void myDetecHarrisCorner(const Mat& src,Mat &R,Mat &LAMDA_1)
{
    Mat gray;
    if (src.channels() == 3)
	{
		cvtColor(src, gray, COLOR_BGR2GRAY);
	}
    gray.convertTo(gray,CV_64F);
    R = Mat(Size(gray.cols,gray.rows),CV_64F);
    LAMDA_1 = Mat(Size(gray.cols,gray.rows),CV_64F);
    // imwrite("frame0.jpg",gray);
    //Ix，为Iy为梯度矩阵
    Mat Ix, Iy;
    //调用Sobel算子获取梯度值
    Sobel(gray, Ix, CV_64F, 1, 0, 3);
    Sobel(gray, Iy, CV_64F, 0, 1, 3);
    //计算I_xx,I_yy,I_xy
	Mat Ix2, Iy2, Ixy;
    Ix2 = Ix.mul(Ix);// 执行两个矩阵按元素相乘 获取Mat_xx。
	Iy2 = Iy.mul(Iy);// 执行两个矩阵按元素相乘 获取Mat_yy。
	Ixy = Ix.mul(Iy);// 执行两个矩阵按元素相乘 获取Mat_xy。
    
    //计算M
   
    Mat M = Mat(Size(2,2),CV_64F);
    // cout <<gray.cols<<endl;
    for(int i=0;i<gray.rows;i++){
        for(int j=0;j<gray.cols;j++){
            // cout <<i<<","<<j<<endl;

            M.at<double>(0,0)=0;
            M.at<double>(0,1)=0;
            M.at<double>(1,0)=0;
            M.at<double>(1,1)=0;
            for(int p=max(i-w_size/2,0);p<=min(i+w_size/2,gray.rows-1);p++)
            {
                for(int q=max(j-w_size/2,0);q<=min(j+w_size/2,gray.cols-1);q++)
                {
                    M.at<double>(0,0)+=Ix2.at<double>(p,q);
                    M.at<double>(0,1)+=Ixy.at<double>(p,q);
                    M.at<double>(1,0)+=Ixy.at<double>(p,q);
                    M.at<double>(1,1)+=Iy2.at<double>(p,q);
                }
            }
            cv::Mat eValuesMat;
	        cv::Mat eVectorsMat;
            cv::eigen(M, eValuesMat, eVectorsMat);
            // double lamda_1=
            // double lamda_2=
            double lamda_1 = -1000000000;
            double lamda_2 = 100000000000;
            for(auto ii=0; ii<eValuesMat.rows; ii++){
		        for(auto jj=0; jj<eValuesMat.cols; jj++){
                    if(eValuesMat.at<double>(ii,jj)>lamda_1){
                        lamda_1 = eValuesMat.at<double>(ii,jj);
                    }
                     if(eValuesMat.at<double>(ii,jj)<lamda_2){
                        lamda_2 = eValuesMat.at<double>(ii,jj);
                    }
	        	}
        	}
            LAMDA_1.at<double>(i,j) = lamda_1;
            double det_M =lamda_1*lamda_2;
            double trace_M = lamda_1+lamda_2;
            double k = 0.05; 
            if((det_M-k*trace_M*trace_M)/1e5>threshold)
                R.at<double>(i,j) = (det_M-k*trace_M*trace_M)/1e5;
            // cout <<R.at<double>(i,j)<<endl;
        }
    }
    // cout<<R.size()<<endl;
    // Mat IMG_OUT ;
    // cvtColor(R, IMG_OUT, cv::COLOR_GRAY2RGB);
    // IMG_OUT.convertTo(IMG_OUT,CV_64F);
    R.convertTo(R,CV_8U);
    LAMDA_1.convertTo(LAMDA_1,CV_8U);
    //Take points of local maxima of R
    for(int i=0;i<gray.rows;i++){
        for(int j=0;j<gray.cols;j++){
             double  max_r=-1000000000000;
             int max_p;
             int max_q;
             for(int p=max(i-w_size/2,0);p<=min(i+w_size/2,gray.rows-1);p++)
            {
                for(int q=max(j-w_size/2,0);q<=min(j+w_size/2,gray.cols-1);q++)
                {
                    if(R.at<uchar>(p,q) >  max_r){
                        max_r = R.at<uchar>(p,q);
                        max_p = p;
                        max_q = q;
                    }
                }
            }
              for(int p=max(i-w_size/2,0);p<=min(i+w_size/2,gray.rows-1);p++)
            {
                for(int q=max(j-w_size/2,0);q<=min(j+w_size/2,gray.cols-1);q++)
                {
                    if(p != max_p || q != max_q)
                    {
                        R.at<uchar>(p,q) = 0;
                    }
                }
            }
        }
    }
    // 转为color img
    Mat color_Img = cv::Mat(Size(gray.cols,gray.rows), CV_8UC3, cv::Scalar(0, 0, 0));
    for(int i=0;i<gray.rows;i++){
        for(int j=0;j<gray.cols;j++){
            color_Img.at<Vec3b>(i,j)[0] = R.at<uchar>(i,j);
            color_Img.at<Vec3b>(i,j)[1] = R.at<uchar>(i,j);
            // color_Img.at<Vec3b>(i,j)[2] = 2*R.at<uchar>(i,j);
        }
    }
    R = color_Img;
    // return color_Img;
}

int main(int argc, char* argv[])
{
    cv::VideoCapture capture = cv::VideoCapture("/home/zhengyihao/course/cv/hk2/file/test.avi");
    VideoWriter ca_1 = cv::VideoWriter("test1.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), capture.get(CAP_PROP_FPS), Size(1920,1080), true);;
    VideoWriter ca_2 = cv::VideoWriter("test2.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), capture.get(CAP_PROP_FPS),  Size(1920,1080), false);;
    vector<Mat> ori_ca_frames;
    
    int i=0;
    while(true)
    {
        Mat frame;
        capture >> frame;
        if (frame.empty())
            break;
        ori_ca_frames.push_back(frame);
    }

    for(int i=0;i<ori_ca_frames.size();i++){
        Mat R ;
        Mat LAMDA_1 ;
        myDetecHarrisCorner(ori_ca_frames[i],R,LAMDA_1);
        Mat s1 = R;
        Mat res;
        s1.convertTo(s1,CV_8UC3);
        Mat s2;
        ori_ca_frames[i].convertTo(ori_ca_frames[i],CV_8UC3);
        // cout << s1.size()<<endl;
        // cout << ori_ca_frames[0].size()<<endl;
        
        addWeighted(s1, 5, ori_ca_frames[i], 0.8, 1, res);
        ca_1.write(res);
        ca_2.write(LAMDA_1);
        // imwrite("frame0.png",res);
    }
     ca_1.release();
     ca_2.release();
    // imwrite("frame0.jpg",ori_ca_frames[0]);
    // waitKey(0);
    // namedWindow("image",0);
    // imshow("image",ori_ca_frames[0]);
    // waitKey();
}