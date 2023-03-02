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
#include<fstream>  
#include <json/json.h>
#include <string>
using namespace cv;
using namespace std;

#define num_of_persons 41
#define faces_per_person 5

const int WIDTH = 73;
const int HEIGHT = 89;

class img_face {
public:
	Mat ori_Img;
	Mat gray_Img;
	int x1, y1, x2, y2;
	Mat_<double> img_vect;
	void load_img(std::string& path);
};

void img_face::load_img(std::string& path) {
    
	Json::Reader reader;
	Json::Value root;
    ifstream in(path+".json", ios::binary);
    if (reader.parse(in, root))
	{
        x1 = root["centre_of_left_eye"][0].asInt();
        y1 = root["centre_of_left_eye"][1].asInt();
        x2 = root["centre_of_right_eye"][0].asInt();
        y2 = root["centre_of_right_eye"][1].asInt();
    }
	ori_Img = cv::imread(path + ".pgm");
	gray_Img = imread(path + ".pgm", cv::IMREAD_GRAYSCALE);
	Point center((x1 + x2) / 2, (y1 + y2) / 2);
	double angle = atan((double)(y2 - y1) / (double)(x2 - x1)) * 180.0 / CV_PI;
    Mat trans_mat;
	trans_mat = getRotationMatrix2D(center, angle, 1.0);
	trans_mat.at<double>(0, 2) += 37.0 - center.x;
	trans_mat.at<double>(1, 2) += 30.0 - center.y;
    Mat trans;
	warpAffine(gray_Img, trans, trans_mat, gray_Img.size() * 4 / 5);
	cv::equalizeHist(trans, trans);
	trans.copyTo(img_vect);
	img_vect = img_vect.reshape(1, 1).t();
}

Mat reconstruct(int k,Mat e_vector_mat, Mat e_value_mat,Mat mean,Mat d){
   e_vector_mat=e_vector_mat.rowRange(0, k);
    d = d.rowRange(0, k);
    Mat _result(Size(WIDTH, HEIGHT), CV_64FC1);
    Mat result1 =e_vector_mat.t() * d ;
    result1 =  result1+mean;
    Mat _temp;
    result1.copyTo(_temp);
    _temp = _temp.t();
    for (int i = 0; i < HEIGHT; ++i) {
		_temp.colRange(i * WIDTH, (i + 1) * WIDTH).convertTo(_result.row(i), CV_64FC1);
    }
    normalize(_result, _result, 1.0, 0.0, NORM_MINMAX);
	_result.convertTo(_result, CV_8U, 255);
    return _result;
}

int main(int argc, char** argv){
    char* model_name;
    string file_name;
    if (argc <= 2) {
        cout<<"the num of args is wrong !" <<endl;
		return 0;
	}else{
		file_name = argv[1];
        model_name = argv[2];
    }
    FileStorage model(model_name, FileStorage::READ |  FileStorage::BASE64);
	Mat e_vector_mat, e_value_mat,mean;
    model["e_vector_mat"] >> e_vector_mat;
	model["e_value_mat"] >> e_value_mat;
    model["mean"] >> mean;
    img_face face;
    face.load_img(file_name);
    Mat temp;
    face.img_vect.copyTo(temp);
 
    Mat d = e_vector_mat*(temp-mean)  ;


    Mat result1 = reconstruct(10,e_vector_mat, e_value_mat,mean,d);
    Mat result2 = reconstruct(25,e_vector_mat, e_value_mat,mean,d);
    Mat result3 = reconstruct(50,e_vector_mat, e_value_mat,mean,d);
    Mat result4 = reconstruct(100,e_vector_mat, e_value_mat,mean,d);

    vector<Mat> rec4;
    rec4.push_back(result1);
    rec4.push_back(result2);
    rec4.push_back(result3);
    rec4.push_back(result4);

    Mat output1;
	hconcat(rec4, output1);
	imwrite("rec4.png", output1);


}