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
class img_face;
void load_data(std::string& path);
std::vector<img_face> faces;
std::vector<cv::Mat_<double>> imgs_list;
cv::Mat_<double> imgs_1;

class img_face {
public:
	Mat ori_Img;
	Mat gray_Img;
	int x1, y1, x2, y2;
	Mat_<double> img_vect;
	void load_img(std::string& path);
};

void load_data(std::string& dirPath) {
    for (int i = 1; i <= num_of_persons; i++)
    {
        for (int j = 1; j <= faces_per_person; ++j) {
            std::string entry_path = dirPath + "/s" + std::to_string(i) + "/" + (std::to_string(j));
            img_face face;
            face.load_img(entry_path);
            faces.push_back(face);
            imgs_list.push_back(face.img_vect);
        }
    }
	hconcat(imgs_list, imgs_1);
}

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
    // cout<<img_vect.size()<<endl;
}
int main(int argc, char** argv) {
    char* model_name;
	double energy = 0.95;
	if (argc <= 2) {
        cout<<"the num of args is wrong !" <<endl;
		return 0;
	}
	else if (argc >= 3) {
		model_name = argv[2];
		energy = atof(argv[1]);
	}
    std::string dirPath("/home/zhengyihao/course/cv/hk3/dataset");

    load_data(dirPath);
	Mat _imgs_1;
    imgs_1.copyTo(_imgs_1);
    Mat cov;
	
    PCA pca(_imgs_1, Mat(), CV_PCA_DATA_AS_COL, energy);
    Mat mean = pca.mean.clone();
    Mat_<float> e_value_mat = pca.eigenvalues;
    Mat e_vector_mat = pca.eigenvectors.clone();

    cout<<e_vector_mat.size()<<endl;

	FileStorage model(model_name, FileStorage::WRITE |FileStorage::BASE64);
	model << "e_vector_mat" << e_vector_mat;
	model << "e_value_mat" << e_value_mat;
    model << "mean" << mean;
	model.release();

	vector<Mat> top10;
	for (int i = 0; i < 10; ++i) {
        Mat result(Size(WIDTH, HEIGHT), CV_64FC1);
        Mat temp;
        e_vector_mat.row(i).copyTo(temp);
        cout<<"temp:"<<temp.size()<<endl;

	    for (int i = 0; i < HEIGHT; ++i) {
		temp.colRange(i * WIDTH, (i + 1) * WIDTH).convertTo(result.row(i), CV_64FC1);
	    }
    	normalize(result, result, 1.0, 0.0, NORM_MINMAX);

        top10.push_back(result);
	}
	Mat output1;
	hconcat(top10, output1);
	output1.convertTo(output1, CV_8U, 255);
	imwrite("top10.png", output1);
    Mat _result(Size(WIDTH, HEIGHT), CV_64FC1);
    Mat temp;
    mean.copyTo(temp);
    temp = temp.t();

    for (int i = 0; i < HEIGHT; ++i) {
		temp.colRange(i * WIDTH, (i + 1) * WIDTH).convertTo(_result.row(i), CV_64FC1);
    }
    normalize(_result, _result, 1.0, 0.0, NORM_MINMAX);
    
	_result.convertTo(_result, CV_8U, 255);
	imwrite("mean.png", _result);
	return 0;
}

