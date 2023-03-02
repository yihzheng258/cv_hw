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
#define faces_per_person_total 10
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
}


int main(int argc, char** argv) {	
    if (argc <= 2) {
		 cout<<"the num of args is wrong !" <<endl;
		return 0;
	}
    std::string dirPath("/home/zhengyihao/course/cv/hk3/dataset");
	load_data(dirPath);

	char* model_name ;
	string file_name ;

	if (argc >= 3) {
        file_name = argv[1];
		model_name = argv[2];
    }
    img_face face;
    Mat origin_mat = imread(file_name+".pgm");
    face.load_img(file_name);
    Mat e_vector_mat, e_value_mat,mean;
    FileStorage model(model_name, FileStorage::READ |  FileStorage::BASE64);
    model["e_vector_mat"] >> e_vector_mat;
	model["e_value_mat"] >> e_value_mat;
    model["mean"] >> mean;


    Mat testvector = e_vector_mat * (face.img_vect - mean);
    
    for(int i=0;i<imgs_1.cols;i++)
    {
        imgs_1.col(i) = imgs_1.col(i) - mean;
    }
    Mat distance = e_vector_mat * imgs_1;
    double min_d = norm(testvector, distance.col(0), NORM_L2);
    double temp_d = 0;
    int min_i = 0;
    for (int i = 0; i < distance.cols; ++i) {
        temp_d = norm(testvector, distance.col(i), NORM_L2);
        
        if (temp_d <= min_d) {
            min_d = temp_d;
            min_i = i;
        }
    }
    int index = (min_i+1)/5+1;
    
    string text = "result:" + to_string(index);
    putText(origin_mat, text, Point(10, 95), CV_FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, 8);
    imwrite("result.png",origin_mat);
    int k[7] ={10,20,40,60,100,130,160};
    Mat temp;
    distance.copyTo(temp);
    for(int p=0;p<7;p++)
    {
        temp = distance.rowRange(0,k[p]);
        cout <<temp.size() <<endl;
        int correct =0;
        for (int i = 1; i <= num_of_persons; ++i) {
            for (int j = 6; j <= faces_per_person_total; j++)
            {
                string facePath("/home/zhengyihao/course/cv/hk3/dataset/s" + to_string(i) +"/" + std::to_string(j));
                img_face face;
                face.load_img(facePath);
                testvector = e_vector_mat.rowRange(0,k[p]) * (face.img_vect - mean);

                double min_d = norm(testvector, temp.col(0), NORM_L2);

                for (int i = 0; i < temp.cols; ++i) {
                    temp_d = norm(testvector, temp.col(i), NORM_L2);
                    if (temp_d <= min_d) {
                        min_d = temp_d;
                        min_i = i;
                    }
                }
                int index = (min_i+1)/5+1;
                if(index == i )
                { 
                    correct++;
                }
            }
        }
        float rank1 = float(correct)/(num_of_persons*5);
        cout << "rank1:" <<rank1 <<endl;
    }
}


