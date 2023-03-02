//算法原理步骤
//1.计算x y 方向的梯度值Mat_x,Mat_y
//2.计算Mat_xx,Mat_yy,Mat_xy
//3.利用高斯函数对Mat_xx,Mat_yy,Mat_xy进行滤波
//4.计算局部特征结果矩阵M的特征值和响应函数
//C(i,j) = Det(M) - k(trace(M)^2) k∈(0.04,0.06]
//5.将计算完的响应函数的值C进行非极大值抑制，滤除边缘与非角点的点，保留满足大于设定的阈值的区域
//6.找出角点
 
 
void myDetecHarrisCornerAlgorithm(const Mat& src, vector<cv::Point> &points, double k)
{
	Mat gray;
	if (src.channels() == 3)
	{
		cvtColor(src, gray, COLOR_BGR2GRAY);
	}
	else if (src.channels() == 1)
	{
		gray = src.clone();
	}
	else
	{
		cout << "Image channnels is Error! " << endl;
		return ;
	}
 
	gray.convertTo(gray,CV_64F);
	
	//Step1
	//1.1 创建x与y方向内核
	Mat xKernel = (Mat_<double>(1, 3) << -1, 0, 1);
	//[-1,0,1] x方向
	Mat yKernel = xKernel.t();//反转矩阵。 该方法通过矩阵表达式进行矩阵求逆。
	//Mat yKernel = (Mat_<double>(3, 1) << -1, 0, 1);
	//[-1
	//	0
	//	1] y方向
	//1.2卷积获取x与y方向的梯度值
	Mat Ix, Iy;
	filter2D(gray, Ix, CV_64F, xKernel);
	filter2D(gray, Iy, CV_64F, yKernel);
	
	//Step2
	//计算Mat_xx,Mat_yy,Mat_xy
	Mat Ix2, Iy2, Ixy;
	Ix2 = Ix.mul(Ix);// 执行两个矩阵按元素相乘 获取Mat_xx。
	Iy2 = Iy.mul(Iy);// 执行两个矩阵按元素相乘 获取Mat_yy。
	Ixy = Ix.mul(Iy);// 执行两个矩阵按元素相乘 获取Mat_xy。
 
	//Step3
	//3.1获取高斯滤波内核
	Mat gaussKernel = getGaussianKernel(7,1);
	//3.2利用高斯函数对Mat_xx,Mat_yy,Mat_xy进行滤波
	filter2D(Ix2, Ix2, CV_64F, gaussKernel);
	filter2D(Iy2, Iy2, CV_64F, gaussKernel);
	filter2D(Ixy, Ixy, CV_64F, gaussKernel);
	
	//Step4
	//计算局部特征结果矩阵M的特征值和响应函数
	//C(i,j) = Det(M) - k(trace(M)^2) k∈(0.04,0.06]
	Mat cornerStrength(gray.size(),CV_64F);
	int width = gray.size().width;
	int height = gray.size().height;
	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			//M = [Ix2,Ixy
			//	   Ixy,Iy2]
			//det = Ix2 * Ix2 - Ixy^2
			//trace = Ix2 + Iy2
			//C = det - k * trace
			double det_m = Ix2.at<double>(h, w) * Iy2.at<double>(h, w) - pow(Ixy.at<double>(h, w), 2);
			double trace_m = Ix2.at<double>(h, w) + Iy2.at<double>(h, w);
			cornerStrength.at<double>(h, w) = det_m - k * trace_m * trace_m;
		}
	}
	
	//Step5
	//5.1寻找最大值
	double maxStrength;
	minMaxLoc(cornerStrength,NULL,&maxStrength,NULL,NULL);
	//5.2非极大值抑制
	Mat dilated;
	dilate(cornerStrength,dilated,Mat());
	Mat localMax;
	compare(cornerStrength,dilated,localMax,CMP_EQ);
	//5.3保留满足大于设定的阈值
	Mat cornerMap;
	double qualityLevel = 0.01;
	double thresh = qualityLevel * maxStrength;
	cornerMap = cornerStrength > thresh;// 大于标识符重载函数
	// 等同于threshold(cornerStrength,cornerMap,thresh,255,THRESH_BINARY)
	bitwise_and(cornerMap,localMax,cornerMap);
	
	//Step6
	// Iterate over the pixels to obtain all feature points 迭代像素以获得所有特征点
	for (int y = 0; y < cornerMap.rows; y++) {
 
		const uchar* rowPtr = cornerMap.ptr<uchar>(y); //行指针
 
		for (int x = 0; x < cornerMap.cols; x++) {
 
			// if it is a feature point 如果是特征点（像素值非0值为特征点）
			if (rowPtr[x]) {
 
				points.push_back(cv::Point(x, y));
			}
		}
	}
 
}
int main()
{
    
}