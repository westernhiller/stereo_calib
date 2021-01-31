#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <iostream>
using namespace std;
using namespace cv;

bool checkEllipseShape(const vector<Point>& contour, const RotatedRect& ellipse,double ratio=0.01)
{
	//get all the point on the ellipse point
	vector<Point> ellipse_point;

	//get the parameter of the ellipse
	Point2f center = ellipse.center;
	double a_2 = pow(ellipse.size.width*0.5,2);
	double b_2 = pow(ellipse.size.height*0.5,2);
	double ellipse_angle = (ellipse.angle*3.1415926)/180;	

	//the uppart
	for(int i=0;i<ellipse.size.width;i++)
	{
		double x = -ellipse.size.width*0.5+i;
		double y_left = sqrt( (1 - (x*x/a_2))*b_2 );

		//rotate
		//[ cos(seta) sin(seta)]
		//[-sin(seta) cos(seta)]
		cv::Point2f rotate_point_left;
		rotate_point_left.x =  cos(ellipse_angle)*x - sin(ellipse_angle)*y_left;
		rotate_point_left.y = +sin(ellipse_angle)*x + cos(ellipse_angle)*y_left;

		//trans
		rotate_point_left += center;

		//store
		ellipse_point.push_back(Point(rotate_point_left));
	}
	//the downpart
	for(int i=0;i<ellipse.size.width;i++)
	{
		double x = ellipse.size.width*0.5-i;
		double y_right = -sqrt( (1 - (x*x/a_2))*b_2 );

		//rotate
		//[ cos(seta) sin(seta)]
		//[-sin(seta) cos(seta)]
		cv::Point2f rotate_point_right;
		rotate_point_right.x =  cos(ellipse_angle)*x - sin(ellipse_angle)*y_right;
        	rotate_point_right.y = +sin(ellipse_angle)*x + cos(ellipse_angle)*y_right;

		//trans
		rotate_point_right += center;

		//store
		ellipse_point.push_back(Point(rotate_point_right));
	}

	//match shape
	return matchShapes(ellipse_point,contour,CV_CONTOURS_MATCH_I1,0) > ratio;  

}

bool findEllipses(Mat& imgSrc, const Size& boardSize, vector<Point2f>& corners)
{
    Mat gray;
    cvtColor(imgSrc, gray, COLOR_BGR2GRAY);

	Mat threshold_output;
	vector<vector<Point> > contours;

	// find contours
	int threshold_value = threshold(gray, threshold_output, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
	findContours( threshold_output, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

  	for( int i = 0; i < contours.size(); i++ )
	{ 
		//point size check
		if(contours[i].size()<10)
			continue;
		//point area
		if(contourArea(contours[i])<10)
			continue;
	
		RotatedRect minEllipse = fitEllipse(Mat(contours[i]));
		//check shape
		if(checkEllipseShape(contours[i], minEllipse))
			continue;

		// draw ellipse
		ellipse(imgSrc, minEllipse, Scalar( 0, 0, 255), 2);

		corners.push_back(minEllipse.center);
	}
	int num = boardSize.width * boardSize.height;
	if(corners.size() != num)
		return false;

	return true;
}

bool findCircles(const Mat& imgSrc, const Size& boardSize, vector<Point2f>& corners)
{
    Mat gray;
    cvtColor(imgSrc, gray, COLOR_BGR2GRAY);
	medianBlur(gray, gray, 5);
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 30,
                 gray.rows/16,  // change this value to detect circles with different distances to each other
                 100, 30, 
				 30, 100 // change the last two parameters (min_radius & max_radius) to detect larger circles
    );
	int num = boardSize.width * boardSize.height;
	if(circles.size() < num)
		return false;

	// calculate avaerage radius, to remove outliers
	float avg_radius = 0;
    for( size_t i = 0; i < circles.size(); i++ )
		avg_radius += circles[i][2];
	avg_radius /= circles.size();
	corners.clear();
    for( size_t i = 0; i < circles.size(); i++ )
	{
		if(abs(avg_radius - circles[i][2]) < 10)
		{
			Point center = Point(circles[i][0], circles[i][1]);
			corners.push_back(center);
		}
	}
	return corners.size() == num;
}


vector<vector<Point2f> >corners_l_array, corners_r_array;
int array_index = 0;

int  main(int argc, char** argv)
{
	if(argc != 3)
	{
		cout << "Usage: StereoCalib + image folder + image number" <<endl;
		return -1;
	}

	Size boardSize(9, 4);
	const float squareSize = 200.f;

	vector<vector<Point2f> > imagePoints_l;
	vector<vector<Point2f> > imagePoints_r;

	int nimages = atoi(argv[2]);
	int nvalidimages = 0;
	for(int i = 0; i < nimages; i++)
	{
		char idx[4];
		sprintf(idx, "%02d", i);

		string leftimagefile = string(argv[1]) + "/left_" + idx + ".bmp";
		string rightimagefile = string(argv[1]) + "/right_" + idx + ".bmp";
cout << "left: " << leftimagefile <<" , right: " << rightimagefile << endl;
		Mat image_left = imread(leftimagefile, IMREAD_COLOR);
		Mat image_right = imread(rightimagefile, IMREAD_COLOR);
double start = static_cast<double>(getTickCount());

		bool found_l = false, found_r = false;
		vector<Point2f> corners_l, corners_r;

		found_l = findEllipses(image_left, boardSize, corners_l);
		found_r = findEllipses(image_right, boardSize, corners_r);
		if (found_l && found_r) 
		{
			imagePoints_l.push_back(corners_l);
			imagePoints_r.push_back(corners_r);

			imshow("Left Camera", image_left);
			imshow("Right Camera", image_right);

			char c = (char)waitKey(500);
			if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
				exit(-1);
			nvalidimages++;

		}
		else
		{
			cout << i << ": failed finding chess board corners" << endl;
		}
		

double time = ((double)getTickCount() - start) / getTickFrequency();
cout << "it takes " << time << " seconds" << endl;
	}

	vector<vector<Point2f> > imagePoints[2] = { imagePoints_l, imagePoints_r };
	vector<vector<Point3f> > objectPoints;
	objectPoints.resize(nvalidimages);

	for (int i = 0; i < nvalidimages; i++)
	{
		for (int j = 0; j < boardSize.height; j++)
			for (int k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}
	cout << "Running stereo calibration ..." << endl;

	Size imageSize(1920, 1080);

	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints_l, imageSize, 0);
	cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints_r, imageSize, 0);

	Mat R, T, E, F;	

	double rms = stereoCalibrate(objectPoints, imagePoints_l, imagePoints_r,
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F, //ͼ��ߴ� ��ת���� ƽ�ƾ��� ��֡���� ��������
		CALIB_FIX_ASPECT_RATIO + //��������˸ñ�־λ����ô�ڵ��ñ궨����ʱ���Ż�����ֻͬʱ�ı�fx��fy�����̶�intrinsic_matrix������ֵ����� CV_CALIB_USE_INTRINSIC_GUESSҲû�б����ã���intrinsic_matrix�е�fx��fy����Ϊ�κ�ֵ����������أ�
		CALIB_ZERO_TANGENT_DIST + //�ñ�־�ڱ궨�߼��������ʱ��Ƚ���Ҫ����Ϊ��ȷ���������º�С�ľ�����䡣��ͼ���������0�ᵼ���������ź���ֵ���ȶ���ͨ�����øñ�־���Թر�����������p1��p2����ϣ���������������Ϊ0//��ֵ�����ؽǵ�
		CALIB_USE_INTRINSIC_GUESS + // cvCalibrateCamera2()�����ڲ��������ʱ��ͨ������Ҫ�������Ϣ��������˵������cx��cy��ͼ�����ģ��ĳ�ʼֵ����ֱ�Ӵӱ���image_size�еõ�����(H-1)/2,(W-1)/2)������������˸ñ�����ôinstrinsic_matrix���������ȷ��ֵ������������ʼ�²⣬ΪcvCalibrateCamera2()���Ż�ʱ����
		CALIB_SAME_FOCAL_LENGTH + //�ñ�־λ���Ż���ʱ��ֱ��ʹ��intrinsic_matrix���ݹ�����fx��fy��
		CALIB_RATIONAL_MODEL +
		CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,//�̶��������k1,k2,k3����������������ͨ�������Щ��־����Ϊ����ֵ��һ������һ������Ӧ����Ϊ0����ʼʹ������͸����
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));//��ֵ�����ؽǵ�

	cout << "done with RMS error=" << rms << endl;

	double err = 0;
	int npoints = 0;
	//���㼫������
	vector<Vec3f> lines[2]; //����
	for (int i = 0; i < nvalidimages; i++)
	{
		//��ĳͼ���нǵ�����
		int npt = (int)imagePoints_l[i].size();
		Mat imgpt[2];
		imgpt[0] = Mat(imagePoints_l[i]);
		undistortPoints(imgpt[0], imgpt[0], cameraMatrix[0], distCoeffs[0], Mat(), cameraMatrix[0]);
		computeCorrespondEpilines(imgpt[0], 0 + 1, F, lines[0]);

		imgpt[1] = Mat(imagePoints_r[i]); //ĳͼ�Ľǵ���������
		undistortPoints(imgpt[1], imgpt[1], cameraMatrix[1], distCoeffs[1], Mat(), cameraMatrix[1]); //����У����Ľǵ�����
		computeCorrespondEpilines(imgpt[1], 1 + 1, F, lines[1]); //���㼫��

		for (int j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
					imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average epipolar err = " << err / npoints << endl;


	Mat R1, R2, P1, P2, Q; //���������
	Rect validRoi[2];

	stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1], //���ڲ������� //��������
		imageSize, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]); //ͼ��ߴ� ��ת���� ƽ�ƾ��� ����ת�������� ����ת�������� ��ƽ�ƽ������� ��ƽ�ƽ������� ��Ƚ�������
cout << "validRoi[0] = (" << validRoi[0].x <<","<<validRoi[0].y<<","<<validRoi[0].width<<","<<validRoi[0].height<<")"<<endl;
cout << "validRoi[1] = (" << validRoi[1].x <<","<<validRoi[1].y<<","<<validRoi[1].width<<","<<validRoi[1].height<<")"<<endl;

		Rect overlapRoi = validRoi[0] & validRoi[1];
cout << "overlapRoi = (" << overlapRoi.x <<","<<overlapRoi.y<<","<<overlapRoi.width<<","<<overlapRoi.height<<")"<<endl;

	FileStorage fs("calib.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
                fs<< "S" << imageSize;
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" 
			<< P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the calibration parameters\n";


	// COMPUTE AND DISPLAY RECTIFICATION
	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;
	double sf;
	int w, h;

	sf = 1200. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width*sf);
	h = cvRound(imageSize.height*sf);
	canvas.create(h, w * 2, CV_8UC3);

	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 256, 7,
		10 * 7 * 7,
		40 * 7 * 7,
		1, 63, 10, 200, 32, StereoSGBM::MODE_SGBM);

	for(int i = 0; i < nimages; i++)
	{
		Mat rimg, cimg;
		char idx[4];
		sprintf(idx, "%02d", i + 1);

		string leftimagefile = string(argv[1]) + "/left_" + idx + ".bmp";
		string rightimagefile = string(argv[1]) + "/right_" + idx + ".bmp";

		Mat image_left = imread(leftimagefile, IMREAD_COLOR);
		Mat image_right = imread(rightimagefile, IMREAD_COLOR);

		remap(image_left, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
		rimg.copyTo(cimg);

		Rect validRoi2 = validRoi[0] & validRoi[1];
		Mat rectified_left = rimg(validRoi2);
//		imwrite(string("rected_left") + idx + string(".png"), rectified_left);

		Mat canvasPart1 = canvas(Rect(w * 0, 0, w, h));
		resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
		Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
				cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

		remap(image_right, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
		rimg.copyTo(cimg);

		Mat rectified_right = rimg(validRoi2);
//		imwrite(string("rected_right") + idx + string(".png"), rectified_right);

		Mat canvasPart2 = canvas(Rect(w * 1, 0, w, h));
		resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
		Rect vroi2 = Rect(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
			cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));

		Rect vroi = vroi1 & vroi2;

		rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
		rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

		for (int j = 0; j < canvas.rows; j += 32)
			line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);

		imshow("rectified", canvas);//��������������ͼ
		imwrite(string("rected_") + idx + string(".png"), canvas);

		Mat imgLeft = canvasPart1(vroi).clone();
		Mat imgRight = canvasPart2(vroi).clone();

		cvtColor(imgLeft, imgLeft, COLOR_BGR2GRAY);
		cvtColor(imgRight, imgRight, COLOR_BGR2GRAY);
		Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
		Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

		sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

		sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
		Mat Mask;
		cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
		applyColorMap(sgbmDisp8U, sgbmDisp8U, COLORMAP_HSV);
		Mat  sgbmDisparityShow, sgnm[3];
		sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

		split(sgbmDisparityShow, sgnm);
		imshow("sgbmDisparity", sgbmDisparityShow);//����ͼ��ʾ
		imshow("sgbm", sgnm[1]);//�Ҷ���ʾ
		
//		imwrite(string("rected_") + idx + string(".png"), canvas);
		char c = (char)waitKey(500);
		if (c == 27 || c == 'q' || c == 'Q')
			break;


	}

	return 0;

}
