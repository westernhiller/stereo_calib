#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <iostream>
using namespace std;
using namespace cv;

bool checkEllipseShape(const vector<Point>& contour, const RotatedRect& ellipse,double ratio=0.05)
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

bool LocalBinarize(const Mat &imgSrc, Mat &imgDst, const Size &szWindow)
{
    if ((NULL == imgSrc.data) || (imgSrc.channels() != 1) 
        || (szWindow.width <= 0 || szWindow.height <= 0))
    {
        return false;
    }

    if ((NULL == imgDst.data) || (imgDst.size() != imgSrc.size()))
    {
        imgDst = Mat::zeros(imgSrc.size(), CV_8UC1);
    }

    if (szWindow.width >= imgSrc.cols || szWindow.height >= imgSrc.rows)
    {
        threshold(imgSrc, imgDst, 0, 255, THRESH_OTSU);
        return true;
    }

    int iWinWid = szWindow.width;
    int iWinHgt = szWindow.height;
    int iGridWid = imgSrc.cols / iWinWid;
    int iGridHgt = imgSrc.rows / iWinHgt;
    int iLastGridWid = imgSrc.cols - ((iWinWid - 1) * iGridWid);
    int iLastGridHgt = imgSrc.rows - ((iWinHgt - 1) * iGridHgt);

    for (int i = 0; i < iWinHgt; i++)
    {
        for (int j = 0; j < iWinWid; j++)
        {
            // get grid rectangle
            Rect rctGrid;
            rctGrid.x = j * iGridWid;
            rctGrid.y = i * iGridHgt;
            if (j == (iWinWid - 1))
            {
                // last horizontal grid
                rctGrid.width = iLastGridWid;
            }
            else
            {
                rctGrid.width = iGridWid;
            }

            if (i == (iWinHgt - 1))
            {
                // last vertical grid
                rctGrid.height = iLastGridHgt;
            }
            else
            {
                rctGrid.height = iGridHgt;
            }

            // binarize the grid area
            Mat imgSrcGrid = imgSrc(rctGrid);
            Mat imgDstGrid = imgDst(rctGrid);
            threshold(imgSrcGrid, imgDstGrid, 0, 255, THRESH_OTSU);
			
//    		erode(imgDstGrid, imgDstGrid, Mat());
//   		dilate(imgDstGrid, imgDstGrid, Mat());
        }
    }

    return true;
}


bool findEllipses(Mat& imgSrc, const Size& boardSize, vector<Point2f>& corners)
{
    Mat gray;
    cvtColor(imgSrc, gray, COLOR_BGR2GRAY);

	Mat threshold_output;
	vector<vector<Point> > contours;

	// find contours
	imshow("gray", gray);
	LocalBinarize(gray, threshold_output, cv::Size(15, 15));
//	int threshold_value = threshold(gray, threshold_output, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
	imshow("binary", threshold_output);

	Mat gray2 = 255 - gray;
	Mat bin2;
	LocalBinarize(gray2, bin2, cv::Size(9, 9));
	imshow("gray2", gray2);
	imshow("binary2", bin2);

//	findContours( bin2, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	findContours( threshold_output, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

  	for( int i = 0; i < contours.size(); i++ )
	{ 
		//point size check
		if(contours[i].size()<32)
			continue;
		//point area
		if(contourArea(contours[i])<300)
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

			cv::circle(imgSrc, center, circles[i][2], Scalar( 0, 0, 255));
		}
	}
	return corners.size() == num;
}

int  main(int argc, char*argv[])
{
	if(argc != 2)
	{
		cout << "Usage: irtest + image" <<endl;
		return -1;
	}

	Size boardSize(9, 4);
	const float squareSize = 200.f;
	vector<Point2f> corners;

	Mat input_image = imread(argv[1], IMREAD_COLOR);

	findEllipses(input_image, boardSize, corners);
//	findCircles(input_image, boardSize, corners);
	imshow("ellipse", input_image);
	waitKey(0);

	return 0;
}
