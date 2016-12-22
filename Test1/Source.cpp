#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <windows.h>
#include <ios>

using namespace cv;
using namespace std;

int photocount = 0;
int key;

int x = 0;
int y = 0;
int z = 0;

double k;
double l;
double m;
double n;

Rect roi_b;
Rect roi_c;

size_t ic = 0; // ic is index of current element
int ac = 0; // ac is area of current element

size_t ib = 0; // ib is index of biggest element
int ab = 0; // ab is area of biggest element

struct EyeSystem //utk storage time shj
{

	int a;
	int b;
	int c;

	float d;
	float e;
	float f;
	float g;

}info[100];


void process(Mat quadimage)
{
	std::ofstream log("D:/fyptest/test.txt", std::ios_base::app | std::ios_base::out);

	imshow("quad", quadimage);

	uint8_t* pixelPtr = (uint8_t*)quadimage.data;
	int cn = quadimage.channels();
	Scalar_<uint8_t> bgrPixel;

	for (int i = 0; i < quadimage.rows; i++)
	{
		for (int j = 0; j < quadimage.cols; j++)
		{
			uchar b = bgrPixel.val[0];
			uchar g = bgrPixel.val[1];
			uchar r = bgrPixel.val[2];

			int B = (int)(b);
			int G = (int)(g);
			int R = (int)(r);

			bgrPixel.val[0] = pixelPtr[i*quadimage.cols*cn + j*cn + 0]; // B
			bgrPixel.val[1] = pixelPtr[i*quadimage.cols*cn + j*cn + 1]; // G
			bgrPixel.val[2] = pixelPtr[i*quadimage.cols*cn + j*cn + 2]; // R

			x = x + R;
			y = y + G;
			z = z + B;

			k = sqrt(x / 1000);
			l = sqrt(y / 1000);
			m = sqrt(z / 1000);
			n = cubeRoot((k / 3) + (2 * l) - (2 * m / 5));
		}
	}

	log << " R" << x << " G" << y << " B" << z << " k" << k << " l" << l << " m" << m << " n" << n << "\n";
	cout << " R" << x << " G" << y << " B" << z << " k" << k << " l" << l << " m" << m << " n" << n << "\n";

	info[0].a = x;
	info[1].b = y;
	info[2].c = z;
	info[3].d = k;
	info[4].e = l;
	info[5].f = m;
	info[6].g = n;


	log.close();

	printf(" Name is: %d \n", info[0].a);

}


void quad(Mat cropImage)
{
	Mat top_left = cropImage(cv::Range(0, cropImage.rows / 2 - 1), cv::Range(0, cropImage.cols / 2 - 1));
	Mat top_right = cropImage(cv::Range(0, cropImage.rows / 2 - 1), cv::Range(cropImage.cols / 2, cropImage.cols - 1));
	Mat bottom_left = cropImage(cv::Range(cropImage.rows / 2, cropImage.rows - 1), cv::Range(0, cropImage.cols / 2 - 1));
	Mat bottom_right = cropImage(cv::Range(cropImage.rows / 2, cropImage.rows - 1), cv::Range(cropImage.cols / 2, cropImage.cols - 1));

	
	imshow("TL", top_left);
	/*imshow("TR", top_right);
	imshow("BL", bottom_left);
	imshow("BR", bottom_right);*/


	for (int i = 0; i < 4; i++)
	{
		if (i == 0)
		{
			process(top_left);
		}
		else if (i == 1)
		{
			process(top_right);
		}
		else if (i == 2)
		{
			process(bottom_left);
		}
		else if (i == 3)
		{
			process(top_right);
		}

	}
}


void region(Mat img)
{
	//destroyWindow("detected");
	//destroyWindow("Camera");
	destroyAllWindows;


	Mat gray;
	cvtColor(img, gray, CV_RGB2GRAY);
	blur(gray, gray, Size(5, 5));
	Canny(gray, gray, 100,100);
	Mat bw = gray > 70;

	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(bw, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat mask = Mat::zeros(gray.rows, gray.cols, CV_8UC1);

	int w_threshold = 20;
	int h_threshold = 20;
	vector<int> selected;

	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(0, 0, 0);
		Rect R = boundingRect(contours[i]);
		if (R.width > w_threshold && R.height > h_threshold)
		{
			selected.push_back(i);
			drawContours(img, contours, i, color, 2, 8, hierarchy, 0, Point());
			Mat cropImage = img(R);
			imshow("iris crop image from eye ", cropImage); //executed
			quad(cropImage);
		}
	}
	
	
}

void regionon(Mat img)
{
	//destroyWindow("detected");
	//destroyWindow("Camera");
	destroyAllWindows;

	// on time detect boundary circle
	Mat gray;
	cvtColor(img, gray, CV_RGB2GRAY);
	blur(gray, gray, Size(9, 9));
	Canny(gray, gray, 40, 40);
	Mat bw = gray > 60;

	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(bw, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat mask = Mat::zeros(gray.rows, gray.cols, CV_8UC1);

	int w_threshold = 50;
	int h_threshold = 50;
	vector<int> selected;

	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(0, 0, 0);
		Rect R = boundingRect(contours[i]);

		if (R.width > w_threshold && R.height > h_threshold)
		{
			selected.push_back(i);
			drawContours(img, contours, i, color, 2, 8, hierarchy, 0, Point());
			Mat cropImage = img(R);
			quad(cropImage);
		}
	}
	
}

String inttostr(int input)
{
	stringstream ss;
	ss << input;
	return ss.str();
}

void detectAndDisplay(Mat frame)
{

	std::vector<Rect> eyes;
	Mat frame_gray;
	Mat crop;
	Mat res;
	Mat gray;


	//int photocount = 0;
	String imagename;


	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	CascadeClassifier eyes_cascade;
	eyes_cascade.load("C:/Users/Aliyah/Desktop/coding_fyp/Test1/Test1/haarcascade_eye.xml");

	eyes_cascade.detectMultiScale(frame, eyes, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));


	// Set Region of Interest
	/*Rect roi_b;
	Rect roi_c;

	size_t ic = 0; // ic is index of current element
	int ac = 0; // ac is area of current element

	size_t ib = 0; // ib is index of biggest element
	int ab = 0; // ab is area of biggest element*/

	for (ic = 0; ic < eyes.size(); ic++) // Iterate through all current elements (detected eyes)

	{
		roi_c.x = eyes[ic].x;
		roi_c.y = eyes[ic].y;
		roi_c.width = (eyes[ic].width);
		roi_c.height = (eyes[ic].height);

		ac = roi_c.width * roi_c.height; // Get the area of current element (detected eyes)

		roi_b.x = eyes[ib].x;
		roi_b.y = eyes[ib].y;
		roi_b.width = (eyes[ib].width);
		roi_b.height = (eyes[ib].height);

		ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

		if (ac > ab)
		{
			ib = ic;
			roi_b.x = eyes[ib].x;
			roi_b.y = eyes[ib].y;
			roi_b.width = (eyes[ib].width);
			roi_b.height = (eyes[ib].height);
		}

		crop = frame(roi_b);
		/*resize(crop, res, Size(100, 100), 0, 0, INTER_LINEAR); // This will be needed later while saving images
		cvtColor(crop, gray, COLOR_BGR2GRAY); // Convert cropped image to Grayscale		*/
		Mat tmp;
		Mat dst;

		tmp = crop;
		dst = tmp;
		pyrUp(tmp, dst, Size(tmp.cols * 2, tmp.rows * 2));
		//imshow("detected", crop);
		imshow("detected", dst);
		destroyAllWindows;
	}


	if (key == 'c')
	{
		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(100);


		photocount++;// increment image number for each capture
		imagename = "D:/fyptest/ " + inttostr(photocount) + ".jpg";
		imwrite(imagename, crop, compression_params);
		region(crop);
	}

}


void takepix()
{

	Mat frame;

	VideoCapture cap(0); //open camera no.0  0=internal 1=external
	while ((key = waitKey(10)) != 27)
	{
		cap >> frame;
		imshow("Camera", frame);
		detectAndDisplay(frame);
	}
}

void newuser()
{
	takepix();

}

int main()
{
	int k = 2;
	RNG rng(12345);
	int key;

	while (k != 5000)
	{
		if (k == 1) //coding on time jadi 
		{
			Mat img = imread("D:/fyptest/Step1.jpg", 1);
			regionon(img);
		}

		else
		{
			newuser(); //new application
		}
	}


	waitKey(0);
	return 0;
	
}







