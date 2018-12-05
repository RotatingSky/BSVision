#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace Vision {
	enum FitMode
	{
		FIT_ELLIPSE_ORI,
		FIT_ELLIPSE_AMS,
		FIT_ELLIPSE_DIRECT,
		FIT_ELLIPSE_ITER
	};
	enum CamIndex
	{
		ON_LEFT,
		ON_RIGHT
	};
	enum ChannelMode
	{
		CHANNEL_ALL,
		CHANNEL_BLUE,
		CHANNEL_GREEN,
		CHANNEL_RED,
		CHANNEL_MERGE
	};
	void readStereoParams(std::string fileName, cv::Mat &cameraMatrixL, cv::Mat &distCoeffL, cv::Mat &cameraMatrixR, cv::Mat &distCoeffR, cv::Mat &R, cv::Mat &T);
	void saveStereoParams(std::string fileName, cv::Mat cameraMatrix[], cv::Mat distCoeffs[], cv::Mat R, cv::Mat T);
	void adaptiveMedianFilter(cv::Mat src, cv::Mat &dst, int kSize);
	void gammaFilter(cv::Mat src, cv::Mat dst, double fGamma);
	void computeThresholds(cv::Mat src, double &threshold1, double &threshold2);
	void preprocess(cv::Mat src, cv::Mat &dst, cv::Mat &srcROI, double scope, Vision::ChannelMode channelMode, double scale, double fGamma, double kernelSize, double fBilateral, double threshold1, double threshold2);
	void preprocess(cv::Mat src, cv::Mat &dst, int kernelSize, double fBilateral);
	void grabContours(cv::Mat src, cv::Mat &dst, std::vector<std::vector<cv::Point>> &contours);
	void shuffle(std::vector<int> &indexVector);
	void fitEllipse(std::vector<cv::Point> points, cv::RotatedRect &rRect, int batchSize, int epoch);
	void compatiblePoints(std::vector<cv::Point> points, std::vector<int> &coIndexes, cv::RotatedRect rRect, double delta);
	void fitEllipse(std::vector<cv::Point> points, cv::RotatedRect &rRect);
	void computeDis(cv::RotatedRect rect, std::vector<cv::Point> points, std::vector<double> &distances);
	void kMeansEllipses(std::vector<cv::Point> points, std::vector<int> &labels, cv::RotatedRect rect, int maxEpoches);
	void getEllipse(cv::Mat src, std::vector<cv::RotatedRect> &rectSet, int curveNum, int contourSize, Vision::FitMode fitMethod);
	void getEllipse(cv::Mat src, std::vector<cv::RotatedRect> &rectSet, int curveNum, int contourSize);
	void pyrEllipse(cv::Mat src, std::vector<cv::RotatedRect>& rectSet, int rectNum);
	void computeEllipseParams(std::vector<cv::Point> points, cv::RotatedRect rRect, double &ovality, double &cRate);
	void conicReconstruction(std::vector<cv::RotatedRect> rects, double &d, double &E, cv::Point3d xyz[], cv::Mat R, cv::Mat T, cv::Mat cameraMatrix[]);
	void selectPoints(cv::RotatedRect rect, std::vector<cv::Point2d> &points, double scale, int pointNum);
	void selectPoints(cv::RotatedRect rect, std::vector<cv::Point2f> &points, int pointNum);
	void crossPoints(cv::RotatedRect rect, cv::Vec3f lineCoeff, std::vector<cv::Point2d> &interPoints);
	void getTarPoint(cv::RotatedRect refRect, cv::RotatedRect tarRect, cv::Vec3f lineCoeff, cv::Point2f refPoint, cv::Point2f &tarPoint);
	void computeF(cv::Mat Al, cv::Mat Ar, cv::Mat R, cv::Mat T, cv::Mat &F);
	void computeEpipolar(std::vector<cv::Point2d> points, Vision::CamIndex index, cv::Mat F, std::vector<cv::Vec3d> &lines);
	void computeXYZ(cv::Point pl, cv::Point pr, cv::Mat Al, cv::Mat Ar, cv::Mat R, cv::Mat T, cv::Point3d &xyzl, cv::Point3d &xyzr);
	void computeCD(cv::RotatedRect rects[], cv::Point3d xyz[], double &diameter, double scale, cv::Mat cameraMatrix[], cv::Mat R, cv::Mat T);
	void fitPlane(std::vector<cv::Point3d> points, cv::Vec4d& planeCoeff);
	void computeCentDia(std::vector<cv::Point3d> points, cv::Vec4d plane, cv::Point3d &center, double &diameter);
	void computeCD(cv::RotatedRect rects[], cv::Vec4d plane[], cv::Point3d xyz[], double &diameter, int pointNum, cv::Mat cameraMatrix[], cv::Mat R, cv::Mat T);
	void computeDelDeep(cv::Vec4d plane1, cv::Vec4d plane2, double d1, double d2, double theta, double &delta, double &deep);
	void computeHoleParameters(double d1, double d2, double angle, double D, double &h, double &delta);
	void saveHoleParameters(std::string fileName, cv::Point3d center[][2], double diameter[], double delta, double deep);
	void adjustUnilinear(std::vector<cv::Point2f> basePoints, int holeNum, std::vector<cv::Point2f> &adjustedPoints);
	void adjustBilinear(std::vector<cv::Point2f> basePoints, int rowNum, int colNum, std::vector<std::vector<cv::Point2f>> &adjustedPoints);
};