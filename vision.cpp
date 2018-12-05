/*
 *	Create by Sny on 2018-06-04.
 */

#include <ctime>
#include "vision.h"

void Vision::readStereoParams(std::string fileName, cv::Mat &cameraMatrixL, cv::Mat &distCoeffL, cv::Mat &cameraMatrixR, cv::Mat &distCoeffR, cv::Mat &R, cv::Mat &T)
{
	cv::FileStorage fs(fileName, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		std::cout << "Failed to open " << fileName << "!" << std::endl;
		return;
	}
	fs["Intrinsics_Camera_Left"] >> cameraMatrixL;
	fs["Distortion_Left"] >> distCoeffL;
	fs["Intrinsics_Camera_Right"] >> cameraMatrixR;
	fs["Distortion_Right"] >> distCoeffR;
	fs["Rotated_Matrix"] >> R;
	fs["Translation"] >> T;
}

void Vision::saveStereoParams(std::string fileName, cv::Mat cameraMatrix[], cv::Mat distCoeffs[], cv::Mat R, cv::Mat T)
{
	// Save calibration parameters
	cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
	if (!fs.isOpened())
	{
		std::cout << "Error: can not save calibration parameters" << std::endl;
		return;
	}
	fs << "Intrinsics_Camera_Left" << cameraMatrix[0]
		<< "Distortion_Left" << distCoeffs[0]
		<< "Intrinsics_Camera_Right" << cameraMatrix[1]
		<< "Distortion_Right" << distCoeffs[1]
		<< "Rotated_Matrix" << R
		<< "Translation" << T;
	fs.release();
}

void Vision::adaptiveMedianFilter(cv::Mat src, cv::Mat &dst, int kSize)
{
	if (src.channels() == 3)
	{
		cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	}
	else
	{
		dst = src.clone();
	}
	
	if (kSize % 2 == 0)
	{
		return;
	}
	else
	{
		int padding = (kSize - 1) / 2;
		std::vector<uchar> kernelVec;
		for (int j = padding; j < src.rows - padding; ++j)
		{
			for (int i = padding; i < src.cols - padding; ++i)
			{
				// Get all kernel ROI pixel values.
				for (int m = 0; m < kSize; ++m)
				{
					uchar *data = src.ptr<uchar>(j - padding + m);
					for (int n = 0; n < kSize; ++n)
					{
						kernelVec.push_back(data[i - padding + n]);
					}
				}
				// Sort the vector.
				uchar temp;
				for (size_t p = 0; p < kernelVec.size() - 1; ++p)
				{
					for (size_t q = 0; q < kernelVec.size() - p - 1; ++q)
					{
						if (kernelVec[q] > kernelVec[q + 1])
						{
							temp = kernelVec[q];
							kernelVec[q] = kernelVec[q + 1];
							kernelVec[q + 1] = temp;
						}
					}
				}
				// Adaptive method.
				if (src.at<uchar>(i, j) == kernelVec[0] ||
					src.at<uchar>(i, j) == kernelVec[kernelVec.size() - 1])
				{
					dst.at<uchar>(i, j) = kernelVec[(kernelVec.size() + 1) / 2];
				}
				else
				{
					// Keep original pixel value.
				}
			}
		}
	}
}

void Vision::gammaFilter(cv::Mat src, cv::Mat dst, double fGamma)
{
	uchar lut[256];
	for (int i = 0; i < 256; i++) {
		lut[i] = cv::saturate_cast<uchar>(pow((double)(i / 255.0), fGamma) * 255.0);
	}
	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1: {
		cv::MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++) {
			*it = lut[(*it)];
		}
		break;
	}
	case 3: {
		cv::MatIterator_<cv::Vec3b> it, end;
		for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++) {
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}
		break;
	}
	}
}

// Compute the 2 thresholds for Canny filter.
void Vision::computeThresholds(cv::Mat src, double &threshold1, double &threshold2)
{
	// Convert to gray image.
	if (src.channels() == 3)
	{
		cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
	}

	// Get histogram of src.
	cv::MatND dstHist;
	int channels = 0;
	int dims = 1;
	int size = 256;
	float hranges[] = { 0, 255 };
	const float *ranges[] = { hranges };
	cv::calcHist(&src, 1, &channels, cv::Mat(), dstHist, dims, &size, ranges);

	// Compute prob[] and uT.
	float sum = 0;
	for (int i = 0; i < 256; i++)
	{
		sum += dstHist.at<float>(i);
	}
	std::vector<float> prob(256, 0);
	float uT = 0;
	for (int i = 0; i < 256; i++)
	{
		prob[i] = dstHist.at<float>(i) / sum;
		uT += i * prob[i];
	}

	//// Single threshold method.
	//float maxSigma2 = 0;
	//int bestT = 0;
	//float P0 = 0;
	//float u = 0;
	//for (int T = 0; T < 256; T++)
	//{
	//	P0 += prob[T];
	//	u += T * prob[T];
	//	float u0 = u / P0;
	//	float u1 = (uT - u) / (1 - P0);
	//	float sigma2 = P0 * (1 - P0) * (u0 - u1) * (u0 - u1);
	//	if (sigma2 > maxSigma2)
	//	{
	//		maxSigma2 = sigma2;
	//		bestT = T;
	//	}
	//}
	//threshold1 = 1.8 * bestT;
	//threshold2 = 0.6 * bestT;

	// Compute thresholds.
	float w[3] = { 0 };
	float u_[3] = { 0 };
	float u[3] = { 0 };
	float maxSigma2 = 0;
	for (int i = 0; i < 255; i++)
	{
		w[0] += prob[i];
		u_[0] += i * prob[i];
		u[0] = u_[0] / w[0];
		for (int j = i + 1; j < 256; j++)
		{
			w[1] += prob[j];
			u_[1] += j * prob[j];
			u[1] = u_[1] / w[1];
			w[2] = 1 - w[0] - w[1];
			u_[2] = uT - u_[0] - u_[1];
			u[2] = u_[2] / w[2];
			float sigma2 = w[0] * (u[0] - uT) * (u[0] - uT)
				+ w[1] * (u[1] - uT) * (u[1] - uT)
				+ w[2] * (u[2] - uT) * (u[2] - uT);
			if (sigma2 > maxSigma2)
			{
				maxSigma2 = sigma2;
				threshold1 = i;
				threshold2 = j;
			}
		}
	}
}

void Vision::preprocess(cv::Mat src, cv::Mat &dst, cv::Mat &srcROI, double scope, Vision::ChannelMode channelMode, double scale, double fGamma, double kernelSize, double fBilateral, double threshold1, double threshold2)
{
	// Clip image and resize the image.
	cv::Rect imgROI;
	int border = MIN(src.cols, src.rows);
	imgROI.width = int(border * scope);
	imgROI.height = int(border * scope);
	imgROI.x = (src.cols - imgROI.width) / 2;
	imgROI.y = (src.rows - imgROI.height) / 2;
	srcROI = src(imgROI);
	cv::resize(srcROI, srcROI, cv::Size(int(scale * srcROI.cols), int(scale * srcROI.rows)));

	cv::Mat grayImg;
	std::vector<cv::Mat> rgbChannels(3);
	if (src.channels() == 3)
	{
		cv::split(srcROI, rgbChannels);
		switch (channelMode)
		{
		case Vision::CHANNEL_BLUE: grayImg = rgbChannels[0]; break;
		case Vision::CHANNEL_GREEN: grayImg = rgbChannels[1]; break;
		case Vision::CHANNEL_RED: grayImg = rgbChannels[2]; break;
		case Vision::CHANNEL_MERGE:
		case Vision::CHANNEL_ALL:
		default: cvtColor(srcROI, grayImg, cv::COLOR_BGR2GRAY); break;
		}
	}
	else if(src.channels() == 1)
	{
		grayImg = src.clone();
	}

	// Gamma transformation.
	gammaFilter(grayImg, grayImg, fGamma);

	cv::Mat filterImg;
	cv::Mat midImg;
	cv::Mat edgeImg;
	cv::GaussianBlur(grayImg, filterImg, cv::Size(kernelSize, kernelSize), 0, 0);
	cv::bilateralFilter(filterImg, midImg, (int)fBilateral, fBilateral * 2, fBilateral / 2);
	cv::Canny(midImg, edgeImg, threshold1, threshold2);
	// Merge all channel results.
	if (channelMode == CHANNEL_MERGE)
	{
		std::vector<cv::Mat> filterImgs(3);
		std::vector<cv::Mat> midImgs(3);
		std::vector<cv::Mat> edgeImgs(3);
		for (int k = 0; k < 3; k++)
		{
			cv::GaussianBlur(rgbChannels[k], filterImgs[k], cv::Size(kernelSize, kernelSize), 0, 0);
			cv::bilateralFilter(filterImgs[k], midImgs[k], (int)fBilateral, fBilateral * 2, fBilateral / 2);
			cv::Canny(midImgs[k], edgeImgs[k], threshold1, threshold2);
		}
		for (int j = 0; j < edgeImg.rows; j++)
		{
			uchar* edgePtr = edgeImg.ptr<uchar>(j);
			uchar* pixelPtrR = edgeImgs[0].ptr<uchar>(j);
			uchar* pixelPtrG = edgeImgs[1].ptr<uchar>(j);
			uchar* pixelPtrB = edgeImgs[2].ptr<uchar>(j);
			for (int i = 0; i < edgeImg.cols; i++)
			{
				if (pixelPtrB[i] > 0 || pixelPtrR[i] > 0)
				{
					edgePtr[i] = 255;
				}
			}
		}
	}
	dst = edgeImg.clone();
}

// Override
void Vision::preprocess(cv::Mat src, cv::Mat& dst, int kernelSize, double fBilateral)
{
	// Convert the color image to gray image.
	cv::Mat edgeImg;
	if (src.channels() == 3)
	{
		// Merge the edge images.
		std::vector<cv::Mat> rgbChannels(3);
		cv::split(src, rgbChannels);
		std::vector<cv::Mat> filterImgs(3);
		std::vector<cv::Mat> midImgs(3);
		std::vector<cv::Mat> edgeImgs(3);
		for (int k = 0; k < 3; k++)
		{
			cv::GaussianBlur(rgbChannels[k], filterImgs[k], cv::Size(kernelSize, kernelSize), 0, 0);
			cv::bilateralFilter(filterImgs[k], midImgs[k], (int)fBilateral, fBilateral * 2, fBilateral / 2);
			double threshold1 = 0;
			double threshold2 = 0;
			computeThresholds(midImgs[k], threshold1, threshold2);
			cv::Canny(midImgs[k], edgeImgs[k], threshold1, threshold2);
		}
		cv::cvtColor(src, edgeImg, cv::COLOR_BGR2GRAY);
		for (int j = 0; j < edgeImg.rows; j++)
		{
			uchar* edgePtr = edgeImg.ptr<uchar>(j);
			uchar* pixelPtrR = edgeImgs[0].ptr<uchar>(j);
			uchar* pixelPtrG = edgeImgs[1].ptr<uchar>(j);
			uchar* pixelPtrB = edgeImgs[2].ptr<uchar>(j);
			for (int i = 0; i < edgeImg.cols; i++)
			{
				if (pixelPtrB[i] > 0 || pixelPtrR[i] > 0)
				{
					edgePtr[i] = 255;
				}
				else
				{
					edgePtr[i] = 0;
				}
			}
		}
		dst = edgeImg.clone();
	}
	else if (src.channels() == 1)
	{
		// Get edge image.
		cv::Mat filterImg;
		cv::Mat midImg;
		cv::GaussianBlur(src, filterImg, cv::Size(kernelSize, kernelSize), 0, 0);
		cv::bilateralFilter(filterImg, midImg, (int)fBilateral, fBilateral * 2, fBilateral / 2);
		double threshold1 = 0;
		double threshold2 = 0;
		computeThresholds(midImg, threshold1, threshold2);
		cv::Canny(midImg, edgeImg, threshold1, threshold2);
		dst = edgeImg.clone();
	}
	else
	{
		return;
	}

	/*cv::imshow("Edge", dst);
	cv::waitKey(0);*/
}

void Vision::grabContours(cv::Mat src, cv::Mat &dst, std::vector<std::vector<cv::Point>> &contours)
{
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(src, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	std::vector<std::vector<cv::Point>>::iterator itr = contours.begin();
	float contourSize = 50;
	float rateThreshold = 0.1;
	while (itr != contours.end())
	{
		float rateAL = cv::contourArea(*itr) / cv::arcLength(*itr, true);
		if (itr->size() < contourSize || rateAL < rateThreshold)
		{
			itr = contours.erase(itr);
		}
		else
		{
			itr++;
		}
	}

	dst = src.clone();

	for (size_t i = 0; i < contours.size(); ++i)
	{
		cv::drawContours(dst, contours, i, cv::Scalar(0, 0, 255), 1, 8, hierarchy, 0, cv::Point());
	}
}

void Vision::shuffle(std::vector<int> &indexVector)
{
	size_t curSize = indexVector.size();
	srand((unsigned)time(NULL));
	for (size_t i = 0; i < indexVector.size(); ++i)
	{
		int index = rand() % curSize;
		int temp = indexVector[curSize - 1];
		indexVector[curSize - 1] = indexVector[index];
		indexVector[index] = temp;
		curSize--;
	}
}

void Vision::fitEllipse(std::vector<cv::Point> points, cv::RotatedRect &rRect, int batchSize, int epoch)
{
	if (epoch <= 0 || points.size() < 5)
	{
		return;
	}
	else
	{
		// Convert to operate index vector of points.
		std::vector<int> indexPoints(points.size());
		for (size_t i = 0; i < points.size(); ++i)
		{
			indexPoints[i] = i;
		}
		// Fit ellipse.
		int numSelected = 0;
		int numFited = 0;
		double disThreshold = 0.01;
		std::vector<cv::Point> selectedPoints;
		cv::RotatedRect tempRect;
		for (int k = 0; k < epoch; ++k)
		{
			// Select the remained points.
			std::vector<int> indexRestPoints;
			for (size_t j = numSelected; j < points.size(); ++j)
			{
				indexRestPoints.push_back(indexPoints[j]);
			}
			shuffle(indexRestPoints);
			// Record the remained points.
			for (size_t j = numSelected; j < points.size(); ++j)
			{
				indexPoints[j] = indexRestPoints[j - numSelected];
			}
			// Select new points to add.
			for (int j = 0; j < batchSize; ++j)
			{
				if ((size_t)j < indexRestPoints.size())
				{
					selectedPoints.push_back(points[indexRestPoints[j]]);
					numSelected++;
				}
				else
				{
					break;
				}
			}
			// Compute number of fited points.
			tempRect = cv::fitEllipse(selectedPoints);
			double cosA = cos(tempRect.angle / 180 * CV_PI);
			double sinA = sin(tempRect.angle / 180 * CV_PI);
			double a = tempRect.size.height / 2;
			double b = tempRect.size.width / 2;
			int numCurFited = 0;
			for (size_t i = 0; i < points.size(); ++i)
			{
				// The distance is a value to describe 2 similar ellipses.
				double x = (points[i].x - tempRect.center.x) * sinA - (points[i].y - tempRect.center.y) * cosA;
				double y = (points[i].x - tempRect.center.x) * cosA + (points[i].y - tempRect.center.y) * sinA;
				double distance = pow(x / a, 2) + pow(y / b, 2) - 1;
				if (abs(distance) < disThreshold)
				{
					numCurFited++;
				}
			}
			if (numCurFited < numFited)
			{
				for (int j = 0; j < batchSize; ++j)
				{
					selectedPoints.pop_back();
				}
				numSelected -= batchSize;
			}
			else
			{
				// Update the number of fited points.
				numFited = numCurFited;
			}
		}
		rRect = tempRect;
	}
}

void Vision::compatiblePoints(std::vector<cv::Point> points, std::vector<int> &coIndexes, cv::RotatedRect rRect, double delta)
{
	// Parameters of the ellipse.
	float cosA = (float)cos(rRect.angle / 180 * CV_PI);
	float sinA = (float)sin(rRect.angle / 180 * CV_PI);
	float a = rRect.size.height / 2;
	float b = rRect.size.width / 2;

	// Record the index of compatible point.
	for (size_t i = 0; i < points.size(); ++i)
	{
		float x = (points[i].x - rRect.center.x) * sinA - (points[i].y - rRect.center.y) * cosA;
		float y = (points[i].x - rRect.center.x) * cosA + (points[i].y - rRect.center.y) * sinA;
		float distance = pow(x / a, 2) + pow(y / b, 2) - 1;
		if (abs(distance) < delta)
		{
			coIndexes.push_back((int)i);
		}
	}
}

// RANSAC method.
void Vision::fitEllipse(std::vector<cv::Point> points, cv::RotatedRect &rRect)
{
	// Too few points return error.
	if (points.size() < 6)
	{
		return;
	}

	// Get index of all points.
	std::vector<int> indexPoints(points.size());
	for (int i = 0; i < points.size(); ++i)
	{
		indexPoints[i] = i;
	}

	// Initialize parameters.
	// K = ln(1-Pc) / ln(1-Pk)
	// Pk = q^m
	int epoch = 50;
	int batchSize = 6;
	size_t maxCoNum = 0;
	float disThreshold = 0.01f;
	std::vector<int> bestIndexes;
	cv::RotatedRect tempRect;
	std::vector<cv::Point> tempPoints;

	// Get the best randomly selected points.
	for (int k = 0; k < epoch; ++k)
	{
		size_t curSize = indexPoints.size();
		srand((unsigned)time(NULL));
		std::vector<cv::Point> selectPoints;
		for (int i = 0; i < batchSize; ++i)
		{
			int index = rand() % curSize;
			selectPoints.push_back(points[indexPoints[index]]);
			int temp = indexPoints[curSize - 1];
			indexPoints[curSize - 1] = indexPoints[index];
			indexPoints[index] = temp;
			curSize--;
		}
		tempRect = cv::fitEllipseDirect(selectPoints);
		// Compute number of compatible points.
		std::vector<int> coIndexes;
		compatiblePoints(points, coIndexes, tempRect, disThreshold);
		if (coIndexes.size() > maxCoNum)
		{
			maxCoNum = coIndexes.size();
			bestIndexes = coIndexes;
			tempPoints = selectPoints;
		}
	}

	// Fit ellipse with more points.
	size_t curSize = indexPoints.size() - batchSize;
	int epochAdd = MIN((int)indexPoints.size() / batchSize - 1, 50);
	for (int k = 0; k < epochAdd; ++k)
	{
		srand((unsigned)time(NULL));
		for (int i = 0; i < batchSize; ++i)
		{
			int index = rand() % curSize;
			tempPoints.push_back(points[indexPoints[index]]);
			int temp = indexPoints[curSize - 1];
			indexPoints[curSize - 1] = indexPoints[index];
			indexPoints[index] = temp;
			curSize--;
		}
		tempRect = cv::fitEllipseDirect(tempPoints);
		// Compute number of compatible points.
		std::vector<int> coIndexes;
		compatiblePoints(points, coIndexes, tempRect, disThreshold);
		if (coIndexes.size() > maxCoNum)
		{
			maxCoNum = coIndexes.size();
			bestIndexes = coIndexes;
		}
		else
		{
			for (int i = 0; i < batchSize; ++i)
			{
				tempPoints.pop_back();
			}
			curSize += batchSize;
		}
	}

	// Get the final result.
	std::vector<cv::Point> bestPoints;
	for (size_t i = 0; i < bestIndexes.size(); ++i)
	{
		bestPoints.push_back(points[bestIndexes[i]]);
	}
	rRect = cv::fitEllipseDirect(bestPoints);
}

void Vision::computeDis(cv::RotatedRect rect, std::vector<cv::Point> points, std::vector<double> &distances)
{
	// Parameter of ellipse.
	double cosA = cos(rect.angle / 180 * CV_PI);
	double sinA = sin(rect.angle / 180 * CV_PI);
	double a = rect.size.height / 2;
	double b = rect.size.width / 2;

	// Compute distances.
	for (size_t i = 0; i < points.size(); i++)
	{
		double x = (points[i].x - rect.center.x) * sinA - (points[i].y - rect.center.y) * cosA;
		double y = (points[i].x - rect.center.x) * cosA + (points[i].y - rect.center.y) * sinA;
		distances[i] = pow(x / a, 2) + pow(y / b, 2) - 1;
	}
}

// Get ellipses by k-means method.
void Vision::kMeansEllipses(std::vector<cv::Point> points, std::vector<int> &labels, cv::RotatedRect rect, int maxEpoches)
{
	if (labels.empty())
	{
		labels = std::vector<int>(points.size());
	}
	if (maxEpoches <= 0)
	{
		maxEpoches = 5;
	}

	// Get distance.
	std::vector<double> distances(points.size());
	Vision::computeDis(rect, points, distances);

	// Initialize the centers.
	int clusterNum = 2;
	std::vector<double> center(clusterNum);
	std::vector<double> d(clusterNum);
	center[0] = -0.1;
	center[1] = 0.1;

	// K-means method.
	for (int k = 0; k < maxEpoches; k++)
	{
		std::vector<double> sumD(clusterNum, 0);
		std::vector<int> countNum(clusterNum, 0);
		for (size_t i = 0; i < distances.size(); i++)
		{
			d[0] = abs(center[0] - distances[i]);
			d[1] = abs(center[1] - distances[i]);
			if (d[0] < d[1])
			{
				labels[i] = 0;
			}
			else
			{
				labels[i] = 1;
			}
			sumD[labels[i]] += distances[i];
			countNum[labels[i]]++;
		}
		center[0] = countNum[0] == 0 ? center[0] : sumD[0] / countNum[0];
		center[1] = countNum[1] == 0 ? center[1] : sumD[1] / countNum[1];
	}
}

void Vision::getEllipse(cv::Mat src, std::vector<cv::RotatedRect> &rectSet, int curveNum, int contourSize, Vision::FitMode fitMethod)
{
	// Grab contours from the image.
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(src, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

	// Delete undesirable contours.
	std::vector<std::vector<cv::Point>>::iterator itr = contours.begin();
	while (itr != contours.end())
	{
		if (itr->size() < contourSize)
		{
			itr = contours.erase(itr);
		}
		else
		{
			itr++;
		}
	}

	// Get the ellipses of hole contours.
	std::vector<bool> usedFlag(contours.size(), false);
	double ovalityThreshold = 0.05;
	double rateThreshold = 0.6;
	for (int k = 0; k < curveNum; k++)
	{
		std::vector<cv::Point> tempContour;
		cv::RotatedRect rect;
		for (size_t i = 0; i < contours.size(); i++)
		{
			if (!usedFlag[i])
			{
				tempContour.insert(tempContour.end(), contours[i].begin(), contours[i].end());
				switch (fitMethod)
				{
				case Vision::FIT_ELLIPSE_AMS: rect = cv::fitEllipseAMS(tempContour); break;
				case Vision::FIT_ELLIPSE_DIRECT: rect = cv::fitEllipseDirect(tempContour); break;
				case Vision::FIT_ELLIPSE_ITER: fitEllipse(tempContour, rect, 10, (int)tempContour.size() / 20); break;
				case Vision::FIT_ELLIPSE_ORI:
				default: rect = cv::fitEllipse(tempContour); break;
				}
				// Compute ellipse parameters.
				double ovality = 0;
				double cRate = 0;
				Vision::computeEllipseParams(tempContour, rect, ovality, cRate);
				if (ovality > ovalityThreshold || cRate < rateThreshold)
				{
					tempContour.erase(tempContour.end() - contours[i].size(), tempContour.end());
				}
				else
				{
					usedFlag[i] = true;
				}
			}
		}
		fitEllipse(tempContour, rect, 10, (int)tempContour.size() / 20);
		rectSet.push_back(rect);
	}
}

// Override
void Vision::getEllipse(cv::Mat src, std::vector<cv::RotatedRect> &rectSet, int curveNum, int contourSize)
{
	// Grab contours from the edge image.
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(src, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

	// Delete undesirable contours.
	std::vector<std::vector<cv::Point>>::iterator itr = contours.begin();
	while (itr != contours.end())
	{
		if (itr->size() < contourSize)
		{
			itr = contours.erase(itr);
		}
		else
		{
			itr++;
		}
	}

	// Get ellipses of the hole contours.
	if (curveNum == 2)
	{
		std::vector<std::vector<cv::Point>> resultContours(curveNum);
		std::vector<cv::Point> tempContour;
		cv::RotatedRect rect;
		for (size_t j = 0; j < contours.size(); j++)
		{
			tempContour.insert(tempContour.end(), contours[j].begin(), contours[j].end());
		}
		rect = cv::fitEllipseDirect(tempContour);
		/*cv::Mat drawImg1;
		cv::cvtColor(src, drawImg1, cv::COLOR_GRAY2BGR);
		for (size_t j = 0; j < contours.size(); j++)
		{
			cv::drawContours(drawImg1, contours, j, cv::Scalar(255, 0, 0), 1);
		}*/
		// Remove the bad points.
		std::vector<double> distances(tempContour.size());
		Vision::computeDis(rect, tempContour, distances);
		double ThresholdPD = 0.5;
		std::vector<cv::Point> midContour = tempContour;
		std::vector<cv::Point>::iterator itrMid = midContour.begin();
		for (size_t i = 0; i < midContour.size(); i++)
		{
			if (abs(distances[i]) > ThresholdPD)
			{
				itrMid = midContour.erase(itrMid);
			}
			else
			{
				itrMid++;
			}
		}
		if (midContour.size() < size_t(0.9 * tempContour.size()))
		{
			rect.size.height *= 0.9;
			Vision::computeDis(rect, tempContour, distances);
			std::vector<cv::Point>::iterator itrTemp = tempContour.begin();
			for (size_t i = 0; i < tempContour.size(); i++)
			{
				if (abs(distances[i]) > ThresholdPD)
				{
					itrTemp = tempContour.erase(itrTemp);
				}
				else
				{
					itrTemp++;
				}
			}
		}
		else
		{
			tempContour = midContour;
		}
		/*cv::ellipse(drawImg1, rect, cv::Scalar(0, 0, 255), 1, 8);
		cv::imshow("Ellipse", drawImg1);
		cv::waitKey(0);*/
		/*cv::Mat drawImg2 = cv::Mat::zeros(src.size(), CV_8UC3);
		for (size_t i = 0; i < tempContour.size(); i++)
		{
			cv::circle(drawImg2, tempContour[i], 2, cv::Scalar(0, 255, 0), 1);
		}
		cv::imshow("Rest", drawImg2);
		cv::waitKey(0);*/
		/*double cosA = cos(rect.angle / 180 * CV_PI);
		double sinA = sin(rect.angle / 180 * CV_PI);
		double a = rect.size.width / 2;
		double b = rect.size.height / 2;
		double rangeThreshold = 0.05;
		for (size_t i = 0; i < tempContour.size(); i++)
		{
			double x = (tempContour[i].y - rect.center.y) * sinA + (tempContour[i].x - rect.center.x) * cosA;
			double y = (tempContour[i].y - rect.center.y) * cosA - (tempContour[i].x - rect.center.x) * sinA;
			double distance = pow(x / a, 2) + pow(y / b, 2) - 1;
			if (distance > rangeThreshold)
			{
				resultContours[0].push_back(tempContour[i]);
			}
			else if (distance < -rangeThreshold)
			{
				resultContours[1].push_back(tempContour[i]);
			}
		}*/
		// Devide the rest points into 2 clusters.
		std::vector<int> labels(tempContour.size());
		Vision::kMeansEllipses(tempContour, labels, rect, 15);
		for (size_t i = 0; i < tempContour.size(); i++)
		{
			resultContours[labels[i]].push_back(tempContour[i]);
		}
		/*cv::Mat drawImg3 = cv::Mat::zeros(src.size(), CV_8UC3);
		for (size_t i = 0; i < resultContours[0].size(); i++)
		{
			circle(drawImg3, resultContours[0][i], 2, cv::Scalar(0, 255, 255), -1);
		}
		for (size_t i = 0; i < resultContours[1].size(); i++)
		{
			circle(drawImg3, resultContours[1][i], 2, cv::Scalar(255, 255, 0), -1);
		}
		cv::imshow("Points", drawImg3);
		cv::waitKey(0);*/
		// Fit the points.
		for (int k = 0; k < curveNum; k++)
		{
			if (resultContours[k].size() > 4)
			{
				Vision::fitEllipse(resultContours[k], rect);
				rectSet.push_back(rect);
			}
		}
	}
	else if (curveNum == 1)
	{
		std::vector<cv::Point> tempContour;
		cv::RotatedRect rect;
		for (size_t j = 0; j < contours.size(); j++)
		{
			tempContour.insert(tempContour.end(), contours[j].begin(), contours[j].end());
		}
		Vision::fitEllipse(tempContour, rect);
		rectSet.push_back(rect);
	}
}

// Get ellipse by Gaussian pyramid.
void Vision::pyrEllipse(cv::Mat src, std::vector<cv::RotatedRect> &rectSet, int rectNum)
{
	// Generate images of all layers.
	cv::Mat G[4];
	G[0] = src;
	cv::pyrDown(G[0], G[1], cv::Size(G[0].cols / 2, G[0].rows / 2));
	cv::pyrDown(G[1], G[2], cv::Size(G[1].cols / 2, G[1].rows / 2));
	cv::pyrDown(G[2], G[3], cv::Size(G[2].cols / 2, G[2].rows / 2));

	// Parameters for image process.
	int kernelSize = 5;
	double fBilateral = 1.0;
	double ffB = 2.0;

	// Detect ellipses in G[3].
	cv::Mat edgeImgG3;
	Vision::preprocess(G[3], edgeImgG3, kernelSize, fBilateral);
	std::vector<cv::RotatedRect> rectSetG3;
	Vision::getEllipse(edgeImgG3, rectSetG3, rectNum, 50);
	cv::Mat testImg;
	cv::cvtColor(edgeImgG3, testImg, cv::COLOR_GRAY2BGR);
	cv::ellipse(testImg, rectSetG3[0], cv::Scalar(0, 0, 255), 1, 8);
	cv::ellipse(testImg, rectSetG3[1], cv::Scalar(0, 0, 255), 1, 8);
	/*cv::imshow("TEST", testImg);
	cv::waitKey(0);*/

	// Get ROI of all images and edge images.
	double tempB0 = MAX(rectSetG3[0].size.width, rectSetG3[0].size.height);
	double tempB1 = MAX(rectSetG3[1].size.width, rectSetG3[1].size.height);
	int borderG3 = 0;
	int fSize = 20;
	cv::Rect rectG[4];
	if (tempB0 > tempB1)
	{
		borderG3 = ((int)tempB0 / fSize + 1) * fSize;
		rectG[3].x = (int)(rectSetG3[0].center.x - borderG3 / 2);
		rectG[3].y = (int)(rectSetG3[0].center.y - borderG3 / 2);
	}
	else
	{
		borderG3 = ((int)tempB1 / fSize + 1) * fSize;
		rectG[3].x = (int)(rectSetG3[1].center.x - borderG3 / 2);
		rectG[3].y = (int)(rectSetG3[1].center.y - borderG3 / 2);
	}
	rectG[3].width = borderG3;
	rectG[3].height = borderG3;
	cv::Mat roiG[4];
	cv::Mat roiEdge[4];
	roiG[3] = G[3](rectG[3]);
	roiEdge[3] = edgeImgG3(rectG[3]);
	for (int l = 2; l >= 0; l--)
	{
		rectG[l].x = rectG[l + 1].x * 2;
		rectG[l].y = rectG[l + 1].y * 2;
		rectG[l].width = rectG[l + 1].width * 2;
		rectG[l].height = rectG[l + 1].height * 2;
		roiG[l] = G[l](rectG[l]);
		fBilateral *= ffB;
		Vision::preprocess(roiG[l], roiEdge[l], kernelSize, fBilateral);
	}

	// Grab points of ellipses in each layer.
	double rangeThreshold = 0.1;
	double fRange = 4;
	std::vector<cv::RotatedRect> tempRects(2);
	for (size_t k = 0; k < rectSetG3.size(); k++)
	{
		tempRects[k].center.x = rectSetG3[k].center.x - rectG[3].x;
		tempRects[k].center.y = rectSetG3[k].center.y - rectG[3].y;
		tempRects[k].size.width = rectSetG3[k].size.width;
		tempRects[k].size.height = rectSetG3[k].size.height;
		tempRects[k].angle = rectSetG3[k].angle;
	}
	for (int l = 3; l >= 0; l--)
	{
		if (l < 3)
		{
			for (size_t k = 0; k < tempRects.size(); k++)
			{
				tempRects[k].center.x = tempRects[k].center.x * 2;
				tempRects[k].center.y = tempRects[k].center.y * 2;
				tempRects[k].size.width = tempRects[k].size.width * 2;
				tempRects[k].size.height = tempRects[k].size.height * 2;
			}
			rangeThreshold /= fRange;
		}
		std::vector<std::vector<cv::Point>> curves(2);
		for (size_t k = 0; k < tempRects.size(); k++)
		{
			for (int i = 0; i < roiEdge[l].rows; i++)
			{
				uchar* data = roiEdge[l].ptr<uchar>(i);
				for (int j = 0; j < roiEdge[l].cols; j++)
				{
					if (data[j] > 128)
					{
						double cosA = cos(tempRects[k].angle / 180 * CV_PI);
						double sinA = sin(tempRects[k].angle / 180 * CV_PI);
						double a = tempRects[k].size.height / 2;
						double b = tempRects[k].size.width / 2;
						double x = (j - tempRects[k].center.x) * sinA - (i - tempRects[k].center.y) * cosA;
						double y = (j - tempRects[k].center.x) * cosA + (i - tempRects[k].center.y) * sinA;
						double distance = pow(x / a, 2) + pow(y / b, 2) - 1;
						if (abs(distance) < rangeThreshold)
						{
							curves[k].push_back(cv::Point(j, i));
						}
					}
				}
			}
			if (curves[k].size() > 4)
			{
				cv::RotatedRect rect = cv::fitEllipseDirect(curves[k]);
				tempRects[k] = rect;
			}
		}
		/*cv::Mat drawImg;
		cv::cvtColor(roiG[l], drawImg, cv::COLOR_GRAY2BGR);
		for (size_t k = 0; k < tempRects.size(); k++)
		{
			cv::ellipse(drawImg, tempRects[k], cv::Scalar(0, 255, 0), 1, 8);
		}
		cv::imshow("TEST", drawImg);
		cv::waitKey(0);*/
	}
	for (size_t k = 0; k < tempRects.size(); k++)
	{
		tempRects[k].center.x += rectG[0].x;
		tempRects[k].center.y += rectG[0].y;
	}
	rectSet = tempRects;
}

void Vision::computeEllipseParams(std::vector<cv::Point> points, cv::RotatedRect rRect, double &ovality, double &cRate)
{
	double cosA = cos(rRect.angle / 180 * CV_PI);
	double sinA = sin(rRect.angle / 180 * CV_PI);
	double a = rRect.size.height / 2;
	double b = rRect.size.width / 2;
	cRate = MIN(rRect.size.width, rRect.size.height) / MAX(rRect.size.width, rRect.size.height);
	ovality = 0;
	for (size_t i = 0; i < points.size(); ++i)
	{
		double x = (points[i].x - rRect.center.x) * sinA - (points[i].y - rRect.center.y) * cosA;
		double y = (points[i].x - rRect.center.x) * cosA + (points[i].y - rRect.center.y) * sinA;
		double distance = pow(x / a, 2) + pow(y / b, 2) - 1;
		ovality += abs(distance);
	}
	ovality /= points.size();
}

// Conic reconstruction algorithm.
void Vision::conicReconstruction(std::vector<cv::RotatedRect> rects, double &d, double &E, cv::Point3d xyz[], cv::Mat R, cv::Mat T, cv::Mat cameraMatrix[])
{
	// Reference: S.D.Ma, Conics-Based Stereo, Motion Estimation, and Pose Determination.
	// Compute Q of 2 ellipses in the images.
	cv::RotatedRect rect;
	std::vector<cv::Mat> Q(2);
	for (int i = 0; i < 2; i++)
	{
		rect = rects[i];
		double cosA = cos(rect.angle / 180 * CV_PI);
		double sinA = sin(rect.angle / 180 * CV_PI);
		double a = rect.size.height / 2;
		double b = rect.size.width / 2;
		double c2A = cosA * cosA;
		double s2A = sinA * sinA;
		double scA = sinA * cosA;
		double _a2 = 1 / (a * a);
		double _b2 = 1 / (b * b);
		double A = s2A * _a2 + c2A * _b2;
		double B = 2 * scA * (_b2 - _a2);
		double C = c2A * _a2 + s2A * _b2;
		double D = -2 * rect.center.x * A - rect.center.y * B;
		double E = -2 * rect.center.y * C - rect.center.x * B;
		double F = rect.center.x * rect.center.x * A + rect.center.y * rect.center.y * C + rect.center.x * rect.center.y * B - 1;
		Q[i] = (cv::Mat_<double>(3, 3) <<
			A, B / 2, D / 2,
			B / 2, C, E / 2,
			D / 2, E / 2, F);
	}
	Q[0] = cameraMatrix[0].t() * Q[0] * cameraMatrix[0];
	Q[1] = cameraMatrix[1].t() * Q[1] * cameraMatrix[1];

	// Compute eigenvalues and eigenvectors of M.
	cv::Mat M = (R.t() * Q[1] * R).inv() * Q[0];
	cv::Mat eigenvaluesM;
	cv::Mat eigenvectorsM;
	cv::eigenNonSymmetric(M, eigenvaluesM, eigenvectorsM);
	/*std::cout << M << std::endl;
	std::cout << eigenvaluesM << std::endl;
	std::cout << eigenvectorsM << std::endl;*/
	double k = eigenvaluesM.at<double>(0);

	// C = Q1 - k * R' * Q2 * R
	// Compute eigenvalues and eigenvectors of C.
	// Suppose that l1, l2 are 2 nonzero eigenvalues of C.
	// s1, s2 are the correspondent eigenvectors.
	// The third column of R1 is given by:
	// r13 = +-norm[(sqrt(|l1|) * s1 +- sqrt(|l2|) * s2)]
	cv::Mat C = Q[0] - k * R.t() * Q[1] * R;
	cv::Mat eigenvaluesC;
	cv::Mat eigenvectorsC;
	cv::eigen(C, eigenvaluesC, eigenvectorsC);
	/*std::cout << C << std::endl;
	std::cout << eigenvaluesC << std::endl;
	std::cout << eigenvectorsC << std::endl;*/
	std::vector<cv::Mat> R1s(3);
	R1s[2] = sqrt(abs(eigenvaluesC.at<double>(0))) * eigenvectorsC.row(0).t() + sqrt(abs(eigenvaluesC.at<double>(2))) * eigenvectorsC.row(2).t();
	cv::normalize(R1s[2], R1s[2], 1, 0, cv::NORM_L2);

	// H = Q1 - r13 * r13' * Q1.
	// Because H * r11 = (r11' * Q1 * r11) * r11
	// and H * r12 = (r12' * Q1 * r12) * r12
	// Then r11, r12 are 2 eigenvectors of H.
	// lh1 = r11' * Q1 * r11, lh2 = r12' * Q1 * r12 are correspondent eigenvalues.
	cv::Mat H = Q[0] - R1s[2] * R1s[2].t() * Q[0];
	cv::Mat eigenvaluesH;
	cv::Mat eigenvectorsH;
	cv::eigenNonSymmetric(H, eigenvaluesH, eigenvectorsH);
	/*std::cout << H << std::endl;
	std::cout << eigenvaluesH << std::endl;
	std::cout << eigenvectorsH << std::endl;*/
	R1s[0] = eigenvectorsH.row(0).t();
	R1s[1] = eigenvectorsH.row(1).t();
	//R1s[1] = R1s[2].cross(R1s[0]);
	/*std::cout << R1s[0] << std::endl;
	std::cout << R1s[1] << std::endl;*/

	// Obtain R2 by R2 = R * R1
	cv::Mat R1;
	cv::Mat R2;
	cv::hconcat(R1s, R1);
	R2 = R * R1;
	/*std::cout << R1 << std::endl;
	std::cout << R2 << std::endl;*/

	// Because it is ellipse, we can get 3 equations:
	// t1' * Q1 * r11 = 0
	// t1' * Q1 * r12 = 0
	// t2' * Q2 * r21 = 0
	// And we already have t2 = R * t1 + t
	// k1 = -t1' * Q1 * t1
	// k2 = k / k1
	// a^2 = k1 / (r11' * Q1 * r11)
	// b^2 = k1 / (r12' * Q1 * r12)
	cv::Mat A(4, 3, CV_64FC1);
	A.row(0) = R1s[0].t() * Q[0];
	A.row(1) = R1s[1].t() * Q[0];
	A.row(2) = R2.col(0).t() * Q[1] * R;
	A.row(3) = R2.col(1).t() * Q[1] * R;
	cv::Mat beta_2 = -R2.col(0).t() * Q[1] * T;
	cv::Mat beta_3 = -R2.col(1).t() * Q[1] * T;
	cv::Mat beta = (cv::Mat_<double>(4, 1) << 0, 0, beta_2.at<double>(0), beta_3.at<double>(0));
	/*std::cout << A << std::endl;
	std::cout << beta << std::endl;*/
	cv::Mat t1 = (A.t() * A).inv() * A.t() * beta;
	cv::Mat t2 = R * t1 + T;
	cv::Mat temp = -t1.t() * Q[0] * t1;
	/*std::cout << t1 << std::endl;
	std::cout << t2 << std::endl;*/
	double k1 = temp.at<double>(0);
	double k2 = k1 / k;
	temp = R1s[0].t() * Q[0] * R1s[0];
	double a = sqrt(k1 / temp.at<double>(0));
	temp = R1s[1].t() * Q[0] * R1s[1];
	double b = sqrt(k1 / temp.at<double>(0));
	d = a + b;
	//std::cout << a << " x " << b << std::endl;
	xyz[0] = cv::Point3d(t1.at<double>(0), t1.at<double>(1), t1.at<double>(2));
	xyz[1] = cv::Point3d(t2.at<double>(0), t2.at<double>(1), t2.at<double>(2));

	// Compute errors.
	std::vector<cv::Mat> G1s;
	G1s.push_back(R1s[0]);
	G1s.push_back(R1s[1]);
	G1s.push_back(t1);
	cv::Mat G1;
	cv::hconcat(G1s, G1);
	std::vector<cv::Mat> G2s;
	G2s.push_back(R2.col(0));
	G2s.push_back(R2.col(1));
	G2s.push_back(t2);
	cv::Mat G2;
	cv::hconcat(G2s, G2);
	cv::Mat E1 = G1.t() * Q[0] * G1;
	cv::Mat E2 = G2.t() * Q[1] * G2;
	cv::Mat Qs = (cv::Mat_<double>(3, 3) <<
		1 / (a * a), 0, 0,
		0, 1 / (b * b), 0,
		0, 0, -1);
	/*std::cout << G1 << std::endl;
	std::cout << G2 << std::endl;
	std::cout << E1 << std::endl;
	std::cout << k1 * Qs << std::endl;
	std::cout << E2 << std::endl;
	std::cout << k2 * Qs << std::endl;*/
	E = cv::norm(E1 - k1 * Qs) + cv::norm(E2 - k2 * Qs);
}

void Vision::selectPoints(cv::RotatedRect rect, std::vector<cv::Point2d> &points, double scale, int pointNum)
{
	double cosA = cos(rect.angle / 180 * CV_PI);
	double sinA = sin(rect.angle / 180 * CV_PI);
	double a = rect.size.height / 2;
	double b = rect.size.width / 2;
	double dA = 2 * CV_PI / pointNum;
	cv::Point2d tempPoint;
	cv::Point2d resPoint;
	for (int i = 0; i < pointNum; i++)
	{
		tempPoint.x = a * cos(dA * i);
		tempPoint.y = b * sin(dA * i);
		resPoint.x = tempPoint.y * cosA + tempPoint.x * sinA + rect.center.x;
		resPoint.y = tempPoint.y * sinA - tempPoint.x * cosA + rect.center.y;
		resPoint /= scale;
		points.push_back(resPoint);
	}
}

// Override
void Vision::selectPoints(cv::RotatedRect rect, std::vector<cv::Point2f> &points, int pointNum)
{
	float cosA = (float)cos(rect.angle / 180 * CV_PI);
	float sinA = (float)sin(rect.angle / 180 * CV_PI);
	float a = rect.size.height / 2;
	float b = rect.size.width / 2;
	float dA = float(2 * CV_PI / pointNum);
	cv::Point2f tempPoint;
	cv::Point2f resPoint;
	for (int i = 0; i < pointNum; i++)
	{
		tempPoint.x = a * cos(dA * i);
		tempPoint.y = b * sin(dA * i);
		resPoint.x = tempPoint.y * cosA + tempPoint.x * sinA + rect.center.x;
		resPoint.y = tempPoint.y * sinA - tempPoint.x * cosA + rect.center.y;
		points.push_back(resPoint);
	}
}

void Vision::crossPoints(cv::RotatedRect rect, cv::Vec3f lineCoeff, std::vector<cv::Point2d> &interPoints)
{
	// Convert to ellipse coordinate.
	double cosA = cos(rect.angle / 180 * CV_PI);
	double sinA = sin(rect.angle / 180 * CV_PI);
	double A = lineCoeff[0] * sinA - lineCoeff[1] * cosA;
	double B = lineCoeff[0] * cosA + lineCoeff[1] * sinA;
	double C = lineCoeff[0] * rect.center.x + lineCoeff[1] * rect.center.y + lineCoeff[2];

	// Compute crossover points.
	// A * x + B * y + C = 0
	// x^2 / a^2 + y^2 / b^2 = 1
	double a = rect.size.height / 2;
	double b = rect.size.width / 2;
	double a_ = (B * B) / (a * a) + (A * A) / (b * b);
	double b_ = 2 * A * C / (b * b);
	double c_ = (C * C) / (b * b) - B * B;
	double delta = b_ * b_ - 4 * a_ * c_;
	std::vector<cv::Point2d> tempSet;
	cv::Point2d tempPoint;
	if (delta > 0)
	{
		tempPoint.x = (-b_ - sqrt(delta)) / (2 * a_);
		tempPoint.y = -(C + A * tempPoint.x) / B;
		tempSet.push_back(tempPoint);
		tempPoint.x = (-b_ + sqrt(delta)) / (2 * a_);
		tempPoint.y = -(C + A * tempPoint.x) / B;
		tempSet.push_back(tempPoint);
	}
	else if (delta == 0)
	{
		tempPoint.x = -b_ / (2 * a_);
		tempPoint.y = -(C + A * tempPoint.x) / B;
		tempSet.push_back(tempPoint);
	}

	// Transform the points to original coordinate.
	for (size_t i = 0; i < tempSet.size(); i++)
	{
		tempPoint.x = tempSet[i].y * cosA + tempSet[i].x * sinA + rect.center.x;
		tempPoint.y = tempSet[i].y * sinA - tempSet[i].x * cosA + rect.center.y;
		interPoints.push_back(tempPoint);
	}
}

// A new method to get the coordinates of cross points.
void Vision::getTarPoint(cv::RotatedRect refRect, cv::RotatedRect tarRect, cv::Vec3f lineCoeff, cv::Point2f refPoint, cv::Point2f &tarPoint)
{
	// Compute theta0 of reference point.
	cv::Point2f refPoint_;
	float cosA = cosf(float(refRect.angle / 180 * CV_PI));
	float sinA = sinf(float(refRect.angle / 180 * CV_PI));
	refPoint_.x = (refPoint.x - refRect.center.x) * sinA - (refPoint.y - refRect.center.y) * cosA;
	refPoint_.y = (refPoint.x - refRect.center.x) * cosA + (refPoint.y - refRect.center.y) * sinA;
	float theta0 = 0;
	theta0 = atan2f(2 * refPoint_.y / refRect.size.height, 2 * refPoint_.x / refRect.size.width);

	// Convert to ellipse coordinate.
	cosA = cosf(float(tarRect.angle / 180 * CV_PI));
	sinA = sinf(float(tarRect.angle / 180 * CV_PI));
	float A = lineCoeff[0] * sinA - lineCoeff[1] * cosA;
	float B = lineCoeff[0] * cosA + lineCoeff[1] * sinA;
	float C = lineCoeff[0] * tarRect.center.x + lineCoeff[1] * tarRect.center.y + lineCoeff[2];

	// Compute crossover points.
	// A * x + B * y + C = 0
	// x = a * cos(theta);
	// y = b * sin(theta);
	float a = tarRect.size.height / 2;
	float b = tarRect.size.width / 2;
	float a_ = (A * a) * (A * a) + (B * b) * (B * b);
	float b_ = 2 * A * a * C;
	float c_ = (C * C) - (B * b) * (B * b);
	float delta = b_ * b_ - 4 * a_ * c_;
	cv::Point2f tempPoint;
	if (delta > 0)
	{
		float theta1 = 0;
		float theta2 = 0;
		float cosT1, sinT1;
		float cosT2, sinT2;
		cosT1 = (-b_ - sqrt(delta)) / (2 * a_);
		sinT1 = -(C + A * a * cosT1) / (B * b);
		theta1 = atan2f(sinT1, cosT1);
		cosT2 = (-b_ + sqrt(delta)) / (2 * a_);
		sinT2 = -(C + A * a * cosT2) / (B * b);
		theta2 = atan2f(sinT2, cosT2);
		// Here can be improve.
		/*if ((refRect.size.width >= refRect.size.height) && (a < b))
		{
			theta1 += (float)CV_PI / 2;
			theta2 += (float)CV_PI / 2;
			theta1 = theta1 > CV_PI ? float(theta1 - 2 * CV_PI) : theta1;
			theta2 = theta2 > CV_PI ? float(theta2 - 2 * CV_PI) : theta2;
		}
		else if ((refRect.size.width < refRect.size.height) && (a >= b))
		{
			theta1 -= (float)CV_PI / 2;
			theta2 -= (float)CV_PI / 2;
			theta1 = theta1 < -CV_PI ? float(theta1 + 2 * CV_PI) : theta1;
			theta2 = theta2 < -CV_PI ? float(theta2 + 2 * CV_PI) : theta2;
		}*/
		if (theta0 < 0)
		{
			theta0 = float(theta0 - CV_PI);
			if (abs(theta1 - theta0) < abs(theta2 - theta0))
			{
				tempPoint.x = a * cosT1;
				tempPoint.y = b * sinT1;
			}
			else
			{
				tempPoint.x = a * cosT2;
				tempPoint.y = b * sinT2;
			}
		}
		else if (theta0 > 0)
		{
			theta0 = float(theta0 + CV_PI);
			if (abs(theta1 - theta0) < abs(theta2 - theta0))
			{
				tempPoint.x = a * cosT1;
				tempPoint.y = b * sinT1;
			}
			else
			{
				tempPoint.x = a * cosT2;
				tempPoint.y = b * sinT2;
			}
		}
		else
		{
			if (abs(theta1) > abs(theta2))
			{
				tempPoint.x = a * cosT1;
				tempPoint.y = b * sinT1;
			}
			else
			{
				tempPoint.x = a * cosT2;
				tempPoint.y = b * sinT2;
			}
		}
	}
	else if (delta = 0)
	{
		float cosT = -b_ / (2 * a_);
		float sinT = -(C + A * a * cosT) / (B * b);
		tempPoint.x = a * cosT;
		tempPoint.y = b * sinT;
	}
	else
	{
		return;
	}

	// Transform the points to original coordinate.
	tarPoint.x = tempPoint.y * cosA + tempPoint.x * sinA + tarRect.center.x;
	tarPoint.y = tempPoint.y * sinA - tempPoint.x * cosA + tarRect.center.y;
}

void Vision::computeF(cv::Mat Al, cv::Mat Ar, cv::Mat R, cv::Mat T, cv::Mat &F)
{
	// E = R * S
	// F = Ar' \ S * R / Al
	// S = [0, -T(3), T(2); T(3), 0, -T(1); -T(2), T(1), 0]
	cv::Mat S = (cv::Mat_<double>(3, 3) <<
		0, -T.at<double>(2, 0), T.at<double>(1, 0),
		T.at<double>(2, 0), 0, -T.at<double>(0, 0),
		-T.at<double>(1, 0), T.at<double>(0, 0), 0);
	F = Ar.t().inv() * S * R * Al.inv();
}

void Vision::computeEpipolar(std::vector<cv::Point2d> points, Vision::CamIndex index, cv::Mat F, std::vector<cv::Vec3d> &lines)
{
	if (index == Vision::ON_LEFT)
	{
		for (size_t i = 0; i < points.size(); i++)
		{
			cv::Mat pl = (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, 1);
			cv::Mat Lr(3, 1, CV_32FC1);
			cv::Vec3d lineR;
			Lr = F * pl;
			lineR[0] = Lr.at<double>(0, 0);
			lineR[1] = Lr.at<double>(1, 0);
			lineR[2] = Lr.at<double>(2, 0);
			lines.push_back(lineR);
		}
	}
	else if (index == Vision::ON_RIGHT)
	{
		for (size_t i = 0; i < points.size(); i++)
		{
			cv::Mat pr = (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, 1);
			cv::Mat Ll(3, 1, CV_32FC1);
			cv::Vec3d lineL;
			Ll = F.t() * pr;
			lineL[0] = Ll.at<double>(0, 0);
			lineL[1] = Ll.at<double>(1, 0);
			lineL[2] = Ll.at<double>(2, 0);
			lines.push_back(lineL);
		}
	}
}

void Vision::computeXYZ(cv::Point pl, cv::Point pr, cv::Mat Al, cv::Mat Ar, cv::Mat R, cv::Mat T, cv::Point3d &xyzl, cv::Point3d &xyzr)
{
	// v0 = -Ar * R \ Al * Pl
	// v1 = Ar * T
	// A = [v0, Pr]
	// zs = [zl, zr]'
	// A * zs = v1
	// zs = (A' * A) \ A' * b
	cv::Mat v0(3, 1, CV_32FC1);
	cv::Mat v1(3, 1, CV_32FC1);
	cv::Mat Pl = (cv::Mat_<double>(3, 1) << pl.x, pl.y, 1);
	cv::Mat Pr = (cv::Mat_<double>(3, 1) << pr.x, pr.y, 1);
	v0 = -Ar * R * Al.inv() * Pl;
	v1 = Ar * T;
	cv::Mat A(3, 2, CV_32FC1);
	cv::Mat zs(2, 1, CV_32FC1);
	hconcat(v0, Pr, A);
	solve(A, v1, zs, cv::DECOMP_SVD);
	double zl = zs.at<double>(0, 0);
	double zr = zs.at<double>(1, 0);
	// zl * Pl = Al * xyzl
	// zr * Pr = Ar * xyzr
	cv::Mat PcL(3, 1, CV_32FC1);
	cv::Mat PcR(3, 1, CV_32FC1);
	PcL = zl * Al.inv() * Pl;
	PcR = zr * Ar.inv() * Pr;
	xyzl = cv::Point3d(PcL.at<double>(0, 0), PcL.at<double>(1, 0), PcL.at<double>(2, 0));
	xyzr = cv::Point3d(PcR.at<double>(0, 0), PcR.at<double>(1, 0), PcR.at<double>(2, 0));
}

void Vision::computeCD(cv::RotatedRect rects[], cv::Point3d xyz[], double &diameter, double scale, cv::Mat cameraMatrix[], cv::Mat R, cv::Mat T)
{
	// Compute xyz of center of hole.
	// It needs some modifications to coordinate of hole centers.
	cv::Point2d pl, pr;
	pl = rects[0].center / scale;
	pr = rects[1].center / scale;
	computeXYZ(pl, pr, cameraMatrix[0], cameraMatrix[1], R, T, xyz[0], xyz[1]);

	/* EPIPOLAR GEOMETRY */
	// Compute fundamental matrix.
	cv::Mat F;
	computeF(cameraMatrix[0], cameraMatrix[1], R, T, F);
	// Select points of the left image.
	std::vector<cv::Point2d> points;
	int pointNum = 53;
	selectPoints(rects[0], points, scale, pointNum);
	// Compute correspond epipolar lines.
	std::vector<cv::Vec3d> lines;
	computeCorrespondEpilines(points, 1, F, lines);
	// Draw lines, compute crossover points and 3D coordinates.
	std::vector<cv::Point3d> points3DL;
	std::vector<cv::Point3d> points3DR;
	cv::Point3d xyzl;
	cv::Point3d xyzr;
	for (size_t i = 0; i < lines.size(); i++)
	{
		// Draw the points on left image.
		cv::Point temp;
		temp.x = int(points[i].x * scale);
		temp.y = int(points[i].y * scale);
		// You should better to modify this function for drawing epipolar lines.
		lines[i][2] *= scale;
		// Compute crossover points and 3D coordinate.
		std::vector<cv::Point2d> interPoints;
		crossPoints(rects[1], lines[i], interPoints);
		// Check the neighbor points for better fitting.
		/* TODO: ADD the code here! */
		pl = points[i];
		if (interPoints.size() == 1)
		{
			pr = interPoints[0] / scale;
			computeXYZ(pl, pr, cameraMatrix[0], cameraMatrix[1], R, T, xyzl, xyzr);
			points3DL.push_back(xyzl);
			points3DR.push_back(xyzr);
		}
		else if (interPoints.size() == 2)
		{
			cv::Point3d xyzl0, xyzr0;
			cv::Point3d xyzl1, xyzr1;
			pr = interPoints[0] / scale;
			computeXYZ(pl, pr, cameraMatrix[0], cameraMatrix[1], R, T, xyzl0, xyzr0);
			pr = interPoints[1] / scale;
			computeXYZ(pl, pr, cameraMatrix[0], cameraMatrix[1], R, T, xyzl1, xyzr1);
			if (abs(xyzl0.z - xyzr0.z) < abs(xyzl1.z - xyzr1.z))
			{
				points3DL.push_back(xyzl0);
				points3DR.push_back(xyzr0);
			}
			else {
				points3DL.push_back(xyzl1);
				points3DR.push_back(xyzr1);
			}
		}
	}

	// Compute diameter of the ellipse.
	std::vector<cv::Point2f> holePointsL;
	for (size_t i = 0; i < points3DL.size(); i++)
	{
		cv::Point2f tempHolePoint;
		tempHolePoint.x = (float)points3DL[i].x;
		tempHolePoint.y = (float)points3DL[i].y;
		holePointsL.push_back(tempHolePoint);
	}
	if (holePointsL.size() > 4)
	{
		cv::RotatedRect holeRectL = fitEllipse(holePointsL);
		diameter = MAX(holeRectL.size.width, holeRectL.size.height);
	}
}

void Vision::fitPlane(std::vector<cv::Point3d> points, cv::Vec4d& planeCoeff)
{
	// A * x + B * y + C * z + D = 0
	// M * [A, B, C, D]' = 0
	// M' * M * [A, B, C, D]' = 0
	// v = [A, B, C, D]'
	// [U, S, V] = SVD(M);
	// v = V(:, 4)

	// Initialize the points matrix.
	std::vector<cv::Mat> matPoints;
	for (size_t i = 0; i < points.size(); i++)
	{
		cv::Mat tempMat = (cv::Mat_<double>(1, 4) << points[i].x, points[i].y, points[i].z, 1.0);
		matPoints.push_back(tempMat);
	}
	cv::Mat M;
	cv::vconcat(matPoints, M);

	// Compute coefficients by LSM.
	cv::Mat v(4, 1, CV_32FC1);
	cv::SVD::solveZ(M.t() * M, v);
	planeCoeff[0] = v.at<double>(0, 0);
	planeCoeff[1] = v.at<double>(1, 0);
	planeCoeff[2] = v.at<double>(2, 0);
	planeCoeff[3] = v.at<double>(3, 0);

	// Compute coefficients by SVD method.
	//Mat W, U, Vt;
	//SVDecomp(M, W, U, Vt, 0);
	//planeCoeff[0] = Vt.at<double>(3, 0);
	//planeCoeff[1] = Vt.at<double>(3, 1);
	//planeCoeff[2] = Vt.at<double>(3, 2);
	//planeCoeff[3] = Vt.at<double>(3, 3);

	// Normalization of first 3 coefficients.
	double denominator = -sqrt(planeCoeff[0] * planeCoeff[0] + planeCoeff[1] * planeCoeff[1] + planeCoeff[2] * planeCoeff[2]);
	planeCoeff[0] /= denominator;
	planeCoeff[1] /= denominator;
	planeCoeff[2] /= denominator;
	planeCoeff[3] /= denominator;
}

void Vision::computeCentDia(std::vector<cv::Point3d> points, cv::Vec4d plane, cv::Point3d &center, double &diameter)
{
	std::vector<cv::Point2f> holePoints;

	// Project points.
	// theta_ = k x n / |k x n| * theta
	// k = [0, 0, 1]'
	// n = [A, B, C]'
	// A^2 + B^2 + C^2 = 1
	// s = k x n = [-B, A, 0]'
	// R = rodrigues(theta_)
	// p1 = R * p2
	// p2 = R' * p1

	// Compute rotated matrix.
	double A = plane[0];
	double B = plane[1];
	double C = plane[2];
	double angle = acos(C);
	double factor = angle / sqrt(B * B + A * A);
	cv::Mat s = (cv::Mat_<double>(3, 1) << -B * factor, A * factor, 0);
	cv::Mat R(3, 3, CV_32FC1);
	cv::Rodrigues(s, R);

	// Transformate points.
	for (size_t i = 0; i < points.size(); i++)
	{
		cv::Mat tempMat = (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z);
		tempMat = R.t() * tempMat;
		cv::Point2f tempPoint;
		tempPoint.x = (float)tempMat.at<double>(0, 0);
		tempPoint.y = (float)tempMat.at<double>(1, 0);
		holePoints.push_back(tempPoint);
	}

	// Get coordinate of hole center.
	if (holePoints.size() > 4)
	{
		cv::RotatedRect holeRect = fitEllipse(holePoints);
		diameter = (holeRect.size.height + holeRect.size.width) / 2;
		// n' * p1 = -D
		// p1 = R * p2
		// n' * R * p2 = -D
		cv::Point3d holeCenter;
		cv::Mat rowMat = (cv::Mat_<double>(1, 3) << A, B, C);
		rowMat = rowMat * R;
		holeCenter.x = holeRect.center.x;
		holeCenter.y = holeRect.center.y;
		holeCenter.z = -(rowMat.at<double>(0) * holeCenter.x + rowMat.at<double>(1) * holeCenter.y + plane[3]) / rowMat.at<double>(2);
		cv::Mat colMat = (cv::Mat_<double>(3, 1) << holeCenter.x, holeCenter.y, holeCenter.z);
		colMat = R * colMat;
		center.x = colMat.at<double>(0);
		center.y = colMat.at<double>(1);
		center.z = colMat.at<double>(2);
	}
}

// Override
void Vision::computeCD(cv::RotatedRect rects[], cv::Vec4d plane[], cv::Point3d xyz[], double &diameter, int pointNum, cv::Mat cameraMatrix[], cv::Mat R, cv::Mat T)
{
	//// Compute xyz of center of hole.
	//// It needs some modifications to coordinate of hole centers.
	//cv::Point2d pl, pr;
	//pl = rects[0].center;
	//pr = rects[1].center;
	//computeXYZ(pl, pr, cameraMatrix[0], cameraMatrix[1], R, T, xyz[0], xyz[1]);

	// Compute fundamental matrix.
	cv::Mat F;
	computeF(cameraMatrix[0], cameraMatrix[1], R, T, F);
	// Select points of the left image.
	std::vector<cv::Point2f> points;
	Vision::selectPoints(rects[0], points, pointNum);
	// Compute correspond epipolar lines.
	std::vector<cv::Vec3f> lines;
	computeCorrespondEpilines(points, 1, F, lines);
	// Compute crossover points.
	std::vector<cv::Point2f> pointsL;
	std::vector<cv::Point2f> pointsR;
	for (int i = 0; i < pointNum; i++)
	{
		// Compute crossover points and 3D coordinate.
		cv::Point2f interPoint;
		getTarPoint(rects[0], rects[1], lines[i], points[i], interPoint);
		if (interPoint.x != 0 && interPoint.y != 0)
		{
			pointsL.push_back(points[i]);
			pointsR.push_back(interPoint);
		}
	}
	// compute 3D coordinates.
	std::vector<cv::Point3d> points3DL;
	std::vector<cv::Point3d> points3DR;
	for (size_t i = 0; i < pointsL.size(); i++)
	{
		cv::Point3d xyzl;
		cv::Point3d xyzr;
		computeXYZ(pointsL[i], pointsR[i], cameraMatrix[0], cameraMatrix[1], R, T, xyzl, xyzr);
		points3DL.push_back(xyzl);
		points3DR.push_back(xyzr);
	}

	//// Compute diameter of the ellipse.
	//std::vector<cv::Point2f> holePointsL;
	//for (size_t i = 0; i < points3DL.size(); i++)
	//{
	//	cv::Point2f tempHolePoint;
	//	tempHolePoint.x = (float)points3DL[i].x;
	//	tempHolePoint.y = (float)points3DL[i].y;
	//	holePointsL.push_back(tempHolePoint);
	//}
	//if (holePointsL.size() > 4)
	//{
	//	cv::RotatedRect holeRectL = fitEllipse(holePointsL);
	//	diameter = MAX(holeRectL.size.width, holeRectL.size.height);
	//}

	// Compute diameter of the ellipse and coordinate of hole center.
	Vision::fitPlane(points3DL, plane[0]);
	Vision::fitPlane(points3DR, plane[1]);
	double tempD[2] = { 0 };
	Vision::computeCentDia(points3DL, plane[0], xyz[0], tempD[0]);
	Vision::computeCentDia(points3DR, plane[1], xyz[1], tempD[1]);
	diameter = (tempD[0] + tempD[1]) / 2;
}

void Vision::computeDelDeep(cv::Vec4d plane1, cv::Vec4d plane2, double d1, double d2, double theta, double &delta, double &deep)
{
	// Compute verticality of the hole.
	double n1n2 = plane1[0] * plane2[0] + plane1[1] * plane2[1] + plane1[2] * plane2[2];
	delta = acos(n1n2);

	// Compute deep of the hole.
	theta = theta / 360 * CV_PI;
	if (d2 < d1)
	{
		std::swap(d1, d2);
	}
	deep = (d2 - d1) / (2 * tan(theta));
}

// Compute parameters of the hole.
void Vision::computeHoleParameters(double d1, double d2, double theta, double D, double &h, double &delta)
{
	// Compute deep of the hole.
	theta = theta / 360 * CV_PI;
	if (d2 < d1)
	{
		std::swap(d1, d2);
	}
	h = (d2 - d1) / (2 * tan(theta));

	// Compute verticality of the hole.
	delta = (D * sin(theta * 2)) / (d1 * sin(theta) * sin(theta) + (d2 - d1));
}

void Vision::saveHoleParameters(std::string fileName, cv::Point3d center[][2], double diameter[], double delta, double deep)
{
	// Save calibration parameters
	cv::FileStorage fs(fileName, cv::FileStorage::WRITE || cv::FileStorage::APPEND);
	if (!fs.isOpened())
	{
		std::cout << "Error: can not save calibration parameters" << std::endl;
		return;
	}
	// Here should not exist spaces in the name.
	fs << "Left_Center_1" << center[0][0]
		<< "Left_Center_2" << center[1][0]
		<< "Right_Center_1" << center[0][1]
		<< "Right_Center_2" << center[1][1];
	fs << "Diameter_1" << diameter[0]
		<< "Diameter_2" << diameter[1]
		<< "Delta" << delta
		<< "Deep" << deep;
	fs.release();
}

void Vision::adjustUnilinear(std::vector<cv::Point2f> basePoints, int holeNum, std::vector<cv::Point2f> &adjustedPoints)
{
	if (basePoints.size() != 2 || holeNum < 2)
	{
		return;
	}
	else
	{
		float ratio1, ratioN;
		cv::Point2f tempPoint;
		for (int i = 1; i <= holeNum; ++i)
		{
			ratio1 = (float)(holeNum - i) / (holeNum - 1);
			ratioN = (float)(i - 1) / (holeNum - 1);
			tempPoint.x = ratio1 * basePoints[0].x + ratioN * basePoints[1].x;
			tempPoint.y = ratio1 * basePoints[0].y + ratioN * basePoints[1].y;
			adjustedPoints.push_back(tempPoint);
		}
	}
}

void Vision::adjustBilinear(std::vector<cv::Point2f> basePoints, int rowNum, int colNum, std::vector<std::vector<cv::Point2f>> &adjustedPoints)
{
	if (basePoints.size() != 4 || rowNum < 2 || colNum < 2)
	{
		return;
	}
	else
	{
		float ratioC1, ratioCN;
		float ratioR1, ratioRN;
		cv::Point2f tempPoint1;
		cv::Point2f tempPoint2;
		cv::Point2f tempPoint;
		for (int j = 0; j < rowNum; ++j)
		{
			for (int i = 0; i < colNum; ++i)
			{
				ratioC1 = (float)(colNum - i) / (colNum - 1);
				ratioCN = (float)(i - 1) / (colNum - 1);
				ratioR1 = (float)(rowNum - j) / (rowNum - 1);
				ratioRN = (float)(j - 1) / (rowNum - 1);
				tempPoint1.x = ratioC1 * basePoints[0].x + ratioCN * basePoints[1].x;
				tempPoint1.y = ratioC1 * basePoints[0].y + ratioCN * basePoints[1].y;
				tempPoint2.x = ratioC1 * basePoints[3].x + ratioCN * basePoints[2].x;
				tempPoint2.y = ratioC1 * basePoints[3].y + ratioCN * basePoints[2].y;
				tempPoint.x = ratioR1 * tempPoint1.x + ratioRN * tempPoint2.x;
				tempPoint.y = ratioR1 * tempPoint1.y + ratioRN * tempPoint2.y;
			}
		}
	}
}