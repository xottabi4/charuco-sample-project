#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <algorithm>
#include <opencv/cv.hpp>
#include "Charuco.h"

using namespace cv;
using namespace std;

void drawOnTopOfBoard(Mat &testImageCopy, vector<Point2f> &charucoCorners, vector<int> &charucoIds, const Mat &logo);

void render(Mat &inputImage, Vec3d &vec, Vec3d &tvec, const Mat &matrix, const Mat &coeffs);

void createCubePoints(vector<Point3d> &bottomPoints, vector<Point3d> &topPoints);

void drawCube(Mat &inputImage, vector<Point2d> &bottomPointsProjected, vector<Point2d> &topPointsProjected);

int main() {
    std::string calibrationVideoFileName = "resources/charuco_board_2_calibration_video.mp4";
    std::string cameraParametersFileName = "charuco_board_2_calibration_params.yml";
    //    std::unique_ptr<Charuco> charuco(new Charuco(calibrationVideoFileName, cameraParametersFileName));
    std::unique_ptr<Charuco> charuco(new Charuco(cameraParametersFileName));

    VideoCapture cap;
    cap.open("resources/video_for_test.mp4");
    if (!cap.isOpened()) {
        return -1;
    }
    cv::Mat frame = imread("resources/test_image3.jpg");
    Mat logo = imread("resources/logo.png");

    namedWindow("transformed", WINDOW_AUTOSIZE);
    namedWindow("pose", WINDOW_AUTOSIZE);

//    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
//    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
//    VideoWriter video("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height), true);
    while (1) {
        cap >> frame;
        if (frame.empty()) {
            cerr << "empty video frame !!!" << endl;
            break;
        }
        //        cv::resize(frame, frame, cv::Size(frame.cols / 1.5, frame.rows / 1.5));
        cv::Mat testImageCopy = frame.clone();

        std::vector<cv::Point2f> charucoCorners;
        std::vector<int> charucoIds;
        charuco->detectBoardMarkersFromImageCalibrated(frame, charucoCorners, charucoIds);

        cv::Vec3d rvec, tvec;
        try {
            charuco->estimatePose(frame, charucoCorners, charucoIds, rvec, tvec);
        } catch (std::exception const &e) {
            cerr << e.what() << endl;
            imshow("transformed", testImageCopy);
//            video.write(testImageCopy);
            continue;
        }
        drawOnTopOfBoard(testImageCopy, charucoCorners, charucoIds, logo);

        render(testImageCopy, rvec, tvec, charuco->getCameraMatrix(), charuco->getDistCoeffs());

        imshow("transformed", testImageCopy);
//        video.write(testImageCopy);
        imshow("pose", frame);
        if (waitKey(30) >= 0) {
            continue;
        }
    }
    return 0;
}

void render(Mat &inputImage, Vec3d &vec, Vec3d &tvec, const Mat &matrix, const Mat &coeffs) {
    vector<Point3d> bottomPoints;
    vector<Point3d> topPoints;
    createCubePoints(bottomPoints, topPoints);

    vector<Point2d> bottomPointsProjected;
    vector<Point2d> topPointsProjected;
    projectPoints(bottomPoints, vec, tvec, matrix, coeffs, bottomPointsProjected);
    projectPoints(topPoints, vec, tvec, matrix, coeffs, topPointsProjected);

    drawCube(inputImage, bottomPointsProjected, topPointsProjected);
}

void createCubePoints(vector<Point3d> &bottomPoints, vector<Point3d> &topPoints) {
    int squareLength = 40;
    int widthCoeficient = 4;
    int heigthCoeficient = 3;

    int boxHeigth = 40;
    int boxWidth = 40;

    bottomPoints.push_back(Point3i(squareLength,
                                   squareLength, 0));
    bottomPoints.push_back(Point3i(squareLength,
                                   squareLength * widthCoeficient + boxWidth, 0));
    bottomPoints.push_back(Point3i(squareLength * heigthCoeficient + boxWidth,
                                   squareLength * widthCoeficient + boxWidth, 0));
    bottomPoints.push_back(Point3i(squareLength * heigthCoeficient + boxWidth,
                                   squareLength, 0));

    topPoints.push_back(Point3i(squareLength,
                                squareLength,
                                boxHeigth * 2));
    topPoints.push_back(Point3i(squareLength,
                                squareLength * widthCoeficient + boxWidth,
                                boxHeigth * 2));
    topPoints.push_back(Point3i(squareLength * heigthCoeficient + boxWidth,
                                squareLength * widthCoeficient + boxWidth,
                                boxHeigth * 2));
    topPoints.push_back(Point3i(squareLength * heigthCoeficient + boxWidth,
                                squareLength,
                                boxHeigth * 2));
}

void drawCube(Mat &inputImage, vector<Point2d> &bottomPointsProjected, vector<Point2d> &topPointsProjected) {
    Scalar cubeColor = Scalar(0, 0, 255);
    int lineThickness = 2;
    // Draw bottom and top
    vector<vector<Point> > contoursToDraw(2);
    vector<Point> v(bottomPointsProjected.begin(), bottomPointsProjected.end());
    contoursToDraw.push_back(v);
    vector<Point> v2(topPointsProjected.begin(), topPointsProjected.end());
    contoursToDraw.push_back(v2);
    polylines(inputImage, contoursToDraw, true, cubeColor, lineThickness);
    //    drawContours(inputImage,contoursToDraw,-1,cubeColor);

    // Draw side lines
    for (int i = 0; i < 4; i++) {
        line(inputImage, bottomPointsProjected.at(i), topPointsProjected.at(i), cubeColor, lineThickness);
    }
}

void drawOnTopOfBoard(Mat &testImageCopy, vector<Point2f> &charucoCorners, vector<int> &charucoIds, const Mat &logo) {
    // Input Quadilateral or Image plane coordinates
    vector<Point2f> inputQuad;
    std::array<int, 4> neededCornerIds = {19, 3, 0, 16};
    for (int corner : neededCornerIds) {
        int pos = std::find(charucoIds.begin(), charucoIds.end(), corner) - charucoIds.begin();
        if (pos >= charucoIds.size()) {
            cerr << "Mandatory corner " + std::to_string(corner) + " not detected !" << endl;
            return;
        } else {
            inputQuad.push_back(charucoCorners.at(pos));
        }
    }
    //    inputQuad.push_back(Point(charucoCorners.at(19).x - (charucoCorners.at(15).x - charucoCorners.at(19).x),
    //                              charucoCorners.at(19).y - (charucoCorners.at(18).y - charucoCorners.at(19).y)));
    //    inputQuad.push_back(Point(charucoCorners.at(3).x + (charucoCorners.at(3).x - charucoCorners.at(7).x),
    //                              charucoCorners.at(3).y - (charucoCorners.at(2).y - charucoCorners.at(3).y)));
    //    inputQuad.push_back(Point(charucoCorners.at(0).x + (charucoCorners.at(0).x - charucoCorners.at(4).x),
    //                              charucoCorners.at(0).y + (charucoCorners.at(0).y - charucoCorners.at(1).y)));
    //    inputQuad.push_back(Point(charucoCorners.at(16).x - (charucoCorners.at(12).x - charucoCorners.at(16).x),
    //                              charucoCorners.at(16).y + (charucoCorners.at(16).y - charucoCorners.at(17).y)));

    //    double d = 1.1;
    //    inputQuad.push_back(charucoCorners.at(19) * d);
    //    inputQuad.push_back(charucoCorners.at(3) * d);
    //    inputQuad.push_back(charucoCorners.at(0) * d);
    //    inputQuad.push_back(charucoCorners.at(16) * d);


//    inputQuad.push_back(charucoCorners.at(19));
//    inputQuad.push_back(charucoCorners.at(3));
//    inputQuad.push_back(charucoCorners.at(0));
//    inputQuad.push_back(charucoCorners.at(16));
    // Output Quadilateral or World plane coordinates
    vector<Point2f> outputQuad;
    outputQuad.push_back(Point(0, 0));
    outputQuad.push_back(Point(logo.cols, 0));
    outputQuad.push_back(Point(logo.cols, logo.rows));
    outputQuad.push_back(Point(0, logo.rows));
    
    Mat transformedLogo(2, 4, CV_32FC1);
    // Set the lambda matrix the same type and size as input
    transformedLogo = Mat::zeros(logo.rows, logo.cols, logo.type());

    // Mat lambda = getPerspectiveTransform( inputQuad, outputQuad );
    Mat lambda = findHomography(outputQuad, inputQuad);
    warpPerspective(logo, transformedLogo, lambda, testImageCopy.size());

    //    namedWindow("a", WINDOW_AUTOSIZE);
    //    imshow("a", testImageCopy + transformedLogo);
    testImageCopy += transformedLogo;
}
