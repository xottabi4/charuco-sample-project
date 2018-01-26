//
// Created by xottabi4 on 11/20/17.
//
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include "Charuco.h"


using namespace cv;
using namespace std;



//TODO board 1
//int squareLength = 40;   //Here, our measurement unit is millimeter.
//int markerLength = 30; // Here, our measurement unit is millimeter.

//TODO board 2
int squareLength = 40;   //Here, our measurement unit is millimeter.
int markerLength = 20; // Here, our measurement unit is millimeter.

//int squareLength = 25;   //Here, our measurement unit is millimeter.
//int markerLength = 13; // Here, our measurement unit is millimeter.

Charuco::Charuco() {
    generateBoard();
}

Charuco::Charuco(std::string &calibrationVideoFileName, std::string &cameraParametersFileName) : Charuco() {
    calibrateCamera(calibrationVideoFileName, cameraParametersFileName);
}

Charuco::Charuco(std::string &cameraParametersFileName) : Charuco() {
    loadCameraCalibrationParams(cameraParametersFileName);
}

const Ptr<aruco::CharucoBoard> &Charuco::getBoard() const {
    return board;
}

const Mat &Charuco::getCameraMatrix() const {
    return cameraMatrix;
}

const Mat &Charuco::getDistCoeffs() const {
    return distCoeffs;
}

void Charuco::generateBoard() {
    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(
            aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
    board = aruco::CharucoBoard::create(5, 6, squareLength, markerLength, markerDictionary);

//    Ptr<aruco::CharucoBoard> boardPointer = aruco::CharucoBoard::create(5, 6, squareLength, markerLength, markerDictionary);
//    board = boardPointer.operator*();
}

void Charuco::saveBoardAsImage() {
    Mat boardImage;
    board->draw(cv::Size(500, 600), boardImage, 10, 1);

    namedWindow("charuco board", WINDOW_AUTOSIZE);
    imshow("charuco board", boardImage);
    waitKey(0);

    imwrite("./charuco_board.jpg", boardImage);
}

void Charuco::detectBoardMarkersFromImageCalibrated(Mat &inputImage, std::vector<cv::Point2f> &charucoCorners,
                                                    std::vector<int> &charucoIds) {
    vector<int> markerIds;
    vector<vector<Point2f> > markerCorners;
    aruco::detectMarkers(inputImage, board->dictionary, markerCorners, markerIds);

    if (!markerIds.empty()) {
        aruco::interpolateCornersCharuco(markerCorners, markerIds, inputImage, board, charucoCorners, charucoIds,
                                         cameraMatrix, distCoeffs);
        if (!charucoIds.empty()) {
            cv::aruco::drawDetectedCornersCharuco(inputImage, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
        }
    }
}

void Charuco::detectCharucoCornersFromImage(Mat &inputImage, vector<int> &charucoIds,
                                            vector<Point2f> &charucoCorners) {
    vector<int> markerIds;
    vector<vector<Point2f> > markerCorners;
    cv::aruco::detectMarkers(inputImage, board->dictionary, markerCorners, markerIds);

    // refind strategy to detect more markers
    //    if(refindStrategy) aruco::refineDetectedMarkers(image, board, corners, ids, rejected);

    if (!markerIds.empty()) {
        aruco::interpolateCornersCharuco(markerCorners, markerIds, inputImage, board, charucoCorners, charucoIds);
        if (!charucoIds.empty()) {
            aruco::drawDetectedCornersCharuco(inputImage, charucoCorners, charucoIds, Scalar(255, 0, 0));
        }
    }
}

void Charuco::estimatePose(Mat &inputImage, vector<Point2f> charucoCorners, vector<int> charucoIds, cv::Vec3d &rvec, cv::Vec3d &tvec) {
    bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix,
                                                     distCoeffs, rvec, tvec);
    if (valid) {
        cv::aruco::drawAxis(inputImage, cameraMatrix, distCoeffs, rvec, tvec, 100);
    } else {
        throw std::runtime_error("estimatePoseCharucoBoard invalid !!!");
    }
}

void Charuco::calibrateCamera(std::string &calibrationVideoFileName, std::string &cameraParametersFileName) {
    VideoCapture inputVideo;
    int waitTime;
    if (!calibrationVideoFileName.empty()) {
        inputVideo.open(calibrationVideoFileName);
        waitTime = 0;
    } else {
        cerr << "Unable to open video file !" << endl;
        int camId = 0;
        inputVideo.open(camId);
        waitTime = 10;
    }

    // collect data from each frame
    vector<vector<Point2f> > allCharucoCorners;
    vector<vector<int> > allCharucoIds;
    vector<Mat> allImgs;
    Size imgSize;

    while (inputVideo.grab()) {
        Mat image;
        inputVideo.retrieve(image);
cout<< image.size()<<endl;
        Mat imageCopy = image.clone();

        vector<int> charucoIds;
        vector<Point2f> charucoCorners;

        detectCharucoCornersFromImage(imageCopy, charucoIds, charucoCorners);

        putText(imageCopy, "Press 'c' to add current frame. 'ESC' to finish and calibrate",
                Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

        imshow("out", imageCopy);
        char key = (char) waitKey(waitTime);
        if (key == 27) {
            break;
        }
        if (key == 'c' && !charucoIds.empty()) {
            cout << "Frame captured" << endl;
            allCharucoCorners.push_back(charucoCorners);
            allCharucoIds.push_back(charucoIds);
            allImgs.push_back(image);
            imgSize = image.size();
        }
    }

    if (allCharucoIds.empty()) {
        cerr << "Not enough captures for calibration" << endl;
        return;
    }

    // prepare data for calibration
//    vector< vector< Point2f > > allCornersConcatenated;
//    vector< int > allIdsConcatenated;
//    vector< int > markerCounterPerFrame;
//    markerCounterPerFrame.reserve(allCharucoCorners.size());
//    for(unsigned int i = 0; i < allCharucoCorners.size(); i++) {
//        markerCounterPerFrame.push_back((int)allCharucoCorners[i].size());
//        for(unsigned int j = 0; j < allCharucoCorners[i].size(); j++) {
//            allCornersConcatenated.push_back(allCharucoCorners[i][j]);
//            allIdsConcatenated.push_back(allCharucoIds[i][j]);
//        }
//    }

    // calibrate camera using charuco
//    double repError = aruco::calibrateCameraCharuco(allCornersConcatenated, allIdsConcatenated, &board, imgSize,
//                                                    cameraMatrix, distCoeffs);

    double repError = aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, board, imgSize,
                                                    cameraMatrix, distCoeffs);
    cout << "Rep Error: " << repError << endl;

    saveCameraCalibrationParams(cameraParametersFileName, repError);
}


void Charuco::saveCameraCalibrationParams(std::string &cameraParametersFileName, double repError) {
//    ofstream out(cameraParametersFileName.c_str());
    FileStorage fs(cameraParametersFileName, FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw runtime_error("can't save camera calibration parameters to file!");
    }
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "reprojection_error" << repError;
    cout << "Calibration saved to " << cameraParametersFileName << endl;
}

void Charuco::loadCameraCalibrationParams(std::string &cameraParametersFileName) {
//    ifstream in(cameraParametersFileName.c_str());
//    in >> "camera_matrix" >> cameraMatrix;
//    in >> "distortion_coefficients" >> distCoeffs;

    FileStorage fs(cameraParametersFileName, FileStorage::READ);
    if(!fs.isOpened()){
        throw runtime_error("can't save camera calibration parameters to file!");
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >>distCoeffs;
    cout << "Calibration camera parameters loaded from " << cameraParametersFileName << endl;
}
