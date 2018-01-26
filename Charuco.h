//
// Created by xottabi4 on 11/20/17.
//

#ifndef ARUCO_TEST_CHARUCO_H
#define ARUCO_TEST_CHARUCO_H

#include <opencv2/aruco/charuco.hpp>

class Charuco {
public:

    Charuco(std::string &calibrationVideoFileName, std::string &cameraParametersFileName);

    Charuco(std::string &cameraParametersFileName);

    void saveBoardAsImage();

    void detectBoardMarkersFromImageCalibrated(cv::Mat &inputImage, std::vector<cv::Point2f> &charucoCorners,
                                               std::vector<int> &charucoIds);

    void estimatePose(cv::Mat &inputImage, std::vector<cv::Point2f> charucoCorners, std::vector<int> charucoIds, cv::Vec3d &rvec, cv::Vec3d &tvec);

    const cv::Ptr<cv::aruco::CharucoBoard> &getBoard() const;

    const cv::Mat &getCameraMatrix() const;

    const cv::Mat &getDistCoeffs() const;

private:

    cv::Ptr<cv::aruco::CharucoBoard> board;
    //aruco::CharucoBoard board;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    Charuco();

    void generateBoard();

    void calibrateCamera(std::string &calibrationVideoFileName, std::string &cameraParametersFileName);

    void saveCameraCalibrationParams(std::string &cameraParametersFileName, double repError);

    void loadCameraCalibrationParams(std::string &cameraParametersFileName);

    void detectCharucoCornersFromImage(cv::Mat &inputImage, std::vector<int> &charucoIds,
                                       std::vector<cv::Point2f> &charucoCorners);
};


#endif //ARUCO_TEST_CHARUCO_H
