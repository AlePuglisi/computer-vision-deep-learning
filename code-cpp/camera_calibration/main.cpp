#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <fstream>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace cv;

int main(int argc, char** argv)
{

    (void) argc;
    (void) argv; 

    std::vector<cv::String> fileNames; 

    cv::String image_path_pattern = "/home/ale/tutorials/computer-vision-deep-learning/code-cpp/camera_calibration/calibration_images/image*.png";
    cv::glob(image_path_pattern , fileNames, false);
    cv::Size patternSize(14 -1, 10 -1);


    std::vector<std::vector<cv::Point2f>> q(fileNames.size());
    std::vector<std::vector<cv::Point3f>> Q;

    int checkerBoard[2] = {14,10} ; 
    int fieldSize = 20; 

    std::vector<cv::Point3f> objp;
    for(int i = 1; i < checkerBoard[1]; i++){
        for(int j = 1; j < checkerBoard[0]; j++){
            objp.push_back(cv::Point3f(j*fieldSize, i*fieldSize, 0)); 
        }    
    }

    std::vector<cv::Point2f> imgPoint;
    std::size_t i = 0; 
    for (auto const &f : fileNames){
        std::cout << std::string(f) << std::endl; 

        cv::Mat img = cv::imread(fileNames[i]); 
        cv::Mat gray; 

        cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

        bool patternFound = cv::findChessboardCorners(gray, patternSize, q[i], cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

        if(patternFound){
            cv::cornerSubPix(gray, q[i], cv::Size(11,11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            Q.push_back(objp);
        }

        cv::drawChessboardCorners(img, patternSize, q[i], patternFound);
        
        cv::namedWindow("chessboard detection", cv::WINDOW_AUTOSIZE);
        cv::imshow("chessboard detection", img);
        cv::waitKey(0);

        i++;
    }

    cv::Matx33f K(cv::Matx33f::eye());
    cv::Vec<float, 5> k(0, 0, 0, 0, 0);

    std::vector<cv::Mat> rvecs, tvecs; 
    std::vector<double> stdIntrinsic, stdExtrinsic, perViewErrors; 

    int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT; 
    cv::Size frameSize(1200, 800); 

    std::cout << "Calibrating..." << std::endl; 

    float error = cv::calibrateCamera(Q, q, frameSize, K, k, rvecs, tvecs, flags); 

    std::cout << "Reprojection Error = " << error << "\nK = \n" << K << "\nk = \n" << k << std::endl; 

    
    std::cout << "\nrvecs = \nrvec.size() = " << rvecs.size() << std::endl;
    for (int i = 0; i < rvecs.size() ; i++){
        std::cout << "\nImage" << i << std::endl;
        std::cout << rvecs[i] << std::endl;
    }
    std::cout << "\ntvecs = \ntvecs.size() = " << tvecs.size() << std::endl;
    for (int i = 0; i < tvecs.size() ; i++){
        std::cout << "\nImage" << i << std::endl;
        std::cout << tvecs[i] << std::endl;
    }
    
    std::ofstream poseFile("camera_poses.csv");
    poseFile << "image_id,rx,ry,rz,tx,ty,tz\n";
    for (size_t i = 0; i < rvecs.size(); i++) {
        poseFile << i << ","
                << rvecs[i].at<double>(0) << ","
                << rvecs[i].at<double>(1) << ","
                << rvecs[i].at<double>(2) << ","
                << tvecs[i].at<double>(0) << ","
                << tvecs[i].at<double>(1) << ","
                << tvecs[i].at<double>(2) << "\n";
    }
    poseFile.close();

    cv::Mat mapX, mapY; 
    cv::initUndistortRectifyMap(K, k, cv::Matx33f::eye(), K, frameSize, CV_32FC1, mapX, mapY); 


    for (auto const &f : fileNames){
        std::cout << std::string(f) << std::endl; 

        cv::Mat img = cv::imread(f, cv::IMREAD_COLOR); 
        
        cv::Mat imgUndistorted; 
        cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);

        // cv::namedWindow("Original Imag", cv::WINDOW_AUTOSIZE);
        // cv::imshow("Original Image", img);

        cv::namedWindow("Undistorted Imag", cv::WINDOW_AUTOSIZE);
        cv::imshow("Undistorted Image", imgUndistorted);

        cv::waitKey(0);
    }

    return 0;
}