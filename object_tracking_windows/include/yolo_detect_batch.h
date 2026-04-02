#pragma once

#ifndef YOLO_DETECT_BATCH_H
#define YOLO_DETECT_BATCH_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/optflow/rlofflow.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/ocl.hpp"
#include "iostream"
#include "vector"
#include "queue"
#include <chrono>
#include "include/global_parameters.h"

/*  YOLO class definition  */
class YOLODetect_batch
{
private:
    const bool _debug_yolo = false;

    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;

    
    std::string _yolofilePath;
    std::vector<size_t> _object_indices;
    double _frameWidth;
    double _frameHeight;
    double _yoloWidth;
    double _yoloHeight;
    cv::Size _YOLOSize;
    double _IoUThreshold;
    double _ConfThreshold;

    /* initialize function */
    void initializeDevice()
    {
        // set device

        if (torch::cuda::is_available())
        {
            // device = new torch::Device(devicetype, 0);
            device = new torch::Device(torch::kCUDA);
            std::cout << "set cuda" << std::endl;
        }
        else
        {
            device = new torch::Device(torch::kCPU);
            std::cout << "set CPU" << std::endl;
        }
        //device = new torch::Device(torch::kCPU);
        //std::cout << "set CPU" << std::endl;
    }

    void loadModel()
    {
        // read param
        mdl = torch::jit::load(_yolofilePath, *device);
        mdl.to(*device);
        mdl.eval();
        std::cout << "load model" << std::endl;
    }

public:
    // constructor for YOLODetect
    YOLODetect_batch(const int frameWidth, const int frameHeight, const int yoloWidth, const int yoloHeight,
        const double conf_threshold, const double iou_threhold,
        const std::vector<size_t> object_indices,
        const std::string yolo_path
        )
    {
        _yolofilePath = yolo_path;
        _object_indices = object_indices;
        _frameWidth = static_cast<double>(frameWidth);
        _frameHeight = static_cast<double>(frameHeight);
        _yoloWidth = static_cast<double>(yoloWidth);
        _yoloHeight = static_cast<double>(yoloHeight);
        _YOLOSize = { yoloWidth, yoloHeight };
        _ConfThreshold = conf_threshold;
        _IoUThreshold = iou_threhold;
        
        initializeDevice();
        loadModel();
        std::cout << "YOLO construtor has finished!" << std::endl;
    };
    ~YOLODetect_batch() { delete device; }; // Deconstructor
    
    void run();

	void preprocessRGB(cv::Mat& frame, torch::Tensor& imgTensor);
	
    std::pair<std::vector<cv::Rect2d>, std::vector<int>> detectFrame(cv::Mat& frame, int& counter, const bool bool_color=true);

	void roiSetting(
		std::vector<torch::Tensor>& detectedBoxes, std::vector<int>& labels,
		std::vector<cv::Rect2d>& newRoi, std::vector<int>& newClass
	);

    //preprocess img
    void preprocessImg(cv::Mat& frame, torch::Tensor& imgTensor);
};

#endif