#include "include/yolo_detect_batch.h"

void YOLODetect_batch::run() {

    std::cout << "[yolo] run() started" << std::endl;
    
    // Warmup
    {
        cv::Mat warmup_img = cv::Mat::zeros(_yoloHeight, _yoloWidth, CV_8UC3);
        int warmup_counter = -1;

        for (int i = 0; i < 5; ++i) {
            try {
                auto warmup_result = detectFrame(warmup_img, warmup_counter, true);
                std::cout << i << "-th warmup :: " << warmup_result.first.size() << " ROIs" << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "[yolo] warmup failed at iteration "
                    << i << ": " << e.what() << std::endl;
            }
        }
        std::cout << "[yolo] warmup finished." << std::endl;
    }

    while (true) {
        if (!q_cam2yolo.empty()) {
            break;
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }

        if (!q_end.empty()) {
            std::cout << "[yolo] end flag before start" << std::endl;
            break;
        }
    }

    if (!q_end.empty())
        return;

    int counter = 0;
    double time_elapsed = 0.0;
    while (true) {
        if (!q_end.empty()) {
            std::cout << "[yolo] end flag received" << std::endl;
            break;
        }

        if (!q_cam2yolo.empty()) {
            auto now = std::chrono::steady_clock::now();
            Cam2Yolo data_fromCam = q_cam2yolo.front();
            q_cam2yolo.pop();

            cv::Mat image_raw = data_fromCam.img_raw;
            double time_img = data_fromCam.time_img;

        
            auto trackers = detectFrame(image_raw, counter);
            std::vector<cv::Rect2d> rois = trackers.first;
            std::vector<int> labels = trackers.second;


            if (!rois.empty()) {
                Yolo2Tracking data_toTracking;
                data_toTracking.img_raw = image_raw;
                data_toTracking.rois = rois;
                data_toTracking.labels = labels;
                data_toTracking.time_detect = time_img;

                q_yolo2tracking.push(data_toTracking);
            }
            // Association with hungarian algorithm.
            auto end_iteration = std::chrono::steady_clock::now();
            double time_inference = std::chrono::duration_cast<std::chrono::milliseconds>(end_iteration - now).count();
            time_elapsed += time_inference;
            if (counter % 50 == 0 && counter <= 300)
                std::cout << "processing time= " << time_inference << " ms" << std::endl;

            counter++;
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    double processing_speed = counter / (time_elapsed / 1000.0);
    std::cout << "[YOLO detection] processing speed = " << processing_speed << " Hz" << std::endl;
}

void YOLODetect_batch::preprocessRGB(cv::Mat& frame, torch::Tensor& imgTensor)
{
    // run
    cv::Mat yoloimg; // define yolo img type
    //cv::imwrite("input.jpg", frame);
    //cv::cvtColor(frame, yoloimg, cv::COLOR_GRAY2RGB);
    cv::resize(frame, yoloimg, _YOLOSize);
    //cv::imwrite("yoloimg.jpg", yoloimg);
    //std::cout << "yoloImg.height" << yoloimg.rows << ", yoloimg.width" << yoloimg.cols << std::endl;
    imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); // vector to tensor
    imgTensor = imgTensor.permute({ 2, 0, 1 });                                                  // Convert shape from (H,W,C) -> (C,H,W)
    imgTensor = imgTensor.toType(torch::kFloat);                                               // convert to float type
    imgTensor = imgTensor.div(255);                                                            // normalization
    imgTensor = imgTensor.unsqueeze(0);                                                        //(1,3,320,320)
    imgTensor = imgTensor.to(*device);                                                         // transport data to GPU
}

//#(objects), (left,top,width,height)
std::pair<std::vector<cv::Rect2d>, std::vector<int>> YOLODetect_batch::detectFrame(cv::Mat& frame, int& counter, const bool bool_color)
{
     /* preprocess img */
    torch::Tensor imgTensor;
    if (bool_color)
        preprocessRGB(frame, imgTensor);
    else
        preprocessImg(frame, imgTensor);

    /* inference */
    torch::Tensor preds;

    auto start_inf = std::chrono::high_resolution_clock::now();

    /* inference */
    /* wrap to disable grad calculation */
    {
        torch::NoGradGuard no_grad;
        preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,300,6]
    }

    if (_debug_yolo) {
        auto stop_inf = std::chrono::high_resolution_clock::now();
        auto duration_inf = std::chrono::duration_cast<std::chrono::microseconds>(stop_inf - start_inf);
        std::cout << "** YOLO inference time = " << duration_inf.count() << " microseconds **" << std::endl;
    }

    //POST PROCESS
    auto start_postprocess = std::chrono::high_resolution_clock::now();
    //STEP1 :: divide detections into balls ans boxes
    std::vector<torch::Tensor> rois; //detected rois.(n,6),(m,6) :: including both left and right objects
    std::vector<int> labels;//detected labels.
    torch::Tensor preds_good = preds.select(2, 4) > _ConfThreshold; // Extract the high score detections. :: xc is "True" or "False"
    torch::Tensor x0 = preds.index_select(1, torch::nonzero(preds_good[0]).select(1, 0)); // box, x0.shape : (1,n,6) : n: number of candidates
    x0 = x0.squeeze(0); //(1,n,6) -> (n,6) (left,top,right,bottom)
    int size = x0.size(0);//num of detections
    torch::Tensor bbox, pred;
    double left, top, right, bottom;
    int label;

    if (size == 1) {
        pred = x0[0].cpu();
        bbox = pred.slice(0, 0, 4);//x.slice(dim,start,end);
        bbox[0] = std::max(bbox[0].item<double>(), 0.0);//left
        bbox[1] = std::max(bbox[1].item<double>(), 0.0);//top
        bbox[2] = std::min(bbox[2].item<double>(), _yoloWidth);//right
        bbox[3] = std::min(bbox[3].item<double>(), _yoloHeight);//bottom
        label = pred[5].item<int>();//label
        rois.push_back(bbox);
        labels.push_back(label);
    }
    else if (size >= 2) {
        for (int i = 0; i < size; i++) {
            pred = x0[i].cpu();
            bbox = pred.slice(0, 0, 4);//x.slice(dim,start,end);
            bbox[0] = std::max(bbox[0].item<double>(), 0.0);//left
            bbox[1] = std::max(bbox[1].item<double>(), 0.0);//top
            bbox[2] = std::min(bbox[2].item<double>(), _yoloWidth);//right
            bbox[3] = std::min(bbox[3].item<double>(), _yoloHeight);//bottom
            label = pred[5].item<int>();//label
            rois.push_back(bbox);
            labels.push_back(label);
        }
    }

	std::vector<cv::Rect2d> rois_new; 
    std::vector<int> classes;
	roiSetting(rois, labels, rois_new, classes); //separate detection into left and right

    return std::make_pair(rois_new,classes);
}

void YOLODetect_batch::roiSetting(
    std::vector<torch::Tensor>& detectedBoxes, std::vector<int>& labels,
    std::vector<cv::Rect2d>& newRoi, std::vector<int>& newClass
)
{
    // std::cout << "bboxesYolo size=" << detectedBoxes.size() << std::endl;
    /* detected by Yolo */
    if (!detectedBoxes.empty())
    {
        //std::cout << "No TM tracker exist " << std::endl;
        int numBboxes = detectedBoxes.size(); // num of detection
        int left, top, right, bottom;         // score0 : ball , score1 : box
        cv::Rect2d roi;

        /* convert torch::Tensor to cv::Rect2d */
        std::vector<cv::Rect2d> bboxesYolo_left, bboxesYolo_right;
        for (int i = 0; i < numBboxes; ++i)
        {
			if (std::find(_object_indices.begin(),_object_indices.end(),labels[i]) != _object_indices.end()){//detect only designated objects
				float expandrate[2] = { _frameWidth / _yoloWidth, _frameHeight / _yoloHeight }; // resize bbox to fit original img size
				
				// std::cout << "expandRate :" << expandrate[0] << "," << expandrate[1] << std::endl;
				left = static_cast<int>(detectedBoxes[i][0].item().toFloat() * expandrate[0]);
				top = static_cast<int>(detectedBoxes[i][1].item().toFloat() * expandrate[1]);
				right = static_cast<int>(detectedBoxes[i][2].item().toFloat() * expandrate[0]);
				bottom = static_cast<int>(detectedBoxes[i][3].item().toFloat() * expandrate[1]);
				
				
				newRoi.emplace_back(left, top, (right - left), (bottom - top));
				newClass.push_back(labels[i]);
			}
        }
    }
    /* No object detected in Yolo -> return -1 class label */
    else
    {
        /* nothing to do */
    }
}

void YOLODetect_batch::preprocessImg(cv::Mat& frame, torch::Tensor& imgTensor)
{
    // run
    //std::cout << frame.size() << std::endl;
    cv::Mat yoloimg; // define yolo img type
    cv::resize(frame, yoloimg, _YOLOSize);
    cv::cvtColor(yoloimg, yoloimg, cv::COLOR_GRAY2RGB);
    imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); // vector to tensor
    imgTensor = imgTensor.permute({ 2, 0, 1 });                                                  // Convert shape from (H,W,C) -> (C,H,W)
    imgTensor = imgTensor.toType(torch::kFloat);                                               // convert to float type
    imgTensor = imgTensor.div(255);                                                            // normalization
    imgTensor = imgTensor.unsqueeze(0);                                                        // expand dims for Convolutional layer (height,width,1)
    imgTensor = imgTensor.to(*device);                                                         // transport data to GPU
}