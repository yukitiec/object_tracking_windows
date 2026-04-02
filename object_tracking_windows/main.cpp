//installed files.
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
#include <queue>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <thread>
#include <iostream>
#include <vector>
#include <array>
#include <ctime>
#include <direct.h>
#include <sys/stat.h>
#include <algorithm>
//custom files.
#include "include/global_parameters.h"
#include "include/manage_tracker.h"
#include "include/yolo_detect_batch.h"
#include "include/utils.h"

//make namespace.
using Track2DEntry = std::tuple<double, int, cv::Rect2d>;   // time, label, bbox
using Track2DSeq = std::vector<Track2DEntry>;


int main()
{
    //Load parameters for tracking.
    Config cfg;
    try
    {
        cfg = load_config("C:/Users/kawaw/cpp/object_tracking_windows/object_tracking_windows/config/default.txt");

        std::cout << std::boolalpha;
        std::cout << "display        : " << cfg.display << std::endl;
        std::cout << "time_capture   : " << cfg.time_capture << std::endl;
        std::cout << "yolo_path      : " << cfg.yolo_path << std::endl;
        std::cout << "yoloWidth      : " << cfg.yoloWidth << std::endl;
        std::cout << "yoloHeight     : " << cfg.yoloHeight << std::endl;
        std::cout << "IoU_threshold  : " << cfg.IoU_threshold << std::endl;
        std::cout << "conf_threshold : " << cfg.conf_threshold << std::endl;

        std::cout << "object_index   : ";
        for (size_t v : cfg.object_index) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    unsigned int idx_mode = 3;
	//0:skeleton, 1:corner detection, 2:cube detection. 
	//3: skeleton and ball tracking, 4: skeleton and cube tracking.
	const int pixel_neighbor_ = 2;

    // Object list
    std::vector<std::vector<cv::Rect2d>> ps_2d; // (n_obj, n_seq, 4) - bounding boxes for balls over time
    std::vector<std::vector<Track2DEntry>> storage_2d; // (n_obj, n_seq, 4) - timestamped bounding boxes
    
    double total_time = 0.0;

    // Storage for color images (add this vector to save images over time)
    std::vector<cv::Mat> stored_color_images;
    std::vector<double> time_list;
    int counter_deb = 0;
    bool roi_fixed = false;
    const bool _bool_kf = false;
    try
    {
        int camIndex = 0;      // default internal webcam
        
        cv::VideoCapture cap(camIndex);

        if (!cap.isOpened()) {
            throw runtime_error("Failed to open internal webcam.");
        }

        // Read default camera properties
        int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fps = cap.get(cv::CAP_PROP_FPS);

        std::cout << "Default webcam settings:" << std::endl;
        std::cout << "Width  : " << frameWidth << std::endl;
        std::cout << "Height : " << frameHeight << std::endl;
        std::cout << "FPS    : " << fps << std::endl;

        cv::namedWindow("PC Webcam RGB", cv::WINDOW_AUTOSIZE);

        int counter = 0;
        auto start_time = std::chrono::steady_clock::now();
        double time_current = 0.0;

        // Thread initialization for detection.
        std::vector<std::thread> worker_threads;

        // Construct required instances.
        TrackerManager _tracker_manager;

        // Setup YOLO detection batch class.
        YOLODetect_batch yolo_detect(
            frameWidth, frameHeight,
            cfg.yoloWidth, cfg.yoloHeight,
            cfg.conf_threshold, cfg.IoU_threshold,
            cfg.object_index, cfg.yolo_path
        );

        // Launch YOLO detection in a separate thread.
        worker_threads.emplace_back([&yolo_detect]() {
            yolo_detect.run();
        });

        while (cv::waitKey(1) < 0 && cv::getWindowProperty("PC Webcam RGB", cv::WND_PROP_VISIBLE) >= 1)
        {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - start_time;
            time_current = elapsed.count();
            //save time
            time_list.push_back(time_current);

            if (time_current > cfg.time_capture) {
                q_end.push(true);//terminal flag.
                break;
            }

            if (counter_deb == 0)
                cout << "start" << endl;

            counter_deb += 1;
            auto st_iteration = std::chrono::steady_clock::now();

            try {
                cv::Mat color_image;
                cap.read(color_image);

                if (color_image.empty()) {
                    cerr << "Failed to capture frame from webcam." << endl;
                    continue;
                }

                if (color_image.rows == frameHeight && color_image.cols == frameWidth) {

                    // Remove old data if q_cam2yolo size is greater than 2
                    while (q_cam2yolo.size() > 2) {
                        q_cam2yolo.pop();
                    }

                    Cam2Yolo data_toYolo;
                    data_toYolo.img_raw = color_image;
                    data_toYolo.time_img = time_current;
                    q_cam2yolo.push(data_toYolo);
                }

                if (!q_yolo2tracking.empty()) {
                    Yolo2Tracking data_fromYolo = q_yolo2tracking.front();
                    q_yolo2tracking.pop();
                    std::vector<cv::Rect2d> rois_2d = data_fromYolo.rois;  //ROI (Region of Interest) =  (N_object, cv::Rect(left,top,width,height)
                    std::vector<int> labels = data_fromYolo.labels;
                    double time_detect = data_fromYolo.time_detect;
                    cv::Mat img_detect = data_fromYolo.img_raw;

                    // 3D triangulation with fallback/dummy handling and out-of-bounds check
                    if (!rois_2d.empty()) {
                        
                        // 2D & 3D
                        std::vector<int> index_delete_storage = _tracker_manager.update2D(time_detect, rois_2d,labels,storage_2d);
                        
                        //move data to storage.
                        ps_2d.emplace_back(std::move(rois_2d));
                    
                    }
                }
                
                if (cfg.display) {
                    for (int i_obj = 0; i_obj < storage_2d.size(); i_obj++) { // for each object
                        double t_detect = std::get<0>(storage_2d[i_obj].back()); // time_last_update [second]
                        int label_object = std::get<1>(storage_2d[i_obj].back()); // label
                        cv::Rect2d bbox = std::get<2>(storage_2d[i_obj].back());
                        double box_width = bbox.width;
                        double box_height = bbox.height;
                        cv::rectangle(color_image, bbox, cv::Scalar(0, 0, 255), 2);
                        // Write i_obj, t_detect and bounding box size on top of the bounding box
                        std::ostringstream label_stream;
                        label_stream << "ID: " << i_obj <<" Label: "<<label_object;
                        std::string label = label_stream.str();
                        int baseLine = 0;
                        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                        int text_x = static_cast<int>(bbox.x);
                        int text_y = std::max(static_cast<int>(bbox.y) - label_size.height - 4, 0);
                        cv::rectangle(color_image,
                            cv::Point(text_x, text_y),
                            cv::Point(text_x + label_size.width, text_y + label_size.height + baseLine),
                            cv::Scalar(0, 0, 255), cv::FILLED);
                        cv::putText(color_image, label, cv::Point(text_x, text_y + label_size.height),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    }

                    cv::imshow("PC Webcam RGB", color_image);
                }
                stored_color_images.push_back(color_image.clone());

                // Association with hungarian algorithm.
                auto end_iteration = std::chrono::steady_clock::now();
                double time_associate = std::chrono::duration_cast<std::chrono::milliseconds>(end_iteration - now).count();
                if (time_associate < 100.0)
                    total_time += time_associate;
                if (counter % 50 == 0 && counter <= 300)
                    std::cout << "processing time= " << time_associate << " ms" << std::endl;
                    
            }
            catch (const std::exception& e) {
                cerr << "Error during webcam frame capture: " << e.what() << endl;
                q_end.push(true);
                return -1;
            }
            ++counter;
        }
        cap.release();
        cv::destroyAllWindows();

        double processing_speed = counter / (total_time / 1000.0);
        std::cout << "processing speed = " << processing_speed << " Hz" << std::endl;


        // Add current day-hour-min-sec timestamp to directory name as "realsense/dd_hh_mm_ss"
        auto now_time = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now_time);
        std::tm now_tm;
#ifdef _WIN32
        localtime_s(&now_tm, &time_t_now); // Windows
#else
        localtime_r(&time_t_now, &now_tm); // Linux/Unix
#endif
        char time_str[20];
        std::snprintf(time_str, sizeof(time_str), "%02d_%02d_%02d_%02d", 
            now_tm.tm_mday, now_tm.tm_hour, now_tm.tm_min, now_tm.tm_sec);
        std::string rootDir_ = std::string("C:/Users/kawaw/cpp/object_tracking_windows/object_tracking_windows/csv/") + time_str;
        std::filesystem::create_directories(rootDir_);

        // Save time_list in a CSV file
        {
            std::string time_list_path = rootDir_ + "/time_list.csv";
            std::ofstream time_list_file(time_list_path);
            if (time_list_file.is_open()) {
                for (size_t i = 0; i < time_list.size(); ++i) {
                    time_list_file << time_list[i];
                    if (i != time_list.size() - 1)
                        time_list_file << ",";
                }
                time_list_file << std::endl;
                time_list_file.close();
            } else {
                std::cerr << "Unable to open " << time_list_path << " for writing" << std::endl;
            }
        }


		//Save object tracking
        if (!storage_2d.empty()) {
            for (size_t i = 0; i < storage_2d.size(); i++) {
                _tracker_manager.saved_data.push_back(storage_2d[i]);
            }
        }

        if (!_tracker_manager.storage_2d_kf.empty()) {
            for (size_t i = 0; i < _tracker_manager.storage_2d_kf.size(); i++) {
                _tracker_manager.saved_2d_kf.push_back(_tracker_manager.storage_2d_kf[i]);
            }
        }
        // Save _tracker_manager.saved_data (2D rectangles: time, label, left, top, width, height)
        std::string saved_data_2d_path = rootDir_ + "/object_2d.csv";
        std::ofstream saved_data_2d_file(saved_data_2d_path);

        if (saved_data_2d_file.is_open()) {
            for (const auto& obj_seq : _tracker_manager.saved_data) {
                for (size_t j = 0; j < obj_seq.size(); ++j) {
                    const auto& item = obj_seq[j];

                    const double time = std::get<0>(item);
                    const int label = std::get<1>(item);
                    const cv::Rect2d& bbox = std::get<2>(item);

                    saved_data_2d_file
                        << time << ","
                        << label << ","
                        << bbox.x << ","
                        << bbox.y << ","
                        << bbox.width << ","
                        << bbox.height;

                    if (j != obj_seq.size() - 1)
                        saved_data_2d_file << ",";
                }
                saved_data_2d_file << std::endl;
            }
            saved_data_2d_file.close();
        }
        else {
            std::cerr << "Unable to open " << saved_data_2d_path << " for writing" << std::endl;
        }
        //==================

        //Filtered data.
        std::string saved_data_2d_kf_path = rootDir_ + "/object_2d_kf.csv";
        std::ofstream saved_data_2d_kf_file(saved_data_2d_kf_path);

        if (saved_data_2d_kf_file.is_open()) {
            for (const auto& obj_seq : _tracker_manager.saved_2d_kf) {
                for (size_t j = 0; j < obj_seq.size(); ++j) {
                    const auto& item = obj_seq[j];

                    const double time = std::get<0>(item);
                    const int label = std::get<1>(item);
                    const cv::Rect2d& bbox = std::get<2>(item);

                    saved_data_2d_kf_file
                        << time << ","
                        << label << ","
                        << bbox.x << ","
                        << bbox.y << ","
                        << bbox.width << ","
                        << bbox.height;

                    if (j != obj_seq.size() - 1)
                        saved_data_2d_kf_file << ",";
                }
                saved_data_2d_kf_file << std::endl;
            }
            saved_data_2d_kf_file.close();
        }
        else {
            std::cerr << "Unable to open " << saved_data_2d_kf_path << " for writing" << std::endl;
        }
        //=========================================

        // Save images to rootDir + "/images"
        {
            // Save video as mp4 using H.264 codec if available (fallback to MJPG or other codec if needed)
            std::string video_path = rootDir_ + "/output_video.mp4";
            if (!stored_color_images.empty()) {
                int frame_width = stored_color_images[0].cols;
                int frame_height = stored_color_images[0].rows;

                // Try H.264 ('avc1') first for mp4, fallback to MJPG if not available
#ifdef CV_FOURCC
                int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // H.264
#else
                int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
#endif
                cv::VideoWriter video_writer(
                    video_path,
                    fourcc,
                    fps, // set FPS to match acquisition
                    cv::Size(frame_width, frame_height)
                );
                // If H.264 codec not supported, fallback to MJPG (but note that .mp4 might not play with MJPG)
                if (!video_writer.isOpened()) {
                    std::cerr << "Could not open mp4 file with H.264 (avc1). Falling back to MJPG for .mp4." << std::endl;
                    fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                    video_writer.open(
                        video_path,
                        fourcc,
                        fps,
                        cv::Size(frame_width, frame_height)
                    );
                }
                if (!video_writer.isOpened()) {
                    std::cerr << "Unable to open " << video_path << " for writing video" << std::endl;
                } else {
                    for (const auto& img : stored_color_images) {
                        if (img.empty()) continue;
                        // If needed, convert to 8UC3 BGR for VideoWriter
                        cv::Mat out_img;
                        if (img.type() == CV_8UC3) {
                            out_img = img;
                        } else if (img.type() == CV_8UC1) {
                            cv::cvtColor(img, out_img, cv::COLOR_GRAY2BGR);
                        } else {
                            // Try to convert or skip if type is unsupported
                            img.convertTo(out_img, CV_8UC3);
                        }
                        video_writer.write(out_img);
                    }
                    video_writer.release();
                }
            }
        }

        if (!worker_threads.empty()) {//wait for the threads to finish.
            if (q_end.empty())
                q_end.push(true);
            for (std::thread& th : worker_threads)
                th.join();

            //remove the terminal flag.
            while (!q_end.empty())
                q_end.pop();
        }
    }
    catch (const std::exception& e)
    {
        cerr << "Exception: " << e.what() << endl;
        return -1;
    }

    //threadRobot.join();
    return 0;
}
