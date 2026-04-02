#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
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
#include <vector>
#include <array>
#include <ctime>
#include <direct.h>
#include <sys/stat.h>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <cmath>

//For realSense
#include <sstream>
#include <iomanip>

#include <librealsense2/rs.hpp> // RealSense SDK


//#include <ur_rtde/rtde_control_interface.h>
//#include <ur_rtde/rtde_io_interface.h>
//#include <ur_rtde/rtde_receive_interface.h>
