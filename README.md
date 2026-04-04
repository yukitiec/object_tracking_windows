# Windows Object Tracking Pipeline


- **YOLOv10m-based object detection** for designated object classes
- **Multiple object tracking** using **SORT** (Simple Online and Realtime Tracking)
- **Tracked result visualization** with bounding boxes and IDs
- **CSV export** of tracked trajectories
- **Docker / Docker Compose based execution**

This project was implemented as a ROS object tracking take-home assignment and satisfies the core requirements of:

- ROS2-based modular node design
- Dockerized setup and execution
- Configurable runtime
- Basic testing support
- Output visualization and CSV logging

![Demo image](samples/test1_cpu.png)
### [Demo video 1 (Offline tracking with CPU (AMD Ryzen 5 7645HX with Radeon Graphics) : 2 Hz)](samples/test1_cpu.mp4)
### [Demo video 2 (Real-time tracking with GPU (NVIDIA GeForce RTX 4050 Laptop GPU) : 10 Hz))](samples/test2_gpu.mp4)
---

## 1. Overview

The system consists of two nodes:
- ros topic publish/subscribe is impelemented using std::queue<>

1. **YOLO node**
   - Loads a TorchScript YOLOv10m model
   - Performs object detection
   - Publishes detection results with queue

2. **Tracker node**
   - Send video images to YOLO node every time YOLO detections are subscribed.
   - Subscribes to detections
   - Associates detections across frames using **SORT**
   - Draws tracking results
   - Saves tracked data to CSV
   - Optionally saves an output video

The current implementation supports two input modes:

- **ROS image topic / webcam mode**
- **Video-file mode** using a configured input video path

Because webcam access in Docker Desktop + WSL2 can be limited depending on host configuration, video-file mode is also supported for reproducible testing.

---

## 2. Technologies and environment

### Windows 
- **Visual Studio 2022**

### Language
- **C++17**

### Main libraries
- OpenCV
- LibTorch (TorchScript runtime)

---

## 3. Repository structure

```text
object_tracking_windows/
├── src/
├── include/
├── config/
├── make_torchscript/
│   └── makeTorchScript.ipynb
├── analysis/
|── main.cpp
└── README.md
```

## 4. System Architecture
### Nodes
- yolo_node
	Subscribes to /camera/image_raw
		Loads a TorchScript YOLOv10m model
		Performs object detection for designated object classes
	Publishes detection results to /yolo/detections 

- tracker_node
	Subscribes to /yolo/detections
		Uses ROS image input or a configured video file
		Associates detections across frames using SORT
		Displays tracking results
		Saves tracked trajectories to CSV
		Optionally saves output video
	Publishes video images to /camera/image_raw 
- Topics
	Input image topic
		/camera/image_raw
		Type: sensor_msgs/msg/Image
	Detection topic
		/yolo/detections
		Type: custom message tracker_pkg/msg/Detection2DArray

## 5. Tracking Approach
- Detector: YOLOv10m exported as TorchScript
- Tracker: SORT-based association
	Kalman filter prediction
	distance and size-based association
	Hungarian algorithm matching
- Output
	Bounding box visualization with object IDs
	CSV export of tracked objects
	Optionally save video

## 6. Runtime Configuration
- Main config file
```sh
%/config/default.txt

#contents
display true #display the tracking results in realtime.
time_capture 70
video_path none

yolo_path /config/yolov10m_w640_h480_cpu.torchscript
yoloWidth 640
yoloHeight 480
IoU_threshold 0.7 
conf_threshold 0.3 

object_index 73,76
```
- parameters:
	- display: show tracking results
	- time_capture: maximum processing duration
	- video_path: video file path or none (webcamera)
	- yolo_path: TorchScript model path
	- yoloWidth, yoloHeight: detector input size
	- IoU_threshold: Duplicated objects' IoU threshold (Not used in YOLOv10n) [0,1]
	- conf_threshold: detection confidence threshold [0,1]
	- object_index: target class indices


## 7. TorchScript Model Notes
- The runtime requires a TorchScript YOLO model.
- If Docker uses CPU-only PyTorch / LibTorch, use a CPU-compatible TorchScript model.
	- Example:
	```txt
	yolo_path /config/yolov10m_w640_h480_cpu.torchscript
	```
	- Recommended workflow
	- Trace pretrained .pt to torchscripts in: make_torchscript/makeTorchScript.ipynb
	- Place the generated deployment model in: ros_ws/src/tracker_pkg/config/


## 8. Webcam (Video file-based tracking is not still implemented)
- Webcam mode (not tested in the linux version. [tested with windows](https://github.com/yukitiec/object_tracking_windows.git))
	- Set none in video_path:
	```txt
	video_path none
	```
	- and use images from:
	```text
	/camera/image_raw
	```

## 9. Output Files
- Saved under:
```text
video_path/<timestamp>/
```
- Typical outputs:
	- time_list.csv : image time list (1,N_frames)
	- object_2d.csv : (N_step, 6(time, label, left,top,width,height))
	- object_2d_kf.csv : Kalman filtered data (N_step, 6(time, label, left,top,width,height))
	- output_video.mp4 : captured image

## 10. Design Decisions
- Why YOLOv10m + SORT
	- Strong detector + lightweight tracker
	- Practical for realtime multi-object tracking
	
- Why TorchScript
	- Enables C++ deployment
	- Simplifies runtime inference in Docker

## 11. Future Improvements
- 3D positioning.
- system robust to occlusion.

