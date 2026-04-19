# Vision Framework

This is a computer vision framework built on the FINS architecture, providing modular nodes for image processing, camera sources, calibration, and visualization.

## Architecture Overview

The framework is organized into several categories:
- **Sources**: Camera input modules
- **Compute**: Image processing and calibration algorithms  
- **Sinks**: Output and visualization modules

## Source Modules

### IntegratedCameraSource (`src/source/camera_integrated.hpp`)
- **Function**: Captures video from integrated cameras using OpenCV
- **Output**: `cv::Mat` image frames
- **Category**: `Vision>Source`

### StreamingCameraSource (`src/source/camera_stream.hpp`)
- **Function**: Captures video streams from network URLs
- **Output**: `cv::Mat` image frames
- **Category**: `Vision>Streaming`

## Compute Modules

### Preprocessing Functions (`src/compute/preprocess.hpp`)
A comprehensive collection of image preprocessing algorithms:

#### Color Space Conversion
- **Grey**: Converts images to grayscale
- **HSV**: Converts BGR images to HSV color space

#### Image Enhancement
- **RGBEnhance**: Adjusts individual RGB channel intensities (0-10 range)
- **Contrast**: Adjusts image contrast using multiplicative scaling
- **Brightness**: Adjusts image brightness using additive offset
- **Sharpen**: Applies sharpening filter with adjustable intensity
- **WhiteBalance**: Performs automatic white balance based on gray world assumption

#### Geometric Operations
- **Resize**: Resizes images to specified dimensions
- **HomographyWarp**: Applies perspective transformation using 3x3 homography matrix

#### Filtering Operations
- **GaussianBlur**: Applies Gaussian smoothing filter
- **MedianBlur**: Applies median filter for noise reduction
- **BilateralFilter**: Edge-preserving smoothing filter

#### Morphological Operations
- **Dilate**: Morphological dilation
- **Erode**: Morphological erosion
- **MorphOpen**: Morphological opening (erosion followed by dilation)
- **MorphClose**: Morphological closing (dilation followed by erosion)

#### Feature Detection
- **Canny**: Canny edge detection with dual thresholds
- **Contours**: Detects and draws image contours
- **HoughLinesP**: Probabilistic Hough line detection
- **HoughCircles**: Hough circle detection with radius constraints

#### Segmentation
- **ColorThreshold**: HSV color space segmentation with configurable ranges

#### Annotation
- **PutText**: Overlays text on images (supports multi-line)
- **DrawCross**: Draws cross markers at specified coordinates

#### Noise Generation
- **SaltPepperNoise**: Adds salt and pepper noise with configurable probabilities

### Calibration Module (`src/compute/calibration.hpp`)
#### AprilTagDetection
- **Function**: Detects AprilTag fiducial markers in images
- **Input**: `cv::Mat` undistorted image
- **Output**: `AprilTagPoints` (corner coordinates)
- **Category**: `Vision>Calibration`

#### PoseEstimation
- **Function**: Estimates 6DOF pose from AprilTag detections
- **Input**: `AprilTagPoints` corner coordinates
- **Output**: `geometry_msgs::msg::TransformStamped`
- **Category**: `Vision>Calibration`

### Image Undistortion (`src/compute/undistort.hpp`)
#### ImageUndistort
- **Function**: Removes lens distortion from images using camera calibration parameters
- **Input**: `cv::Mat` distorted image
- **Output**: `cv::Mat` undistorted image
- **Category**: `Vision>Preprocess`

## Sink Modules

### ImageDisplay (`src/sinks/display.hpp`)
#### ImageDisplay
- **Function**: Displays images using OpenCV's imshow functionality
- **Input**: `cv::Mat` image frames
- **Category**: `Vision>Display`

#### DepthDisplay
- **Function**: Specialized display for depth images with pseudo-color mapping
- **Input**: `cv::Mat` depth image
- **Category**: `Vision>Display`

### RtspStreamer (`src/sinks/rtsp_streamer.hpp`)
- **Function**: Streams video to RTSP servers using GStreamer
- **Input**: `cv::Mat` image frames
- **Category**: `Vision>Streaming`