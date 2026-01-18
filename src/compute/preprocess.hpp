/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

// compute/preprocess.hpp

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <fins/functional_node.hpp>

namespace vision_preprocess {

using namespace cv;

using fins::Input;
using fins::Output;
using fins::Parameter;

inline void check_empty_frame(const Mat &frame) {
  if (frame.empty()) {
    throw std::invalid_argument("input image is empty");
  }
}

inline void check_channels(const Mat &frame, int channel) {
  if (frame.channels() != channel) {
    throw std::invalid_argument(
      std::string("image must have ")
    + std::to_string(channel)
    + std::string(" channels"));
  }
}

inline void check_positive(double value) {
  if (value < 0) {
    throw std::invalid_argument("value must be positive");
  }
}

template<typename T1, typename T2, typename T3>
inline void check_range(T1 value, T2 min, T3 max) {
  if (value < min || value > max) {
    throw std::invalid_argument(
      std::string("value must be in range [")
      + std::to_string(min)
      + std::string(", ")
      + std::to_string(max)
      + std::string("]"));
  }
}

inline void check_odd(int value) {
  if (value % 2 == 0) {
    throw std::invalid_argument("value must be odd");
  }
}

inline void check_even(int value) {
  if (value % 2 != 0) {
    throw std::invalid_argument("value must be even");
  }
}

static auto grey = fins::Function("Grey", 
  [](Input<Mat> &input, Output<Mat> &output){
    check_empty_frame(input);
    if(input->channels() == 3)
      cvtColor(*input, *output, COLOR_BGR2GRAY);
    else 
      input->copyTo(*output);
  })
  .with_description("Converts an image to grayscale")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_category("Vision>Preprocess")
  .build();

static auto hsv = fins::Function("HSV", 
  [](Input<Mat> &input, Output<Mat> &output){
    check_empty_frame(input);
    check_channels(input, 3);
    cvtColor(*input, *output, COLOR_BGR2HSV);
  })
  .with_description("Converts an image to HSV")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_category("Vision>Preprocess")
  .build();

static auto resize = fins::Function("Resize", 
    [](Input<Mat> &input, Output<Mat> &output, Parameter<int> &width, Parameter<int> &height) {
    check_empty_frame(input);
    check_positive(width);
    check_positive(height);
    cv::resize(*input, *output, Size(width, height));
  })
  .with_description("Resize an image")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<int>("width", 540)
  .with_parameter<int>("height", 960)
  .with_category("Vision>Preprocess")
  .build();

static auto rgb_enhance = fins::Function("RGBEnhance", 
  [](Input<Mat> &input, Output<Mat> &output, Parameter<double> &R, Parameter<double> &G, Parameter<double> &B) {
    check_empty_frame(input);
    check_channels(input, 3);
    check_range(R, 0, 10);
    check_range(G, 0, 10);
    check_range(B, 0, 10);

    std::vector<Mat> channels(3);
    split(*input, channels);
    channels[0] = channels[0] * B; // OpenCV uses BGR
    channels[1] = channels[1] * G;
    channels[2] = channels[2] * R;
    merge(channels, *output);
  })
  .with_description("Enhance RGB channels")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<double>("R", 1.0)
  .with_parameter<double>("G", 1.0)
  .with_parameter<double>("B", 1.0)
  .with_category("Vision>Preprocess")
  .build();

static auto contrast = fins::Function("Contrast", 
  [](Input<Mat> &input, Output<Mat> &output, Parameter<double> &contrast) {
    check_empty_frame(input);
    input->convertTo(*output, -1, contrast, 0.0);
  })
  .with_description("Adjust image contrast")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<double>("contrast", 1.0)
  .with_category("Vision>Preprocess")
  .build();

static auto brighten = fins::Function("Brightness", 
  [](Input<Mat> &input, Output<Mat> &output, Parameter<double> &brightness) {
    check_empty_frame(input);
    input->convertTo(*output, -1, 1.0, brightness);
  })
  .with_description("Adjust image brightness")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<double>("brightness", 0.0)
  .with_category("Vision>Preprocess")
  .build();

static auto sharpen = fins::Function("Sharpen",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<double> &sharpen_amount) {
    check_empty_frame(input);
    Mat kernel = (Mat_<float>(3,3) << -1, -1, -1, -1,  9, -1, -1, -1, -1);
    Mat laplacian;
    filter2D(*input, laplacian, CV_32F, kernel);
    Mat sharpened;
    input->convertTo(sharpened, CV_32F);
    Mat result = sharpened + sharpen_amount * laplacian;
    result.convertTo(*output, input->type());
  })
  .with_description("Sharpen the image")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<double>("sharpen_amount", 0.0)
  .with_category("Vision>Preprocess")
  .build();

static auto white_balance = fins::Function("WhiteBalance", 
  [](Input<Mat> &input, Output<Mat> &output) {
    check_empty_frame(input);
    if(input->channels() != 3) {
      input->copyTo(*output);
      return;
    }
    std::vector<Mat> channels;
    split(*input, channels);
    double b_avg = mean(channels[0])[0];
    double g_avg = mean(channels[1])[0];
    double r_avg = mean(channels[2])[0];
    double gray_avg = (b_avg + g_avg + r_avg) / 3.0;
    if(b_avg > 0) channels[0] = channels[0] * (gray_avg / b_avg);
    if(g_avg > 0) channels[1] = channels[1] * (gray_avg / g_avg);
    if(r_avg > 0) channels[2] = channels[2] * (gray_avg / r_avg);
    merge(channels, *output);
  })
  .with_description("Perform automatic white balance")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_category("Vision>Preprocess")
  .build();

static auto gaussian_blur = fins::Function("GaussianBlur",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<int> &kernel_size) {
    check_empty_frame(input);
    int k = kernel_size;
    if(k > 0 && k % 2 == 0) k++; 
    if(k <= 0) k = 1;
    GaussianBlur(*input, *output, Size(k, k), 0, 0);
  })
  .with_description("Apply Gaussian blur")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<int>("kernel_size", 3)
  .with_category("Vision>Preprocess")
  .build();

static auto median_blur = fins::Function("MedianBlur",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<int> &kernel_size) {
    check_empty_frame(input);
    int k = kernel_size;
    if(k > 0 && k % 2 == 0) k++;
    if(k <= 0) k = 1;
    medianBlur(*input, *output, k);
  })
  .with_description("Apply median blur")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<int>("kernel_size", 3)
  .with_category("Vision>Preprocess")
  .build();

static auto bilateral_filter = fins::Function("BilateralFilter",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<int> &diameter) {
    check_empty_frame(input);
    bilateralFilter(*input, *output, diameter, diameter * 2, diameter / 2);
  })
  .with_description("Apply bilateral filter")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<int>("diameter", 5)
  .with_category("Vision>Preprocess")
  .build();

static auto dilate = fins::Function("Dilate",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<int> &kernel_size) {
    check_empty_frame(input);
    int k = kernel_size;
    if(k <= 0) k = 1;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(k, k));
    cv::dilate(*input, *output, kernel);
  })
  .with_description("Dilate the image")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<int>("kernel_size", 3)
  .with_category("Vision>Preprocess")
  .build();

static auto erode = fins::Function("Erode",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<int> &kernel_size) {
    check_empty_frame(input);
    int k = kernel_size;
    if(k <= 0) k = 1;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(k, k));
    cv::erode(*input, *output, kernel);
  })
  .with_description("Erode the image")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<int>("kernel_size", 3)
  .with_category("Vision>Preprocess")
  .build();

static auto morph_open = fins::Function("MorphOpen",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<int> &kernel_size) {
    check_empty_frame(input);
    int k = kernel_size;
    if(k <= 0) k = 1;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(k, k));
    morphologyEx(*input, *output, MORPH_OPEN, kernel);
  })
  .with_description("Perform morphological opening")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<int>("kernel_size", 3)
  .with_category("Vision>Preprocess")
  .build();

static auto morph_close = fins::Function("MorphClose",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<int> &kernel_size) {
    check_empty_frame(input);
    int k = kernel_size;
    if(k <= 0) k = 1;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(k, k));
    morphologyEx(*input, *output, MORPH_CLOSE, kernel);
  })
  .with_description("Perform morphological closing")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<int>("kernel_size", 3)
  .with_category("Vision>Preprocess")
  .build();

static auto canny = fins::Function("Canny",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<double> &threshold1, Parameter<double> &threshold2) {
    check_empty_frame(input.value);
    Mat gray;
    if(input.value.channels() == 3)
      cvtColor(*input, gray, COLOR_BGR2GRAY);
    else
      gray = *input;
    Canny(gray, *output, threshold1, threshold2);
  })
  .with_description("Perform Canny edge detection")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<double>("threshold1", 100.0)
  .with_parameter<double>("threshold2", 200.0)
  .with_category("Vision>Preprocess")
  .build();

static auto contours = fins::Function("Contours",
  [](Input<Mat> &input, Output<Mat> &output) {
    check_empty_frame(input);
    Mat src_gray;
    if(input->channels() == 3)
      cvtColor(*input, src_gray, COLOR_BGR2GRAY);
    else
      src_gray = *input;
         
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    // Canny edge detection usually preceeds findContours or simple threshold
    Mat canny_output;
    Canny(src_gray, canny_output, 100, 200);
    findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
      Scalar color = Scalar(0, 255, 0);
      drawContours(drawing, contours, (int)i, color, 1, LINE_8, hierarchy, 0);
    }
    output = drawing;
  })
  .with_description("Detect and draw contours")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_category("Vision>Preprocess")
  .build();

static auto put_text = fins::Function("PutText",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<std::string> &text) {
    check_empty_frame(input);
    input->copyTo(*output);
    std::vector<std::string> lines;
    std::string current_line;
    std::istringstream stream(text);
    while (std::getline(stream, current_line)) {
      lines.push_back(current_line);
    }
    int y_offset = 40, line_height = 30;
    for (size_t i = 0; i < lines.size(); ++i) {
      putText(*output, lines[i], Point(10, y_offset + i * line_height),
        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
    }
  })
  .with_description("Put text on the image")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<std::string>("text", "Text")
  .with_category("Vision>Preprocess")
  .build();

static auto draw_cross = fins::Function("DrawCross",
  [](Input<Mat> &input, Output<Mat> &output, Parameter<int> &x, Parameter<int> &y) {
    check_empty_frame(input);
    input->copyTo(*output);
    auto color = Scalar(0, 255, 0);
    int thickness = 2, cross_size = 15;
    line(*output, Point(*x - cross_size, *y), Point(*x + cross_size, *y), color, thickness);
    line(*output, Point(*x, *y - cross_size), Point(*x, *y + cross_size), color, thickness);
  })
  .with_description("Draw a cross at a specified point")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<int>("x", 100)
  .with_parameter<int>("y", 100)
  .with_category("Vision>Preprocess")
  .build();

static auto color_threshold = fins::Function("ColorThreshold",
  [](Input<Mat> &in, Output<Mat> &out, 
     Parameter<int> &h_min, Parameter<int> &h_max,
     Parameter<int> &s_min, Parameter<int> &s_max,
     Parameter<int> &v_min, Parameter<int> &v_max) {
    
    if (in->empty()) return;
    Mat hsv;
    cv::cvtColor(*in, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, 
                cv::Scalar(*h_min, *s_min, *v_min), 
                cv::Scalar(*h_max, *s_max, *v_max), 
                *out);
  })
  .with_description("Segment image based on HSV color range")
  .with_inputs_description({"image"})
  .with_outputs_description({"mask"})
  .with_parameter<int>("h_min", 0)
  .with_parameter<int>("h_max", 180)
  .with_parameter<int>("s_min", 0)
  .with_parameter<int>("s_max", 255)
  .with_parameter<int>("v_min", 0)
  .with_parameter<int>("v_max", 255)
  .with_category("Vision>Preprocess")
  .build();

static auto hough_lines_p = fins::Function("HoughLinesP",
  [](Input<Mat> &in, Output<Mat> &out,
     Parameter<double> &rho, Parameter<double> &theta, Parameter<int> &threshold,
     Parameter<double> &minLineLength, Parameter<double> &maxLineGap) {
       
    if (in->empty()) return;
    Mat gray, edges;
    if (in->channels() == 3) {
      cv::cvtColor(*in, gray, cv::COLOR_BGR2GRAY);
      cv::Canny(gray, edges, 50, 150, 3);
    } else {
      edges = *in; 
    }
    
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, *rho, *theta, *threshold, *minLineLength, *maxLineGap);
    
    if (in->channels() == 1) {
      cv::cvtColor(*in, *out, cv::COLOR_GRAY2BGR);
    } else {
      in->copyTo(*out);
    }
    
    for( size_t i = 0; i < lines.size(); i++ ) {
      cv::line(*out, cv::Point(lines[i][0], lines[i][1]),
          cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0,0,255), 2, 8 );
    }
  })
  .with_description("Detect lines using Hough Transform")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<double>("rho", 1.0)
  .with_parameter<double>("theta", CV_PI / 180.0)
  .with_parameter<int>("threshold", 50)
  .with_parameter<double>("minLineLength", 50.0)
  .with_parameter<double>("maxLineGap", 10.0)
  .with_category("Vision>Preprocess")
  .build();

static auto hough_circles = fins::Function("HoughCircles",
  [](Input<Mat> &in, Output<Mat> &out,
     Parameter<double> &dp, Parameter<double> &minDist,
     Parameter<double> &param1, Parameter<double> &param2,
     Parameter<int> &minRadius, Parameter<int> &maxRadius) {
       
    if (in->empty()) return;
    Mat gray;
    if (in->channels() == 3) {
      cv::cvtColor(*in, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = *in;
    }
    cv::medianBlur(gray, gray, 5);
    
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, *dp, *minDist, 
                     *param1, *param2, *minRadius, *maxRadius);
    
    if (in->channels() == 1) {
      cv::cvtColor(*in, *out, cv::COLOR_GRAY2BGR);
    } else {
      in->copyTo(*out);
    }
    
    for( size_t i = 0; i < circles.size(); i++ ) {
      cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      cv::circle( *out, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
      cv::circle( *out, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );
    }
  })
  .with_description("Detect circles using Hough Transform")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<double>("dp", 1.0)
  .with_parameter<double>("minDist", 20.0)
  .with_parameter<double>("param1", 100.0)
  .with_parameter<double>("param2", 30.0)
  .with_parameter<int>("minRadius", 0)
  .with_parameter<int>("maxRadius", 0)
  .with_category("Vision>Preprocess")
  .build();

static auto homography_warp = fins::Function("HomographyWarp",
  [](Input<Mat> &input, Output<Mat> &output,
     Parameter<std::vector<double>> &H_vec, Parameter<int> &out_w, Parameter<int> &out_h) {
    if (input->empty()) return;
    
    Mat H_mat;
    if (H_vec->size() == 9) {
      H_mat = Mat(3, 3, CV_64F);
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          H_mat.at<double>(r, c) = H_vec->at(r * 3 + c);
        }
      }
    } else {
      H_mat = Mat::eye(3, 3, CV_64F);
    }
    
    int target_w = out_w, target_h = out_h;
    if (target_w <= 0 || target_h <= 0) {
      target_w = input->cols;
      target_h = input->rows;
    }
    
    cv::warpPerspective(*input, *output, H_mat, cv::Size(target_w, target_h), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
  })
  .with_description("Apply homography transform")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<std::vector<double>>("H", {1,0,0, 0,1,0, 0,0,1})
  .with_parameter<int>("out_width", 0)
  .with_parameter<int>("out_height", 0)
  .with_category("Vision>Preprocess")
  .build();
  
static auto salt_pepper_noise = fins::Function("SaltPepperNoise", 
[](Input<Mat> &input, Output<Mat> &output, Parameter<double> &salt_prob, Parameter<double> &pepper_prob) {
  const double pa = salt_prob, pb = pepper_prob;
  if(pa <= 0 && pb <= 0) {
    *output = *input;
    return;
  }
  
  int img_size = input->rows * input->cols;
  int amount1 = static_cast<int>(img_size * pa);
  int amount2 = static_cast<int>(img_size * pb);
  
  *output = input->clone();
  for (int counter = 0; counter < amount1; ++counter) {
    output->at<cv::Vec3b>(rand() % output->rows, rand() % output->cols) = cv::Vec3b(255, 255, 255);
  }
  for (int counter = 0; counter < amount2; ++counter) {
    output->at<cv::Vec3b>(rand() % output->rows, rand() % output->cols) = cv::Vec3b(0, 0, 0);
  }
})
  .with_description("Add salt and pepper noise")
  .with_inputs_description({"image"})
  .with_outputs_description({"image"})
  .with_parameter<double>("salt_prob", 0.0)
  .with_parameter<double>("pepper_prob", 0.0)
  .with_category("Vision>Preprocess")
  .build();

}