/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

// compute/undistort.hpp

#pragma once

#include <fins/node.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <iostream>

class ImageUndistort : public fins::Node {
public:
  void define() override {
    set_name("ImageUndistort");
    set_description("Undistort image using parameters (camera_matrix, dist_coeffs)");
    set_category("Vision>Preprocess");

    register_input<0, cv::Mat>("image", &ImageUndistort::on_image);
    register_output<0, cv::Mat>("image");

    register_parameter<std::vector<double>>("camera_matrix", &ImageUndistort::update_camera_matrix, {1,0,0, 0,1,0, 0,0,1});
    register_parameter<std::vector<double>>("dist_coeffs", &ImageUndistort::update_dist_coeffs, {0,0,0,0,0,0,0,0,0});
  }

  void initialize() override {
    std::lock_guard<std::mutex> lock(mutex_);
    reset_maps();
  }

  void run() override {}

  void pause() override {}
  
  void reset() override {
    std::lock_guard<std::mutex> lock(mutex_);
    reset_maps();
  }

  void update_camera_matrix(const std::vector<double>& vec) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (vec.size() == 9) {
      // Create 3x3 matrix from vector
      camera_matrix_ = cv::Mat(vec, true).reshape(1, 3);
      reset_maps();
      logger->info("ImageUndistort: Loaded camera matrix.");
    } else {
      if (!vec.empty())
        logger->error("ImageUndistort: Invalid camera_matrix size (expected 9): {}", vec.size());

      camera_matrix_.release();
    }
  }

  void update_dist_coeffs(const std::vector<double>& vec) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!vec.empty()) {
      dist_coeffs_ = cv::Mat(vec, true);
      reset_maps();
      logger->info("ImageUndistort: Loaded distortion coefficients.");
    } else {
      dist_coeffs_.release();
    }
  }

  void on_image(const fins::Msg<cv::Mat>& msg) {
    if (!msg || msg->empty()) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Check if parameters are valid. If not, pass through original image.
    if (camera_matrix_.empty() || dist_coeffs_.empty()) {
      send<0>(*msg, msg.event_time);
      return;
    }

    // Init maps if needed or size changed
    if (map1_.empty() || msg->size() != image_size_) {
      image_size_ = msg->size();
      cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat(), 
                                  camera_matrix_, image_size_, CV_16SC2, map1_, map2_);
    }

    cv::Mat undistorted;
    cv::remap(*msg, undistorted, map1_, map2_, cv::INTER_LINEAR);
    send<0>(undistorted, msg.event_time);
  }

private:
  void reset_maps() {
    map1_.release();
    map2_.release();
    image_size_ = cv::Size();
  }

  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Mat map1_, map2_;
  cv::Size image_size_;
  std::mutex mutex_;
};

EXPORT_NODE(ImageUndistort)
