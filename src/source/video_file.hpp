/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

// source/video_file.hpp

#pragma once

#include <atomic>
#include <filesystem>
#include <fins/node.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

class VideoSource : public fins::Node {
public:
  VideoSource() = default;

  void define() override {
    set_name("VideoSource");
    set_description("Reads video frames from file");
    set_category("Vision>Source");

    register_output<0, cv::Mat>("image");
    register_parameter<std::string>("path", &VideoSource::update_path, "/path/to/video.mp4");
  }

  void initialize() override {
    running_ = false;
    std::lock_guard<std::mutex> lock(mutex_);
  }

  void update_path(const std::string &path) {
    std::lock_guard<std::mutex> lock(mutex_);
    path_ = path;
    open_video_internal();
  }

  void run() override {
    running_ = true;
    worker_ = std::thread(&VideoSource::loop, this);
  }

  void pause() override {
    running_ = false;
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void reset() override {
    pause();
    std::lock_guard<std::mutex> lock(mutex_);
    if (video_.isOpened())
      video_.release();
    open_video_internal();
  }

private:
  void open_video_internal() {
    if (path_.empty())
      return;

    if (std::filesystem::exists(path_)) {
      video_.open(path_);
      if (video_.isOpened()) {
        fps_ = video_.get(cv::CAP_PROP_FPS);
        if (fps_ <= 0)
          fps_ = 25.0;
        logger->info("Opened video: {} ({} FPS)", path_, fps_);
      } else {
        logger->error("Failed to open video: {}", path_);
      }
    } else {
      logger->error("File does not exist: {}", path_);
    }
  }

  void loop() {
    while (running_) {
      cv::Mat frame;
      double current_fps = 25.0;
      bool has_frame = false;

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (video_.isOpened()) {
          current_fps = fps_;
          video_ >> frame;
          if (frame.empty()) {
            video_.set(cv::CAP_PROP_POS_FRAMES, 0);
            video_ >> frame;
          }
          if (!frame.empty()) {
            has_frame = true;
          }
        }
      }

      if (has_frame) {
        send<0>(frame, fins::now());
      }

      int sleep_ms = (current_fps > 0) ? static_cast<int>(1000.0 / current_fps) : 40;
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
  }

  std::string path_ = "";
  cv::VideoCapture video_;
  double fps_ = 25.0;
  std::thread worker_;
  std::atomic<bool> running_{false};
  std::mutex mutex_;
};

EXPORT_NODE(VideoSource)