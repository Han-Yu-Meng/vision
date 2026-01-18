/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

// source/camera_stream.hpp

#pragma once

#include <atomic>
#include <chrono>
#include <fins/node.hpp>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

class StreamingCameraSource : public fins::Node {
public:
  void define() override {
    set_name("StreamingCamera");
    set_description("OpenCV Streaming Camera, reads from URL");
    set_category("Vision>Streaming");

    register_output<0, cv::Mat>("image");
    register_parameter<std::string>("url", &StreamingCameraSource::update_url, "https://192.168.1.2:8080/video");
  }

  void initialize() override {
    frame_count_ = 0;
    is_running_ = false;
  }

  void run() override {
    is_running_ = true;
    worker_ = std::thread(&StreamingCameraSource::loop, this);
  }

  void pause() override {
    is_running_ = false;
    if (worker_.joinable()) {
      worker_.join();
    }
    close_camera();
  }

  void reset() override {
    pause();
    frame_count_ = 0;
  }

  void update_url(const std::string &url) {
    std::lock_guard<std::mutex> lock(mutex_);
    url_ = url;
  }

private:
  void loop() {
    while (is_running_) {
      if (!cap_.isOpened()) {
        std::string current_url;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          current_url = url_;
        }

        if (current_url.empty()) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }

        cap_.open(current_url);
        if (!cap_.isOpened()) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          continue;
        }
      }

      cv::Mat frame;
      if (cap_.read(frame)) {
        cv::putText(frame, std::to_string(++frame_count_) + " frame", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        send<0>(frame, fins::now());
      } else {
        close_camera(); // Reconnect on failure
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  }

  void close_camera() {
    if (cap_.isOpened()) {
      cap_.release();
    }
  }

  cv::VideoCapture cap_;
  std::string url_;
  std::mutex mutex_;
  std::thread worker_;
  std::atomic<bool> is_running_{false};
  int frame_count_ = 0;
};

EXPORT_NODE(StreamingCameraSource)