/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

// source/img_file.hpp

#pragma once

#include <atomic>
#include <chrono>
#include <fins/node.hpp>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

class ImgSource : public fins::Node {
public:
  void define() override {
    set_name("ImgSource");
    set_description("Generates images by reading from a specified file at regular intervals.");
    set_category("Vision>Source");

    register_output<0, cv::Mat>("image");
    register_parameter<std::string>("path", &ImgSource::update_path, "/path/to/image.jpg");
    register_parameter<int>("interval_ms", &ImgSource::update_interval, 1000);
  }

  void initialize() override { is_running_ = false; }

  void run() override {
    is_running_ = true;
    worker_ = std::thread(&ImgSource::loop, this);
  }

  void pause() override {
    is_running_ = false;
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void reset() override {
    pause();
    std::lock_guard<std::mutex> lock(mutex_);
    img_ = cv::Mat();
  }

  void update_path(const std::string &path) {
    std::lock_guard<std::mutex> lock(mutex_);
    path_ = path;
    if (!path_.empty()) {
      cv::Mat temp = cv::imread(path_);
      if (!temp.empty()) {
        img_ = temp;
      }
    }
  }

  void update_interval(const int &interval) {
    std::lock_guard<std::mutex> lock(mutex_);
    interval_ms_ = interval > 0 ? interval : 1000;
  }

private:
  void loop() {
    while (is_running_) {
      cv::Mat current_img;
      int current_interval = 1000;

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!img_.empty()) {
          current_img = img_;
        }
        current_interval = interval_ms_;
      }

      if (!current_img.empty()) {
        send<0>(current_img, fins::now());
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(current_interval));
    }
  }

  std::mutex mutex_;
  cv::Mat img_;
  std::string path_;
  int interval_ms_ = 1000;
  std::thread worker_;
  std::atomic<bool> is_running_{false};
};

EXPORT_NODE(ImgSource)