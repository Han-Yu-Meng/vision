/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

// sinks/display.hpp

#pragma once

#include <atomic>
#include <condition_variable>
#include <fins/node.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

class ImageDisplay : public fins::Node {
public:
  ImageDisplay() = default;

  void define() override {
    set_name("ImageDisplay");
    set_description("Display images using OpenCV's imshow function.");
    set_category("Vision>Display");

    register_input<0, cv::Mat>("image", &ImageDisplay::on_image);
    register_parameter<std::string>("title", &ImageDisplay::update_title, "title");
  }

  void initialize() override { is_running_ = false; }

  void update_title(const std::string &title) {
    std::lock_guard<std::mutex> lock(mutex_);
    window_title_ = title;
  }

  void on_image(const fins::Msg<cv::Mat> &msg) {
    if (!is_running_ || !msg)
      return;

    std::unique_lock<std::mutex> lock(mutex_);
    if (frame_queue_.size() > 2) {
      frame_queue_.pop();
    }
    frame_queue_.push(*msg);
    cond_.notify_one();
  }

  void run() override {
    is_running_ = true;
    display_thread_ = std::thread(&ImageDisplay::display_loop, this);
  }

  void pause() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      is_running_ = false;
    }
    cond_.notify_all();
    if (display_thread_.joinable()) {
      display_thread_.join();
    }
    cv::destroyAllWindows();
  }

  void reset() override {
    pause();
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<cv::Mat> empty;
    std::swap(frame_queue_, empty);
  }

protected:
  virtual void process_frame(const cv::Mat &src, cv::Mat &dst) { dst = src; }

  void display_loop() {
    std::string current_title = window_title_;
    if (current_title.empty())
      current_title = "ImageDisplay";

    bool window_created = false;

    while (is_running_) {
      cv::Mat frame;
      bool has_frame = false;
      std::string next_title;

      {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!window_title_.empty())
          next_title = window_title_;

        cond_.wait(lock, [this] { return !frame_queue_.empty() || !is_running_; });

        if (!is_running_)
          break;

        if (!frame_queue_.empty()) {
          frame = frame_queue_.front();
          frame_queue_.pop();
          has_frame = true;
        }
      }

      if (!next_title.empty() && next_title != current_title) {
        if (window_created) {
          cv::destroyWindow(current_title);
          window_created = false;
        }
        current_title = next_title;
      }

      if (has_frame && !frame.empty()) {
        cv::Mat show_frame;
        process_frame(frame, show_frame);
        if (!show_frame.empty()) {
          if (!window_created) {
            cv::namedWindow(current_title, cv::WINDOW_AUTOSIZE);
            window_created = true;
          }
          cv::imshow(current_title, show_frame);
        }
      }

      if (window_created) {
        cv::waitKey(1);
      }
    }

    if (window_created) {
      cv::destroyWindow(current_title);
    }
  }

  std::string window_title_ = "ImageDisplay";
  std::atomic<bool> is_running_{false};
  std::thread display_thread_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::queue<cv::Mat> frame_queue_;
};

class DepthDisplay : public ImageDisplay {
public:
  void define() override {
    set_name("DepthDisplay");
    set_description("Show depth image with pseudo-color colormap");
    set_category("Vision>Display");

    register_input<0, cv::Mat>("depth", &ImageDisplay::on_image);
    register_parameter<std::string>("title", &ImageDisplay::update_title);
  }

protected:
  void process_frame(const cv::Mat &src, cv::Mat &dst) override {
    if (src.empty())
      return;
    cv::Mat depth_8u;
    src.convertTo(depth_8u, CV_8UC1, 0.03);
    cv::applyColorMap(depth_8u, dst, cv::COLORMAP_JET);
  }
};

EXPORT_NODE(ImageDisplay)
EXPORT_NODE(DepthDisplay)