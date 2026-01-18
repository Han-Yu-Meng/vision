/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

// sinks/rtsp_streamer.hpp

#pragma once

#include <fins/node.hpp>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

class RtspStreamer : public fins::Node {
public:
  void define() override {
    set_name("RtspStreamer");
    set_description("Push video stream to RTSP server using GStreamer");
    set_category("Vision>Streaming");

    register_input<0, cv::Mat>("image", &RtspStreamer::on_image);
    register_parameter<std::string>("rtsp_url", &RtspStreamer::update_url, "rtsp://[username:password@]ip_address[:port]/path");
    register_parameter<int>("fps", &RtspStreamer::update_fps, 30);
    register_parameter<int>("bitrate_kbps", &RtspStreamer::update_bitrate, 2048);
  }

  void initialize() override {
    fps_ = 30;
    bitrate_ = 2048;
    restart_needed_ = false;
  }

  void run() override {}

  void pause() override {
    std::lock_guard<std::mutex> lock(mutex_);
    close_writer();
  }

  void reset() override {
    pause();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      url_.clear();
    }
  }

  void update_url(const std::string &url) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (url_ != url) {
      url_ = url;
      restart_needed_ = true;
    }
  }

  void update_fps(const int &fps) {
    std::lock_guard<std::mutex> lock(mutex_);
    int new_fps = fps > 0 ? fps : 30;
    if (fps_ != new_fps) {
      fps_ = new_fps;
      restart_needed_ = true;
    }
  }

  void update_bitrate(const int &bitrate) {
    std::lock_guard<std::mutex> lock(mutex_);
    int new_bitrate = bitrate > 0 ? bitrate : 2048;
    if (bitrate_ != new_bitrate) {
      bitrate_ = new_bitrate;
      restart_needed_ = true;
    }
  }

  void on_image(const fins::Msg<cv::Mat> &msg) {
    if (!msg || msg->empty())
      return;

    std::lock_guard<std::mutex> lock(mutex_);

    if (url_.empty())
      return;

    cv::Size frame_size = msg->size();

    if (restart_needed_ || !writer_.isOpened() || frame_size != current_size_) {
      close_writer();
      current_size_ = frame_size;
      restart_needed_ = false;

      // Build GStreamer pipeline
      // appsrc -> videoconvert -> x264enc -> rtspclientsink
      std::stringstream ss;
      ss << "appsrc ! videoconvert ! video/x-raw,format=BGR ! "
         << "x264enc tune=zerolatency bitrate=" << bitrate_ << " speed-preset=superfast ! "
         << "rtspclientsink location=" << url_;

      std::string pipeline = ss.str();

      try {
        writer_.open(pipeline, cv::CAP_GSTREAMER, 0, fps_, current_size_, true);
        if (!writer_.isOpened()) {
          logger->error("RtspStreamer: Failed to open GStreamer pipeline: {}", pipeline);
        } else {
          logger->info("RtspStreamer: Stream started to {}", url_);
        }
      } catch (const std::exception &e) {
        logger->error("RtspStreamer: Exception opening GStreamer pipeline: {}", e.what());
      }
    }

    if (writer_.isOpened()) {
      writer_.write(*msg);
    }
  }

private:
  void close_writer() {
    if (writer_.isOpened()) {
      writer_.release();
    }
  }

  cv::VideoWriter writer_;
  std::string url_;
  int fps_ = 30;
  int bitrate_ = 2048;
  cv::Size current_size_;
  bool restart_needed_ = false;
  std::mutex mutex_;
};

EXPORT_NODE(RtspStreamer)