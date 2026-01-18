/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

// compute/calibration.hpp

#pragma once

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <vector>
#include "agent/parameter_server.hpp"
#include "executor.hpp"
#include "utils/node_list.hpp"

namespace vision {

  class AprilTagPoints {
  public:
    std::vector<cv::Point2f> points;
  };

  class AprilTagDetection : public ExecutorBase<MMsg<cv::Mat>, MMsg<AprilTagPoints>, MMsg<std::string>> {
  public:
    void initialize() override {
      dict_type_ = "36h11"; // default
      updateDetector();
      logger->info("AprilTagDetection node initialized with dict: " + dict_type_);
    }

    void run() override {}
    void pause() override {}
    void reset() override {}

    template<int Port, typename T>
    void receive(const Msg<T> &msg) {
      if (externs->is_updated) {
        dict_type_ = externs->template get<0>();
        updateDetector();
        externs->update_acknowledge();
        logger->info("Updated AprilTag detector dict to: " + dict_type_);
      }

      if (msg->empty())
        return;

      std::vector<int> markerIds;
      std::vector<std::vector<cv::Point2f>> markerCorners;
      detector_.detectMarkers(*msg, markerCorners, markerIds);

      AprilTagPoints detected_points;
      detected_points.points.clear();
      for (const auto &corners: markerCorners) {
        detected_points.points.insert(detected_points.points.end(), corners.begin(), corners.end());
      }

      publisher->send<0>(detected_points, msg.event_time);
    }

  private:
    void updateDetector() {
      cv::aruco::PredefinedDictionaryType dict_type;
      if (dict_type_ == "36h11") {
        dict_type = cv::aruco::DICT_APRILTAG_36h11;
      } else if (dict_type_ == "25h9") {
        dict_type = cv::aruco::DICT_APRILTAG_25h9;
      } else if (dict_type_ == "16h5") {
        dict_type = cv::aruco::DICT_APRILTAG_16h5;
      } else {
        logger->warn("Unknown dict_type: " + dict_type_ + ", using default 36h11");
        dict_type = cv::aruco::DICT_APRILTAG_36h11;
      }
      detector_ = cv::aruco::ArucoDetector(cv::aruco::getPredefinedDictionary(dict_type));
    }

    cv::aruco::ArucoDetector detector_;
    std::string dict_type_;
  };

  class PoseEstimation
      : public ExecutorBase<MMsg<AprilTagPoints>, MMsg<geometry_msgs::msg::TransformStamped>, MMsg<double>> {
  public:
    void initialize() override {
      tag_size_ = 1.0; // default 1 meter
      std::vector<double> k_vec = PARAMETER_SERVER.get<std::vector<double>>("Calibration.camera_matrix");
      if (k_vec.size() == 9) {
        camera_matrix_ = cv::Mat(3, 3, CV_64F, k_vec.data()).clone();
        logger->info("PoseEstimation node initialized with camera matrix.");
      } else {
        logger->warn("PoseEstimation: Invalid or missing camera_matrix.");
      }
    }

    void run() override {}
    void pause() override {}
    void reset() override {}

    template<int Port, typename T>
    void receive(const Msg<T> &msg) {
      if (externs->is_updated) {
        tag_size_ = externs->template get<0>();
        externs->update_acknowledge();
        logger->info("Updated PoseEstimation tag_size to: " + std::to_string(tag_size_));
      }

      if (msg->points.size() != 4 || camera_matrix_.empty())
        return;

      float half = tag_size_ / 2.0f;
      std::vector<cv::Point3f> object_points = {cv::Point3f(-half, -half, 0), cv::Point3f(half, -half, 0),
                                                cv::Point3f(half, half, 0), cv::Point3f(-half, half, 0)};

      std::vector<cv::Point2f> image_points = msg->points;

      cv::Mat rvec(3, 1, CV_64F), tvec(3, 1, CV_64F);
      cv::solvePnP(object_points, image_points, camera_matrix_, cv::Mat(), rvec, tvec);

      geometry_msgs::msg::TransformStamped transform;
      auto now = std::chrono::system_clock::now();
      auto duration = now.time_since_epoch();
      transform.header.stamp.sec = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
      transform.header.stamp.nanosec =
          std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() % 1000000000;
      transform.header.frame_id = "camera";
      transform.child_frame_id = "tag";

      cv::Mat R;
      cv::Rodrigues(rvec, R);
      tf2::Matrix3x3 tf_rot(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), R.at<double>(1, 0),
                            R.at<double>(1, 1), R.at<double>(1, 2), R.at<double>(2, 0), R.at<double>(2, 1),
                            R.at<double>(2, 2));
      tf2::Quaternion q;
      tf_rot.getRotation(q);

      transform.transform.translation.x = tvec.at<double>(0);
      transform.transform.translation.y = tvec.at<double>(1);
      transform.transform.translation.z = tvec.at<double>(2);
      transform.transform.rotation.x = q.x();
      transform.transform.rotation.y = q.y();
      transform.transform.rotation.z = q.z();
      transform.transform.rotation.w = q.w();

      publisher->send<0>(transform, msg.event_time);
    }

  private:
    cv::Mat camera_matrix_;
    double tag_size_;
  };

  inline auto april_tag_detection =
      Executor<AprilTagDetection>("AprilTagDetection")
          .with_description("Detect AprilTags in undistorted image and output corner points")
          .with_inputs_description({"image"})
          .with_outputs_description({"april_tag_points"})
          .with_externs_description({"dict_type"})
          .with_category("Vision>Calibration");

  inline auto pose_estimation =
      Executor<PoseEstimation>("PoseEstimation")
          .with_description("Estimate pose from AprilTag points using camera matrix from Parameter Server")
          .with_inputs_description({"april_tag_points"})
          .with_outputs_description({"transform"})
          .with_externs_description({"tag_size"})
          .with_category("Vision>Calibration");

  inline auto &calibration_nodes() {
    static auto nodes = make_node_list(april_tag_detection, pose_estimation);
    return nodes;
  }

} // namespace vision