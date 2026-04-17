/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

// compute/quality.hpp

#pragma once

#include <opencv2/opencv.hpp>
#include <tuple>
#include <fins/functional_node.hpp>

namespace vision_quality {

using namespace cv;
using fins::Input;
using fins::Output;
using std::tuple;

inline void check_empty_frames(const tuple<Mat, Mat> &frames) {
  if (std::get<0>(frames).empty() || std::get<1>(frames).empty()) {
    throw std::invalid_argument("input frames must not be empty");
  }
}

static auto psnr = fins::Function("PSNR", 
  [](Input<tuple<Mat, Mat>> &input, Output<double> &psnr) {
    check_empty_frames(*input);
    const auto& frame1 = std::get<0>(*input);
    const auto& frame2 = std::get<1>(*input);
    
    Mat s1;
    absdiff(frame1, frame2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    Scalar s = sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];
    if (sse <= 1e-10) {
      *psnr = 0;
    } else {
      double mse = sse / (double)(frame1.channels() * frame1.total());
      *psnr = 10.0 * log10((255 * 255) / mse);
    }
  })
  .with_description("Peak Signal-to-Noise Ratio")
  .with_inputs_description({"fused_image"})
  .with_outputs_description({"PSNR"})
  .with_category("Vision>Metrics")
  .build();

static auto ssim = fins::Function("SSIM", 
  [](Input<tuple<Mat, Mat>> &input, Output<double> &ssim) {
    check_empty_frames(*input);
    const auto& frame1 = std::get<0>(*input);
    const auto& frame2 = std::get<1>(*input);

    const double C1 = 6.5025, C2 = 58.5225;
    Mat img1, img2;
    frame1.convertTo(img1, CV_32F);
    frame2.convertTo(img2, CV_32F);

    Mat mu1, mu2;
    GaussianBlur(img1, mu1, Size(11, 11), 1.5);
    GaussianBlur(img2, mu2, Size(11, 11), 1.5);

    Mat mu1_sq = mu1.mul(mu1);
    Mat mu2_sq = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    Mat sigma1_sq, sigma2_sq, sigma12;

    GaussianBlur(img1.mul(img1), sigma1_sq, Size(11, 11), 1.5);
    sigma1_sq -= mu1_sq;

    GaussianBlur(img2.mul(img2), sigma2_sq, Size(11, 11), 1.5);
    sigma2_sq -= mu2_sq;

    GaussianBlur(img1.mul(img2), sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    Mat t1 = (2 * mu1_mu2 + C1);
    Mat t2 = (2 * sigma12 + C2);
    Mat t3 = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2);

    Mat ssim_map = (t1.mul(t2)) / t3;
    double s = sum(ssim_map)[0];
    *ssim = s / (double)(ssim_map.rows * ssim_map.cols);
  })
  .with_description("Structural Similarity Index")
  .with_inputs_description({"fused_image"})
  .with_outputs_description({"SSIM"})
  .with_category("Vision>Metrics")
  .build();

static auto mse = fins::Function("MSE", 
  [](Input<tuple<Mat, Mat>> &input, Output<double> &mse) {
    check_empty_frames(*input);
    const auto& frame1 = std::get<0>(*input);
    const auto& frame2 = std::get<1>(*input);

    Mat s1;
    absdiff(frame1, frame2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    Scalar s = sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];
    *mse = sse / (double)(frame1.channels() * frame1.total());
  })
  .with_description("Mean Squared Error")
  .with_inputs_description({"fused_image"})
  .with_outputs_description({"MSE"})
  .with_category("Vision>Metrics")
  .build();

static auto mae = fins::Function("MAE", 
  [](Input<tuple<Mat, Mat>> &input, Output<double> &mae) {
    check_empty_frames(*input);
    const auto& frame1 = std::get<0>(*input);
    const auto& frame2 = std::get<1>(*input);

    Mat s1;
    absdiff(frame1, frame2, s1);
    s1.convertTo(s1, CV_32F);
    Scalar s = sum(s1);
    double sae = s.val[0] + s.val[1] + s.val[2];
    *mae = sae / (double)(frame1.channels() * frame1.total());
  })
  .with_description("Mean Absolute Error")
  .with_inputs_description({"fused_image"})
  .with_outputs_description({"MAE"})
  .with_category("Vision>Metrics")
  .build();

static auto uqi = fins::Function("UQI", 
  [](Input<tuple<Mat, Mat>> &input, Output<double> &uqi) {
    check_empty_frames(*input);
    const auto& frame1 = std::get<0>(*input);
    const auto& frame2 = std::get<1>(*input);

    Mat img1, img2;
    frame1.convertTo(img1, CV_32F);
    frame2.convertTo(img2, CV_32F);

    double mean1 = mean(img1)[0];
    double mean2 = mean(img2)[0];

    Mat numerator = (img1.mul(img2) - mean1 * mean2);
    Mat denominator = (img1.mul(img1) + img2.mul(img2) - mean1 * mean1 - mean2 * mean2);

    Mat uqi_map = numerator / denominator;
    double sum_uqi = sum(uqi_map)[0];
    *uqi = sum_uqi / (double)(img1.rows * img1.cols);
  })
  .with_description("Universal Quality Index")
  .with_inputs_description({"fused_image"})
  .with_outputs_description({"UQI"})
  .with_category("Vision>Metrics")
  .build();

static auto ncc = fins::Function("NCC", 
  [](Input<tuple<Mat, Mat>> &input, Output<double> &ncc) {
    check_empty_frames(*input);
    const auto& frame1 = std::get<0>(*input);
    const auto& frame2 = std::get<1>(*input);

    Mat img1, img2;
    frame1.convertTo(img1, CV_32F);
    frame2.convertTo(img2, CV_32F);

    double mean1 = mean(img1)[0];
    double mean2 = mean(img2)[0];

    Mat numerator = (img1 - mean1).mul(img2 - mean2);
    Mat denominator;
    sqrt((img1 - mean1).mul(img1 - mean1) + (img2 - mean2).mul(img2 - mean2), denominator);

    Mat ncc_map = numerator / denominator;
    double sum_ncc = sum(ncc_map)[0];
    *ncc = sum_ncc / (double)(img1.rows * img1.cols);
  })
  .with_description("Normalized Cross-Correlation")
  .with_inputs_description({"fused_image"})
  .with_outputs_description({"NCC"})
  .with_category("Vision>Metrics")
  .build();

}
