/*******************************************************************************
 * Copyright (c) 2024.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

// compute/quality.hpp

#pragma once

#include <opencv2/opencv.hpp>
#include "func.hpp"
#include "utils/node_list.hpp"
#include "executors/fusion.hpp"

namespace vision {

inline auto& img_quality_assessments_nodes() {
auto psnr = Function("PSNR", 
  [](Input<cv::Mat> &frame_input1, Input<cv::Mat> &frame_input2, Output<double> &psnr) {
    if (frame_input1.value.empty() || frame_input2.value.empty()) return;
    cv::Mat s1;
    cv::absdiff(frame_input1.value, frame_input2.value, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    cv::Scalar s = cv::sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];
    if (sse <= 1e-10) {
      psnr.value = 0;
    } else {
      double mse = sse / (double) (frame_input1.value.channels() * frame_input1.value.total());
      psnr.value = 10.0 * log10((255 * 255) / mse);
    }
  })
  .with_description("Peak Signal-to-Noise Ratio")
  .with_inputs_description({"image1", "image2"})
  .with_outputs_description({"PSNR"});

auto ssim = Function("SSIM", 
  [](Input<cv::Mat> &frame_input1, Input<cv::Mat> &frame_input2, Output<double> &ssim) {
    if (frame_input1.value.empty() || frame_input2.value.empty()) return;
    const double C1 = 6.5025, C2 = 58.5225;
    cv::Mat img1, img2;
    frame_input1.value.convertTo(img1, CV_32F);
    frame_input2.value.convertTo(img2, CV_32F);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;

    cv::GaussianBlur(img1.mul(img1), sigma1_sq, cv::Size(11, 11), 1.5);
    sigma1_sq -= mu1_sq;

    cv::GaussianBlur(img2.mul(img2), sigma2_sq, cv::Size(11, 11), 1.5);
    sigma2_sq -= mu2_sq;

    cv::GaussianBlur(img1.mul(img2), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1 = (2 * mu1_mu2 + C1);
    cv::Mat t2 = (2 * sigma12 + C2);
    cv::Mat t3 = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2);

    cv::Mat ssim_map = (t1.mul(t2)) / t3;
    double s = cv::sum(ssim_map)[0];
    ssim.value = s / (double)(ssim_map.rows * ssim_map.cols);
  })
  .with_description("Structural Similarity Index")
  .with_inputs_description({"image1", "image2"})
  .with_outputs_description({"SSIM"});

auto mse = Function("MSE", 
  [](Input<cv::Mat> &frame_input1, Input<cv::Mat> &frame_input2, Output<double> &mse) {
    if (frame_input1.value.empty() || frame_input2.value.empty()) return;
    cv::Mat s1;
    cv::absdiff(frame_input1.value, frame_input2.value, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    cv::Scalar s = cv::sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];
    mse.value = sse / (double)(frame_input1.value.channels() * frame_input1.value.total());
  })
  .with_description("Mean Squared Error")
  .with_inputs_description({"image1", "image2"})
  .with_outputs_description({"MSE"});

auto mae = Function("MAE", 
  [](Input<cv::Mat> &frame_input1, Input<cv::Mat> &frame_input2, Output<double> &mae) {
    if (frame_input1.value.empty() || frame_input2.value.empty()) return;
    cv::Mat s1;
    cv::absdiff(frame_input1.value, frame_input2.value, s1);
    s1.convertTo(s1, CV_32F);
    cv::Scalar s = cv::sum(s1);
    double sae = s.val[0] + s.val[1] + s.val[2];
    mae.value = sae / (double)(frame_input1.value.channels() * frame_input1.value.total());
  })
  .with_description("Mean Absolute Error")
  .with_inputs_description({"image1", "image2"})
  .with_outputs_description({"MAE"});

auto uqi = Function("UQI", 
  [](Input<cv::Mat> &frame_input1, Input<cv::Mat> &frame_input2, Output<double> &uqi) {
    if (frame_input1.value.empty() || frame_input2.value.empty()) return;
    cv::Mat img1, img2;
    frame_input1.value.convertTo(img1, CV_32F);
    frame_input2.value.convertTo(img2, CV_32F);

    double mean1 = cv::mean(img1)[0];
    double mean2 = cv::mean(img2)[0];

    cv::Mat numerator = (img1.mul(img2) - mean1 * mean2);
    cv::Mat denominator = (img1.mul(img1) + img2.mul(img2) - mean1 * mean1 - mean2 * mean2);

    cv::Mat uqi_map = numerator / denominator;
    double sum_uqi = cv::sum(uqi_map)[0];
    uqi.value = sum_uqi / (double)(img1.rows * img1.cols);
  })
  .with_description("Universal Quality Index")
  .with_inputs_description({"image1", "image2"})
  .with_outputs_description({"UQI"});

auto ncc = Function("NCC", 
  [](Input<cv::Mat> &frame_input1, Input<cv::Mat> &frame_input2, Output<double> &ncc) {
    if (frame_input1.value.empty() || frame_input2.value.empty()) return;
    cv::Mat img1, img2;
    frame_input1.value.convertTo(img1, CV_32F);
    frame_input2.value.convertTo(img2, CV_32F);

    double mean1 = cv::mean(img1)[0];
    double mean2 = cv::mean(img2)[0];

    cv::Mat numerator = (img1 - mean1).mul(img2 - mean2);
    cv::Mat denominator;
    cv::sqrt((img1 - mean1).mul(img1 - mean1) + (img2 - mean2).mul(img2 - mean2), denominator);

    cv::Mat ncc_map = numerator / denominator;
    double sum_ncc = cv::sum(ncc_map)[0];
    ncc.value = sum_ncc / (double)(img1.rows * img1.cols);
  })
  .with_description("Normalized Cross-Correlation")
  .with_inputs_description({"image1", "image2"})
  .with_outputs_description({"NCC"});

auto img_fusion = Executor<Fusion<cv::Mat, cv::Mat>>("ImageFusion")
 .with_description("Fuses two image inputs into a tuple.")
 .with_inputs_description({"image1", "image2"})
 .with_outputs_description({"fused"});
  

static auto nodes = make_node_list(
  psnr, ssim, mse, mae, uqi, ncc, img_fusion
).with_category("Vision>Metrics");
return nodes;
}

}