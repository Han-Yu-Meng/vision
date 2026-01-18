/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include "source/camera_integrated.hpp"
#include "source/camera_stream.hpp"
#include "source/video_file.hpp"
#include "source/img_file.hpp"

#include "compute/preprocess.hpp"
#include "compute/undistort.hpp"

// TODO: to be added later
// #include "compute/calibration.hpp"
// #include "compute/quality.hpp"

#include "sinks/display.hpp"
#include "sinks/rtsp_streamer.hpp"

DEFINE_PLUGIN_ENTRY()