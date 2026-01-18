# Vision 库

依赖 OpenCV（及可选 Python 环境），提供从相机/文件读取、图像预处理、质量评估以及可视化节点。

## 节点分组

### 1. 图像/视频源

| 节点 | 功能 | 输出 | Extern |
| --- | --- | --- | --- |
| `image_source` | 轮询读取单帧图片。 | `cv::Mat` | `path`、`interval_ms` |
| `video_source` | 按 FPS 或指定间隔读取视频帧。 | `cv::Mat` | `path`、`interval_ms` |
| `camera` | 共享 OpenCV 摄像头（默认 `/dev/video0`）。 | `cv::Mat` | `device` |

### 2. 显示

| 节点 | 功能 | 输入 | Extern |
| --- | --- | --- | --- |
| `image_display` | 独立线程弹窗展示图像。 | `cv::Mat` | `title` |

### 3. 预处理（C++）

`img_pretreat_nodes` 内含多种常用算子：
`Grey`、`HSV`、`Resize`、`RGBEnhance`、`Contrast`、`Brightness`、`Sharpen`、`WhiteBalance`、`GaussianBlur`、`MedianBlur`、`BilateralFilter`、`Dilate`、`Erode`、`MorphOpen`、`MorphClose`、`Canny`、`Contours`、`PutText`、`DrawCross`。

### 4. Python 扩展

`img_pretreat_py_nodes` 通过嵌入式 Python 暴露：

- `PyYOLO`、`PyYOLOSeg`、`PyGrey`、`PyRGBEnhance`

### 5. 质量评估

`img_quality_assessments_nodes` 包含 `PSNR`、`SSIM`、`MSE`、`MAE`、`UQI`、`NCC` 等指标，可对两张图像进行数值评价。

所有节点可按需 `NewStep(...)` 组合：例如 “视频源 → Canny → ImageDisplay” 或 “相机 → Python YOLO → ROS 发布”。
