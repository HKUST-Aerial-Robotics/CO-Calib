# OmniCalib Open

[English](README.md) | 简体中文

[项目主页](https://hkust-aerial-robotics.github.io/CO-Calib/)

OmniCalib Open 面向任意数量相机，自动完成输入识别、标定板检测、Datawash、ROS1 标准标定 bag 生成和 Kalibr 标定。

默认 NN-Detector 使用 Task13 权重，并在原图上进行 `cornerSubPix` 细化（窗口半径5、最多30次迭代、精度阈值0.001）。Datawash 和标定共用细化后的角点，置信度保留网络原值。选帧采用开源 CO-Calib 实现；Kalibr 保留随机处理顺序和内部重启策略，不固定随机种子、不增加外层失败补跑。

用户只需要提供：

- 图像序列、ROS1 bag 或 ROS2 bag
- `rig.yaml`
- `datawash.yaml`
- `target.yaml`
- Detector 和运行设备选择

## 1. 安装

要求：Linux、Conda、Docker。使用 NN-Detector GPU 推理时还需要可用的 NVIDIA 驱动。

```bash
cd Opensource
conda env create -f environment.yml
conda activate omnicalib-open
```

拉取项目提供的 Kalibr 镜像，CO-Calib 默认直接使用该镜像：

```bash
docker pull hkustswarm/co-calib:v1.0
```

普通用户不需要编译 Kalibr、重新打标签或挂载本地后端代码。

## 2. 一键标定

### NN-Detector，使用 GPU

这是推荐运行方式：

```bash
omnicalib run \
  --input /path/to/sequence_or_bag \
  --rig /path/to/rig.yaml \
  --datawash /path/to/datawash.yaml \
  --target /path/to/target.yaml \
  --detector NN-Detector \
  --device gpu
```

如果 CUDA Provider 不可用或初始化失败，程序会自动切换到 CPU 并继续执行，不需要修改命令或重新运行。

也可以直接指定 CPU：

```bash
omnicalib run \
  --input /path/to/sequence_or_bag \
  --rig /path/to/rig.yaml \
  --datawash /path/to/datawash.yaml \
  --target /path/to/target.yaml \
  --detector NN-Detector \
  --device cpu
```

`--device auto` 是默认值，会优先尝试 GPU，然后自动回退 CPU。`NN-Detector` 也是默认 Detector，因此这两个参数都可以省略。

### ACV-Detector

```bash
omnicalib run \
  --input /path/to/sequence_or_bag \
  --rig /path/to/rig.yaml \
  --datawash /path/to/datawash.yaml \
  --target /path/to/target.yaml \
  --detector ACV-Detector \
  --device cpu
```

选择的 Detector 会同时用于 Datawash 和最终标定。ACV-Detector 在 Kalibr 容器中使用 CPU 运行。

默认结果写入输入数据旁的 `<input-name>_omnicalib/`。可使用 `--output /path/to/output` 指定目录；目标目录已经存在时，使用 `--overwrite` 重新生成。

## 3. 输入数据

工具根据 `--input` 自动识别以下三种格式。

### 图像序列

```text
sequence/
├── cam0/
│   ├── images/
│   │   ├── 000000.png
│   │   └── ...
│   └── timestamps.csv
├── cam1/
│   ├── images/
│   └── timestamps.csv
└── camN/
    ├── images/
    └── timestamps.csv
```

每个相机都需要独立的图像目录和 `timestamps.csv`：

```csv
frame_id,timestamp_ns,filename
0,1700000000000000000,000000.png
1,1700000000100000000,000001.png
```

支持 PNG、JPEG、BMP 和 TIFF。不同相机不要求完全同频，工具按照 `rig.yaml` 中的同步容差形成共视组。

### ROS1 bag

将单个 `.bag` 文件传给 `--input`。支持 `sensor_msgs/Image` 和 `sensor_msgs/CompressedImage`。

### ROS2 bag

将包含 `metadata.yaml` 的 rosbag2 目录传给 `--input`。支持 `sensor_msgs/Image` 和 `sensor_msgs/CompressedImage`。

仓库的 `examples/stereo_10frames/` 同时提供图像序列、ROS1 bag 和 ROS2 bag 样例，每个相机包含 10 帧相同数据。

## 4. 配置文件

### rig.yaml

`rig.yaml` 定义相机数量、topic、图像序列目录和相机模型：

```yaml
sync_tolerance_ms: 10.0
cameras:
  - id: cam0
    topic: /camera_0/image_compressed
    model: omni-none
    directory: cam0
    frame_id: camera_0
  - id: cam1
    topic: /camera_1/image_compressed
    model: omni-none
    directory: cam1
    frame_id: camera_1
```

多目系统只需继续增加 `cameras` 项。`id` 和 `topic` 必须唯一，`directory` 必须对应图像序列中的目录。支持：

- `pinhole-radtan`
- `pinhole-equi`
- `pinhole-fov`
- `omni-none`
- `omni-radtan`
- `eucm-none`
- `ds-none`

### datawash.yaml

```yaml
detector_confidence: 0.99
sample_min_detection_points: 12

selection:
  anchor:
    radial_span: 0.25
    iso: 0.50
    budget: 0
  covisible:
    radial_span: 0.00
    iso: 0.30
    budget: 100
  mono_fill:
    radial_span: 0.20
    iso: 0.30
    budget: 0
```

- `detector_confidence`：检测点置信度下限。
- `sample_min_detection_points`：单张图进入选择流程所需的最少有效点数。
- `radial_span`：标定板在有效成像区域中的归一化径向跨度门限。
- `iso`：投影 Jacobian 各向同性门限，越接近 `1` 越严格。
- `budget`：anchor 和 mono fill 按相机限制，covisible 按相机对限制；`0` 表示没有硬上限。

### target.yaml

当前 Detector 面向 6 x 6 AprilGrid：

```yaml
target_type: aprilgrid
tagCols: 6
tagRows: 6
tagSize: 0.055
tagSpacing: 0.3
```

`tagSize` 和 `tagSpacing` 必须与实际打印标定板一致。

仓库提供可直接修改的配置：

- `configs/rig_stereo.yaml`
- `configs/datawash.yaml`
- `configs/target_aprilgrid_6x6.yaml`

## 5. 输出结果

```text
<input-name>_omnicalib/
├── datawash/
│   ├── calibration_clean.bag
│   ├── selected_roles.csv
│   └── summary.json
├── kalibr/
│   ├── calibration-camchain.yaml
│   ├── calibration-results-cam.txt
│   └── calibration-report-cam.pdf
└── summary.json
```

`selected_roles.csv` 记录 anchor、covisible 和 mono fill 的选择结果。`active_cameras` 表示该时刻实际参与标定的相机。

## 6. 外参可视化

`visualization/` 中的浏览器工具用于显示 Kalibr `camchain.yaml` 或 `camchain-imucam.yaml` 中的多相机及相机-IMU 外参。YAML 文件仅在浏览器本地解析，不会上传标定数据。

在 `Opensource/` 仓库根目录创建环境并启动可视化工具：

```bash
cd visualization
conda env create -f environment.yml
conda activate kalibr-visualizer
python -m http.server 8765 --bind 127.0.0.1
```

在浏览器中打开 `http://127.0.0.1:8765/`，然后选择或拖入标定生成的文件：

```text
<input-name>_omnicalib/kalibr/calibration-camchain.yaml
# 或 Kalibr 相机-IMU 标定结果
calibration-camchain-imucam.yaml
```

工具会显示相机视锥、相机坐标轴、相邻相机基线及距离、世界坐标网格和位姿表。如果相机节点包含 `T_cam_imu`，还会显示 IMU 本体、IMU 原生坐标轴、参考相机连线、相对旋转角，以及换算为毫秒的 `timeshift_cam_imu`。工具优先使用 `cam0` 作为 IMU 参考，并支持包含连续 `cam0`、`cam1`、...、`camN` 节点的任意数量相机；`cam0` 之后的每个相机需要提供 4 x 4 `T_cn_cnm1` 变换。

操作方式：

- 左键拖动：旋转视角
- 右键拖动或 Alt + 拖动：平移视角
- 鼠标滚轮：缩放
- `Reset view`：恢复默认视角
- `Fit rig`：将全部相机适配到当前视口

使用 `Ctrl+C` 停止本地服务器。

## 7. 许可证

除非文件或目录另有声明，CO-Calib 自有源代码、文档以及自主开发的
`models/nn_detector_aprilgrid_6x6.onnx` 模型权重，均按照顶层
[`LICENSE`](LICENSE) 中的 BSD-4-Clause 许可证分发。模型的权属与许可范围见
[`models/MODEL_CARD.md`](models/MODEL_CARD.md)。

本仓库在 `vendor/kalibr` 中分发经过修改的 Kalibr。再分发时必须保留原始
Kalibr 版权声明、许可条件、免责声明、广告致谢条款和禁止背书条款。任何提及
本软件功能或用途的广告材料都必须包含以下致谢：

> This product includes software developed by the Autonomous Systems Lab and Skybotix AG.

Kalibr 中的 `incremental_calibration` 和 `incremental_calibration_python`
组件声明使用 LGPLv3，并继续受 LGPLv3 约束。完整分发声明和各组件许可条款见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)、
[`vendor/kalibr/LICENSE`](vendor/kalibr/LICENSE) 和 [`LICENSES/`](LICENSES/)。

## 7. 自动生成镜头黑边 Mask

从包含运动的标定图像序列、ROS1 bag 或 ROS2 bag，为每个相机生成 mask：

```bash
omnicalib mask \
  --input /path/to/sequence_or_bag \
  --rig /path/to/rig.yaml \
  --output /path/to/masks \
  --max-frames 300 \
  --dark-threshold 45 \
  --texture-threshold 5 \
  --persistence 0.95 \
  --feature-rate 0.01 \
  --margin 2
```

工具沿每路图像序列均匀采样。只有在至少 95% 的采样帧中偏暗、局部纹理较弱、与图像边界连通且没有重复 Shi–Tomasi 角点的区域，才会被遮掉。额外扩展 2 像素的边缘，同时保护重复出现的角点。无需 GPU 或新增依赖。

输出包括 `camN_mask.png`（uint8：**255 保留，0 排除**）、各相机叠加预览、时间平均图、暗像素频率图、`overview.jpg` 和 `summary.json`。Mask 与原图分辨率一致。输出目录必须为空；每个相机至少需要 10 帧，推荐使用 300 帧内容有变化的图像。

`--dark-threshold` 是 8 位灰度亮度阈值；`--texture-threshold` 是 7 x 7 邻域灰度标准差上限；`--feature-rate` 表示角点出现达到该比例时保护其邻域。提高亮度/纹理阈值或降低持续比例，会使遮罩更激进。使用前应检查叠加图，尤其是迁移到其他数据时。静止的深色遮挡物可能被误认为镜筒黑边；没有检测到角点不代表区域一定不可用。

此命令负责导出 mask，标定与 Datawash 尚未自动读取它们。特征跟踪应将 PNG 作为检测掩膜使用；图像置黑适合预览，但可能在遮罩边界引入人工边缘。
