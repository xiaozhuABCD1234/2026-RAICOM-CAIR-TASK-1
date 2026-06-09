# UGOT-UBT

基于优必选 UGOT 机器人平台的竞赛任务程序，实现从语音指令到自主抓取搬运的完整闭环。


## 模块说明

### 入口

| 文件                | 说明                                                                          |
| ------------------- | ----------------------------------------------------------------------------- |
| `task_1.py`         | 集成任务入口：语音 → 巡线 → YOLO 追踪 → 抓取 → AprilTag 导航 → 卸货，往返双程 |
| `run_linefollow.py` | 纯 PID 巡线演示                                                               |

### 检测（三选一）

| 文件             | 说明                            |
| ---------------- | ------------------------------- |
| `detect_yolo.py` | YOLO PyTorch (ultralytics) 检测 |
| `detect_onnx.py` | ONNX Runtime 推理检测           |
| `detect_cv.py`   | 经典 OpenCV 视觉检测            |

### 追踪 & 抓取

| 文件            | 说明                                      |
| --------------- | ----------------------------------------- |
| `chase_yolo.py` | YOLO 驱动追踪 + PID 控制                  |
| `chase_cv.py`   | 经典视觉追踪 + PID 控制                   |
| `grab_yolo.py`  | YOLO 追踪 + 稳定判定 + 舵机抓取（集成版） |

### 导航

| 文件              | 说明                                |
| ----------------- | ----------------------------------- |
| `nav_unload.py`   | AprilTag 追踪 → 巡线进入 A/B 卸货区 |
| `monitor_line.py` | 车道线实时监测工具                  |

### 机器人控制

| 文件                 | 说明                         |
| -------------------- | ---------------------------- |
| `control_servo.py`   | 三关节舵机控制 (ID 51/52/53) |
| `sensor_distance.py` | 红外距离传感器读取           |
| `voice_command.py`   | 语音指令识别与解析           |

### 工具 & 诊断

| 文件                  | 说明                         |
| --------------------- | ---------------------------- |
| `tool_capture.py`     | 相机图像采集 + YOLO 标注预览 |
| `tool_read_servo.py`  | 读取舵机当前角度             |
| `test_peripherals.py` | 外设连通性测试               |

### 公共

| 文件        | 说明                         |
| ----------- | ---------------------------- |
| `utils.py`  | 共享常量、连接辅助、工具函数 |
| `config.py` | TOML 配置加载                |
| `logger.py` | loguru 彩色日志 + 文件轮转   |

## 快速开始

```bash
# 安装依赖
uv sync

# 修改机器人 IP 地址
vi config.toml

# 运行集成任务
python task_1.py
```

## 配置文件

```toml
[network]
robot_ip = "10.165.165.121"

[logging]
console_level = "TRACE"   # TRACE | DEBUG | INFO | SUCCESS | WARNING | ERROR | CRITICAL
```

## 依赖

- Python >= 3.14
- `ugot` — UGOT 机器人 SDK
- `ultralytics` — YOLO 模型
- `opencv-python` — 图像处理
- `onnxruntime` — ONNX 推理（可选）
- `loguru` — 日志
