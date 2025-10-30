# 🎵 Kunpeng-AED: 声学事件检测系统

[![Platform](https://img.shields.io/badge/Platform-OrangePi%20Kunpeng%20Pro-orange)](https://github.com/kunpeng-aed)
[![OS](https://img.shields.io/badge/OS-openEuler%2022.03-blue)](https://www.openeuler.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)

基于**鲲鹏多核架构**优化的实时声学事件检测系统，运行于 OrangePi Kunpeng Pro (openEuler 22.03)，采用 YAMNet INT8 轻量化模型实现低延迟音频分类。

---

## ✨ 核心特性

- ✅ **实时音频采集**: 支持USB/I2S麦克风，16kHz单声道流式输入
- ✅ **多核优化推理**: INT8量化模型 + 多线程加速 (1/2/4核可调)
- ✅ **Web实时仪表盘**: Flask + SocketIO 实时显示检测结果与性能指标
- ✅ **性能基准测试**: CPU负载、延迟、QPS多维度评估工具
- ✅ **完整文档**: 设计文档、测试报告、部署指南

---

## 🏗️ 系统架构

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 音频采集模块 │───▶│ 特征提取模块  │───▶│ 推理引擎模块  │───▶│  Web展示层   │
│  (stream)   │    │  (feature)   │    │  (infer)     │    │  (Flask UI)  │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
      ▲                    │                    │                    │
      │                    │                    │                    ▼
 USB麦克风            Mel频谱图           YAMNet INT8        实时图表+指标
                   (librosa多核)         (TFLite多线程)
```

---

## 📦 快速开始

### 1. 环境准备

**硬件要求:**
- OrangePi Kunpeng Pro (4核 AArch64 @ 1.6GHz, 8GB RAM)
- USB麦克风 / I2S麦克风阵列

**软件要求:**
- openEuler 22.03 LTS
- Python 3.10+
- Git

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/Hugo0713/Kunpeng-AED.git
cd kunpeng-aed

# 运行环境配置脚本
bash scripts/setup_oe2203.sh

# 激活虚拟环境
source venv/bin/activate
```

### 3. 下载模型

从 [TensorFlow Hub](https://tfhub.dev/google/yamnet/1) 下载 YAMNet INT8 模型:

```bash
mkdir -p models
# 下载 yamnet_int8.tflite 到 models/ 目录
wget -O models/yamnet_int8.tflite <model-url>
```

### 4. 启动系统

```bash
# 默认配置 (4线程, 8080端口)
bash scripts/run.sh

# 自定义配置
THREADS=2 PORT=5000 bash scripts/run.sh

# WAV文件测试模式
WAV_FILE=test_audio.wav bash scripts/run.sh
```

访问仪表盘: `http://<board-ip>:8080`

---

## 🧪 性能测试

### 基准测试

运行CPU多线程性能对比:

```bash
python -m app.bench_cpu \
    --model models/yamnet_int8.tflite \
    --threads 1 2 4 \
    --iterations 100
```

**预期输出:**

```
================================================================================
BENCHMARK SUMMARY
================================================================================
 Threads |  Mean (ms) |   P95 (ms) |      QPS |    CPU %
--------------------------------------------------------------------------------
       1 |      35.42 |      37.89 |    28.23 |    98.50
       2 |      19.67 |      21.34 |    50.84 |   186.30
       4 |      12.15 |      13.78 |    82.30 |   312.45
================================================================================
```

### 功能测试

使用测试音频验证检测能力:

```bash
# 测试警报声检测
WAV_FILE=test_samples/alarm.wav bash scripts/run.sh

# 测试尖叫声检测
WAV_FILE=test_samples/scream.wav bash scripts/run.sh
```

---

## 📊 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| 音频流 | `app/stream.py` | 实时麦克风采集 + 滚动缓冲 |
| 特征提取 | `app/feature.py` | Mel频谱图计算 (多核并行) |
| 推理引擎 | `app/infer_tflite.py` | TFLite INT8模型推理 |
| 性能测试 | `app/bench_cpu.py` | 多线程基准测试工具 |
| Web服务器 | `app/server.py` | Flask + SocketIO 实时推流 |
| 前端页面 | `web/templates/` | Bootstrap + Chart.js仪表盘 |

---

## 🔧 配置选项

### 命令行参数

```bash
python -m app.server \
    --model models/yamnet_int8.tflite \  # 模型路径
    --threads 4 \                        # 推理线程数
    --device 0 \                         # 音频设备索引
    --wav test.wav \                     # WAV文件(测试模式)
    --host 0.0.0.0 \                     # 服务器地址
    --port 8080                          # 服务器端口
```

### 环境变量

```bash
export MODEL_PATH="models/yamnet_int8.tflite"
export THREADS=4
export PORT=8080
```

---

## 📖 文档目录

- [设计文档](docs/design_doc.md) - 架构设计与模块接口
- [测试报告](docs/test_report.md) - 性能测试结果与分析
- [部署指南](docs/deployment.md) - Systemd服务配置
- [API文档](docs/api.md) - Python模块API说明

---

## 🛠️ 开发指南

### 项目结构

```
kunpeng-aed/
├── app/                    # 核心业务逻辑
│   ├── stream.py          # 音频采集
│   ├── feature.py         # 特征提取
│   ├── infer_tflite.py    # 推理引擎
│   ├── bench_cpu.py       # 性能测试
│   └── server.py          # Web服务器
├── web/                    # 前端资源
│   ├── templates/         # HTML模板
│   └── static/            # JS/CSS静态文件
├── models/                 # 模型文件
├── scripts/                # 部署脚本
│   ├── run.sh             # 启动脚本
│   ├── stop.sh            # 停止脚本
│   └── setup_oe2203.sh    # 环境配置
├── docs/                   # 文档
├── requirements.txt        # Python依赖
└── README.md
```

### 代码规范

- 遵循 PEP8 代码风格
- 使用类型注解 (Type Hints)
- 每个模块提供独立的 `test_*()` 函数

---

## 🐛 故障排查

### 问题1: 找不到音频设备

```bash
# 列出可用音频设备
python -c "import sounddevice; print(sounddevice.query_devices())"

# 指定设备索引
DEVICE=1 bash scripts/run.sh
```

### 问题2: TFLite模型加载失败

确保使用 **AArch64** 版本的 `tflite-runtime`:

```bash
pip install tflite-runtime --extra-index-url https://google-coral.github.io/py-repo/
```

### 问题3: CPU使用率过高

降低线程数或减小音频采样率:

```bash
THREADS=2 bash scripts/run.sh
```

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

- 项目主页: https://github.com/Hugo0713/Kunpeng-AED
- 技术支持: support@kunpeng-aed.org

---

**Built with ❤️ on Kunpeng Platform**
