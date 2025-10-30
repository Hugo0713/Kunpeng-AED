# Kunpeng-AED API 文档

**版本**: v1.0  

---

## 目录

1. [Python模块API](#1-python模块api)
2. [Web API](#2-web-api)
3. [WebSocket事件](#3-websocket事件)
4. [命令行接口](#4-命令行接口)
5. [配置说明](#5-配置说明)
6. [错误处理](#6-错误处理)
7. [部署指南](#7-部署指南)
8. [性能优化](#8-性能优化)

---

## 1. Python模块API

### 1.1 AudioStream (音频流模块)

#### 类定义

```python
class AudioStream:
    """实时音频采集与缓冲管理"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        window_duration: float = 0.96,
        hop_duration: float = 0.48,
        device: Optional[int] = None,
        wav_file: Optional[Path] = None
    ) -> None
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sample_rate` | int | 16000 | 采样率 (Hz) |
| `channels` | int | 1 | 声道数 (1=单声道) |
| `window_duration` | float | 0.96 | 滚动窗口大小 (秒) |
| `hop_duration` | float | 0.48 | 帧间跳步 (秒) |
| `device` | Optional[int] | None | 音频设备索引 (None=默认) |
| `wav_file` | Optional[Path] | None | WAV文件路径 (测试模式) |

---

#### 方法

##### `start() -> None`

启动音频采集线程。

**示例**:
```python
stream = AudioStream(sample_rate=16000)
stream.start()
```

---

##### `stop() -> None`

停止音频采集线程。

**示例**:
```python
stream.stop()
```

---

##### `get_frame(timeout: float = 1.0) -> Optional[np.ndarray]`

从队列中获取音频帧。

**参数**:
- `timeout` (float): 最大等待时间 (秒)

**返回**:
- `np.ndarray`: 音频帧 `[window_size]` 或 `None` (超时)

**示例**:
```python
frame = stream.get_frame(timeout=1.0)
if frame is not None:
    print(f"Frame shape: {frame.shape}")  # (15360,)
```

---

### 1.2 FeatureExtractor (特征提取模块)

#### 类定义

```python
class FeatureExtractor:
    """Mel频谱图特征提取器"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 160,
        n_mels: int = 64,
        fmin: float = 125.0,
        fmax: float = 7500.0,
        num_workers: int = 1
    ) -> None
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sample_rate` | int | 16000 | 音频采样率 |
| `n_fft` | int | 2048 | FFT窗口大小 |
| `hop_length` | int | 160 | STFT帧移 |
| `n_mels` | int | 64 | Mel滤波器数量 |
| `fmin` | float | 125.0 | 最低频率 (Hz) |
| `fmax` | float | 7500.0 | 最高频率 (Hz) |
| `num_workers` | int | 1 | 并行worker数量 |

---

#### 方法

##### `extract(audio: np.ndarray) -> np.ndarray`

提取单个音频帧的Log-Mel频谱图。

**参数**:
- `audio` (np.ndarray): 音频波形 `[samples]`

**返回**:
- `np.ndarray`: Log-Mel频谱图 `[n_mels, time_frames]`

**示例**:
```python
extractor = FeatureExtractor(n_mels=64)
audio = np.random.randn(15360)  # 0.96s @ 16kHz
features = extractor.extract(audio)
print(features.shape)  # (64, 96)
```

---

##### `extract_batch(audio_batch: List[np.ndarray]) -> List[np.ndarray]`

批量提取特征 (并行处理)。

**参数**:
- `audio_batch` (List[np.ndarray]): 音频帧列表

**返回**:
- `List[np.ndarray]`: 特征列表

---

##### `close() -> None`

关闭多进程池。

---

##### `set_normalization_stats(mean: float, std: float) -> None`

更新归一化统计量。

**示例**:
```python
extractor.set_normalization_stats(mean=-15.0, std=10.0)
```

---

### 1.3 YAMNetInference (推理引擎模块)

#### 类定义

```python
class YAMNetInference:
    """TFLite INT8推理引擎"""
    
    def __init__(
        self,
        model_path: Path,
        num_threads: int = 4,
        use_npu: bool = False,
        class_names: Optional[List[str]] = None
    ) -> None
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | Path | - | TFLite模型路径 |
| `num_threads` | int | 4 | CPU线程数 |
| `use_npu` | bool | False | 启用NPU加速 (实验性) |
| `class_names` | Optional[List[str]] | None | 521个AudioSet类别名 |

---

#### 方法

##### `predict(features: np.ndarray) -> Tuple[np.ndarray, float]`

执行推理并返回预测结果。

**参数**:
- `features` (np.ndarray): Log-Mel频谱图 `[n_mels, time_frames]`

**返回**:
- `Tuple[np.ndarray, float]`:
  - `predictions`: 类别概率 `[521]`
  - `latency_ms`: 推理延迟 (毫秒)

**示例**:
```python
engine = YAMNetInference(
    model_path=Path("models/yamnet_int8.tflite"),
    num_threads=4
)
features = np.random.randn(64, 96)
predictions, latency = engine.predict(features)
print(f"Latency: {latency:.2f}ms")
print(f"Top class: {np.argmax(predictions)}")
```

---

##### `get_top_k(predictions: np.ndarray, k: int = 5) -> List[Tuple[str, float]]`

获取Top-K预测类别。

**参数**:
- `predictions` (np.ndarray): 类别概率向量 `[521]`
- `k` (int): 返回前K个类别

**返回**:
- `List[Tuple[str, float]]`: `[(class_name, probability), ...]`

**示例**:
```python
top_k = engine.get_top_k(predictions, k=5)
for i, (class_name, prob) in enumerate(top_k):
    print(f"{i+1}. {class_name}: {prob:.4f}")
```

---

### 1.4 CPUBenchmark (性能测试模块)

#### 类定义

```python
class CPUBenchmark:
    """CPU多线程性能基准测试"""
    
    def __init__(
        self,
        model_path: Path,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> None
```

---

#### 方法

##### `run(thread_counts: List[int]) -> dict`

运行性能基准测试。

**参数**:
- `thread_counts` (List[int]): 要测试的线程数列表

**返回**:
- `dict`: 测试结果字典

**示例**:
```python
benchmark = CPUBenchmark(
    model_path=Path("models/yamnet_int8.tflite"),
    num_iterations=100
)
results = benchmark.run(thread_counts=[1, 2, 4])
print(json.dumps(results, indent=2))
```

**输出示例**:
```json
{
  "1": {
    "mean_latency": 35.42,
    "p95_latency": 37.89,
    "qps": 28.23,
    "mean_cpu": 98.5
  },
  "4": {
    "mean_latency": 12.15,
    "p95_latency": 13.78,
    "qps": 82.30,
    "mean_cpu": 312.4
  }
}
```

---

## 2. Web API

### 2.1 HTTP端点

#### `GET /`

返回主仪表盘页面。

**请求**:
```http
GET / HTTP/1.1
Host: localhost:8080
```

**响应**:
```http
HTTP/1.1 200 OK
Content-Type: text/html

<!DOCTYPE html>
<html>
...
</html>
```

---

#### `GET /static/<filename>`

获取静态资源 (JS/CSS)。

**示例**:
```http
GET /static/app.js HTTP/1.1
Host: localhost:8080
```

---

## 3. WebSocket事件

### 3.1 客户端连接

#### `connect`

客户端建立WebSocket连接时触发。

**服务端响应**:
```javascript
// 自动发送系统状态
{
  "event": "system_status",
  "data": {
    "status": "running",
    "model": "/path/to/yamnet_int8.tflite",
    "threads": 4
  }
}
```

---

### 3.2 推理结果推送

#### `inference_result`

服务端实时推送推理结果。

**消息格式**:
```json
{
  "timestamp": 1705747200.123,
  "frame_id": 42,
  "top_class": "Speech",
  "confidence": 0.87,
  "top_k": [
    {"class": "Speech", "prob": 0.87},
    {"class": "Music", "prob": 0.09},
    {"class": "Inside, small room", "prob": 0.03}
  ],
  "latency_ms": 15.32,
  "cpu_percent": 45.6,
  "threads": 4
}
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | float | Unix时间戳 (秒) |
| `frame_id` | int | 帧序号 |
| `top_class` | str | Top-1类别名 |
| `confidence` | float | Top-1置信度 [0, 1] |
| `top_k` | array | Top-5类别列表 |
| `latency_ms` | float | 推理延迟 (毫秒) |
| `cpu_percent` | float | CPU使用率 (%) |
| `threads` | int | 推理线程数 |

---

### 3.3 客户端JavaScript示例

```javascript
// 连接WebSocket
const socket = io.connect('http://localhost:8080');

// 监听连接事件
socket.on('connect', () => {
    console.log('Connected to server');
});

// 监听推理结果
socket.on('inference_result', (data) => {
    console.log('Top class:', data.top_class);
    console.log('Confidence:', data.confidence);
    console.log('Latency:', data.latency_ms, 'ms');
    
    // 更新UI
    document.getElementById('topClass').textContent = data.top_class;
    document.getElementById('confidence').textContent = 
        (data.confidence * 100).toFixed(1) + '%';
});

// 监听系统状态
socket.on('system_status', (data) => {
    console.log('Model:', data.model);
    console.log('Threads:', data.threads);
});

// 监听断开事件
socket.on('disconnect', () => {
    console.log('Disconnected from server');
});
```

---

## 4. 命令行接口

### 4.1 启动服务器

```bash
python -m app.server [OPTIONS]
```

**选项**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | Path | `models/yamnet_int8.tflite` | TFLite模型路径 |
| `--threads` | int | 4 | 推理线程数 |
| `--device` | int | None | 音频设备索引 (None=默认设备) |
| `--wav` | Path | None | WAV文件路径 (测试模式) |
| `--host` | str | `0.0.0.0` | 服务器监听地址 |
| `--port` | int | 8080 | 服务器端口 |

**示例**:

```bash
# 使用默认麦克风
python -m app.server --model models/yamnet_int8.tflite --threads 4

# 使用WAV文件测试
python -m app.server --wav data/test_audio.wav --threads 2

# 指定音频设备和端口
python -m app.server --device 1 --host 127.0.0.1 --port 5000
```

---

### 4.2 性能测试

```bash
python -m app.bench_cpu [OPTIONS]
```

**选项**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | Path | `models/yamnet_int8.tflite` | TFLite模型路径 |
| `--threads` | int[] | `[1, 2, 4]` | 要测试的线程数列表 |
| `--iterations` | int | 100 | 测试迭代次数 |
| `--warmup` | int | 10 | 预热迭代次数 |

**示例**:

```bash
# 测试1/2/4/8线程性能
python -m app.bench_cpu --threads 1 2 4 8 --iterations 200

# 快速测试
python -m app.bench_cpu --iterations 50 --warmup 5
```

**输出示例**:

```
================================================================================
BENCHMARK SUMMARY
================================================================================
 Threads |  Mean (ms) |   P95 (ms) |      QPS |    CPU %
--------------------------------------------------------------------------------
       1 |      35.42 |      37.89 |    28.23 |    98.50
       2 |      18.76 |      20.15 |    53.30 |   187.20
       4 |      12.15 |      13.78 |    82.30 |   312.40
       8 |      11.89 |      14.02 |    84.10 |   485.60
================================================================================
```

---

### 4.3 特征提取测试

```bash
python -m app.feature [OPTIONS]
```

**示例**:

```bash
# 测试特征提取
python -m app.feature

# 输出
# Feature shape: (64, 96)
# Feature range: [-2.34, 3.12]
```

---

### 4.4 音频流测试

```bash
python -m app.stream [OPTIONS]
```

**示例**:

```bash
# 测试音频流 (生成测试WAV)
python -m app.stream

# 输出
# Created test WAV: /tmp/test_audio.wav
# Microphone opened (device=None)
# Frame 0: shape=(15360,), rms=0.3542
# Frame 1: shape=(15360,), rms=0.3489
```

---

### 4.5 推理引擎测试

```bash
python -m app.infer_tflite [OPTIONS]
```

**示例**:

```bash
# 测试推理引擎
python -m app.infer_tflite

# 输出
# YAMNet loaded: input=[1, 96, 64], output=[1, 521], threads=4
# Inference latency: 15.32 ms
#   1. Speech: 0.4523
#   2. Music: 0.2134
#   3. Inside, small room: 0.1245
```

---

## 5. 配置说明

### 5.1 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | 4核 ARM Cortex-A76 | 8核 ARM Cortex-A76 |
| 内存 | 2 GB | 4 GB |
| 存储 | 1 GB | 2 GB |
| 操作系统 | openEuler 22.03 LTS | openEuler 22.03 LTS SP2 |
| Python | 3.9+ | 3.10+ |

---

### 5.2 依赖库版本

```txt
numpy>=1.23.0
librosa>=0.10.0
sounddevice>=0.4.6
soundfile>=0.12.1
psutil>=5.9.0
flask>=2.3.0
flask-socketio>=5.3.0
gevent>=23.9.0
gevent-websocket>=0.10.1
tflite-runtime>=2.13.0  # 或 tensorflow>=2.13.0
```

安装命令:

```bash
pip install -r requirements.txt
```

---

### 5.3 模型配置

#### YAMNet INT8 模型

- **输入形状**: `[1, 96, 64]` (batch, time, freq)
- **输出形状**: `[1, 521]` (batch, classes)
- **量化方式**: INT8 (post-training quantization)
- **文件大小**: ~1.2 MB
- **推理延迟**: ~12ms (4线程 @ Kunpeng Pro)

#### 自定义模型

如需使用自定义模型,需满足:

1. TFLite格式 (.tflite)
2. 输入: `[batch, time, freq]` Log-Mel spectrogram
3. 输出: `[batch, num_classes]` 类别概率

---

### 5.4 音频配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 16000 Hz | YAMNet标准采样率 |
| 声道数 | 1 (单声道) | 自动转换多声道 |
| 位深度 | 32-bit float | 内部处理精度 |
| 窗口大小 | 0.96s (15360 samples) | 滚动缓冲区 |
| 跳步大小 | 0.48s (7680 samples) | 推理间隔 |

---

### 5.5 特征配置

| 参数 | 值 | 说明 |
|------|-----|------|
| n_fft | 2048 | FFT窗口大小 |
| hop_length | 160 | STFT帧移 (10ms) |
| n_mels | 64 | Mel滤波器数量 |
| fmin | 125 Hz | 最低频率 |
| fmax | 7500 Hz | 最高频率 |
| 归一化 | z-score | (x - mean) / std |

---

## 6. 错误处理

### 6.1 常见错误码

#### AudioStream错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `OSError: [Errno -9996] Invalid input device` | 无效音频设备 | 运行 `python -c "import sounddevice; print(sounddevice.query_devices())"` 查看设备列表 |
| `FileNotFoundError: WAV file not found` | WAV文件路径错误 | 检查 `--wav` 参数路径 |
| `ValueError: Sample rate mismatch` | WAV采样率≠16kHz | 使用 `ffmpeg` 转换: `ffmpeg -i input.wav -ar 16000 output.wav` |

---

#### FeatureExtractor错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `ImportError: librosa not found` | librosa未安装 | `pip install librosa` |
| `ValueError: Audio length < hop_size` | 音频过短 | 使用至少0.5s的音频 |
| `MemoryError` | 多进程内存不足 | 减少 `num_workers` 参数 |

---

#### YAMNetInference错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `ValueError: Model file not found` | 模型路径错误 | 检查 `--model` 参数 |
| `RuntimeError: Failed to allocate tensors` | 内存不足 | 关闭其他进程释放内存 |
| `ValueError: Input shape mismatch` | 特征维度错误 | 确保输入为 `[64, >=96]` |

---

### 6.2 日志级别

设置环境变量控制日志详细程度:

```bash
# DEBUG: 详细调试信息
export LOG_LEVEL=DEBUG
python -m app.server

# INFO: 标准信息 (默认)
export LOG_LEVEL=INFO

# WARNING: 仅警告和错误
export LOG_LEVEL=WARNING

# ERROR: 仅错误
export LOG_LEVEL=ERROR
```

---

### 6.3 异常捕获示例

```python
import logging
from pathlib import Path
from app.infer_tflite import YAMNetInference

logger = logging.getLogger(__name__)

try:
    engine = YAMNetInference(
        model_path=Path("models/yamnet_int8.tflite"),
        num_threads=4
    )
except FileNotFoundError as e:
    logger.error(f"Model not found: {e}")
    # 下载模型或使用默认路径
except RuntimeError as e:
    logger.error(f"TFLite initialization failed: {e}")
    # 检查TFLite版本兼容性
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
```

---

## 7. 部署指南

### 7.1 开发环境部署

```bash
# 1. 克隆代码
git clone https://github.com/your-org/kunpeng-aed.git
cd kunpeng-aed

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载模型
mkdir -p models
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/yamnet_int8.tflite \
     -O models/yamnet_int8.tflite

# 5. 启动服务
python -m app.server --threads 4 --port 8080
```

---

### 7.2 生产环境部署 (systemd)

创建服务文件 `/etc/systemd/system/kunpeng-aed.service`:

```ini
[Unit]
Description=Kunpeng-AED Real-time Audio Event Detection
After=network.target sound.target

[Service]
Type=simple
User=kunpeng
WorkingDirectory=/opt/kunpeng-aed
Environment="PATH=/opt/kunpeng-aed/venv/bin"
ExecStart=/opt/kunpeng-aed/venv/bin/python -m app.server \
    --model /opt/kunpeng-aed/models/yamnet_int8.tflite \
    --threads 4 \
    --host 0.0.0.0 \
    --port 8080
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

启动服务:

```bash
sudo systemctl daemon-reload
sudo systemctl enable kunpeng-aed
sudo systemctl start kunpeng-aed

# 查看状态
sudo systemctl status kunpeng-aed

# 查看日志
sudo journalctl -u kunpeng-aed -f
```

---

### 7.3 Docker部署

`Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY app/ ./app/
COPY web/ ./web/
COPY models/ ./models/

EXPOSE 8080

CMD ["python", "-m", "app.server", "--host", "0.0.0.0", "--port", "8080"]
```

构建和运行:

```bash
# 构建镜像
docker build -t kunpeng-aed:latest .

# 运行容器
docker run -d \
    --name kunpeng-aed \
    --device /dev/snd:/dev/snd \
    -p 8080:8080 \
    kunpeng-aed:latest

# 查看日志
docker logs -f kunpeng-aed
```

---

### 7.4 Nginx反向代理

`/etc/nginx/sites-available/kunpeng-aed`:

```nginx
upstream kunpeng_backend {
    server 127.0.0.1:8080;
}

server {
    listen 80;
    server_name aed.example.com;

    location / {
        proxy_pass http://kunpeng_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /socket.io/ {
        proxy_pass http://kunpeng_backend/socket.io/;
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

启用配置:

```bash
sudo ln -s /etc/nginx/sites-available/kunpeng-aed /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 8. 性能优化

### 8.1 线程数调优

根据CPU核心数选择最优线程数:

| CPU核心数 | 推荐线程数 | 预期延迟 | 预期QPS |
|-----------|-----------|----------|---------|
| 4核 | 2-4 | 15-20ms | 50-60 |
| 8核 | 4-8 | 10-15ms | 70-90 |
| 16核 | 8-12 | 8-12ms | 90-120 |

```bash
# 测试最优线程数
python -m app.bench_cpu --threads 1 2 4 8 --iterations 200
```

---

### 8.2 内存优化

```python
# 减少队列大小
audio_stream = AudioStream(frame_queue_size=5)  # 默认10

# 使用单进程特征提取
feature_extractor = FeatureExtractor(num_workers=1)

# 限制批处理大小
batch_size = 4  # 避免过大
```

---

### 8.3 音频设备优化

```bash
# 查看音频设备延迟
python -c "import sounddevice; print(sounddevice.query_devices())"

# 选择低延迟设备
python -m app.server --device 1  # 使用ALSA设备
```

---

### 8.4 系统调优 (openEuler)

```bash
# 1. 关闭CPU频率调节
sudo cpupower frequency-set -g performance

# 2. 禁用IRQ均衡
sudo systemctl stop irqbalance

# 3. 设置进程优先级
sudo renice -n -10 -p $(pgrep -f "app.server")

# 4. 增加文件描述符限制
ulimit -n 65535
```

---

### 8.5 监控指标

使用 `psutil` 监控系统资源:

```python
import psutil

process = psutil.Process()

# CPU使用率
cpu_percent = process.cpu_percent(interval=1.0)

# 内存使用
memory_info = process.memory_info()
print(f"RSS: {memory_info.rss / 1024**2:.2f} MB")

# 线程数
num_threads = process.num_threads()
```

---

## 附录

### A. AudioSet 521类别列表

完整类别列表见: [AudioSet Ontology](https://research.google.com/audioset/ontology/index.html)

常见类别:

| ID | 类别名 | 描述 |
|----|--------|------|
| 0 | Speech | 人类语音 |
| 1 | Music | 音乐 |
| 137 | Dog | 狗叫声 |
| 310 | Alarm | 警报声 |
| 420 | Gunshot | 枪声 |

---

### B. 参考资料

- [TFLite文档](https://www.tensorflow.org/lite)
- [YAMNet论文](https://arxiv.org/abs/2008.00749)
- [AudioSet数据集](https://research.google.com/audioset/)
- [openEuler官网](https://www.openeuler.org/)
- [OrangePi Kunpeng Pro](http://www.orangepi.org/)

---

### C. 许可证

本项目采用 MIT License。详见 `LICENSE` 文件。

---


**文档版本**: v1.0.0  
