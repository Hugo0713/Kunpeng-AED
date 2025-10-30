# Kunpeng-AED 系统设计文档

**版本**: v1.0   
**作者**: Kunpeng-AED Team

---

## 1. 项目概述

### 1.1 背景

Kunpeng-AED 是一个面向**鲲鹏多核架构**优化的实时声学事件检测系统，运行于 OrangePi Kunpeng Pro 开发板 (openEuler 22.03 LTS)。系统采用 Google YAMNet INT8 量化模型，实现低延迟、高吞吐的音频分类任务。

### 1.2 设计目标

- **实时性**: 推理延迟 < 20ms (4核优化)
- **准确性**: AudioSet 521类分类，Top-1准确率 > 70%
- **可扩展性**: 模块化设计，支持自定义模型/特征
- **易用性**: Web可视化仪表盘 + 一键启动脚本

---

## 2. 系统架构

### 2.1 整体架构

```
┌───────────────────────────────────────────────────────────┐
│                    Web Browser                            │
│          (Bootstrap + Chart.js + SocketIO)                │
└─────────────────────┬─────────────────────────────────────┘
                      │ HTTP/WebSocket
                      ▼
┌───────────────────────────────────────────────────────────┐
│               Flask + SocketIO Server                     │
│  - HTTP API: /                                            │
│  - WebSocket: /inference_result, /system_status           │
└─────────────────────┬─────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
┌─────────────┐ ┌──────────┐ ┌───────────┐
│AudioStream  │ │ Feature  │ │ YAMNet    │
│   Module    │ │Extractor │ │ Inference │
└─────────────┘ └──────────┘ └───────────┘
      ▲
      │ sounddevice/soundfile
      │
┌─────────────┐
│ Microphone  │
│  or WAV     │
└─────────────┘
```

### 2.2 数据流

```
音频采集 → 滚动缓冲 → Mel特征 → TFLite推理 → WebSocket推送 → 实时展示
(16kHz)   (0.96s窗)  (64×96)   (521类概率)   (JSON)        (图表)
```

---

## 3. 模块设计

### 3.1 音频采集模块 (`app/stream.py`)

**职责**: 实时获取音频数据，维护滚动缓冲区

**核心类**:
```python
class AudioStream:
    def __init__(
        sample_rate: int = 16000,
        window_duration: float = 0.96,
        hop_duration: float = 0.48,
        device: Optional[int] = None,
        wav_file: Optional[Path] = None
    )
    
    def start() -> None
    def stop() -> None
    def get_frame(timeout: float) -> np.ndarray
```

**设计要点**:
- 使用 `sounddevice.InputStream` 实现非阻塞采集
- 滚动缓冲区大小: `window_duration * sample_rate`
- 帧间跳步: `hop_duration * sample_rate`
- 支持 WAV 文件模拟实时流

### 3.2 特征提取模块 (`app/feature.py`)

**职责**: 计算 Log-Mel 频谱图

**核心类**:
```python
class FeatureExtractor:
    def __init__(
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 160,
        n_mels: int = 64,
        num_workers: int = 1
    )
    
    def extract(audio: np.ndarray) -> np.ndarray
    def extract_batch(audio_batch: List[np.ndarray]) -> List[np.ndarray]
```

**参数说明**:
| 参数 | 值 | 说明 |
|------|-----|------|
| `n_fft` | 2048 | FFT窗口大小 (128ms @ 16kHz) |
| `hop_length` | 160 | 帧移 (10ms) |
| `n_mels` | 64 | Mel滤波器数量 |
| `fmin` | 125 Hz | 最低频率 |
| `fmax` | 7500 Hz | 最高频率 |

**输出格式**:
- Shape: `[n_mels, time_frames]` = `[64, 96]`
- 数据类型: `float32`
- 归一化: 均值0, 标准差1

### 3.3 推理引擎模块 (`app/infer_tflite.py`)

**职责**: 运行 TFLite INT8 模型推理

**核心类**:
```python
class YAMNetInference:
    def __init__(
        model_path: Path,
        num_threads: int = 4,
        use_npu: bool = False
    )
    
    def predict(features: np.ndarray) -> Tuple[np.ndarray, float]
    def get_top_k(predictions: np.ndarray, k: int) -> List[Tuple[str, float]]
```

**模型输入**:
- Shape: `[1, 96, 64]` (Batch × Time × Freq)
- 数据类型: `float32`

**模型输出**:
- Shape: `[521]` (AudioSet类别概率)
- 数据类型: `float32`

**多线程优化**:
```python
interpreter = tflite.Interpreter(
    model_path=model_path,
    num_threads=num_threads  # 1/2/4核可调
)
```

### 3.4 Web服务器模块 (`app/server.py`)

**职责**: 提供 HTTP/WebSocket 接口

**API端点**:
| 端点 | 类型 | 功能 |
|------|------|------|
| `/` | HTTP GET | 返回主页面 |
| `/connect` | WebSocket | 客户端连接 |
| `inference_result` | WebSocket Event | 推送推理结果 |
| `system_status` | WebSocket Event | 推送系统状态 |

**推理结果格式**:
```json
{
  "timestamp": 1705747200.123,
  "frame_id": 42,
  "top_class": "Speech",
  "confidence": 0.87,
  "top_k": [
    {"class": "Speech", "prob": 0.87},
    {"class": "Music", "prob": 0.09}
  ],
  "latency_ms": 15.32,
  "cpu_percent": 45.6,
  "threads": 4
}
```

---

## 4. 性能优化策略

### 4.1 CPU多核优化

**TFLite线程配置**:
```python
# 单线程: 延迟~35ms, CPU~100%
# 双线程: 延迟~20ms, CPU~190%
# 四线程: 延迟~12ms, CPU~310%
num_threads = 4  # 推荐值
```

### 4.2 特征提取并行化

使用 Python `multiprocessing.Pool`:
```python
pool = Pool(processes=2)
features = pool.map(extract_fn, audio_batch)
```

### 4.3 INT8量化

- 模型大小: FP32 3.9MB → INT8 1.0MB
- 推理速度提升: ~3.5×
- 精度损失: < 2%

---

## 5. 接口规范

### 5.1 音频流接口

```python
stream = AudioStream(sample_rate=16000)
stream.start()
audio_frame = stream.get_frame(timeout=1.0)  # [15360] float32
stream.stop()
```

### 5.2 特征提取接口

```python
extractor = FeatureExtractor(n_mels=64)
features = extractor.extract(audio_frame)  # [64, 96] float32
```

### 5.3 推理接口

```python
engine = YAMNetInference(model_path, num_threads=4)
predictions, latency_ms = engine.predict(features)  # [521], float
top_k = engine.get_top_k(predictions, k=5)  # [(class, prob), ...]
```

---

## 6. 错误处理

### 6.1 音频设备异常

```python
try:
    stream = AudioStream(device=0)
    stream.start()
except Exception as e:
    logger.error(f"Microphone error: {e}")
    # 降级到 WAV 文件模式
```

### 6.2 推理超时

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Inference timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(1)  # 1秒超时
predictions, _ = engine.predict(features)
signal.alarm(0)
```

---

## 7. 安全性考虑

### 7.1 输入验证

```python
def validate_audio(audio: np.ndarray) -> bool:
    if audio.ndim != 1:
        return False
    if not np.isfinite(audio).all():
        return False
    if np.abs(audio).max() > 1.0:
        audio = np.clip(audio, -1.0, 1.0)
    return True
```

### 7.2 资源限制

```python
MAX_QUEUE_SIZE = 10  # 防止内存溢出
MAX_THREADS = 8      # 防止CPU过载
```

---

## 8. 未来扩展

### 8.1 NPU加速

```python
# CANN delegate (Kunpeng平台)
from tflite_runtime.interpreter import load_delegate

delegate = load_delegate('libcann_delegate.so')
interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[delegate]
)
```

### 8.2 模型切换

支持加载自定义 TFLite 模型:
```bash
python -m app.server --model custom_model.tflite
```

### 8.3 分布式部署

多节点协同检测:
```
边缘节点1 → 本地推理 → 结果上传 → 中心服务器
边缘节点2 → 本地推理 → 结果上传 → 中心服务器
```

---

## 9. 参考资料

- [YAMNet Model Card](https://www.kaggle.com/models/google/yamnet/tensorFlow2/yamnet/1)
- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [AudioSet Ontology](https://research.google.com/audioset/ontology/index.html)
- [OrangePi Kunpeng Pro Specs](http://www.orangepi.org/)

---

**文档版本历史**:
- v1.0 : 初始版本
