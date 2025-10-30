# Kunpeng-AED 部署指南

**目标平台**: OrangePi Kunpeng Pro  
**操作系统**: openEuler 22.03 LTS  
**部署方式**: Systemd服务 + 开机自启

---

## 1. 环境准备

### 1.1 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | 2核 AArch64 | 4核 @ 1.6GHz |
| 内存 | 2GB | 8GB |
| 存储 | 2GB 可用空间 | 10GB SSD |
| 操作系统 | openEuler 22.03 | openEuler 22.03 SP3 |
| Python | 3.10+ | 3.10.8 |

---

### 1.2 依赖安装

#### 方法1: 一键安装脚本 (推荐)

```bash
cd /path/to/kunpeng-aed
bash scripts/setup_oe2203.sh
```

#### 方法2: 手动安装

```bash
# 更新系统
sudo dnf update -y

# 安装系统依赖
sudo dnf install -y \
    python3 python3-pip python3-devel \
    gcc gcc-c++ make cmake \
    portaudio-devel libsndfile-devel \
    alsa-lib-devel git

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装Python包
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### 1.3 模型下载

#### 从Kaggle下载YAMNet模型

```bash
# 安装Kaggle CLI
pip install kaggle

# 配置API密钥 (从 https://www.kaggle.com/settings 获取)
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<EOF
{"username":"your_username","key":"your_api_key"}
EOF
chmod 600 ~/.kaggle/kaggle.json

# 下载模型
mkdir -p models
cd models
kaggle models instances versions download google/yamnet/tensorFlow2/yamnet/1

# 解压并转换为INT8 TFLite (如需要)
# 假设下载的是SavedModel格式
unzip yamnet.zip
python ../scripts/convert_to_int8.py \
    --input yamnet/ \
    --output yamnet_int8.tflite
cd ..
```

**注意**: 如果Kaggle提供的是TFLite格式，直接使用；否则需要转换。

---

## 2. 快速部署

### 2.1 开发模式部署

适用于开发测试，支持实时日志查看。

```bash
# 克隆项目
git clone https://github.com/Hugo0713/Kunpeng-AED.git
cd kunpeng-aed

# 运行环境配置
bash scripts/setup_oe2203.sh
source venv/bin/activate

# 启动服务 (前台运行)
bash scripts/run.sh

# 访问仪表盘
# 浏览器打开: http://<board-ip>:8080
```

---

### 2.2 生产模式部署 (Systemd)

适用于生产环境，支持开机自启和进程守护。

#### 步骤1: 创建系统用户

```bash
# 创建专用用户 (可选)
sudo useradd -r -m -s /bin/bash kunpeng
sudo usermod -aG audio kunpeng  # 添加音频权限
```

#### 步骤2: 部署项目文件

```bash
# 复制项目到目标目录
sudo cp -r /path/to/kunpeng-aed /home/kunpeng/
sudo chown -R kunpeng:kunpeng /home/kunpeng/kunpeng-aed
```

#### 步骤3: 配置Systemd服务

```bash
# 复制服务文件
sudo cp scripts/aed.service /etc/systemd/system/kunpeng-aed.service

# 编辑服务文件 (根据实际路径调整)
sudo nano /etc/systemd/system/kunpeng-aed.service
```

**服务文件内容** (`/etc/systemd/system/kunpeng-aed.service`):

```ini
[Unit]
Description=Kunpeng-AED Acoustic Event Detection Service
After=network.target sound.target
Wants=sound.target

[Service]
Type=simple
User=kunpeng
Group=kunpeng
WorkingDirectory=/home/kunpeng/kunpeng-aed

# 环境变量
Environment="PATH=/home/kunpeng/kunpeng-aed/venv/bin:/usr/local/bin:/usr/bin"
Environment="MODEL_PATH=/home/kunpeng/kunpeng-aed/models/yamnet_int8.tflite"
Environment="THREADS=4"
Environment="PORT=8080"
Environment="DEVICE="

# 启动命令
ExecStart=/home/kunpeng/kunpeng-aed/venv/bin/python -m app.server \
    --model ${MODEL_PATH} \
    --threads ${THREADS} \
    --port ${PORT}

# 进程管理
Restart=on-failure
RestartSec=10
KillMode=mixed
TimeoutStopSec=30

# 日志配置
StandardOutput=journal
StandardError=journal
SyslogIdentifier=kunpeng-aed

# 资源限制
LimitNOFILE=65536
MemoryLimit=1G

[Install]
WantedBy=multi-user.target
```

#### 步骤4: 启动服务

```bash
# 重载Systemd配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start kunpeng-aed

# 查看状态
sudo systemctl status kunpeng-aed

# 查看日志
sudo journalctl -u kunpeng-aed -f

# 设置开机自启
sudo systemctl enable kunpeng-aed
```

#### 步骤5: 验证部署

```bash
# 检查服务状态
systemctl is-active kunpeng-aed  # 应返回 active

# 检查端口监听
sudo netstat -tulnp | grep 8080  # 应显示python进程

# 访问Web界面
curl http://localhost:8080
```

---

## 3. 配置管理

### 3.1 环境变量配置

编辑 `/etc/systemd/system/kunpeng-aed.service`:

```ini
[Service]
Environment="THREADS=2"           # 推理线程数 (1/2/4)
Environment="PORT=5000"           # HTTP端口
Environment="MODEL_PATH=/path"    # 模型路径
Environment="DEVICE=0"            # 音频设备索引
Environment="WAV_FILE=/path"      # WAV文件路径 (测试模式)
```

修改后重启服务:
```bash
sudo systemctl daemon-reload
sudo systemctl restart kunpeng-aed
```

---

### 3.2 日志配置

#### 查看实时日志

```bash
# 实时日志 (Ctrl+C退出)
sudo journalctl -u kunpeng-aed -f

# 最近100行日志
sudo journalctl -u kunpeng-aed -n 100

# 今天的日志
sudo journalctl -u kunpeng-aed --since today

# 带时间戳的日志
sudo journalctl -u kunpeng-aed -o short-iso
```

#### 日志持久化

```bash
# 启用持久化日志
sudo mkdir -p /var/log/journal
sudo systemctl restart systemd-journald

# 设置日志大小限制 (编辑 /etc/systemd/journald.conf)
SystemMaxUse=500M
SystemKeepFree=1G
```

---

### 3.3 防火墙配置

如果需要外部访问Web界面:

```bash
# 开放8080端口
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# 或关闭防火墙 (不推荐生产环境)
sudo systemctl stop firewalld
sudo systemctl disable firewalld
```

---

## 4. 音频设备配置

### 4.1 列出可用设备

```bash
# 使用Python脚本
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# 输出示例:
#   0 USB Audio Device, ALSA (2 in, 0 out)
#   1 Built-in Audio, ALSA (2 in, 2 out)
```

### 4.2 设置默认设备

```bash
# 方法1: 通过环境变量
export DEVICE=0

# 方法2: 修改Systemd服务
# Environment="DEVICE=0"
```

### 4.3 I2S麦克风配置 (ReSpeaker等)

```bash
# 加载I2S驱动
sudo modprobe snd_soc_wm8960

# 验证设备
arecord -l
# 应显示 card 0: seeed2micvoicec [seeed-2mic-voicecard]

# 设置采样率
sudo nano /etc/asound.conf
```

---

## 5. 性能调优

### 5.1 CPU亲和性设置

绑定进程到特定CPU核心:

```bash
# 在服务文件中添加
[Service]
CPUAffinity=0 1 2 3  # 使用核心0-3
```

### 5.2 实时优先级

提升进程优先级 (需要root权限):

```ini
[Service]
Nice=-10
IOSchedulingClass=realtime
IOSchedulingPriority=0
```

### 5.3 内存锁定

防止内存交换:

```ini
[Service]
LimitMEMLOCK=infinity
```

---

## 6. 监控与维护

### 6.1 健康检查脚本

创建 `/usr/local/bin/aed_healthcheck.sh`:

```bash
#!/bin/bash
# Kunpeng-AED健康检查

STATUS=$(systemctl is-active kunpeng-aed)
if [ "$STATUS" != "active" ]; then
    echo "Service down, restarting..."
    systemctl restart kunpeng-aed
    exit 1
fi

# 检查HTTP端点
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080)
if [ "$HTTP_CODE" != "200" ]; then
    echo "HTTP check failed: $HTTP_CODE"
    exit 1
fi

echo "Service healthy"
exit 0
```

### 6.2 Cron定时检查

```bash
# 每5分钟检查一次
sudo crontab -e
# 添加:
*/5 * * * * /usr/local/bin/aed_healthcheck.sh >> /var/log/aed_health.log 2>&1
```

---

### 6.3 资源监控

使用Prometheus + Grafana (可选):

```bash
# 安装node_exporter
sudo dnf install -y golang-github-prometheus-node-exporter
sudo systemctl start node_exporter
sudo systemctl enable node_exporter

# Grafana Dashboard模板: #12486
```

---

## 7. 故障排查

### 7.1 服务无法启动

```bash
# 查看详细错误日志
sudo journalctl -u kunpeng-aed -xe

# 常见问题:
# 1. 模型文件不存在
ls -lh /home/kunpeng/kunpeng-aed/models/yamnet_int8.tflite

# 2. Python虚拟环境损坏
cd /home/kunpeng/kunpeng-aed
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. 端口被占用
sudo netstat -tulnp | grep 8080
sudo kill <PID>
```

---

### 7.2 音频设备错误

```bash
# 检查设备权限
ls -l /dev/snd/*
# 应显示 crw-rw----+ 1 root audio

# 将用户添加到audio组
sudo usermod -aG audio kunpeng

# 重新登录使组权限生效
```

---

### 7.3 推理性能下降

```bash
# 检查CPU频率
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# 设置性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 检查CPU温度
cat /sys/class/thermal/thermal_zone0/temp
```

---

## 8. 升级与回滚

### 8.1 版本升级

```bash
cd /home/kunpeng/kunpeng-aed

# 备份当前版本
sudo tar czf /tmp/kunpeng-aed-backup.tar.gz .

# 拉取最新代码
git pull origin main

# 更新依赖
source venv/bin/activate
pip install -r requirements.txt --upgrade

# 重启服务
sudo systemctl restart kunpeng-aed
```

---

### 8.2 版本回滚

```bash
# 停止服务
sudo systemctl stop kunpeng-aed

# 恢复备份
cd /home/kunpeng
sudo rm -rf kunpeng-aed
sudo tar xzf /tmp/kunpeng-aed-backup.tar.gz -C kunpeng-aed/

# 重启服务
sudo systemctl start kunpeng-aed
```

---

## 9. 安全加固

### 9.1 限制网络访问

```bash
# 仅允许本地访问
# 修改 scripts/run.sh 或服务文件
--host 127.0.0.1  # 替代 0.0.0.0

# 使用Nginx反向代理 (推荐)
sudo dnf install -y nginx
```

### 9.2 HTTPS配置

Nginx配置示例 (`/etc/nginx/conf.d/kunpeng-aed.conf`):

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 10. 卸载

```bash
# 停止并禁用服务
sudo systemctl stop kunpeng-aed
sudo systemctl disable kunpeng-aed
sudo rm /etc/systemd/system/kunpeng-aed.service
sudo systemctl daemon-reload

# 删除项目文件
sudo rm -rf /home/kunpeng/kunpeng-aed

# 删除用户 (可选)
sudo userdel -r kunpeng
```

---

## 附录A: 系统服务命令速查

| 命令 | 说明 |
|------|------|
| `systemctl start kunpeng-aed` | 启动服务 |
| `systemctl stop kunpeng-aed` | 停止服务 |
| `systemctl restart kunpeng-aed` | 重启服务 |
| `systemctl status kunpeng-aed` | 查看状态 |
| `systemctl enable kunpeng-aed` | 开机自启 |
| `systemctl disable kunpeng-aed` | 取消自启 |
| `journalctl -u kunpeng-aed -f` | 实时日志 |

---

## 附录B: 常用脚本路径

| 脚本 | 路径 | 用途 |
|------|------|------|
| 环境配置 | `scripts/setup_oe2203.sh` | 安装依赖 |
| 启动服务 | `scripts/run.sh` | 前台运行 |
| 停止服务 | `scripts/stop.sh` | 停止进程 |
| Systemd服务 | `scripts/aed.service` | 系统服务 |

---

**部署支持**: support@kunpeng-aed.org  
**文档版本**: v1.0 
