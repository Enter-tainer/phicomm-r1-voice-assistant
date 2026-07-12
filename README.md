# 斐讯 R1 语音助手

把斐讯 R1 蓝牙音箱变成一个可编程的语音助手。

## 这是什么

斐讯 R1 是一个带 4-mic 阵列的蓝牙音箱，内置 Android 5.1 系统。本项目通过以下步骤将其改造为自定义语音助手：

1. **连接 ADB** — R1 出厂自带 ADB 调试，无需拆机或获取 root
2. **Android App** — 在 R1 上运行的语音客户端（唤醒词检测 + 录音 + 播放）
3. **Voice Server** — 在局域网服务器上运行的 WebSocket 后端（ASR + LLM + TTS）

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                    R1 Device (Android 5.1)                │
│                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │ WakeWord    │───▶│ AudioRecorder │───▶│ WsClient    │─┼── WebSocket
│  │ Detector    │    │ (16kHz PCM)  │    │ (binary)    │ │
│  │ (ONNX,      │    └──────────────┘    └─────────────┘ │
│  │  on-device) │                           ▲            │
│  └─────────────┘                           │            │
│         ▲                                  │            │
│         │                          ┌───────┴───────┐    │
│  ┌──────┴──────┐                   │ AudioPlayer    │    │
│  │ LedController│◀─────────────────│ (48kHz PCM)   │    │
│  └─────────────┘                   └───────────────┘    │
└──────────────────────────────────────────────────────────┘
                                    │
                    WebSocket (ws://server:8090)
                                    │
┌───────────────────────────────────┼──────────────────────┐
│                    Voice Server (Linux x86_64)            │
│                                   │                      │
│  ┌────────┐  ┌──────────┐  ┌──────┴──────┐  ┌─────────┐ │
│  │ VAD    │  │ ASR      │  │ WebSocket   │  │ TTS     │ │
│  │(Silero)│  │(Qwen3-ASR)│  │ Server      │  │(Edge-TTS│ │
│  │        │  │          │  │ (state      │  │ streaming│ │
│  │speech  │  │transcribe│  │  machine)   │  │ per     │ │
│  │detect  │  │          │  │             │  │ sentence)│ │
│  └────────┘  └────┬─────┘  └─────────────┘  └─────────┘ │
│                     │                     ▲             │
│               ┌─────┴─────┐               │             │
│               │ Hermes API│───────────────┘             │
│               │(GLM-5.2 / │  (OpenAI-compatible)        │
│               │ any LLM)  │                             │
│               └───────────┘                             │
└──────────────────────────────────────────────────────────┘
```

## 语音交互流程

```
用户: "Hey Jarvis"
  ↓
R1: 播放 "滴滴" 提示音 (350ms)
  ↓ 等待 500ms (让 beep 播完)
R1: 开始录音 → PCM 流发送到服务器
  ↓
Server: VAD 检测语音开始/结束
  ↓ 语音结束后
Server: ASR 识别 → "上海天气适合跑步吗？"
Server: 播放 "正在思考" 提示音
Server: 调用 Hermes API → 获取回复
Server: 按句分割 → 逐句 Edge-TTS → 流式发送 PCM
  ↓
R1: 边收边播 → 语音回复
Server: 播放 "好了" 提示音
  ↓
R1: 回到唤醒词检测状态
```

## 项目结构

```
├── server/                    # Python WebSocket 语音服务器
│   ├── server.py              #   WebSocket 服务器 + 状态机
│   ├── pipeline.py            #   ASR → LLM → TTS 流水线
│   ├── vad_silero.py          #   Silero VAD 语音活动检测
│   ├── config.py              #   配置文件
│   ├── kws.py                 #   服务端唤醒词检测 (备用)
│   ├── run.sh                 #   启动脚本
│   └── sounds/                #   状态提示音 (thinking/done/error)
│
├── android-app/               # R1 上运行的 Android 应用
│   ├── app/src/main/java/com/mgt/r1voice/
│   │   ├── WakeWordDetector.java  # 端侧唤醒词检测 (OpenWakeWord ONNX)
│   │   ├── VoiceService.java      # 前台服务 + 状态机
│   │   ├── AudioRecorder.java     # 录音 (16kHz PCM)
│   │   ├── AudioPlayer.java       # 播放 (48kHz PCM)
│   │   ├── WsClient.java          # WebSocket 客户端
│   │   ├── LedController.java     # RGB LED 控制
│   │   └── MainActivity.java      # UI (输入服务器地址 + 启动)
│   └── app/src/main/assets/models/
│       ├── melspectrogram.onnx    # OpenWakeWord 模型
│       ├── embedding_model.onnx
│       └── hey_jarvis.onnx
│
└── README.md
```

## 连接 R1

### 硬件信息
- SoC: Rockchip RK3229 (ARM Cortex-A7 quad @ 1.5GHz)
- RAM: 512MB
- OS: Android 5.1 (API 22)
- Audio: 4-mic array, mono speaker
- Network: WiFi 2.4GHz

### 连接 ADB

R1 出厂自带 ADB 调试，无需拆机或获取 root 权限。直接通过 WiFi 连接即可：

1. 确保电脑和 R1 在同一局域网
2. 获取 R1 的 IP 地址（在 R1 屏幕上查看，或路由器后台查看）
3. 连接 ADB：`adb connect <R1_IP>:5555`

### 关键 ADB 命令
```bash
# 连接 R1
adb connect 192.168.1.152:5555

# 查看日志
adb logcat -s VoiceService:V OpenWakeWord:V

# 安装 APK（注意：adb install 在 R1 上会挂起，需用 push + pm install）
adb -s 192.168.1.152:5555 push app-debug.apk /data/local/tmp/r1voice.apk
adb -s 192.168.1.152:5555 shell "pm install -r /data/local/tmp/r1voice.apk"

# 杀掉 Phicomm 原生服务（防止音频冲突）
adb -s 192.168.1.152:5555 shell am force-stop com.phicomm.speaker

# 重启
adb -s 192.168.1.152:5555 shell reboot
```

## 服务器端

### 依赖
- Python 3.11+
- ffmpeg
- 一个 OpenAI-compatible LLM API (如 Hermes Agent)
- 一个 ASR 服务 (如 Qwen3-ASR, 运行在 :8083)

### 安装
```bash
cd server
uv sync  # 或 pip install -r requirements.txt
```

### 配置
编辑 `config.py`：
```python
WS_PORT = 8090              # WebSocket 端口
HERMES_BASE = "http://localhost:8642/v1"  # LLM API 地址
HERMES_MODEL = "glm-5.2-fast"             # LLM 模型
ASR_BASE = "http://localhost:8083/v1"     # ASR 服务地址
TTS_VOICE = "zh-CN-XiaoxiaoNeural"        # Edge TTS 语音
```

### 启动
```bash
./run.sh
# 或
python server.py
```

### 组件说明

#### VAD (语音活动检测)
- 使用 Silero VAD (PyTorch JIT 模型)
- 检测语音开始/结束，自动截取用户话语
- 1.5 秒静默 = 语音结束
- 500ms grace period 避免唤醒词回声被误检

#### ASR (语音识别)
- OpenAI-compatible API，支持任意 ASR 后端
- 当前使用 Qwen3-ASR (本地部署)
- 输入: 16kHz WAV，输出: 文本

#### LLM (大语言模型)
- OpenAI-compatible API
- 当前使用 GLM-5.2 (通过 Hermes Agent)
- 无状态模式（每次请求独立）

#### TTS (文本转语音)
- Edge TTS (微软免费 TTS)
- **流式分句生成**: 按句号分割，每句独立生成后立即推送
- 避免长文本一次性生成导致超时

## Android App

### 编译
```bash
cd android-app

# 配置 SDK 路径
cp local.properties.example local.properties
# 编辑 local.properties 指向你的 Android SDK

# 编译 (需要 Android SDK + build-tools 30.0.3)
./gradlew assembleDebug

# APK 输出在 app/build/outputs/apk/debug/app-debug.apk
```

### 部署到 R1
```bash
adb connect 192.168.1.152:5555

# 注意：adb install 在 R1 (Android 5.1) 上会挂起，必须用 push + pm install
adb -s 192.168.1.152:5555 push app/build/outputs/apk/debug/app-debug.apk /data/local/tmp/r1voice.apk
adb -s 192.168.1.152:5555 shell "pm install -r /data/local/tmp/r1voice.apk"

# 启动
adb -s 192.168.1.152:5555 shell am start -n com.mgt.r1voice/.MainActivity
# 启动后，在 R1 屏幕上输入服务器地址，点击"启动语音服务"
```

### 唤醒词检测
- 使用 OpenWakeWord (ONNX Runtime，完全端侧运行)
- 唤醒词: "Hey Jarvis"
- 三个 ONNX 模型: melspectrogram → embedding → wakeword classifier
- 固定增益 30x (R1 麦克风灵敏度低)
- 阈值 0.15 (较低，适合 R1 麦克风)

## 已知问题和坑

### 1. AudioFlinger 死锁 (最严重)
**问题**: R1 的 Android 5.1 音频 HAL 非常脆弱。如果创建/释放 AudioRecord 太快，或者与 Phicomm 原生音频服务同时运行，AudioFlinger 会永久死锁，需要重启设备。

**解决方案**:
- 不要释放+重建 AudioRecord。唤醒词检测和录音共用同一个 AudioRecord 实例
- 用 `pm disable` 禁用 Phicomm 音频服务（但可能导致其他问题）
- 重启 R1 是唯一的恢复手段
- BootReceiver 已禁用（开机自启动会与 Phicomm 竞争音频 HAL）

### 2. STREAM_MUSIC 不输出到扬声器
**问题**: R1 的音频路由配置异常，STREAM_MUSIC 不走扬声器。

**解决方案**: 所有音频播放使用 `AudioManager.STREAM_SYSTEM`。

### 3. AudioRecord.getMinBufferSize() 可能阻塞
**问题**: 在某些状态下，`getMinBufferSize()` 会无限阻塞。

**解决方案**: 硬编码 buffer size (2560 bytes)。

### 4. 麦克风增益低
**问题**: R1 的 4-mic 阵列灵敏度很低，录音音量极小。

**解决方案**: 固定增益 30x (`FIXED_GAIN = 30.0f`)。

### 5. TTS 长文本超时
**问题**: edge-tts 生成超过 60 秒的音频时，WebSocket 可能断开。

**解决方案**: 流式分句 TTS — 按句号分割文本，每句独立生成后立即推送。

### 6. System.arraycopy 引用别名
**问题**: `System.arraycopy` 在 `float[][]` 上复制的是引用，不是值。多次 shift 后所有行指向同一个 `float[]`，导致所有 melspec 帧变成相同的值，唤醒词 score 永远为 0。

**解决方案**: 使用 flat `float[]` 数组 + stride 而不是 `float[][]`。

### 7. ONNX Runtime 模型加载顺序
**问题**: initBuffers() 耗时约 4 秒（ONNX 噪声预填充），期间 Phicomm 可能抢占音频 HAL。

**解决方案**: 先 startRecording()（占用音频输入），再 initBuffers()。

## 技术栈

- **端侧唤醒词**: OpenWakeWord (ONNX Runtime Android 1.16.3)
- **WebSocket**: Java-WebSocket 1.5.6 (Android) / websockets 16.0 (Python)
- **VAD**: Silero VAD (PyTorch JIT)
- **ASR**: Qwen3-ASR (OpenAI-compatible API)
- **LLM**: GLM-5.2 (OpenAI-compatible API via Hermes Agent)
- **TTS**: Microsoft Edge TTS (edge-tts Python)
- **音频处理**: ffmpeg (MP3→PCM 转换)

## License

MIT
