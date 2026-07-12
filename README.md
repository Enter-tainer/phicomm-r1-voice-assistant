# 斐讯 R1 语音助手

把斐讯 R1 蓝牙音箱变成一个可编程的语音助手。

## 这是什么

斐讯 R1 是一个带 4-mic 阵列的蓝牙音箱，内置 Android 5.1 系统。本项目通过以下步骤将其改造为自定义语音助手：

1. **连接 ADB** — R1 出厂自带 ADB 调试，无需拆机或获取 root
2. **Android App** — 在 R1 上运行的语音客户端（持续推流麦克风音频 + 播放服务器返回的音频）
3. **Voice Server** — 在局域网服务器上运行的 WebSocket 后端（唤醒词检测 + ASR + LLM + TTS）

## 架构

**核心设计**：唤醒词检测运行在**服务端**（openWakeWord），R1 app 是一个"哑终端"——持续向服务器推流 16kHz PCM 音频，播放服务器返回的 48kHz TTS PCM + 提示音。服务器控制整个状态机。

```
┌──────────────────────────────────────────────────────────┐
│                    R1 Device (Android 5.1)                │
│                                                          │
│  ┌──────────────┐    ┌─────────────┐                    │
│  │ AudioRecorder │───▶│ WsClient    │─┼── WebSocket
│  │ (16kHz PCM,  │    │ (binary)    │ │  (持续推流)
│  │  80ms frames) │    └─────────────┘ │
│  └──────────────┘                     │                 │
│         ▲                              │                 │
│  ┌──────┴──────┐              ┌───────┴───────┐         │
│  │ LedController│◀─────────────│ AudioPlayer    │         │
│  └─────────────┘              │ (48kHz PCM)   │         │
│                               └───────────────┘         │
└──────────────────────────────────────────────────────────┘
                                    │
                    WebSocket (ws://server:8090)
                                    │
┌───────────────────────────────────┼──────────────────────┐
│                    Voice Server (Linux x86_64)            │
│                                   │                      │
│  ┌────────────┐  ┌────────┐  ┌───┴──────┐  ┌─────────┐  │
│  │openWakeWord│  │ VAD    │  │WebSocket │  │ TTS     │  │
│  │(hey_jarvis,│  │(Silero)│  │ Server   │  │(Edge-TTS│  │
│  │ ONNX,      │  │        │  │ (state   │  │ streaming│  │
│  │ threshold  │  │speech  │  │  machine)│  │ per     │  │
│  │ 0.3,       │  │detect  │  │          │  │ sentence)│  │
│  │ gain 30x)  │  │        │  │          │  │         │  │
│  └────────────┘  └───┬────┘  └──────────┘  └─────────┘  │
│                      │                     ▲             │
│                ┌─────┴─────┐               │             │
│                │ ASR       │  ┌────────────┘             │
│                │(Qwen3-ASR)│  │                          │
│                └─────┬─────┘  │                          │
│                      │  ┌─────┴─────┐                    │
│                      └─▶│ Hermes API│                    │
│                         │(GLM-5.2)  │                    │
│                         └───────────┘                    │
└──────────────────────────────────────────────────────────┘
```

## 语音交互流程

```
用户: "Hey Jarvis"
  ↓
Server: openWakeWord 检测到唤醒词 (score > 0.3)
Server: 发送 wake beep (0.15s 短促上升音) → R1 播放
Server: 等待 beep 播完 (beep时长 + 150ms) → 启动 VAD
  ↓
R1: 继续推流麦克风音频
Server: VAD 检测语音开始/结束
  ↓ 语音结束后
Server: 发送 thinking beep → R1 播放
Server: ASR 识别 → "上海天气适合跑步吗？"
Server: 调用 Hermes API → 获取回复
Server: 按句分割 → 逐句 Edge-TTS → 流式发送 PCM
  ↓
R1: 边收边播 → 语音回复
Server: 发送 done beep (0.33s 下降挂断音) → R1 播放
  ↓
Server: 回到唤醒词检测状态
```

## 提示音系统

所有提示音为 48kHz 16-bit mono PCM WAV，服务端启动时预加载到内存，通过 WebSocket binary chunk 发送给 R1。

| 事件 | 文件 | 时长 | 说明 |
|------|------|------|------|
| 唤醒词检测到 | `wake.wav` | 0.15s | 短促双音 beep（上升音调） |
| 开始思考 | `thinking.wav` | 1.10s | Edge TTS 生成 "正在思考" |
| TTS 播放完成 | `done.wav` | 0.33s | 挂断音 beep（1000Hz → 700Hz，先高后低） |
| 出错/没听清 | `error.wav` | 2.54s | Edge TTS 生成 "没听清，请再说一遍" |

## 项目结构

```
├── server/                    # Python WebSocket 语音服务器
│   ├── server.py              #   WebSocket 服务器 + 状态机 + 音频路由
│   ├── pipeline.py            #   ASR → LLM → TTS 流水线 + 心跳保活
│   ├── wake_word.py           #   openWakeWord 封装 (hey_jarvis, ONNX)
│   ├── vad_silero.py          #   Silero VAD 语音活动检测
│   ├── config.py              #   配置文件
│   ├── kws.py / main.py       #   备用/测试代码
│   ├── run.sh                 #   启动脚本
│   └── sounds/                #   状态提示音 (wake/done/thinking/error)
│
├── android-app/               # R1 上运行的 Android 应用
│   ├── app/src/main/java/com/mgt/r1voice/
│   │   ├── VoiceService.java      # 前台服务 + WebSocket 生命周期 + 状态处理
│   │   ├── WsClient.java          # 自动重连 WebSocket 客户端
│   │   ├── AudioRecorder.java     # 录音 (16kHz PCM, 80ms 帧, 持续推流)
│   │   ├── AudioPlayer.java       # 播放 (48kHz PCM, STREAM_SYSTEM)
│   │   ├── LedController.java     # RGB LED 状态指示
│   │   ├── MainActivity.java      # 启动界面
│   │   ├── BootReceiver.java      # 开机自启 (已禁用)
│   │   └── WakeWordDetector.java  # 端侧唤醒词 (已弃用，改用服务端检测)
│   └── app/src/main/assets/models/
│       ├── melspectrogram.onnx    # OpenWakeWord 模型 (端侧备用)
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
adb logcat -s VoiceService:V WsClient:V AudioPlayer:V AudioRecorder:V

# 安装 APK（注意：adb install 在 R1 上会挂起，需用 push + 后台 pm install）
adb -s 192.168.1.152:5555 push app-debug.apk /data/local/tmp/r1voice.apk
adb -s 192.168.1.152:5555 shell "sh -c 'pm install -r /data/local/tmp/r1voice.apk > /data/local/tmp/install_result.txt 2>&1; echo DONE >> /data/local/tmp/install_result.txt &'"

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
uv sync
```

### 配置
编辑 `config.py`：
```python
WS_PORT = 8090              # WebSocket 端口
HERMES_BASE = "http://localhost:8642/v1"  # LLM API 地址
HERMES_MODEL = "glm-5.2-fast"             # LLM 模型
ASR_BASE = "http://localhost:8083/v1"     # ASR 服务地址
TTS_VOICE = "zh-CN-XiaoxiaoNeural"        # Edge TTS 语音
WAKE_WORD_THRESHOLD = 0.3                 # 唤醒词阈值
MIC_GAIN = 30.0                           # 麦克风增益 (R1 灵敏度低)
```

### 启动
```bash
./run.sh
# 或
python server.py
```

### 组件说明

#### 唤醒词检测 (服务端)
- openWakeWord (hey_jarvis 模型, ONNX 推理)
- 阈值 0.3 (R1 麦克风质量较差，score 不会太高)
- 麦克风增益 30x (在预测前应用到 PCM)
- 检测到后播放 wake beep → 等待 beep 播完 → 启动 VAD

#### VAD (语音活动检测)
- Silero VAD (PyTorch JIT 模型)
- 检测语音开始/结束，自动截取用户话语
- 静默检测后截取语音，发送给 ASR

#### ASR (语音识别)
- OpenAI-compatible API
- 当前使用 Qwen3-ASR (本地部署)
- 输入: 16kHz WAV，输出: 文本

#### LLM (大语言模型)
- OpenAI-compatible API
- 当前使用 GLM-5.2 (通过 Hermes Agent)

#### TTS (文本转语音)
- Edge TTS (微软免费 TTS)
- **流式分句生成**: 按中文标点 (。！？) 分割，每句独立生成后立即推送
- 单句失败自动跳过，不影响后续句子
- 避免长文本一次性生成导致超时

## Android App

### 编译
```bash
cd android-app

# 配置 SDK 路径
cp local.properties.example local.properties
# 编辑 local.properties 指向你的 Android SDK

# 编译 (需要 Android SDK + build-tools 30.0.3, Java 8)
# R1 是 Android 5.1 = API 22, 所以 compileSdk/targetSdk = 22
gradle assembleDebug  # 没有 gradlew wrapper，使用系统 gradle

# APK 输出在 app/build/outputs/apk/debug/app-debug.apk
```

### 部署到 R1

**CRITICAL**: R1 运行 Android 5.1。现代 ADB (v34+) 的 `adb install` 会无限挂起（streaming install 协议不兼容），`adb shell pm install` 前台执行也会断开连接。必须用 **push + 后台 pm install**：

```bash
# 1. 重连 ADB
adb kill-server; sleep 2
adb connect 192.168.1.152:5555; sleep 2

# 2. 停止 app
adb -s 192.168.1.152:5555 shell am force-stop com.mgt.r1voice

# 3. Push APK
adb -s 192.168.1.152:5555 push app/build/outputs/apk/debug/app-debug.apk /data/local/tmp/r1voice.apk

# 4. 后台执行 pm install (前台会断开 ADB 连接)
adb -s 192.168.1.152:5555 shell "sh -c 'pm install -r /data/local/tmp/r1voice.apk > /data/local/tmp/install_result.txt 2>&1; echo DONE >> /data/local/tmp/install_result.txt &'"

# 5. 等待 10 秒
sleep 10

# 6. 重连 ADB (步骤 4 后连接通常会断开)
adb kill-server; sleep 2
adb connect 192.168.1.152:5555; sleep 2

# 7. 检查结果
adb -s 192.168.1.152:5555 shell "cat /data/local/tmp/install_result.txt"
# 应显示: Success\nDONE

# 8. 启动 app
adb -s 192.168.1.152:5555 shell am start -n com.mgt.r1voice/.MainActivity
```

## WebSocket 协议

```
Client → Server (binary): 16kHz 16bit mono PCM (80ms = 2560 bytes, 持续推流)
Client → Server (text):   JSON {"type": "stop"|"bye"}
Server → Client (text):   JSON {"type":"state","state":"idle"|"listening"|"thinking"|"speaking"}
Server → Client (text):   JSON {"type":"tts_done"} / {"type":"asr_result","text":"..."}
Server → Client (binary): 48kHz 16bit mono PCM (TTS + 提示音, 1920 bytes/chunk)
```

## 状态机

```
IDLE → (唤醒词检测到) → 播放 wake beep → 等待 beep 播完 → LISTENING
LISTENING → (VAD 语音结束) → THINKING → (TTS 开始) → SPEAKING → (TTS 完成) → 播放 done beep → IDLE
```

- **IDLE**: 服务端运行 openWakeWord，增益 30x，阈值 0.3
- **LISTENING**: 服务端运行 Silero VAD，缓冲语音直到检测到静默
- **THINKING**: ASR + Hermes LLM，播放 thinking 提示音，心跳每 3s 保活
- **SPEAKING**: 流式 TTS（逐句生成），客户端静音麦克风防回声，心跳继续

### VAD 延迟启动
检测到唤醒词后，服务端发送 wake beep，然后**等待 beep 播放完毕**（beep 时长 + 150ms 缓冲）才切换到 LISTENING 状态启动 VAD。防止麦克风收到 beep 回声导致 VAD 误触发。

## 已知问题和坑

### 1. AudioFlinger 死锁 (最严重)
**问题**: R1 的 Android 5.1 音频 HAL 非常脆弱。创建/释放 AudioRecord 太快会导致永久死锁。

**解决方案**: 重启 R1 是唯一恢复手段。BootReceiver 已禁用，手动启动。

### 2. adb install 挂起
**问题**: R1 的 adbd 不兼容现代 ADB 的 streaming install 协议。`adb shell pm install` 前台执行也会断开连接。

**解决方案**: push APK → 后台执行 `pm install`（输出重定向到文件）→ 等 10s → 重连 ADB → 读结果。

### 3. STREAM_MUSIC 不输出到扬声器
**问题**: R1 的音频路由配置异常，STREAM_MUSIC 不走扬声器。

**解决方案**: 所有音频播放使用 `AudioManager.STREAM_SYSTEM`。

### 4. TTS 长文本超时
**问题**: edge-tts 生成超过 60 秒的音频时，WebSocket 可能断开。

**解决方案**: 流式分句 TTS + 心跳保活（thinking 和 speaking 阶段都每 3s 发送状态）。

### 5. VAD 被 beep 回声误触发
**问题**: 唤醒词检测后立即启动 VAD，麦克风收到 beep 回声导致误触发。

**解决方案**: 发送 wake beep 后等待 beep 时长 + 150ms 再启动 VAD。

## 技术栈

- **唤醒词**: openWakeWord (ONNX, 服务端运行)
- **WebSocket**: Java-WebSocket 1.5.6 (Android) / websockets 16.0 (Python)
- **VAD**: Silero VAD (PyTorch JIT)
- **ASR**: Qwen3-ASR (OpenAI-compatible API)
- **LLM**: GLM-5.2 (OpenAI-compatible API via Hermes Agent)
- **TTS**: Microsoft Edge TTS (edge-tts Python)
- **音频处理**: ffmpeg (MP3→PCM 转换)

## License

MIT
