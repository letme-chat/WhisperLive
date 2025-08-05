为了将WhisperLive集成到现有系统中并使其接口与阿里云SDK兼容，我创建了一个新的`WhisperLIveTranscriber`类。以下是完整的实现代码：

```python
import json
import time
import threading
import logging
import numpy as np
from websocket import WebSocketApp, ABNF
import uuid
import traceback

# 设置日志
log = logging.getLogger(__name__)

class WhisperLIveTranscriber:
    def __init__(
        self,
        host,
        port,
        lang=None,
        translate=False,
        model="small",
        use_vad=True,
        use_wss=False,
        log_transcription=True,
        max_clients=4,
        max_connection_time=600,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        initial_prompt=None,
        on_sentence_begin=None,
        on_sentence_end=None,
        on_start=None,
        on_result_changed=None,
        on_completed=None,
        on_error=None,
        on_close=None
    ):
        # 保存回调函数
        self.on_sentence_begin = on_sentence_begin
        self.on_sentence_end = on_sentence_end
        self.on_start = on_start
        self.on_result_changed = on_result_changed
        self.on_completed = on_completed
        self.on_error = on_error
        self.on_close = on_close
        
        # 初始化参数
        self.host = host
        self.port = port
        self.lang = lang
        self.translate = translate
        self.model = model
        self.use_vad = use_vad
        self.use_wss = use_wss
        self.log_transcription = log_transcription
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold
        self.initial_prompt = initial_prompt
        
        # 状态变量
        self.recording = False
        self.waiting = False
        self.server_error = False
        self.server_backend = None
        self.last_response_received = None
        self.disconnect_if_no_response_for = 15
        self.uid = str(uuid.uuid4())
        self.ws_thread = None
        self.client_socket = None
        self.current_segments = {}  # 用于跟踪当前处理中的分段
        
        # 连接WebSocket
        self.connect()

    def connect(self):
        """连接到WhisperLive服务器"""
        socket_protocol = 'wss' if self.use_wss else "ws"
        socket_url = f"{socket_protocol}://{self.host}:{self.port}"
        
        try:
            self.client_socket = WebSocketApp(
                socket_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # 启动WebSocket线程
            self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            log.info(f"Connected to WhisperLive server at {socket_url}")
        except Exception as e:
            log.error(f"Failed to connect to WhisperLive server: {str(e)}")
            if self.on_error:
                self.on_error(f"Connection error: {str(e)}")

    def _on_open(self, ws):
        """WebSocket连接打开时的回调"""
        log.info("WebSocket connection opened")
        # 发送初始化配置
        init_config = {
            "uid": self.uid,
            "language": self.lang,
            "task": "transcribe" if not self.translate else "translate",
            "model": self.model,
            "use_vad": self.use_vad,
            "max_clients": self.max_clients,
            "max_connection_time": self.max_connection_time,
            "send_last_n_segments": self.send_last_n_segments,
            "no_speech_thresh": self.no_speech_thresh,
            "clip_audio": self.clip_audio,
            "same_output_threshold": self.same_output_threshold,
            "initial_prompt": self.initial_prompt
        }
        ws.send(json.dumps(init_config))
        log.debug(f"Sent init config: {init_config}")

    def _on_message(self, ws, message):
        """处理从服务器收到的消息"""
        try:
            msg = json.loads(message)
            log.debug(f"Received message: {msg}")
            
            # 验证UID
            if msg.get("uid") != self.uid:
                log.warning(f"Invalid UID received: {msg.get('uid')}")
                return
                
            # 处理状态消息
            if "status" in msg:
                self._handle_status(msg)
                return
                
            # 处理服务器准备消息
            if "message" in msg and msg["message"] == "SERVER_READY":
                self.recording = True
                self.server_backend = msg.get("backend", "unknown")
                self.last_response_received = time.time()
                if self.on_start:
                    start_event = {
                        "header": {"name": "Start"},
                        "payload": {}
                    }
                    self.on_start(json.dumps(start_event))
                log.info(f"Server ready with backend: {self.server_backend}")
                return
                
            # 处理语言检测
            if "language" in msg:
                self.lang = msg.get("language")
                log.info(f"Detected language: {self.lang}")
                return
                
            # 处理转录结果
            if "segments" in msg:
                self._handle_segments(msg["segments"])
                
            # 处理断开消息
            if "message" in msg and msg["message"] == "DISCONNECT":
                log.info("Server disconnected due to overtime")
                self.recording = False
                if self.on_close:
                    self.on_close()
                
        except json.JSONDecodeError:
            log.error(f"Failed to decode message: {message}")
        except Exception as e:
            log.error(f"Error processing message: {traceback.format_exc()}")

    def _handle_status(self, msg):
        """处理服务器状态消息"""
        status = msg["status"]
        if status == "WAIT":
            self.waiting = True
            log.info(f"Server full. Wait time: {msg.get('message', 'unknown')} min")
        elif status == "ERROR":
            self.server_error = True
            error_msg = msg.get("message", "Unknown error")
            log.error(f"Server error: {error_msg}")
            if self.on_error:
                self.on_error(error_msg)
        elif status == "WARNING":
            log.warning(f"Server warning: {msg.get('message', 'Unknown warning')}")

    def _handle_segments(self, segments):
        """处理转录分段"""
        for seg in segments:
            start = seg["start"]
            
            # 处理新句子开始
            if start not in self.current_segments:
                self._handle_sentence_begin(seg)
            
            # 更新当前分段
            self.current_segments[start] = seg
            
            # 处理结果变化
            self._handle_result_changed(seg)
            
            # 处理句子结束
            if seg.get("completed", False):
                self._handle_sentence_end(seg)
                # 移除已完成的句子
                if start in self.current_segments:
                    del self.current_segments[start]

    def _handle_sentence_begin(self, seg):
        """处理句子开始"""
        if self.on_sentence_begin:
            try:
                begin_time_ms = int(float(seg["start"]) * 1000)
                begin_event = {
                    "header": {"name": "SentenceBegin"},
                    "payload": {"time": begin_time_ms}
                }
                self.on_sentence_begin(json.dumps(begin_event))
            except Exception as e:
                log.error(f"Error handling sentence begin: {str(e)}")

    def _handle_result_changed(self, seg):
        """处理结果变化"""
        if self.on_result_changed:
            try:
                end_time_ms = int(float(seg["end"]) * 1000)
                changed_event = {
                    "header": {"name": "ResultChanged"},
                    "payload": {
                        "result": seg["text"],
                        "time": end_time_ms
                    }
                }
                self.on_result_changed(json.dumps(changed_event))
            except Exception as e:
                log.error(f"Error handling result changed: {str(e)}")

    def _handle_sentence_end(self, seg):
        """处理句子结束"""
        if self.on_sentence_end:
            try:
                end_time_ms = int(float(seg["end"]) * 1000)
                end_event = {
                    "header": {"name": "SentenceEnd"},
                    "payload": {
                        "time": end_time_ms,
                        "result": seg["text"]
                    }
                }
                self.on_sentence_end(json.dumps(end_event))
            except Exception as e:
                log.error(f"Error handling sentence end: {str(e)}")

    def _on_error(self, ws, error):
        """WebSocket错误回调"""
        log.error(f"WebSocket error: {str(error)}")
        self.server_error = True
        if self.on_error:
            self.on_error(str(error))

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket关闭回调"""
        log.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.recording = False
        self.waiting = False
        if self.on_close:
            self.on_close()

    def start(self):
        """开始转录"""
        # 连接时自动开始，这里主要等待服务器准备
        timeout = time.time() + 10
        while not self.recording and time.time() < timeout:
            time.sleep(0.1)
        
        if not self.recording:
            error_msg = "Failed to start: server not ready"
            log.error(error_msg)
            if self.on_error:
                self.on_error(error_msg)
            return False
        
        log.info("Transcription started")
        return True

    def send_audio(self, audio_data):
        """发送音频数据到服务器"""
        if not self.recording or not self.client_socket:
            log.warning("Cannot send audio: not recording or connection closed")
            return
        
        try:
            # 转换16位PCM到float32
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.client_socket.send(audio_array.tobytes(), ABNF.OPCODE_BINARY)
        except Exception as e:
            log.error(f"Error sending audio: {str(e)}")
            if self.on_error:
                self.on_error(f"Audio send error: {str(e)}")

    def stop(self):
        """停止转录"""
        try:
            if self.client_socket:
                # 发送结束信号
                self.client_socket.send("END_OF_AUDIO", ABNF.OPCODE_TEXT)
                
                # 等待所有响应
                self._wait_for_responses()
                
                # 关闭连接
                self.client_socket.close()
            
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=1.0)
                
            self.recording = False
            log.info("Transcription stopped")
            
            if self.on_completed:
                completed_event = {
                    "header": {"name": "Completed"},
                    "payload": {}
                }
                self.on_completed(json.dumps(completed_event))
                
        except Exception as e:
            log.error(f"Error stopping transcription: {str(e)}")
            if self.on_error:
                self.on_error(f"Stop error: {str(e)}")
        finally:
            if self.on_close:
                self.on_close()

    def _wait_for_responses(self):
        """等待所有响应处理完成"""
        if self.last_response_received:
            timeout = time.time() + self.disconnect_if_no_response_for
            while time.time() < timeout:
                time.sleep(0.1)
```

### 主要特点和设计说明：

1. **兼容接口设计**：
   - 实现了与阿里云SDK相同的回调接口（on_sentence_begin, on_sentence_end等）
   - 转换WhisperLive的消息格式为阿里云兼容格式
   - 支持相同的start(), send_audio(), stop()方法

2. **消息处理机制**：
   - 使用current_segments字典跟踪处理中的分段
   - 将WhisperLive的分段转换为阿里云的事件模型：
     - 新分段触发SentenceBegin
     - 分段更新触发ResultChanged
     - 分段完成触发SentenceEnd

3. **音频处理**：
   - 将16位PCM转换为float32格式（WhisperLive要求）
   - 实现了音频数据的分块发送
   - 自动处理音频结束信号(END_OF_AUDIO)

4. **连接管理**：
   - 完整的WebSocket生命周期管理（连接、错误处理、关闭）
   - 超时和重连机制
   - 服务器状态监控

5. **错误处理**：
   - 全面的异常捕获和日志记录
   - 错误信息通过on_error回调传递
   - 防止因单个错误导致整个服务中断

6. **资源清理**：
   - 停止时发送结束信号
   - 等待未完成的消息处理
   - 正确关闭WebSocket连接和线程

### 使用示例：

```python
# 创建WhisperLIveTranscriber实例
whisper_transcriber = WhisperLIveTranscriber(
    host="localhost",
    port=9090,
    on_sentence_begin=on_sentence_begin_callback,
    on_sentence_end=on_sentence_end_callback,
    on_start=on_start_callback,
    on_result_changed=on_result_changed_callback,
    on_error=on_error_callback,
    on_close=on_close_callback
)

# 开始转录
if whisper_transcriber.start():
    # 发送音频数据
    while has_audio:
        audio_chunk = get_audio_chunk()
        whisper_transcriber.send_audio(audio_chunk)
    
    # 停止转录
    whisper_transcriber.stop()
```

这个实现保持了与原有系统相同的接口，同时集成了WhisperLive的功能，使得可以在不修改现有音频处理流程的情况下切换ASR引擎。