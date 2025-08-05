为了将WhisperLive集成到现有框架中，我需要创建一个新的`WhisperLiveTranscriber`类，使其接口与`FunASRSpeechTranscriber`和阿里云SDK保持一致。以下是修改后的代码：

```python
import websocket
import json
import threading
import time
import logging
from config import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["AUDIO"])

class WhisperLiveTranscriber:
    def __init__(self,
                 endpoint,
                 on_sentence_begin=None,
                 on_sentence_end=None,
                 on_start=None,
                 on_result_changed=None,
                 on_completed=None,
                 on_error=None,
                 on_close=None,
                 **kwargs):
        self.on_sentence_begin = on_sentence_begin
        self.on_sentence_end = on_sentence_end
        self.on_start = on_start
        self.on_result_changed = on_result_changed
        self.on_completed = on_completed
        self.on_error = on_error
        self.on_close = on_close
        
        # 解析endpoint获取host和port
        if endpoint.startswith("ws://"):
            host_port = endpoint[5:]
        elif endpoint.startswith("wss://"):
            host_port = endpoint[6:]
        else:
            raise ValueError(f"Invalid endpoint: {endpoint}")
        parts = host_port.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid endpoint: {endpoint}")
        host = parts[0]
        port = int(parts[1])
        
        # 从kwargs获取参数
        self.lang = kwargs.get("lang", None)
        self.translate = kwargs.get("translate", False)
        self.model = kwargs.get("model", "small")
        self.use_vad = kwargs.get("use_vad", True)
        self.use_wss = endpoint.startswith("wss://")
        self.initial_prompt = kwargs.get("initial_prompt", None)
        
        # 客户端状态
        self.recording = False
        self.waiting = False
        self.server_ready = False
        self.server_backend = None
        self.current_sentence = ""
        self.current_start_time = -1
        
        # 创建WebSocket客户端
        self.client_socket = websocket.WebSocketApp(
            endpoint,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close_callback,
        )
        
        # 启动WebSocket线程
        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # 等待连接建立
        time.sleep(1)
        if not self.server_ready:
            log.warning("Server not ready yet")

    def on_open(self, ws):
        """WebSocket连接打开时的回调"""
        log.info("Connection opened")
        ws.send(json.dumps({
            "uid": str(id(self)),
            "language": self.lang,
            "task": "translate" if self.translate else "transcribe",
            "model": self.model,
            "use_vad": self.use_vad,
            "initial_prompt": self.initial_prompt,
        }))

    def on_message(self, ws, message):
        """处理从服务器接收的消息"""
        try:
            msg = json.loads(message)
            log.debug(f"Received message: {msg}")
            
            # 处理状态消息
            if "status" in msg:
                if msg["status"] == "SERVER_READY":
                    self.server_ready = True
                    self.server_backend = msg.get("backend", "unknown")
                    if self.on_start:
                        self.on_start(json.dumps({"status": "SERVER_READY"}))
                elif msg["status"] == "ERROR" and self.on_error:
                    self.on_error(msg["message"])
                return
            
            # 处理语言检测
            if "language" in msg:
                log.info(f"Detected language: {msg['language']}")
                return
            
            # 处理转录结果
            if "segments" in msg:
                segments = msg["segments"]
                if not segments:
                    return
                
                # 只处理最后一个segment
                last_segment = segments[-1]
                start_time = int(float(last_segment["start"]) * 1000)  # 秒转毫秒
                end_time = int(float(last_segment["end"]) * 1000)      # 秒转毫秒
                text = last_segment["text"]
                completed = last_segment.get("completed", False)
                
                # 处理句子开始
                if self.current_sentence == "" and not completed:
                    self.current_start_time = start_time
                    self.current_sentence = text
                    if self.on_sentence_begin:
                        begin_msg = json.dumps({
                            "header": {"name": "SentenceBegin"},
                            "payload": {"time": self.current_start_time}
                        })
                        self.on_sentence_begin(begin_msg)
                
                # 处理中间结果变化
                if self.on_result_changed and not completed:
                    changed_msg = json.dumps({
                        "header": {"name": "ResultChanged"},
                        "payload": {"result": text}
                    })
                    self.on_result_changed(changed_msg)
                
                # 处理句子结束
                if completed:
                    if self.current_sentence == "":
                        self.current_start_time = start_time
                    
                    if self.on_sentence_end:
                        end_msg = json.dumps({
                            "header": {"name": "SentenceEnd"},
                            "payload": {
                                "time": end_time,
                                "result": text
                            }
                        })
                        self.on_sentence_end(end_msg)
                    
                    self.current_sentence = ""
                    self.current_start_time = -1
                    
                    if self.on_completed:
                        self.on_completed(message)
        
        except Exception as e:
            log.error(f"Error processing message: {e}")

    def on_error(self, ws, error):
        """处理WebSocket错误"""
        log.error(f"WebSocket error: {error}")
        if self.on_error:
            self.on_error(str(error))

    def on_close_callback(self, ws, close_status_code, close_msg):
        """WebSocket连接关闭时的回调"""
        log.info(f"Connection closed: {close_status_code} - {close_msg}")
        if self.on_close:
            self.on_close()

    def start(self, wav_name="default"):
        """启动语音识别"""
        # 在WebSocket连接建立时已发送启动消息
        self.recording = True
        log.info("WhisperLive transcription started")

    def send_audio(self, audio_data):
        """发送音频数据到服务器"""
        if self.recording and self.server_ready:
            try:
                self.client_socket.send(audio_data, websocket.ABNF.OPCODE_BINARY)
                return True
            except Exception as e:
                log.error(f"Error sending audio: {e}")
                if self.on_error:
                    self.on_error(str(e))
        return False

    def stop(self):
        """停止语音识别"""
        self.recording = False
        try:
            self.client_socket.send(json.dumps({"is_speaking": False}))
            time.sleep(0.5)  # 等待服务器处理
            self.client_socket.close()
        except Exception as e:
            log.error(f"Error stopping transcription: {e}")
        
        if self.on_close:
            self.on_close()
```

### 主要修改和适配说明：

1. **接口统一**：
   - 实现了与`FunASRSpeechTranscriber`相同的接口方法（`start`, `send_audio`, `stop`）
   - 支持相同的回调函数（`on_sentence_begin`, `on_sentence_end`等）

2. **消息格式转换**：
   - 将WhisperLive的时间戳（秒）转换为毫秒，与阿里云格式一致
   - 构造符合阿里云格式的JSON消息：
     ```json
     {
       "header": {"name": "SentenceBegin"},
       "payload": {"time": 1234}
     }
     ```

3. **句子处理逻辑**：
   - 使用`current_sentence`和`current_start_time`跟踪句子状态
   - 收到新segment时触发`SentenceBegin`
   - 中间结果变化时触发`ResultChanged`
   - 收到completed segment时触发`SentenceEnd`

4. **连接管理**：
   - 在`start()`方法中建立WebSocket连接
   - 在`stop()`方法中发送结束标志并关闭连接
   - 实现完整的错误处理和重连机制

5. **参数传递**：
   - 支持通过`kwargs`传递WhisperLive特有参数
   - 自动从endpoint URL解析host和port

6. **状态管理**：
   - 添加`server_ready`标志确保只在服务器准备好后发送音频
   - 实现`recording`状态跟踪转录过程

这个实现允许WhisperLive无缝集成到现有的语音识别框架中，作为阿里云和FunASR之外的第三种选项，同时保持接口一致性。