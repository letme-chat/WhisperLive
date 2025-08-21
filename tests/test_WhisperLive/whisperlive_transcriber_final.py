
import websocket
import json
import threading
import time
import uuid
import numpy as np
import logging
import traceback
from websocket import ABNF

LOG_ALL_TRANSCRIPT = True # TODO * @jjm for test

class WhisperLiveTranscriber:
    def __init__(self,
                 endpoint,
                 language=None,
                 initial_prompt="下面是一些IT面试问题。",
                 on_sentence_begin=None,
                 on_sentence_end=None,
                 on_start=None,
                 on_result_changed=None,
                 on_completed=None,
                 on_error=None,
                 on_close=None):
        """
        初始化WhisperLive转录器
        
        参数:
            endpoint: WebSocket服务器地址
            on_sentence_begin: 句子开始回调
            on_sentence_end: 句子结束回调
            on_start: 转录开始回调
            on_result_changed: 结果变化回调
            on_completed: 转录完成回调
            on_error: 错误回调
            on_close: 连接关闭回调
            model: Whisper模型大小
            use_vad: 是否使用语音活动检测
        """
        self.endpoint = endpoint
        self.model = "large-v3-turbo"
        self.use_vad = True
        # self.max_clients = 20
        # self.max_connection_time = 600
        # self.send_last_n_segments = 3
        self.language = language
        self.initial_prompt=initial_prompt
        self.uid = str(uuid.uuid4())
        self.last_send_segment = None
        self.same_output_count = 0
        
        # 回调函数
        self.on_sentence_begin = on_sentence_begin
        self.on_sentence_end = on_sentence_end
        self.on_start = on_start
        self.on_result_changed = on_result_changed
        self.on_completed = on_completed
        self.on_error = on_error
        self.on_close = on_close
        
        # WebSocket连接状态
        self.websocket = None
        self.thread = None
        self.running = False
        self.connected = False
        self.stop_event = threading.Event()
        self.completed_event = threading.Event()

        self.transcript = []
        self.all_transcript_text = ""

    def _on_open(self, ws):
        """WebSocket连接打开时的回调"""
        try:
            # 发送初始化配置
            init_msg = json.dumps({
                "uid": self.uid,
                # "language": None,  # 自动检测语言
                "language": self.language,
                "task": "transcribe",
                "model": self.model,
                "use_vad": self.use_vad,
                # "max_clients": self.max_clients,
                # "max_connection_time": self.max_connection_time,
                # "send_last_n_segments": self.send_last_n_segments,
                "no_speech_thresh": 0.45,
                "clip_audio": False,
                "same_output_threshold": 10,
                "initial_prompt": self.initial_prompt
            })
            ws.send(init_msg)
            
            # 触发开始回调
            if self.on_start:
                self.on_start(init_msg)
            
            self.connected = True
        except Exception as e:
            if self.on_error:
                self.on_error(f"Connection open error: {str(e)}")

    def _send_seg_as_sentence_end(self, seg):
        seg_text = seg.get("text", "")
        start_ms = int(float(seg.get("start", 0)) * 1000)
        end_ms = int(float(seg.get("end", 0)) * 1000)
        if self.on_sentence_begin:
            begin_msg = {
                "header": {"name": "SentenceBegin"},
                "payload": {"time": start_ms}
            }
            self.on_sentence_begin(json.dumps(begin_msg))
        if self.on_sentence_end:
            end_msg = {
                "header": {"name": "SentenceEnd"},
                "payload": {
                    "time": end_ms,
                    "result": seg_text
                }
            }
            self.on_sentence_end(json.dumps(end_msg))
    def _on_message(self, ws, message):
        """收到消息时的回调"""
        try:
            message = json.loads(message)
            print("received message:", message)

            if self.uid != message.get("uid"):
                print("[ERROR]: invalid client uid")
                return
            
            # 处理状态消息
            if "status" in message:
                status = message["status"]
                if status == "WAIT":
                    print(f"[INFO]: Server is full. Estimated wait time {round(message['message'])} minutes.")
                if status == "ERROR":
                    error_msg = message.get("message", "Unknown error")
                    if self.on_error:
                        self.on_error(f"Server error: {error_msg}")
                elif status == "WARNING":
                    # 服务器就绪，可以开始发送音频
                    print(f"Message from Server: {message['message']}")
                return
            
            if "message" in message.keys() and message["message"] == "DISCONNECT":
                print("[INFO]: Server disconnected due to overtime.")
            # TODO*@jjm finish then close?

            if "message" in message.keys() and message["message"] == "SERVER_READY":
                self.server_backend = message["backend"]
                print(f"[INFO]: Server Running with backend {self.server_backend}")
                return
            
            # 处理语言检测消息
            if "language" in message:
                self.language = message.get("language")
                lang_prob = message.get("language_prob")
                print(
                    f"[INFO]: Server detected language {self.language} with probability {lang_prob}"
                )
                return
            
            # 处理转录结果
            if "segments" in message:
                segments = message["segments"]

                texts = []
                for i, seg in enumerate(segments):
                    seg_text = seg.get("text", "")
                    if not texts or texts[-1] != seg_text:
                        texts.append(seg_text.strip())
                        if i == len(segments) - 1 and not seg.get("completed", False):
                            # TODO * @jjm send ResultChanged
                            # 结果变化回调
                            if self.on_result_changed:
                                changed_msg = {
                                    "header": {"name": "ResultChanged"},
                                    # "payload": {"result": seg_text}
                                    "text": seg_text,
                                    "is_partial": True # use this to indicate that frontend should not append but replace the last sentence with this one.
                                }
                                self.on_result_changed(json.dumps(changed_msg))
                                self.last_send_segment = seg
                        elif self.server_backend == "faster_whisper" and seg.get("completed", False):
                            if (not self.transcript or float(seg['start']) >= float(self.transcript[-1]['end'])):
                                self.transcript.append(seg)
                                if LOG_ALL_TRANSCRIPT:
                                    self.all_transcript_text += seg_text
                                    print(f"[INFO] self.uid:{self.uid} all transcript:{self.all_transcript_text}")
                                KEEP_LAST_CNT = 3
                                if len(self.transcript) > KEEP_LAST_CNT: # keep last 2 segements only
                                    self.transcript = self.transcript[-KEEP_LAST_CNT:]
                                self._send_seg_as_sentence_end(seg)
                                self.last_send_segment = seg
                
                # # 处理每个片段
                # for i, seg in enumerate(segments):

                #     start_ms = int(float(seg.get("start", 0)) * 1000)
                #     end_ms = int(float(seg.get("end", 0)) * 1000)
                #     text = seg.get("text", "")
                #     completed = seg.get("completed", False)
                    
                #     # 如果是最后一个未完成的片段
                #     is_last_uncompleted = (i == len(segments) - 1 and not completed)
                    
                #     # 句子开始回调
                #     if self.on_sentence_begin and not is_last_uncompleted:
                #         begin_msg = {
                #             "header": {"name": "SentenceBegin"},
                #             "payload": {"time": start_ms}
                #         }
                #         self.on_sentence_begin(json.dumps(begin_msg))
                    
                #     # 处理结果变化（未完成的片段）
                #     if is_last_uncompleted:
                #         # 检测结果是否稳定
                #         if text == self.last_segment_text:
                #             self.same_output_count += 1
                #         else:
                #             self.same_output_count = 0
                #             self.last_segment_text = text
                        
                #         # 结果变化回调
                #         if self.on_result_changed:
                #             changed_msg = {
                #                 "header": {"name": "ResultChanged"},
                #                 "payload": {"result": text}
                #             }
                #             self.on_result_changed(json.dumps(changed_msg))
                #     # 处理完成的句子
                #     else:
                #         # 句子结束回调
                #         if self.on_sentence_end:
                #             end_msg = {
                #                 "header": {"name": "SentenceEnd"},
                #                 "payload": {
                #                     "time": end_ms,
                #                     "result": text
                #                 }
                #             }
                #             self.on_sentence_end(json.dumps(end_msg))
                        
                #         # 重置状态
                #         self.last_segment_text = ""
                #         self.same_output_count = 0
                
                # # 处理完成消息
                # if msg.get("message") == "DISCONNECT":
                #     self.completed_event.set()
                #     if self.on_completed:
                #         self.on_completed(message)
        except Exception as e:
            error_msg = f"Message handling error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            if self.on_error:
                self.on_error(error_msg)

    def _on_error(self, ws, error):
        """发生错误时的回调"""
        if self.on_error:
            self.on_error(f"WebSocket error: {str(error)}")

    def _on_close(self, ws, close_status_code, close_msg):
        """连接关闭时的回调"""
        if self.last_send_segment is not None:
            completed = self.last_send_segment.get("completed", False)
            if not completed:
                print(f"Transcription finished incomplete, sending last_send_segment: {self.last_send_segment}")
                self._send_seg_as_sentence_end(self.last_send_segment)
        self.connected = False
        self.running = False
        if self.on_close:
            self.on_close()
        self.completed_event.set()

    def start(self):
        """开始转录"""
        if self.running:
            return
            
        self.running = True
        self.completed_event.clear()
        self.stop_event.clear()
        
        # 创建WebSocket连接
        self.websocket = websocket.WebSocketApp(
            self.endpoint,
            on_open=lambda ws: self._on_open(ws),
            on_message=lambda ws, msg: self._on_message(ws, msg),
            on_error=lambda ws, err: self._on_error(ws, err),
            on_close=lambda ws, code, msg: self._on_close(ws, code, msg)
        )
        
        # 启动线程运行WebSocket
        self.thread = threading.Thread(target=self.websocket.run_forever)
        self.thread.daemon = True
        self.thread.start()

    def send_audio(self, audio_data):
        """发送音频数据"""
        if not self.connected or not self.running:
            return False
        
        try:
            # 将PCM音频转换为float32格式
            raw_data = np.frombuffer(buffer=audio_data, dtype=np.int16)
            float_audio = raw_data.astype(np.float32) / 32768.0
            audio_bytes = float_audio.tobytes()
            
            # 发送音频数据
            self.websocket.send(audio_bytes, ABNF.OPCODE_BINARY)
            return True
        except Exception as e:
            if self.on_error:
                self.on_error(f"Audio sending error: {str(e)}")
            return False

    def stop(self):
        """停止转录"""
        if not self.running:
            return
            
        try:
            # 发送结束信号
            if self.connected:
                self.websocket.send("END_OF_AUDIO", ABNF.OPCODE_TEXT)
            
            # 等待完成或超时
            self.completed_event.wait(timeout=2.0)
            
            # 关闭连接
            if self.websocket:
                self.websocket.close()
            
            # 等待线程结束
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2)
                
            self.running = False
        except Exception as e:
            if self.on_error:
                self.on_error(f"Stop error: {str(e)}")
        finally:
            if self.on_close:
                self.on_close()


if __name__ == "__main__":
    def on_sentence_begin(msg):
        print("on_sentence_begin:", json.loads(msg))

    def on_sentence_end(msg):
        print("on_sentence_end:", json.loads(msg))

    def on_start(msg):
        print("on_start:", json.loads(msg))

    def on_result_changed(msg):
        print("on_result_changed:", json.loads(msg))

    def on_completed(msg):
        print("on_completed:", json.loads(msg))

    def on_error(msg):
        print("on_error:", json.loads(msg))

    def on_close():
        print("on_close")

    ws_sr = WhisperLiveTranscriber(
        endpoint="ws://localhost:10090",
        language="zh",
        # initial_prompt="下面是一些IT面试问题。",
        initial_prompt="下面是一些技术面试问题。",
        on_sentence_begin=on_sentence_begin,
        on_sentence_end=on_sentence_end,
        on_start=on_start,
        on_result_changed=on_result_changed,
        on_completed=on_completed,
        on_error=on_error,
        on_close=on_close,
    )
    ws_sr.start()
    import wave
    # wav_path="/root/workroot/opensources/funasr-runtime-resources/samples/audio/asr_example.wav"
    # wav_path = "../../../samples/audio/中英混杂技术名词-男.wav"
    wav_path = "./中英混杂技术名词-男.wav"
    #wav_path="/root/workroot/opensources/funasr-runtime-resources/samples/audio/asr_example.pcm"
    with wave.open(wav_path, "rb") as wav_file:
                params = wav_file.getparams()
                frames = wav_file.readframes(wav_file.getnframes())
                audio_bytes = bytes(frames)
    stride = int(60 * 10 / 10 / 1000 * 16000 * 2)
    chunk_num = (len(audio_bytes) - 1) // stride + 1
    # loop to send chunk
    TIMES = 1
    for j in range(TIMES):
        for i in range(chunk_num):
            beg = i * stride
            data = audio_bytes[beg:beg + stride]
            ws_sr.send_audio(data)
            print("Sent chunk:", i)
            time.sleep(0.05)
        # print("start sleep 10s")
        time.sleep(10)
    print("will stop")
    ws_sr.stop()
    print("stopped")