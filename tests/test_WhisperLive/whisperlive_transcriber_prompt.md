```from apps.audio.funasr_speech_trancriber import END_PUNCTUATIONS, FunASRSpeechTranscriber
from apps.audio.speaker_annotator import SpeakerAnnotator
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState
import nls
import time
import asyncio
import json
import traceback

from threading import Lock # TODO check

# from funasr import AutoModel
# import torch
from apps.audio.aliyun_nls_auth import FUNASR_APPKEY, token_manager, audio_usage_tracker #, get_access_token

import logging
from config import (
    SRC_LOG_LEVELS,
)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["AUDIO"])

AUDIO_INTERNAL_ERROR = "Internal error when sending audio"
AUDIO_USER_BUDGET_NOT_ENOUGH_ERROR = "Failed to transcribe audio because you account balance is not sufficient, please recharge."
# AUDIO_SERVER_RESOUCE_NOT_ENOUGH_ERROR = "Server running out of audio transcribe resources, please try again later."

URL = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
# URL = "wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1"
# URL = "wss://nls-gateway.aliyuncs.com/ws/v1"
# URL = "ws://nls-gateway-cn-shanghai-internal.aliyuncs.com:80/ws/v1"

# SampleRate=16000
# BitRate=16
# 每次发送指定二进制数据长度的数据帧，单位字节，16位pcm取值必须为2的整数倍，8位随意。
# 16位16khz的pcm 1秒有：16000hz*16位/8比特=32000字节的数据，默认配置3200字节每秒发送大约10次
# PerFrameSize=3200 

precision = 4

# interviewee_spk_embedding = torch.tensor([[ 0.3490, -0.8557,  1.3099,  1.7994,  0.9181, -0.4321,  1.1471,  0.7920,
#           0.4836, -0.7618, -0.3851,  0.0086, -1.2008,  1.1426,  1.3981, -2.6441,
#          -0.0192, -0.0465, -1.0596, -0.2632,  0.4163, -3.4917, -0.7099, -1.1280,
#          -0.2510,  0.2414,  1.3904, -0.1866,  1.8514, -2.2951, -1.1105, -1.5920,
#           0.8469,  1.5451, -0.5729, -0.5412,  0.4049, -0.8136,  1.5685, -0.7517,
#           0.8238,  1.4476, -0.6843, -1.0967,  1.0473, -1.5515,  1.5384,  0.6609,
#           1.7859,  0.8731,  0.1816,  0.2160,  1.0084,  0.0221, -0.9063, -0.5129,
#           1.5144, -0.3319,  0.8385, -1.4368, -0.3501,  1.5388,  0.9537, -1.2098,
#           1.6867,  0.8655, -1.4847,  0.4344,  0.4424, -0.4636, -0.5684,  0.9449,
#          -2.3738, -0.4810, -0.9811,  1.2695,  1.3298, -2.0937, -0.0773,  0.6623,
#           0.3236,  0.6093,  0.1519, -0.1525,  0.3088, -1.0113, -0.6500, -1.8507,
#           0.5418,  1.1588,  0.8548, -0.7341, -0.2101, -0.5052,  0.6879,  0.4991,
#          -1.1363, -0.0884, -0.8005, -0.6668,  0.3867, -0.8531,  0.3135,  0.3606,
#          -0.7785,  0.0765, -2.9924, -0.3698,  1.5121, -0.1429, -1.1050,  0.3264,
#           1.2279, -1.9989,  1.2535, -0.2557, -2.1247, -0.2526,  0.6476, -0.6763,
#          -0.9398, -0.0532,  0.2633,  1.0180,  0.8914, -0.4000,  0.9657, -1.8915,
#          -0.8787, -0.8566,  0.5988,  0.7630, -0.0933,  1.0303,  2.9155, -0.3794,
#          -0.6012, -0.2799,  1.1247, -0.0911,  0.1609,  0.4285,  3.4473,  0.0041,
#          -0.0965,  2.2729,  0.4694, -0.1362, -0.4178,  1.4197,  0.4858,  0.5969,
#          -1.1854, -0.3463,  0.2626,  1.0203,  0.5193, -2.0661, -0.8096,  0.1149,
#           0.1069,  1.9632,  0.2765,  1.6805,  0.2470,  0.3458, -0.6960, -0.0359,
#           1.0913, -0.9482, -0.0682,  1.4116,  1.3037,  0.3083, -0.4993, -1.7365,
#           0.0863, -1.2328,  0.2504, -0.4596,  1.4052, -0.9903, -1.0625, -2.3904,
#          -0.4057, -0.8973,  1.9315, -1.6916,  1.3337,  0.2078,  1.2506,  1.8627]])

# similarity_function = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
# spk_model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", disable_pbar=True, device='cpu')

class WebSocketSr:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        # TODO * @jjm what if token is None? put this in an isolated init() method
        token, expireTime, appkey, ak_id = token_manager.getToken()
        # token, expireTime, appkey, ak_id = None, None, None, None
        self.use_funasr = appkey == FUNASR_APPKEY
        log.info(f"getting new TOKEN: {token}, expireTime: {expireTime}, appkey: {appkey}, ak_id: {ak_id}")
        self.ak_id = ak_id
        if not self.ak_id:
            self.sr = None
        elif not self.use_funasr:
            self.sr = nls.NlsSpeechTranscriber(
                url=URL,
                token=token,
                appkey=appkey,
                on_sentence_begin=self.on_sentence_begin,
                on_sentence_end=self.on_sentence_end,
                on_start=self.on_start,
                on_result_changed=self.on_result_changed,
                on_completed=self.on_completed,
                on_error=self.on_error,
                on_close=self.on_close,
                callback_args=['thread-tmp0']
            )
            # nls.enableTrace(True)
        else:
            self.sr = FunASRSpeechTranscriber(
                # host="127.0.0.1",
                # port="10096",
                # is_ssl=False,
                endpoint=token,
                on_sentence_begin=self.on_sentence_begin,
                on_sentence_end=self.on_sentence_end,
                on_start=self.on_start,
                on_result_changed=self.on_result_changed,
                on_completed=self.on_completed,
                on_error=self.on_error,
                on_close=self.on_close,
                # callback_args=['thread-tmp0']
            )
        self.queue = asyncio.Queue()
        self.consumer_task = asyncio.create_task(self.consume_messages())
        self.consumer_task_until_empty = asyncio.create_task(self.consume_messages_until_empty())
        self.total_audio_length_ms = 0
        self.audio_cache = b""
        self.audio_cache_lock = Lock()
        self.audio_cache_start_time_ms = 0
        self.last_response_time = time.time()
        self.last_crop_time = time.time()
        self.current_sentence_begin_time_ms = 0
        self.current_sentence_end_time_ms = 0
        self.return_voice_embedding = False
        self.user_voice_signature = None
        self.last_speaker_name = "user"
        self.is_last_sentence_appended_to_speaker_info_queue = False
        self.stopped = False

        self.speaker_annotator = SpeakerAnnotator()

    def set_user_voice_signature(self, user_voice_signature):
        self.user_voice_signature = user_voice_signature
        self.speaker_annotator.set_user_embedding(user_voice_signature)
    
    def on_sentence_begin(self, message, *args):
        with self.audio_cache_lock:
            self.last_response_time = time.time()
        message_json = json.loads(message)
        self.current_sentence_begin_time_ms = int(message_json['payload']['time'])
        log.debug(f"on_sentence_begin put in queue: {message}")
        self.queue.put_nowait(("sentence_begin", message))

    def on_sentence_end(self, message, *args):
        try:
            message_json = json.loads(message)
            log.info(f"on_sentence_end message: {message_json}")
            self.current_sentence_end_time_ms = int(message_json['payload']['time'])
            speaker_name = "user"
            similarity_score = 0.0

            need_generate_voice_signature = self.user_voice_signature is not None or self.return_voice_embedding
            if need_generate_voice_signature:
                # TODO * @jjm this is just a temporary fix for start=-1, end=-1
                if self.current_sentence_end_time_ms < self.audio_cache_start_time_ms:
                    log.warn(f"on_sentence_end current_sentence_end_time_ms < audio_cache_start_time_ms, will return")
                    return
                gap_to_last_sentence_ms = self.current_sentence_begin_time_ms - self.audio_cache_start_time_ms
                start_offset = int(gap_to_last_sentence_ms * 32)  # 32b per ms, for 16000Hz, 16 bits, 1 channel
                end_offset = int((self.current_sentence_end_time_ms - self.audio_cache_start_time_ms) * 32)
                # log.debug(f"on_sentence_end start_offset: {start_offset}, end_offset: {end_offset}")
                current_sentence_time_ms = self.current_sentence_end_time_ms - self.current_sentence_begin_time_ms
                if (current_sentence_time_ms >= 200 and gap_to_last_sentence_ms >= 1500) or current_sentence_time_ms >= 400:
                # split the sentece audio chunk from audio_cache, 
                # based on current_sentence_end_time_ms, current_sentence_begin_time_ms, and audio_cache_start_time_ms
                    sentence_audio_chunk = self.audio_cache[start_offset:end_offset]
                    embedding_2d_np_array = None
                    if self.return_voice_embedding or self.user_voice_signature is not None:
                        embedding_2d_np_array = self.speaker_annotator.get_embedding(sentence_audio_chunk)
                    log.debug(f"get_embedding res: {embedding_2d_np_array}")
                    # time1 = time.time()
                    # res = spk_model.generate(sentence_audio_chunk)
                    # time2 = time.time()
                    # log.debug(f"spk_model.generate time: {time2 - time1}, audio length ms: {self.current_sentence_end_time_ms - self.current_sentence_begin_time_ms}")
                    # embedding_2d_np_array = res[0]['spk_embedding']
                    if self.return_voice_embedding:
                        # formatted_embedding_2d_np_array_json = [[round(x.item(), precision) for x in row] for row in embedding_2d_np_array]
                        formatted_embedding_2d_np_array_json = [[round(x.item(), precision) for x in row] for row in embedding_2d_np_array]
                        formatted_embedding_2d_np_array_json_str = json.dumps(formatted_embedding_2d_np_array_json)
                        log.debug('generated voice signature:', formatted_embedding_2d_np_array_json_str)
                        message_json['payload']['voice_embedding'] = formatted_embedding_2d_np_array_json

                    if self.user_voice_signature is not None:
                        text = message_json['payload']['result']
                        audio_length_ms = self.current_sentence_end_time_ms - self.current_sentence_begin_time_ms
                        speaker_name, similarity_score, self.is_last_sentence_appended_to_speaker_info_queue = self.speaker_annotator.annotate(embedding_2d_np_array, text, audio_length_ms)
                        # similarity_score = similarity_function(self.user_voice_signature, embedding_2d_np_array)
                        # speaker_name = "user" if similarity_score >= same_people_similarity_score_threshold else "other"
                else: # too short sentence, and too close to last sentence, use last speaker name | or just punct
                    speaker_name = self.last_speaker_name
                    text = message_json['payload']['result']
                    log.info(f"speaker text: {text}, speaker_name: {speaker_name}, current_sentence_time_ms: {current_sentence_time_ms}, gap_to_last_sentence_ms: {gap_to_last_sentence_ms}, set to last speaker name")
                    if text in END_PUNCTUATIONS and self.is_last_sentence_appended_to_speaker_info_queue:
                        # append the punct to the last sentence for better llm speaker annotation
                        self.speaker_annotator.append_punct_to_last_sentence(text)

                    self.is_last_sentence_appended_to_speaker_info_queue = False
                # in case memory leak, clear audio_cache
                with self.audio_cache_lock:
                    self.last_response_time = time.time()
                    # log.debug(f"before audio_cache size: {len(self.audio_cache)}")
                    self.audio_cache = self.audio_cache[end_offset:]
                    # log.debug(f"after audio_cache size: {len(self.audio_cache)}")
                    self.audio_cache_start_time_ms = self.current_sentence_end_time_ms
                # 保存为文件
                # filename = f"sentence_{self.current_sentence_begin_time_ms}_{self.current_sentence_end_time_ms}.pcm"
                # save_pcm_data(sentence_audio_chunk, filename)
            message_json['payload']['speaker'] = {"name": speaker_name, "sim": round(similarity_score, precision)}
            self.last_speaker_name = speaker_name
            updated_message = json.dumps(message_json, ensure_ascii=False)
            log.debug(f"on_sentence_end put in queue: {updated_message}")
            self.queue.put_nowait(("sentence_end", updated_message))
        except json.JSONDecodeError:
            log.info(f"Failed to decode message: {message}")
            return
        except Exception as e:
            log.error(f"Failed to handle message: {traceback.format_exc()}")
            # log.error(f"Failed to handle message: {e}")
            return
    
    def on_start(self, message, *args):
        log.debug(f"on_start put in queue: {message}")
        with self.audio_cache_lock:
            self.last_response_time = time.time()
        # asyncio.create_task(self.websocket.send_text("Start: {}".format(message)))
        self.queue.put_nowait(("start", message))
    
    def on_error(self, message, *args):
        log.error(f"on_error put in queue: {message}")
        # asyncio.create_task(self.websocket.send_text("Error: {}".format(message)))
        self.queue.put_nowait(("error", message))
    
    def on_close(self, *args):
        log.error(f"on_close put in queue: {args}")
        # asyncio.create_task(self.websocket.send_text("Close: {}".format(args)))
        self.queue.put_nowait(("close", AUDIO_INTERNAL_ERROR))
    
    def on_result_changed(self, message, *args):
        log.debug(f"on_result_changed put in queue: {message}")
        with self.audio_cache_lock:
            self.last_response_time = time.time()
        # asyncio.create_task(self.websocket.send_text("Result Changed: {}".format(message)))
        self.queue.put_nowait(("result_changed", message))
    
    def on_completed(self, message, *args):
        log.debug(f"on_completed put in queue: {message}")
        # asyncio.create_task(self.websocket.send_text("Completed: {}".format(message)))
        self.queue.put_nowait(("completed", message))

    def start_sr(self):
        with self.audio_cache_lock:
            self.last_response_time = time.time()
        if self.ak_id is None:
            log.error("ak_id is None, will not start sr")
            return False
        if not self.use_funasr:
            self.sr.start(aformat="pcm", 
                enable_intermediate_result=True,
                enable_punctuation_prediction=True,
                enable_inverse_text_normalization=True,
                # ex={"hello":123}
                # ex={"enable_semantic_sentence_detection": True},
                # ex={"enable_words": True},
                # ex={"vocabulary_id": "4f75339128f947a29db9670c4643a653"},
                # ex={"vocabulary_id": "ceab523f58454379a4ba116686cf6541"}, # large vocabulary
                )
        else:
            self.sr.start()
        # asyncio.create_task(self.websocket.send_text("{}: session start".format('WS')))
        self.queue.put_nowait(("start_sr", None))
        return True

    def send_audio(self, audio_data, user_id=None):
        if not self.sr:
            log.error("sr is None, will not send audio")
            return False, AUDIO_INTERNAL_ERROR
        if user_id is None:
            log.error("user_id is None, will not send audio")
            return False, AUDIO_INTERNAL_ERROR
        if self.stopped:
            log.error("asr already stopped, will not send audio")
            return False, AUDIO_INTERNAL_ERROR
        self.sr.send_audio(audio_data)
        self.total_audio_length_ms += len(audio_data) / 32  # 32B per ms, for 16000Hz, 16 bits, 1 channel
        balance_sufficient = audio_usage_tracker.add_active_user(user_id)
        if not balance_sufficient:
            log.error("balance not sufficient, will stop_sr")
            return False, AUDIO_USER_BUDGET_NOT_ENOUGH_ERROR
        log.debug(f"total_audio_length_ms: {self.total_audio_length_ms}")
        # save audio data
        need_generate_voice_signature = self.user_voice_signature is not None or self.return_voice_embedding
        if need_generate_voice_signature:
            #TODO * @jjm handle audio_cache size increase when user is not speaking, about 2MB per minute
            timestamp = time.time()
            with self.audio_cache_lock:
                self.audio_cache += audio_data
                # log.info(f"send_audio finish, audio_cache size: {len(self.audio_cache)}")
                if timestamp - self.last_response_time > 60 and timestamp - self.last_crop_time > 10:
                    # keep the last 30s audio data
                    keep_audio_ms = 30000
                    original_audio_length = len(self.audio_cache)
                    self.audio_cache = self.audio_cache[-keep_audio_ms * 32:]
                    tmp = self.audio_cache_start_time_ms
                    self.audio_cache_start_time_ms += original_audio_length / 32 - keep_audio_ms
                    self.last_crop_time = timestamp
                    # log.info(f"[test2]send_audio long time since last_response_time: {timestamp - self.last_response_time}, original start time ms: {tmp}, new start time ms: {self.audio_cache_start_time_ms}, original audio length: {original_audio_length}, new audio length: {len(self.audio_cache)}")
        return True, "OK"

    
    def stop_sr(self):
        if self.stopped:
            log.debug("already stopped")
            return
        self.sr.stop()
        # TODO * @jjm assure release token.
        token_manager.releaseToken(self.ak_id)
        # asyncio.create_task(self.websocket.send_text("SR session stopped."))
        log.debug("stop_sr put in queue: None")
        self.queue.put_nowait(("stop_sr", None))
        self.stopped = True

    async def process_message(self, event_type, message):
        # log.error(f'self.websocket.client_state: {self.websocket.client_state}, self.websocket.application_state: {self.websocket.application_state}')
        if self.stopped:
            log.debug('already stopped when process_message, will not send message.')
            return
        # if self.websocket.client_state == WebSocketState.DISCONNECTED:
        #     print('[test]websocket is closed when process_message, will stop_sr.')
        #     self.stop_sr()
        #     return
        try:
            log.debug(f"consume_messages event_type: {event_type}, message: {message}")
            message_json = None
            if event_type in ["start_sr", "stop_sr"]:
                return # do nothing
            elif event_type in ["close"]:
                message_json = {"event_type": event_type, "message": message}
            else: # ["start", "result_changed", "completed", "sentence_begin", "sentence_end", "error"]
                message_json = json.loads(message)
                message_json['event_type'] = event_type
            message = json.dumps(message_json, ensure_ascii=False)
            await self.websocket.send_text(f"{message}")
        except Exception as e:
            log.error(f"Error process_message, will stop_sr, error: {e}, detail: {traceback.format_exc()}")
            self.stop_sr()

    async def consume_messages(self):
        log.debug('consume_messages start')
        while True:
            event_type, message = await self.queue.get()
            await self.process_message(event_type, message)
            if event_type in ["close"]:
                break
        log.debug('consume_messages end')
    
    # consume messages until queue is empty, without blocking
    async def consume_messages_until_empty(self):
        # 打印出方法开始的提示信息
        log.debug('consume_messages_until_empty start')
        log.debug('queue size:', self.queue.qsize())
        
        # 首先开始一个循环尝试从队列中获取项目，直到队列为空
        while True:
            try:
                # 尝试立即从队列中获取一个项目
                event_type, message = self.queue.get_nowait()
                await self.process_message(event_type, message)
                # print('consume_messages_until_empty event_type:', event_type, 'message:', message)
                if event_type in ["close"]:
                    break

            except asyncio.QueueEmpty as e:
                # 如果队列为空，抛出QueueEmpty异常，打破循环结束方法
                log.debug(f"QueueEmpty e:{e}")
                break
            except Exception as e:
                # 如果发生其他异常，打印异常信息
                log.error(f"consume_messages_until_empty failed with Exception e:{e}")
                break

        # 打印方法结束的提示信息
        log.debug('consume_messages_until_empty end')
```
上面是websocket_asr.py代码，支持aliyun sdk nls.NlsSpeechTranscriber和使用funasr参考该sdk改造的sdk FunASRSpeechTranscriber。下面是funasr_speech_trancriber.py代码：
```
import ssl
import json
import numpy as np
from websocket import create_connection, ABNF
from queue import Queue
from threading import Thread, Event
import time
import traceback
import re

import logging
from config import (
    SRC_LOG_LEVELS,
)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["AUDIO"])

END_PUNCTUATIONS = ["。", "？", "！", ""]

class FunASRSpeechTranscriber:
    def __init__(self,
                #  host="127.0.0.1",
                #  port="10096",
                #  is_ssl=False,
                 endpoint,
                 on_sentence_begin=None,
                 on_sentence_end=None,
                 on_start=None,
                 on_result_changed=None,
                 on_completed=None,
                 on_error=None,
                 on_close=None):
        self.endpoint = endpoint
        self.is_ssl = endpoint.startswith("wss")
        self.on_sentence_begin = on_sentence_begin
        self.on_sentence_end = on_sentence_end
        self.on_start = on_start
        self.on_result_changed = on_result_changed
        self.on_completed = on_completed
        self.on_error = on_error
        self.on_close = on_close

        # self.msg_queue = Queue()
        self.websocket = None
        self.thread_msg = None
        # self.is_recognizing = False
        self.stop_event = Event()
        self.complete_event = Event()

        self.current_sentence = ""
        self.current_end_time = -1
        self.current_start_time = -1
        self.has_unfinished_sentence = False

    def connect(self):
        try:
            if self.is_ssl:
                ssl_context = ssl.SSLContext()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                uri = self.endpoint # "wss://{}:{}".format(self.host, self.port)
                ssl_opt = {"cert_reqs": ssl.CERT_NONE}
            else:
                uri = self.endpoint # "ws://{}:{}".format(self.host, self.port)
                ssl_context = None
                ssl_opt = None

            self.websocket = create_connection(uri, ssl=ssl_context, sslopt=ssl_opt)
            log.info(f"Connected to: {uri}")

            self.thread_msg = Thread(target=self.thread_rec_msg)
            

        except Exception as e:
            if self.on_error:
                self.on_error("Failed to connect to the server: {}".format(e))

    def _remove_useless_space(self, text):
        # 定义汉字和中文标点符号的正则表达式
        chinese_pattern = r'[\u4e00-\u9fa5]'  # 汉字范围
        chinese_punctuation_pattern = r'[\u3000-\u303f\uff00-\uffa0]'  # 中文标点符号范围

        # 使用正则表达式查找所有空格
        def is_useless_space(match):
            space = match.group(0)
            # 检查空格两边的字符
            left_char = text[match.start() - 1] if match.start() > 0 else None
            right_char = text[match.end()] if match.end() < len(text) else None

            # 如果空格两边一边或两边为汉字或中文标点符号，则删除该空格
            if (left_char and re.match(chinese_pattern + '|' + chinese_punctuation_pattern, left_char)) or \
            (right_char and re.match(chinese_pattern + '|' + chinese_punctuation_pattern, right_char)):
                return ''  # 删除空格
            else:
                return space  # 保留空格

        # 替换文本中的空格
        result = re.sub(r' ', is_useless_space, text)
        return result

    
    def __handle_msg(self, msg_raw):
        log.info(f"Received message: {msg_raw}")
        try:
            msg = json.loads(msg_raw)
            wav_name = msg.get("wav_name", "demo")
            text = msg["text"]
            offline_msg_done = msg.get("is_final", False)
            timestamp = msg.get("timestamp", "")
            # if "mode" not in msg:
            #     return
            mode = msg.get("mode", None)
            if mode == "2pass-online":
                if self.on_result_changed:
                    self.on_result_changed(msg_raw)
            elif mode == "2pass-offline":
                sents_length = len(msg["stamp_sents"])
                # for sent in msg["stamp_sents"]:
                for i in range(sents_length):
                    sent = msg["stamp_sents"][i]
                    punc = sent["punc"]
                    # text_seg = sent["text_seg"].replace(" ", "")
                    # text_seg = sent["text_seg"]
                    text_seg = self._remove_useless_space(sent["text_seg"]) # fix no space between English words
                    punc_start = sent["start"]
                    punc_end = sent["end"]
                    self.current_sentence += text_seg
                    self.current_sentence += punc
                    # TODO * @jjm just a workaround to escape the first punc
                    # if i == 0 and punc_start == -1 and punc_end == -1 and text_seg == "":
                    #     # self.current_sentence += " "
                    #     self.current_sentence += punc
                    # else:
                    #     self.current_sentence += punc
                    log.debug(f" before punc_start: {punc_start}, punc_end: {punc_end}, current_start_time: {self.current_start_time}, current_end_time: {self.current_end_time}")
                    send_single_punc = False
                    if i == 0 and punc_start == -1 and punc_end == -1 and text_seg == "": # punctuation at the beginning
                        punc_start = self.current_end_time
                        punc_end = self.current_end_time
                        send_single_punc = True
                    else:
                        if punc_start == -1:
                            punc_start = self.current_start_time
                        if punc_end == -1:
                            punc_end = self.current_end_time
                    self.current_end_time = punc_end
                    log.debug(f" after punc_start: {punc_start}, punc_end: {punc_end}, current_start_time: {self.current_start_time}, current_end_time: {self.current_end_time}")
                    if self.current_start_time == -1 and punc_start != -1:
                        self.current_start_time = punc_start
                        if self.on_sentence_begin:
                            aliyun_format_json = {
                                "header": {
                                    "name":"SentenceBegin"
                                },
                                "payload": {
                                    "time": punc_start
                                }
                            }
                            aliyun_format_str = json.dumps(aliyun_format_json, ensure_ascii=False)
                            self.on_sentence_begin(aliyun_format_str)
                            self.has_unfinished_sentence = True
                    if punc in END_PUNCTUATIONS or (offline_msg_done and i == sents_length - 1) or send_single_punc:
                        if self.on_sentence_end:
                            aliyun_format_json = {
                                "header": {
                                    "name":"SentenceEnd"
                                },
                                "payload": {
                                    "time": punc_end,
                                    "result": self.current_sentence
                                }
                            }
                            aliyun_format_str = json.dumps(aliyun_format_json, ensure_ascii=False)
                            self.on_sentence_end(aliyun_format_str)
                            self.has_unfinished_sentence = False
                        self.current_sentence = ""
                        self.current_start_time = -1
            else: # mode is None
                if offline_msg_done:
                    text = msg.get("text", "")
                    self.current_sentence += text
                    if self.on_sentence_end and self.has_unfinished_sentence:
                        punc_end = self.current_end_time
                        aliyun_format_json = {
                            "header": {
                                "name":"SentenceEnd"
                            },
                            "payload": {
                                "time": punc_end,
                                "result": self.current_sentence
                            }
                        }
                        aliyun_format_str = json.dumps(aliyun_format_json, ensure_ascii=False)
                        self.on_sentence_end(aliyun_format_str)
                        self.has_unfinished_sentence = False
                

            if offline_msg_done: # TODO check not ever received?
                self.complete_event.set()
                if self.on_completed:
                    self.on_completed(msg_raw)
        except json.JSONDecodeError:
            log.info(f"Failed to decode message: {msg_raw}")
            return
        except Exception as e:
            log.error(f"Failed to handle message: {traceback.format_exc()}")
            # log.error(f"Failed to handle message: {e}")
            return


    def thread_rec_msg(self):
        # while self.is_recognizing:
        # Note: only close websocket will stop the thread
        while not self.stop_event.is_set():
            try:
                msg = self.websocket.recv()
                if msg is None or len(msg) == 0:
                    continue
                self.__handle_msg(msg)
            except Exception as e:
                # log.error(f"Failed to receive message: {traceback.format_exc()}")
                log.error(f"Failed to receive message: {e}")
                break
        log.info(f"Thread stopped")
            # self.msg_queue.put(msg)

    def start(self, mode="2pass", chunk_size=[5, 10, 5], chunk_interval=10, wav_name="default"):
        # TODO connect here
        self.connect()
        message = json.dumps({
            "mode": mode,
            "chunk_size": chunk_size,
            "encoder_chunk_look_back": 4,
            "decoder_chunk_look_back": 1,
            "chunk_interval": chunk_interval,
            "wav_name": wav_name,
            "is_speaking": True
        })
        self.websocket.send(message)
        log.info(f"Send start message: {message}")
        self.is_recognizing = True
        self.thread_msg.start()
        if self.on_start:
            self.on_start(message)

    def send_audio(self, audio_data):
        self.websocket.send(audio_data, ABNF.OPCODE_BINARY)

    def stop(self):
        message = json.dumps({"is_speaking": False})
        self.websocket.send(message)
        log.debug(f"Send stop message: {message}")
        timeout = time.time() + 5
        while not self.complete_event.is_set() and time.time() < timeout:
            time.sleep(0.1)
        self.is_recognizing = False
        self.websocket.close()
        self.stop_event.set()
        if self.thread_msg.is_alive():
            self.thread_msg.join()

        # self.websocket.close()
        if self.on_close:
            self.on_close()
```
现在我想增加对WhisperLive的支持，需要参考funasr_speech_trancriber.py改造下面WhisperLIve库的client.py中的TranscriptionClient代码，使其接口与aliyun sdk相近。
```
import os
import shutil
import wave

import logging
import numpy as np
import pyaudio
import threading
import json
import websocket
import uuid
import time
import av
import whisper_live.utils as utils


class Client:
    """
    Handles communication with a server using WebSocket.
    """
    INSTANCES = {}
    END_OF_AUDIO = "END_OF_AUDIO"

    def __init__(
        self,
        host=None,
        port=None,
        lang=None,
        translate=False,
        model="small",
        srt_file_path="output.srt",
        use_vad=True,
        use_wss=False,
        log_transcription=True,
        max_clients=4,
        max_connection_time=600,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        transcription_callback=None,
        initial_prompt=None,
    ):
        """
        Initializes a Client instance for audio recording and streaming to a server.

        If host and port are not provided, the WebSocket connection will not be established.
        When translate is True, the task will be set to "translate" instead of "transcribe".
        he audio recording starts immediately upon initialization.

        Args:
            host (str): The hostname or IP address of the server.
            port (int): The port number for the WebSocket server.
            lang (str, optional): The selected language for transcription. Default is None.
            translate (bool, optional): Specifies if the task is translation. Default is False.
            model (str, optional): The whisper model to use (e.g., "small", "medium", "large"). Default is "small".
            srt_file_path (str, optional): The file path to save the output SRT file. Default is "output.srt".
            use_vad (bool, optional): Whether to enable voice activity detection. Default is True.
            log_transcription (bool, optional): Whether to log transcription output to the console. Default is True.
            max_clients (int, optional): Maximum number of client connections allowed. Default is 4.
            max_connection_time (int, optional): Maximum allowed connection time in seconds. Default is 600.
            send_last_n_segments (int, optional): Number of most recent segments to send to the client. Defaults to 10.
            no_speech_thresh (float, optional): Segments with no speech probability above this threshold will be discarded. Defaults to 0.45.
            clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
            same_output_threshold (int, optional): Number of repeated outputs before considering it as a valid segment. Defaults to 10.
            transcription_callback (callable, optional): A callback function to handle transcription results. Default is None.
        """
        self.recording = False
        self.task = "transcribe"
        self.uid = str(uuid.uuid4())
        self.waiting = False
        self.last_response_received = None
        self.disconnect_if_no_response_for = 15
        self.language = lang
        self.model = model
        self.server_error = False
        self.srt_file_path = srt_file_path
        self.use_vad = use_vad
        self.use_wss = use_wss
        self.last_segment = None
        self.last_received_segment = None
        self.log_transcription = log_transcription
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold
        self.transcription_callback = transcription_callback
        self.initial_prompt = initial_prompt

        if translate:
            self.task = "translate"

        self.audio_bytes = None

        if host is not None and port is not None:
            socket_protocol = 'wss' if self.use_wss else "ws"
            socket_url = f"{socket_protocol}://{host}:{port}"
            self.client_socket = websocket.WebSocketApp(
                socket_url,
                on_open=lambda ws: self.on_open(ws),
                on_message=lambda ws, message: self.on_message(ws, message),
                on_error=lambda ws, error: self.on_error(ws, error),
                on_close=lambda ws, close_status_code, close_msg: self.on_close(
                    ws, close_status_code, close_msg
                ),
            )
        else:
            print("[ERROR]: No host or port specified.")
            return

        Client.INSTANCES[self.uid] = self

        # start websocket client in a thread
        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        self.transcript = []
        print("[INFO]: * recording")

    def handle_status_messages(self, message_data):
        """Handles server status messages."""
        status = message_data["status"]
        if status == "WAIT":
            self.waiting = True
            print(f"[INFO]: Server is full. Estimated wait time {round(message_data['message'])} minutes.")
        elif status == "ERROR":
            print(f"Message from Server: {message_data['message']}")
            self.server_error = True
        elif status == "WARNING":
            print(f"Message from Server: {message_data['message']}")

    def process_segments(self, segments):
        """Processes transcript segments."""
        text = []
        for i, seg in enumerate(segments):
            if not text or text[-1] != seg["text"]:
                text.append(seg["text"])
                if i == len(segments) - 1 and not seg.get("completed", False):
                    self.last_segment = seg
                elif (self.server_backend == "faster_whisper" and seg.get("completed", False) and
                      (not self.transcript or
                        float(seg['start']) >= float(self.transcript[-1]['end']))):
                    self.transcript.append(seg)
        # update last received segment and last valid response time
        if self.last_received_segment is None or self.last_received_segment != segments[-1]["text"]:
            self.last_response_received = time.time()
            self.last_received_segment = segments[-1]["text"]

        # call the transcription callback if provided
        if self.transcription_callback and callable(self.transcription_callback):
            try:
                self.transcription_callback(" ".join(text), segments) # string, list
            except Exception as e:
                print(f"[WARN] transcription_callback raised: {e}")
            return
        
        if self.log_transcription:
            # Truncate to last 3 entries for brevity.
            # text = text[-3:]
            # utils.clear_screen()
            # utils.print_transcript(text)
            print("[test]result text:", "".join(text))

    def on_message(self, ws, message):
        """
        Callback function called when a message is received from the server.

        It updates various attributes of the client based on the received message, including
        recording status, language detection, and server messages. If a disconnect message
        is received, it sets the recording status to False.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.
            message (str): The received message from the server.

        """
        message = json.loads(message)
        print("received message:", message)

        if self.uid != message.get("uid"):
            print("[ERROR]: invalid client uid")
            return

        if "status" in message.keys():
            self.handle_status_messages(message)
            return

        if "message" in message.keys() and message["message"] == "DISCONNECT":
            print("[INFO]: Server disconnected due to overtime.")
            self.recording = False

        if "message" in message.keys() and message["message"] == "SERVER_READY":
            self.last_response_received = time.time()
            self.recording = True
            self.server_backend = message["backend"]
            print(f"[INFO]: Server Running with backend {self.server_backend}")
            return

        if "language" in message.keys():
            self.language = message.get("language")
            lang_prob = message.get("language_prob")
            print(
                f"[INFO]: Server detected language {self.language} with probability {lang_prob}"
            )
            return

        if "segments" in message.keys():
            self.process_segments(message["segments"])

    def on_error(self, ws, error):
        print(f"[ERROR] WebSocket Error: {error}")
        self.server_error = True
        self.error_message = error

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[INFO]: Websocket connection closed: {close_status_code}: {close_msg}")
        self.recording = False
        self.waiting = False

    def on_open(self, ws):
        """
        Callback function called when the WebSocket connection is successfully opened.

        Sends an initial configuration message to the server, including client UID,
        language selection, and task type.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.

        """
        print("[INFO]: Opened connection")
        ws.send(
            json.dumps(
                {
                    "uid": self.uid,
                    "language": self.language,
                    "task": self.task,
                    "model": self.model,
                    "use_vad": self.use_vad,
                    "max_clients": self.max_clients,
                    "max_connection_time": self.max_connection_time,
                    "send_last_n_segments": self.send_last_n_segments,
                    "no_speech_thresh": self.no_speech_thresh,
                    "clip_audio": self.clip_audio,
                    "same_output_threshold": self.same_output_threshold,
                    "initial_prompt": self.initial_prompt,
                }
            )
        )

    def send_packet_to_server(self, message):
        """
        Send an audio packet to the server using WebSocket.

        Args:
            message (bytes): The audio data packet in bytes to be sent to the server.

        """
        try:
            self.client_socket.send(message, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(e)

    def close_websocket(self):
        """
        Close the WebSocket connection and join the WebSocket thread.

        First attempts to close the WebSocket connection using `self.client_socket.close()`. After
        closing the connection, it joins the WebSocket thread to ensure proper termination.

        """
        try:
            self.client_socket.close()
        except Exception as e:
            print("[ERROR]: Error closing WebSocket:", e)

        try:
            self.ws_thread.join()
        except Exception as e:
            print("[ERROR:] Error joining WebSocket thread:", e)

    def get_client_socket(self):
        """
        Get the WebSocket client socket instance.

        Returns:
            WebSocketApp: The WebSocket client socket instance currently in use by the client.
        """
        return self.client_socket

    def write_srt_file(self, output_path="output.srt"):
        """
        Writes out the transcript in .srt format.

        Args:
            message (output_path, optional): The path to the target file.  Default is "output.srt".

        """
        if self.server_backend == "faster_whisper":
            if not self.transcript and self.last_segment is not None:
                self.transcript.append(self.last_segment)
            elif self.last_segment and self.transcript[-1]["text"] != self.last_segment["text"]:
                self.transcript.append(self.last_segment)
            utils.create_srt_file(self.transcript, output_path)

    def wait_before_disconnect(self):
        """Waits a bit before disconnecting in order to process pending responses."""
        assert self.last_response_received
        while time.time() - self.last_response_received < self.disconnect_if_no_response_for:
            continue


class TranscriptionTeeClient:
    """
    Client for handling audio recording, streaming, and transcription tasks via one or more
    WebSocket connections.

    Acts as a high-level client for audio transcription tasks using a WebSocket connection. It can be used
    to send audio data for transcription to one or more servers, and receive transcribed text segments.
    Args:
        clients (list): one or more previously initialized Client instances

    Attributes:
        clients (list): the underlying Client instances responsible for handling WebSocket connections.
    """
    def __init__(self, clients, save_output_recording=False, output_recording_filename="./output_recording.wav", mute_audio_playback=False):
        self.clients = clients
        if not self.clients:
            raise Exception("At least one client is required.")
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 60000
        self.save_output_recording = save_output_recording
        self.output_recording_filename = output_recording_filename
        self.mute_audio_playback = mute_audio_playback
        self.frames = b""
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
        except OSError as error:
            print(f"[WARN]: Unable to access microphone. {error}")
            self.stream = None

    def __call__(self, audio=None, rtsp_url=None, hls_url=None, save_file=None):
        """
        Start the transcription process.

        Initiates the transcription process by connecting to the server via a WebSocket. It waits for the server
        to be ready to receive audio data and then sends audio for transcription. If an audio file is provided, it
        will be played and streamed to the server; otherwise, it will perform live recording.

        Args:
            audio (str, optional): Path to an audio file for transcription. Default is None, which triggers live recording.

        """
        assert sum(
            source is not None for source in [audio, rtsp_url, hls_url]
        ) <= 1, 'You must provide only one selected source'

        print("[INFO]: Waiting for server ready ...")
        for client in self.clients:
            while not client.recording:
                if client.waiting or client.server_error:
                    self.close_all_clients()
                    return

        print("[INFO]: Server Ready!")
        if hls_url is not None:
            self.process_hls_stream(hls_url, save_file)
        elif audio is not None:
            resampled_file = utils.resample(audio)
            self.play_file(resampled_file)
        elif rtsp_url is not None:
            self.process_rtsp_stream(rtsp_url)
        else:
            self.record()

    def close_all_clients(self):
        """Closes all client websockets."""
        for client in self.clients:
            client.close_websocket()

    def write_all_clients_srt(self):
        """Writes out .srt files for all clients."""
        for client in self.clients:
            client.write_srt_file(client.srt_file_path)

    def multicast_packet(self, packet, unconditional=False):
        """
        Sends an identical packet via all clients.

        Args:
            packet (bytes): The audio data packet in bytes to be sent.
            unconditional (bool, optional): If true, send regardless of whether clients are recording.  Default is False.
        """
        for client in self.clients:
            if (unconditional or client.recording):
                client.send_packet_to_server(packet)

    def play_file(self, filename):
        """
        Play an audio file and send it to the server for processing.

        Reads an audio file, plays it through the audio output, and simultaneously sends
        the audio data to the server for processing. It uses PyAudio to create an audio
        stream for playback. The audio data is read from the file in chunks, converted to
        floating-point format, and sent to the server using WebSocket communication.
        This method is typically used when you want to process pre-recorded audio and send it
        to the server in real-time.

        Args:
            filename (str): The path to the audio file to be played and sent to the server.
        """

        # read audio and create pyaudio stream
        with wave.open(filename, "rb") as wavfile:
            # self.stream = self.p.open(
            #     format=self.p.get_format_from_width(wavfile.getsampwidth()),
            #     channels=wavfile.getnchannels(),
            #     rate=wavfile.getframerate(),
            #     input=True,
            #     output=False,
            #     frames_per_buffer=self.chunk,
            # )
            chunk_duration = self.chunk / float(wavfile.getframerate())
            try:
                while any(client.recording for client in self.clients):
                    data = wavfile.readframes(self.chunk)
                    if data == b"":
                        break

                    audio_array = self.bytes_to_float_array(data)
                    self.multicast_packet(audio_array.tobytes())
                    if self.mute_audio_playback:
                        time.sleep(chunk_duration)
                    else:
                        # self.stream.write(data)
                        print("not impl")
    
                wavfile.close()

                for client in self.clients:
                    client.wait_before_disconnect()
                self.multicast_packet(Client.END_OF_AUDIO.encode('utf-8'), True)
                self.write_all_clients_srt()
                # self.stream.close()
                self.close_all_clients()

            except KeyboardInterrupt:
                wavfile.close()
                # self.stream.stop_stream()
                # self.stream.close()
                self.p.terminate()
                self.close_all_clients()
                self.write_all_clients_srt()
                print("[INFO]: Keyboard interrupt.")

    def process_rtsp_stream(self, rtsp_url):
        """
        Connect to an RTSP source, process the audio stream, and send it for transcription.

        Args:
            rtsp_url (str): The URL of the RTSP stream source.
        """
        print("[INFO]: Connecting to RTSP stream...")
        try:
            container = av.open(rtsp_url, format="rtsp", options={"rtsp_transport": "tcp"})
            self.process_av_stream(container, stream_type="RTSP")
        except Exception as e:
            print(f"[ERROR]: Failed to process RTSP stream: {e}")
        finally:
            for client in self.clients:
                client.wait_before_disconnect()
            self.multicast_packet(Client.END_OF_AUDIO.encode('utf-8'), True)
            self.close_all_clients()
            self.write_all_clients_srt()
        print("[INFO]: RTSP stream processing finished.")

    def process_hls_stream(self, hls_url, save_file=None):
        """
        Connect to an HLS source, process the audio stream, and send it for transcription.

        Args:
            hls_url (str): The URL of the HLS stream source.
            save_file (str, optional): Local path to save the network stream.
        """
        print("[INFO]: Connecting to HLS stream...")
        try:
            container = av.open(hls_url, format="hls")
            self.process_av_stream(container, stream_type="HLS", save_file=save_file)
        except Exception as e:
            print(f"[ERROR]: Failed to process HLS stream: {e}")
        finally:
            for client in self.clients:
                client.wait_before_disconnect()
            self.multicast_packet(Client.END_OF_AUDIO.encode('utf-8'), True)
            self.close_all_clients()
            self.write_all_clients_srt()
        print("[INFO]: HLS stream processing finished.")

    def process_av_stream(self, container, stream_type, save_file=None):
        """
        Process an AV container stream and send audio packets to the server.

        Args:
            container (av.container.InputContainer): The input container to process.
            stream_type (str): The type of stream being processed ("RTSP" or "HLS").
            save_file (str, optional): Local path to save the stream. Default is None.
        """
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if not audio_stream:
            print(f"[ERROR]: No audio stream found in {stream_type} source.")
            return

        output_container = None
        if save_file:
            output_container = av.open(save_file, mode="w")
            output_audio_stream = output_container.add_stream(codec_name="pcm_s16le", rate=self.rate)

        try:
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    audio_data = frame.to_ndarray().tobytes()
                    self.multicast_packet(audio_data)

                    if save_file:
                        output_container.mux(frame)
        except Exception as e:
            print(f"[ERROR]: Error during {stream_type} stream processing: {e}")
        finally:
            # Wait for server to send any leftover transcription.
            time.sleep(5)
            self.multicast_packet(Client.END_OF_AUDIO.encode('utf-8'), True)
            if output_container:
                output_container.close()
            container.close()

    def save_chunk(self, n_audio_file):
        """
        Saves the current audio frames to a WAV file in a separate thread.

        Args:
        n_audio_file (int): The index of the audio file which determines the filename.
                            This helps in maintaining the order and uniqueness of each chunk.
        """
        t = threading.Thread(
            target=self.write_audio_frames_to_file,
            args=(self.frames[:], f"chunks/{n_audio_file}.wav",),
        )
        t.start()

    def finalize_recording(self, n_audio_file):
        """
        Finalizes the recording process by saving any remaining audio frames,
        closing the audio stream, and terminating the process.

        Args:
        n_audio_file (int): The file index to be used if there are remaining audio frames to be saved.
                            This index is incremented before use if the last chunk is saved.
        """
        if self.save_output_recording and len(self.frames):
            self.write_audio_frames_to_file(
                self.frames[:], f"chunks/{n_audio_file}.wav"
            )
            n_audio_file += 1
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.close_all_clients()
        if self.save_output_recording:
            self.write_output_recording(n_audio_file)
        self.write_all_clients_srt()

    def record(self):
        """
        Record audio data from the input stream and save it to a WAV file.

        Continuously records audio data from the input stream, sends it to the server via a WebSocket
        connection, and simultaneously saves it to multiple WAV files in chunks. It stops recording when
        the `RECORD_SECONDS` duration is reached or when the `RECORDING` flag is set to `False`.

        Audio data is saved in chunks to the "chunks" directory. Each chunk is saved as a separate WAV file.
        The recording will continue until the specified duration is reached or until the `RECORDING` flag is set to `False`.
        The recording process can be interrupted by sending a KeyboardInterrupt (e.g., pressing Ctrl+C). After recording,
        the method combines all the saved audio chunks into the specified `out_file`.
        """
        n_audio_file = 0
        if self.save_output_recording:
            if os.path.exists("chunks"):
                shutil.rmtree("chunks")
            os.makedirs("chunks")
        try:
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                if not any(client.recording for client in self.clients):
                    break
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames += data

                audio_array = self.bytes_to_float_array(data)

                self.multicast_packet(audio_array.tobytes())

                # save frames if more than a minute
                if len(self.frames) > 60 * self.rate:
                    if self.save_output_recording:
                        self.save_chunk(n_audio_file)
                        n_audio_file += 1
                    self.frames = b""
            self.write_all_clients_srt()

        except KeyboardInterrupt:
            self.finalize_recording(n_audio_file)

    def write_audio_frames_to_file(self, frames, file_name):
        """
        Write audio frames to a WAV file.

        The WAV file is created or overwritten with the specified name. The audio frames should be
        in the correct format and match the specified channel, sample width, and sample rate.

        Args:
            frames (bytes): The audio frames to be written to the file.
            file_name (str): The name of the WAV file to which the frames will be written.

        """
        with wave.open(file_name, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            wavfile.writeframes(frames)

    def write_output_recording(self, n_audio_file):
        """
        Combine and save recorded audio chunks into a single WAV file.

        The individual audio chunk files are expected to be located in the "chunks" directory. Reads each chunk
        file, appends its audio data to the final recording, and then deletes the chunk file. After combining
        and saving, the final recording is stored in the specified `out_file`.


        Args:
            n_audio_file (int): The number of audio chunk files to combine.
            out_file (str): The name of the output WAV file to save the final recording.

        """
        input_files = [
            f"chunks/{i}.wav"
            for i in range(n_audio_file)
            if os.path.exists(f"chunks/{i}.wav")
        ]
        with wave.open(self.output_recording_filename, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            for in_file in input_files:
                with wave.open(in_file, "rb") as wav_in:
                    while True:
                        data = wav_in.readframes(self.chunk)
                        if data == b"":
                            break
                        wavfile.writeframes(data)
                # remove this file
                os.remove(in_file)
        wavfile.close()
        # clean up temporary directory to store chunks
        if os.path.exists("chunks"):
            shutil.rmtree("chunks")

    @staticmethod
    def bytes_to_float_array(audio_bytes):
        """
        Convert audio data from bytes to a NumPy float array.

        It assumes that the audio data is in 16-bit PCM format. The audio data is normalized to
        have values between -1 and 1.

        Args:
            audio_bytes (bytes): Audio data in bytes.

        Returns:
            np.ndarray: A NumPy array containing the audio data as float values normalized between -1 and 1.
        """
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0


class TranscriptionClient(TranscriptionTeeClient):
    """
    Client for handling audio transcription tasks via a single WebSocket connection.

    Acts as a high-level client for audio transcription tasks using a WebSocket connection. It can be used
    to send audio data for transcription to a server and receive transcribed text segments.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port number to connect to on the server.
        lang (str, optional): The primary language for transcription. Default is None, which defaults to English ('en').
        translate (bool, optional): If True, the task will be translation instead of transcription. Default is False.
        model (str, optional): The whisper model to use (e.g., "small", "base"). Default is "small".
        use_vad (bool, optional): Whether to enable voice activity detection. Default is True.
        save_output_recording (bool, optional): Whether to save the microphone recording. Default is False.
        output_recording_filename (str, optional): Path to save the output recording WAV file. Default is "./output_recording.wav".
        output_transcription_path (str, optional): File path to save the output transcription (SRT file). Default is "./output.srt".
        log_transcription (bool, optional): Whether to log transcription output to the console. Default is True.
        max_clients (int, optional): Maximum number of client connections allowed. Default is 4.
        max_connection_time (int, optional): Maximum allowed connection time in seconds. Default is 600.
        mute_audio_playback (bool, optional): If True, mutes audio playback during file playback. Default is False.
        send_last_n_segments (int, optional): Number of most recent segments to send to the client. Defaults to 10.
        no_speech_thresh (float, optional): Segments with no speech probability above this threshold will be discarded. Defaults to 0.45.
        clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
        same_output_threshold (int, optional): Number of repeated outputs before considering it as a valid segment. Defaults to 10.
        transcription_callback (callable, optional): A callback function to handle transcription results. Default is None.

    Attributes:
        client (Client): An instance of the underlying Client class responsible for handling the WebSocket connection.

    Example:
        To create a TranscriptionClient and start transcription on microphone audio:
        ```python
        transcription_client = TranscriptionClient(host="localhost", port=9090)
        transcription_client()
        ```
    """
    def __init__(
        self,
        host,
        port,
        lang=None,
        translate=False,
        model="small",
        use_vad=True,
        use_wss=False,
        save_output_recording=False,
        output_recording_filename="./output_recording.wav",
        output_transcription_path="./output.srt",
        log_transcription=True,
        max_clients=4,
        max_connection_time=600,
        mute_audio_playback=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        transcription_callback=None,
        initial_prompt=None,
    ):
        self.client = Client(
            host,
            port,
            lang,
            translate,
            model,
            srt_file_path=output_transcription_path,
            use_vad=use_vad,
            use_wss=use_wss,
            log_transcription=log_transcription,
            max_clients=max_clients,
            max_connection_time=max_connection_time,
            send_last_n_segments=send_last_n_segments,
            no_speech_thresh=no_speech_thresh,
            clip_audio=clip_audio,
            same_output_threshold=same_output_threshold,
            transcription_callback=transcription_callback,
            initial_prompt=initial_prompt,
        )

        if save_output_recording and not output_recording_filename.endswith(".wav"):
            raise ValueError(f"Please provide a valid `output_recording_filename`: {output_recording_filename}")
        if not output_transcription_path.endswith(".srt"):
            raise ValueError(f"Please provide a valid `output_transcription_path`: {output_transcription_path}. The file extension should be `.srt`.")
        TranscriptionTeeClient.__init__(
            self,
            [self.client],
            save_output_recording=save_output_recording,
            output_recording_filename=output_recording_filename,
            mute_audio_playback=mute_audio_playback
        )

```
注意：WhisperLIve的输入如：{'uid': 'd9e8bb07-d2cf-485e-bbff-6aef12145652', 'segments': [{'start': '2.048', 'end': '5.048', 'text': '请说说你在项目里如何做JavaScript。', 'completed': False}]} 仅最后一个completed可能为False。
请输出修改后的WhisperLIveTranscriber代码。