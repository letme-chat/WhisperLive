import json
import logging
import threading
import time
import queue
import numpy as np
import re
from whisper_live.backend.hallucination_segments_filter import HallucinationDetectorManager
from whisper_live.transcriber.transcriber_faster_whisper import Segment


class ServeClientBase(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    client_uid: str
    """A unique identifier for the client."""
    websocket: object
    """The WebSocket connection for the client."""
    send_last_n_segments: int
    """Number of most recent segments to send to the client."""
    no_speech_thresh: float
    """Segments with no speech probability above this threshold will be discarded."""
    clip_audio: bool
    """Whether to clip audio with no valid segments."""
    same_output_threshold: int
    """Number of repeated outputs before considering it as a valid segment."""

    def __init__(
        self,
        client_uid,
        websocket,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        translation_queue=None,
    ):
        self.client_uid = client_uid
        self.websocket = websocket
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold

        self.frames = b""
        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ""
        self.prev_out = ""
        self.exit = False
        self.same_output_count = 0
        self.transcript = []
        self.end_time_for_same_output = None
        self.translation_queue = translation_queue

        self.hallucination_detector = HallucinationDetectorManager()

        # threading
        self.lock = threading.Lock()
        # 添加文本规范化相关的配置
        self.normalize_text = True  # 是否启用文本规范化
        self.normalization_pattern = re.compile(r'[\s\-_,.;:!?]+')  # 要移除的字符模式

    def normalize_output_text(self, text):
        """
        规范化文本以便比较，移除大小写、空格、连字符和标点符号差异。
        
        Args:
            text (str): 要规范化的文本
            
        Returns:
            str: 规范化后的文本
        """
        if not self.normalize_text or not text:
            return text
            
        # 转换为小写
        text = text.lower()
        
        # 移除所有空格、连字符和常见标点符号
        text = self.normalization_pattern.sub('', text)
        
        return text
    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

        """
        while True:
            if self.frames_np is not None:
                logging.debug(f"self.timestamp_offset: {self.timestamp_offset}, self.frames_offset: {self.frames_offset}, self.frames_np.shape[0]: {self.frames_np.shape[0]}, self.RATE: {self.RATE}")
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np is None:
                continue

            if self.clip_audio:
                self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if self.frames_np is not None:
                logging.debug(f"self.timestamp_offset: {self.timestamp_offset}, self.frames_offset: {self.frames_offset}, self.frames_np.shape[0]: {self.frames_np.shape[0]}, self.RATE: {self.RATE}, input_bytes: {input_bytes.shape}, duration: {duration}")
            if duration < 1.0:
                time.sleep(0.1)     # wait for audio chunks to arrive
                continue
            try:
                input_sample = input_bytes.copy()
                result = self.transcribe_audio(input_sample)

                if result is None or self.language is None:
                    self.timestamp_offset += duration
                    time.sleep(0.25)    # wait for voice activity, result is None when no voice activity
                    continue
                self.handle_transcription_output(result, duration)

            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)

    def transcribe_audio(self):
        raise NotImplementedError

    def handle_transcription_output(self, result, duration):
        raise NotImplementedError
    
    def format_segment(self, start, end, text, completed=False):
        """
        Formats a transcription segment with precise start and end times alongside the transcribed text.

        Args:
            start (float): The start time of the transcription segment in seconds.
            end (float): The end time of the transcription segment in seconds.
            text (str): The transcribed text corresponding to the segment.

        Returns:
            dict: A dictionary representing the formatted transcription segment, including
                'start' and 'end' times as strings with three decimal places and the 'text'
                of the transcription.
        """
        return {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text,
            'completed': completed
        }

    def add_frames(self, frame_np):
        """
        Add audio frames to the ongoing audio stream buffer.

        This method is responsible for maintaining the audio stream buffer, allowing the continuous addition
        of audio frames as they are received. It also ensures that the buffer does not exceed a specified size
        to prevent excessive memory usage.

        If the buffer size exceeds a threshold (45 seconds of audio data), it discards the oldest 30 seconds
        of audio data to maintain a reasonable buffer size. If the buffer is empty, it initializes it with the provided
        audio frame. The audio stream buffer is used for real-time processing of audio data for transcription.

        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array.

        """
        self.lock.acquire()
        if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30*self.RATE):]
            # check timestamp offset(should be >= self.frame_offset)
            # this basically means that there is no speech as timestamp offset hasnt updated
            # and is less than frame_offset
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)
        self.lock.release()

    def clip_audio_if_no_valid_segment(self):
        """
        Update the timestamp offset based on audio buffer status.
        Clip audio if the current chunk exceeds 30 seconds, this basically implies that
        no valid segment for the last 30 seconds from whisper
        """
        with self.lock:
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > 25 * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - 5

    def get_audio_chunk_for_processing(self):
        """
        Retrieves the next chunk of audio data for processing based on the current offsets.

        Calculates which part of the audio data should be processed next, based on
        the difference between the current timestamp offset and the frame's offset, scaled by
        the audio sample rate (RATE). It then returns this chunk of audio data along with its
        duration in seconds.

        Returns:
            tuple: A tuple containing:
                - input_bytes (np.ndarray): The next chunk of audio data to be processed.
                - duration (float): The duration of the audio chunk in seconds.
        """
        with self.lock:
            samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration

    def prepare_segments(self, last_segment=None):
        """
        Prepares the segments of transcribed text to be sent to the client.

        This method compiles the recent segments of transcribed text, ensuring that only the
        specified number of the most recent segments are included. It also appends the most
        recent segment of text if provided (which is considered incomplete because of the possibility
        of the last word being truncated in the audio chunk).

        Args:
            last_segment (str, optional): The most recent segment of transcribed text to be added
                                          to the list of segments. Defaults to None.

        Returns:
            list: A list of transcribed text segments to be sent to the client.
        """
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]
        return segments

    def get_audio_chunk_duration(self, input_bytes):
        """
        Calculates the duration of the provided audio chunk.

        Args:
            input_bytes (numpy.ndarray): The audio chunk for which to calculate the duration.

        Returns:
            float: The duration of the audio chunk in seconds.
        """
        return input_bytes.shape[0] / self.RATE

    def send_transcription_to_client(self, segments):
        """
        Sends the specified transcription segments to the client over the websocket connection.

        This method formats the transcription segments into a JSON object and attempts to send
        this object to the client. If an error occurs during the send operation, it logs the error.

        Returns:
            segments (list): A list of transcription segments to be sent to the client.
        """
        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "segments": segments,
                })
            )
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")

    def disconnect(self):
        """
        Notify the client of disconnection and send a disconnect message.

        This method sends a disconnect message to the client via the WebSocket connection to notify them
        that the transcription service is disconnecting gracefully.

        """
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.DISCONNECT
        }))

    def cleanup(self):
        """
        Perform cleanup tasks before exiting the transcription service.

        This method performs necessary cleanup tasks, including stopping the transcription thread, marking
        the exit flag to indicate the transcription thread should exit gracefully, and destroying resources
        associated with the transcription process.

        """
        logging.info("Cleaning up.")
        self.exit = True
    
    def get_segment_no_speech_prob(self, segment):
        return getattr(segment, "no_speech_prob", 0)

    def get_segment_start(self, segment):
        return getattr(segment, "start", getattr(segment, "start_ts", 0))

    def get_segment_end(self, segment):
        return getattr(segment, "end", getattr(segment, "end_ts", 0))
    
    def is_segment_invalid(self, segment):
        no_speech_prob = self.get_segment_no_speech_prob(segment)
        if no_speech_prob > self.no_speech_thresh:
            return True
        if segment.avg_logprob < -1.0 - 0.1: # TODO @jjm pass in log_prob_threshold
            return True
        return False


    def split_segments(self, segment):
        """
        按照标点符号分割长段落，优先按句子分割（。？！等），没有句子才按逗号等分割。
        
        Args:
            segment: 要分割的Segment对象
            
        Returns:
            list: 分割后的Segment对象列表
        """
        # 定义分割符优先级：先按句子分割，再按逗号分割
        sentence_delimiters = ['。', '？', '！', ';', '；', '?', '!']
        comma_delimiters = ['，', ',', '、']
        
        text = segment.text
        words = getattr(segment, 'words', [])
        
        # 如果没有words信息，无法准确分割，返回原segment
        if not words:
            return [segment]
        
        # 计算每个字符在words中的位置映射
        char_to_word_idx = []
        for word_idx, word in enumerate(words):
            for char_idx in range(len(word.word)):
                char_to_word_idx.append(word_idx)
        
        # 查找分割点
        split_indices = []
        ignore_last = 10                      # 忽略最后 10 个字符
        effective_len = max(len(text) - ignore_last, 0)
        for i, char in enumerate(text):
            if char in sentence_delimiters and i < effective_len:
                split_indices.append(i + 1)  # 包含分割符
        if not split_indices:
            # 如果没有句子分割符，按逗号分割
            for i, char in enumerate(text):
                if char in comma_delimiters and i < effective_len:
                    split_indices.append(i + 1)
        
        # 如果没有找到分割点，返回原segment
        if not split_indices:
            return [segment]
        
        # 确保最后一个分割点不超过文本长度
        if split_indices[-1] > len(text):
            split_indices[-1] = len(text)
        
        # 创建新的segments
        new_segments = []
        word_start_idx = 0
        last_text_split_idx = 0
        
        for split_idx in split_indices:
            # 确定分割点对应的word索引
            if split_idx >= len(char_to_word_idx):
                word_end_idx = len(words) - 1
            else:
                word_end_idx = char_to_word_idx[split_idx - 1]
            
            # 提取当前分段的words
            segment_words = words[word_start_idx:word_end_idx + 1]
            
            # 提取当前分段的文本
            segment_text = text[last_text_split_idx:split_idx]
            
            # 计算当前分段的开始和结束时间
            segment_start = words[word_start_idx].start if segment_words else 0
            segment_end = words[word_end_idx].end if segment_words else 0
            
            # 创建新的segment对象
            # new_segment = segment.__class__(
            new_segment = Segment(
                id=len(new_segments),
                seek=segment.seek,
                start=segment_start,
                end=segment_end,
                text=segment_text,
                tokens=[],  # 无法准确分割tokens，留空
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
                words=segment_words,
                temperature=segment.temperature
            )
            
            new_segments.append(new_segment)
            word_start_idx = word_end_idx + 1
            last_text_split_idx = split_idx
        
        # 处理最后一段
        if word_start_idx < len(words):
            segment_words = words[word_start_idx:]
            segment_text = text[last_text_split_idx:]
            
            segment_start = words[word_start_idx].start if segment_words else 0
            segment_end = words[-1].end if segment_words else 0
            
            # new_segment = segment.__class__(
            new_segment = Segment(
                id=len(new_segments),
                seek=segment.seek,
                start=segment_start,
                end=segment_end,
                text=segment_text,
                tokens=[],
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
                words=segment_words,
                temperature=segment.temperature
            )
            
            new_segments.append(new_segment)
        
        return new_segments
    
    def update_segments(self, segments, duration):
        """
        Processes the segments from Whisper and updates the transcript.
        Uses helper methods to account for differences between backends.
        
        Args:
            segments (list): List of segments returned by the transcriber.
            duration (float): Duration of the current audio chunk.
        
        Returns:
            dict or None: The last processed segment (if any).
        """
        offset = None
        self.current_out = ''
        last_segment = None
        changing_segments = []

        # Process the last segment if its no_speech_prob is acceptable.
        # if self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
        if not self.is_segment_invalid(segments[-1]):
            maybe_hullucination = False
            start_time = self.get_segment_start(segments[-1])
            end_time = self.get_segment_end(segments[-1])
            if end_time > duration + 0.1:
                logging.info(f"maybe hallucination as segment end time {end_time} is greater than duration {duration} + 0.1")
                maybe_hullucination = True
            if not maybe_hullucination and self.hallucination_detector.detect_hallucination(segments[-1]):
                # print(f"segment maybe hallucination")
                maybe_hullucination = True
            if not maybe_hullucination:
                if len(segments) == 1 and end_time - start_time > 15.0 and self.language in ["zh", "en"]:
                    logging.info(f"[test]long segment duration {end_time - start_time} is greater than 15.0, split to small segments, then process. self.language : {self.language}")
                    # TODO language
                    splited_segments = self.split_segments(segments[0])
                    # 用分割后的segments替换原始segments
                    segments = splited_segments
                self.current_out += segments[-1].text
                with self.lock:
                    last_segment = self.format_segment(
                        self.timestamp_offset + self.get_segment_start(segments[-1]),
                        self.timestamp_offset + min(duration, self.get_segment_end(segments[-1])),
                        self.current_out,
                        completed=False
                    )
        # Process complete segments only if there are more than one
        # and if the last segment's no_speech_prob is below the threshold.
        # if len(segments) > 1 and self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
        if len(segments) > 1 and not self.is_segment_invalid(segments[-1]):
            for s in segments[:-1]:
                text_ = s.text
                self.text.append(text_)
                with self.lock:
                    start = self.timestamp_offset + self.get_segment_start(s)
                    end = self.timestamp_offset + min(duration, self.get_segment_end(s))
                if start >= end:
                    continue
                # if self.get_segment_no_speech_prob(s) > self.no_speech_thresh:
                if self.is_segment_invalid(s):
                    continue
                completed_segment = self.format_segment(start, end, text_, completed=True)
                self.transcript.append(completed_segment)
                changing_segments.append(completed_segment)

                if self.translation_queue:
                    try:
                        self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                    except queue.Full:
                        logging.warning("Translation queue is full, skipping segment")
                offset = min(duration, self.get_segment_end(s))


        # Handle repeated output logic.
        is_same_output = False
        normalized_current = self.normalize_output_text(self.current_out)
        normalized_prev = self.normalize_output_text(self.prev_out)
        # TODO * @jjm transcript the same ignore case and - etc.
        # if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
        if normalized_current == normalized_prev and normalized_current != '':
            is_same_output = True
            self.same_output_count += 1

            # if we remove the audio because of same output on the nth reptition we might remove the 
            # audio thats not yet transcribed so, capturing the time when it was repeated for the first time
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = self.get_segment_end(segments[-1])
            time.sleep(0.1)  # wait briefly for any new voice activity
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # If the same incomplete segment is repeated too many times,
        # append it to the transcript and update the offset.
        not_output_because_same_output = is_same_output and self.same_output_count <= self.same_output_threshold
        logging.debug(f"same_output_count: {self.same_output_count}, is_same_output: {is_same_output}, not_output_because_same_output: {not_output_because_same_output}")
        if self.same_output_count > self.same_output_threshold:
            if not self.text or self.text[-1].strip().lower() != self.current_out.strip().lower():
                self.text.append(self.current_out)
                with self.lock:
                    completed_segment = self.format_segment(
                        self.timestamp_offset,
                        self.timestamp_offset + min(duration, self.end_time_for_same_output),
                        self.current_out,
                        completed=True
                    )
                    self.transcript.append(completed_segment)
                    changing_segments.append(completed_segment)

                    if self.translation_queue:
                        try:
                            self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                        except queue.Full:
                            logging.warning("Translation queue is full, skipping segment")

            self.current_out = ''
            offset = min(duration, self.end_time_for_same_output)
            self.same_output_count = 0
            last_segment = None
            self.end_time_for_same_output = None
        else:
            self.prev_out = self.current_out

        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset

        if last_segment is not None:
            changing_segments = changing_segments + [last_segment]
        logging.info(f"IN: segments: {segments},\n OUT: last_segment: {last_segment}, \nnot_output_because_same_output: {not_output_because_same_output}, \nchanging_segments: {changing_segments}\nuser_id: {self.client_uid}")
        return last_segment, not_output_because_same_output, changing_segments
