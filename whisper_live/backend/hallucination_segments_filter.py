# import unicodedata
from dataclasses import asdict, dataclass
import re
from typing import List, Optional
import logging
# logging.basicConfig(level=logging.INFO) # open the comment if you want to run seperately.

def get_segment_start(segment):
    return getattr(segment, "start", getattr(segment, "start_ts", 0))

def get_segment_end(segment):
    return getattr(segment, "end", getattr(segment, "end_ts", 0))


class InvalidValueHallucinationDetector:
    def is_invalid(self, segment):
        text = segment.text
        if not text:
            print(f"maybe hallucination as text is empty")
            return True
        start = get_segment_start(segment)
        end = get_segment_end(segment)
        if end <= start:
            print(f"maybe hallucination as start timestamp is larger than end")
            return True
        if end <= 0.0 or start < 0.0:
            print(f"maybe hallucination as start or end timestamp is negative")
            return True
        return False
    

class CharsPerSecondHallucinationDetector:
    def __init__(self):
        # 定义Unicode范围对应的语系
        self.language_blocks = {
            'cjk': [
                (0x4E00, 0x9FFF),    # 中文汉字
                (0x3040, 0x309F),    # 平假名
                (0x30A0, 0x30FF),    # 片假名
                (0xAC00, 0xD7AF)     # 韩文字符
            ],
            'cyrillic': [(0x0400, 0x04FF)],  # 俄语
            'latin': [                  # 欧洲语言
                (0x0000, 0x024F),      # 扩展拉丁(含所有西欧字符)
                (0x1E00, 0x1EFF)       # 拉丁扩展附加
            ]
        }
        
        # 各语系字符/秒阈值 (基于正常语速研究)
        self.thresholds = {
            'cjk': 8,     # 中日韩 (汉字/假名/谚文)
            'cyrillic': 35,  # 俄语
            'latin': 30,     # 欧洲语言
            'other': 30      # 默认阈值
        }
        
        # 标点符号停顿权重 (增加预估时间)
        self.punctuation_weights = {
            'full': 0.1,    # 完整停顿: 。？！；:.
            'half': 0.05,   # 半停顿: ，、、
            'minor': 0.02   # 轻微停顿: 空格
        }

    def _get_script(self, char):
        """确定字符所属的语系"""
        cp = ord(char)
        for script, blocks in self.language_blocks.items():
            for start, end in blocks:
                if start <= cp <= end:
                    # print(f"_get_script char:{char}, script:{script}")
                    return script
        # print(f"_get_script char:{char}, script:{script}")
        return 'other'

    def _get_pause_weight(self, char):
        """获取标点符号的停顿权重"""
        # category = unicodedata.category(char)
        # 完整停顿: 句号、问号、感叹号等
        if char in ('。', '！', '？', ';', ':', '.', '!', '?'): # or category == 'Po':
            return self.punctuation_weights['full']
        # 半停顿: 逗号、顿号等
        elif char in ('，', '、', ',', '、'): # or category == 'Pf':
            return self.punctuation_weights['half']
        # 空格和换行
        elif char in (' ', '\t', '\n'): # or category == 'Zs':
            return self.punctuation_weights['minor']
        return 0

    def _is_too_long_maybe_hallucination(self, text, duration):
        """
        判断文本是否超过正常语速范围
        :param text: 待检测文本
        :param duration: 音频时长(秒)
        :return: (是否可能幻觉, 预估所需时长, 实际字符率)
        """
        text = text.strip()
        if len(text) < 8:
            return False
        
        if not text.strip() or duration <= 0:
            return False
        
        # 统计各语系字符数量
        script_counts = {script: 0 for script in self.thresholds}
        total_pause = 0.0

        # 找到最后一个非标点字符的位置
        last_non_punct_index = -1
        for i, char in enumerate(text):
            if self._get_pause_weight(char) == 0:  # 非标点字符
                last_non_punct_index = i
        
        # 如果整个字符串都是标点，则直接返回
        if last_non_punct_index == -1:
            return False
        
        # 只计算到最后一个非标点字符（包含该字符）
        for i, char in enumerate(text[:last_non_punct_index + 1]):
            punct_pause = self._get_pause_weight(char)
            if punct_pause > 0:
                total_pause += punct_pause
            else:
              script = self._get_script(char)
              script_counts[script] += 1
        
        # 计算总预估时间 (字符数/阈值 + 标点停顿)
        required_time = 0.0
        for script, count in script_counts.items():
            if count > 0:
                required_time += count / self.thresholds[script]
        
        # 添加标点停顿时间
        required_time += total_pause
        
        # # 计算实际字符率 (总字符/时长)
        # total_chars = len(text)
        # actual_rate = total_chars / duration if duration > 0 else float('inf')
        
        # 判断是否超速 (实际时长 < 预估所需时长)
        # print(f"text: 【{text}】, script_counts: {script_counts}, total_pause: {total_pause}, required_time: {required_time}, duration: {duration}, is hallucination: {required_time > duration}")
        return required_time > duration

    def is_invalid(self, segment):
        if not segment:
            return False
        text = segment.text
        start = get_segment_start(segment)
        end = get_segment_end(segment)
        duration = end - start
        result = self._is_too_long_maybe_hallucination(text, duration)
        if result:
            print(f"maybe hallucination as text is too long while duration too short.")
        return result


class FirstTokenProbabilityTooLowHallucinationFilter:
    def __init__(self, log_prob_threshold=-1.0, first_word_prob_threshold=0.02):
        self.log_prob_threshold = log_prob_threshold
        self.first_word_prob_threshold = first_word_prob_threshold

    def is_invalid(self, segment):
        if segment.avg_logprob < self.log_prob_threshold:
            print(f"maybe hallucination as avg_logprob is too low")
            return True
        if segment.words:
            if segment.words[0].probability < self.first_word_prob_threshold:
                print(f"maybe hallucination as first word prob is too low")
                return True
        return False
    
class TextPatternMatchHallucinationDetector:
    def __init__(self):
        self.hallucination_text_patterns = [
            r"字幕.*李宗盛", # 中文字幕 李宗盛
            r"中文字幕"
            r"字幕.* ",
            r"字幕志愿者", #中文字幕志愿者 杨栋梁
            r"独播剧场", # 优优独播剧场——YoYo Television Series Exclusive
            r"YoYo",
            r"阿弥陀佛",
            r"点赞.*订阅", # 请不吝点赞 订阅 转发 打赏支持明镜与点点栏目
            r"点赞.*支持",
            r"欢迎.*订阅", # 明镜需要您的支持 欢迎订阅明镜
            r"感谢.*观看",
            r"谢谢大家",
            r"明镜",
            r"接着,接着,接着",
            r"咳嗽",
            r"下面是一些.* ", # 下面是一些IT面试问题。
            r"IT面试问题",
            r"(\w)\1{4}", # 重复字符
            # r"(.{2})\1{3,}",
            # r"(.{3})\1{3,}",
            # r"(.{4})\1{3,}",
            r'(\w{2})(?:\W*\1){3,}',
            r'(\w{3})(?:\W*\1){3,}',
            r'(\w{4})(?:\W*\1){2,}',
            # r"((?:[\u4e00-\u9fff]{2})[^\u4e00-\u9fff]?)\1{3,}",
            # r"((?:[\u4e00-\u9fff]{3})[^\u4e00-\u9fff]?)\1{3,}",
            # r"((?:[\u4e00-\u9fff]{4})[^\u4e00-\u9fff]?)\1{2,}",
        ]
        self.hallucination_text_patterns_re = [re.compile(pattern) for pattern in self.hallucination_text_patterns]

    def is_invalid(self, segment):
        text = segment.text
        result = any(pattern.search(text) for pattern in self.hallucination_text_patterns_re)
        if result:
            print(f"maybe hallucination as matched hallucination_text_patterns.")
        return result

class HallucinationDetectorManager:
    def __init__(self):
        self.invalid_value_hallucination_detector = InvalidValueHallucinationDetector()
        self.chars_per_second_hallucination_detector = CharsPerSecondHallucinationDetector()
        self.first_token_probability_too_low_hallucination_detector = FirstTokenProbabilityTooLowHallucinationFilter()
        self.text_pattern_match_hallucination_detector = TextPatternMatchHallucinationDetector()

    def detect_hallucination(self, segment):
        return self.invalid_value_hallucination_detector.is_invalid(segment) or self.chars_per_second_hallucination_detector.is_invalid(segment) or self.first_token_probability_too_low_hallucination_detector.is_invalid(segment) or self.text_pattern_match_hallucination_detector.is_invalid(segment)
        
# TODO end > duration end
# 使用示例
if __name__ == "__main__":
    @dataclass
    class Segment:
        start: float
        end: float
        text: str
        avg_logprob: float = 0.0
        words: Optional[List] = None

    # 创建检测器管理器
    hallucination_detector = HallucinationDetectorManager()
    
    # 测试用例列表：(测试名称, 预期结果, 测试segment)
    test_cases = [
        # 测试InvalidValueHallucinationDetector
        ("Empty text", True, Segment(start=0.0, end=1.0, text="", avg_logprob=0.0)),
        ("Start > end", True, Segment(start=2.0, end=1.0, text="正常文本", avg_logprob=0.0)),
        ("Zero duration", True, Segment(start=1.0, end=1.0, text="零时长文本", avg_logprob=0.0)),
        
        # 测试CharsPerSecondHallucinationDetector
        ("CJK too fast", True, Segment(start=0.0, end=0.9, text="这是一段超过阈值的中文文本需要检测", avg_logprob=0.0)),
        ("Latin too fast", True, Segment(start=0.0, end=0.5, text="This is an example text that exceeds the character per second limit", avg_logprob=0.0)),
        ("Valid CJK", False, Segment(start=0.0, end=3.0, text="合理速度的中文内容", avg_logprob=0.0)),
        ("Short text", False, Segment(start=0.0, end=0.5, text="短文本", avg_logprob=0.0)),
        ("Valid Mix language", False, Segment(start=0.0, end=3.0, text="这是一段混合文本: English, 日本語, 한국어!", avg_logprob=0.0)),
        ("Invalid Mix language", True, Segment(start=0.0, end=3.0, text="这是一段混合文本: English, 日本語, 한국어!"*3, avg_logprob=0.0)),
        
        # 测试FirstTokenProbabilityTooLowHallucinationFilter
        ("Low avg logprob", True, Segment(start=0.0, end=1.0, text="概率过低", avg_logprob=-2.0)),
        ("Low first word prob", True, Segment(start=0.0, end=1.0, text="概率测试", avg_logprob=0.0, 
            words=[{"word": "prob", "probability": 0.01}])),
        ("Valid probability", False, Segment(start=0.0, end=1.0, text="正常内容", avg_logprob=0.0, 
            words=[{"word": "good", "probability": 0.5}])),
        
        # 测试TextPatternMatchHallucinationDetector
        ("Subtitle pattern", True, Segment(start=0.0, end=1.0, text="中文字幕 李宗盛", avg_logprob=0.0)),
        ("Repetitive pattern", True, Segment(start=0.0, end=1.0, text="咳咳咳咳咳咳", avg_logprob=0.0)),
        ("Subscription pattern", True, Segment(start=0.0, end=1.0, text="欢迎订阅明镜", avg_logprob=0.0)),
        ("Normal text", False, Segment(start=0.0, end=1.0, text="普通正常文本内容", avg_logprob=0.0)),
        ("Repeat normal", True, Segment(start=0.0, end=2.0, text="啊啊啊啊啊", avg_logprob=0.0)),
        ("Repeat normal", False, Segment(start=0.0, end=2.0, text="普通的，普通的，普通的。", avg_logprob=0.0)),
        ("Repeat too many", True, Segment(start=0.0, end=2.0, text="普通的，普通的，普通的，普通的。", avg_logprob=0.0)),
        ("Repeat too many", True, Segment(start=0.0, end=2.0, text="普通的普通的普通的普通的。", avg_logprob=0.0)),
        ("Repeat too many", True, Segment(start=0.0, end=3.0, text="我本身试试试试试试试试试试试试", avg_logprob=0.0)),
        
        
        # 综合测试
        ("Multiple issues", True, Segment(start=2.0, end=1.0, text="字幕志愿者", avg_logprob=-2.0)),
        ("All valid", False, Segment(start=0.0, end=2.0, text="完全正常的文本内容", avg_logprob=0.0, 
            words=[{"word": "good", "probability": 0.8}]))
    ]

    # 运行测试
    total = len(test_cases)
    passed = 0
    failed_cases = []

    print("="*50)
    print("Starting Hallucination Detector Tests")
    print("="*50)
    
    for name, expected, segment in test_cases:
        # 转换words列表为Word对象（如果存在）
        if segment.words:
            words = []
            for w in segment.words:
                # 创建简化版Word对象
                @dataclass
                class Word:
                    word: str
                    probability: float
                words.append(Word(word=w["word"], probability=w["probability"]))
            segment.words = words
        
        result = hallucination_detector.detect_hallucination(segment)
        if result == expected:
            passed += 1
            print(f"✅ PASS: {name} (Expected: {expected}, Result: {result})")
        else:
            failed_cases.append((name, segment, expected, result))
            print(f"❌ FAIL: {name} (Expected: {expected}, Got: {result})")
    
    # 打印统计结果
    print("\n" + "="*50)
    print(f"Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print("="*50)
    
    # 打印失败用例详情
    if failed_cases:
        print("\nFailed Cases Details:")
        for i, (name, segment, expected, result) in enumerate(failed_cases, 1):
            print(f"\n{i}. {name}")
            print(f"   Segment: start={segment.start}, end={segment.end}, text='{segment.text}'")
            print(f"   Expected: {expected}, Actual: {result}")
            if segment.words:
                print(f"   First word: '{segment.words[0].word}' (prob={segment.words[0].probability})")