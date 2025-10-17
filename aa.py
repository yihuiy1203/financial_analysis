import queue
import threading
import time
import sys
from typing import Optional, Tuple
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
import requests

# === 新增 ===
from openai import OpenAI
openai_api_key = "EMPTY"  # 若你的 FastAPI 服务无需密钥，可写任意值
openai_api_base = "http://10.30.129.22:8000/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
# =============

# ========= 可调参数 =========
SAMPLE_RATE = 16000          # Whisper 推荐 16k
BLOCK_SIZE = 30              # 毫秒：VAD 最佳 10/20/30ms，这里取 30ms
VAD_MODE = 2                 # 0~3：越大越“严格”判定为语音（降噪环境建议 2~3）
MAX_SEGMENT_SEC = 12         # 单段最长秒数（防止长句阻塞）
SILENCE_SEC_TO_CUT = 0.6     # 静音这么久就“截断”成一段
LANGUAGE: Optional[str] = None  # None=自动识别；想固定中文可设 "zh"
TASK = "transcribe"          # 或 "translate"（转英文）
MODEL_ID = "Systran/faster-whisper-large-v3"  # 或 "openai/whisper-large-v3", "large-v3"
DEVICE = "cuda"  # "cuda" 或 "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"  # GPU上 float16；CPU 上 int8 更快
# ==========================


def frame_generator(block_ms: int, sample_rate: int, channels: int = 1):
    """sounddevice 回调式音频流，产出 int16 帧（block_ms 毫秒）"""
    q = queue.Queue()

    block_size = int(sample_rate * block_ms / 1000)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # 取第一通道，转 int16
        mono = indata[:, 0].copy()
        mono_i16 = (mono * 32767).astype(np.int16)
        q.put(mono_i16)

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        callback=audio_callback,
        latency="low",
    )
    stream.start()

    try:
        while True:
            yield q.get()
    except GeneratorExit:
        stream.stop()
        stream.close()


def collect_voiced_segments(vad: webrtcvad.Vad, frames: np.ndarray, sample_rate: int, block_ms: int) -> Tuple[bool, float]:
    """基于 VAD 判断当前帧是否语音，同时返回该帧时长（秒）"""
    is_speech = vad.is_speech(frames.tobytes(), sample_rate)
    return is_speech, block_ms / 1000.0

def send_to_llm(text: str):
    """将识别文本发送给本地 LLM"""
    try:
        response = client.chat.completions.create(
            model='/home/yihuiy/model/Qwen3-30B-A3B-Instruct-2507',
            messages=[
                {"role": "system", "content": "你是一个语音助手，请简明回答用户。"},
                {"role": "user", "content": text},
            ],
            max_tokens=8000
        )
        reply = response.choices[0].message.content
        
        data = {"text": reply}
        _ = requests.post("http://127.0.0.1:8001/speak", json=data)
        
        print(f"\033[92mLLM 回复：{reply}\033[0m")  # 绿色输出
    except Exception as e:
        print("LLM 请求错误：", e)
        
        
def transcribe_worker(model: WhisperModel, audio_float32: np.ndarray, seg_id: int):
    """后台线程：对一个完整语音段进行转写"""
    # faster-whisper 支持传 numpy 数组（float32, 16k）
    segments, info = model.transcribe(
        audio=audio_float32,
        language=LANGUAGE,
        task=TASK,
        vad_filter=True,          # 模型内部再做一次 VAD 清理
        beam_size=5,
        best_of=5,
        temperature=0.0,
        no_speech_threshold=0.5,
        condition_on_previous_text=False,  # 独立分段，延迟更稳
    )

    txt = []
    ts = []
    for s in segments:
        # s.start / s.end 单位为秒
        ts.append(f"[{s.start:6.2f} → {s.end:6.2f}]")
        txt.append(s.text.strip())

    line = "".join(txt).strip()
    if line:
        print(f"\n[Segment #{seg_id}] {''.join(ts)}")
        print(line)
        sys.stdout.flush()

        # === 新增：转录后调用 LLM ===
        threading.Thread(target=send_to_llm, args=(line,), daemon=True).start()
        # ===========================

def main():
    print("Loading model... (首次会下载模型，稍等)")
    model = WhisperModel(MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Model ready. Start speaking. Press Ctrl+C to stop.")

    vad = webrtcvad.Vad(VAD_MODE)
    gen = frame_generator(BLOCK_SIZE, SAMPLE_RATE)

    buf = []                  # 当前段的帧缓存（int16）
    voiced = False            # 当前是否处于说话期
    last_voiced_time = 0.0    # 最近一次检测到语音的时间
    seg_start_time = time.time()
    seg_id = 1
    accumulated_sec = 0.0

    workers = []

    try:
        for frame in gen:
            is_speech, dur = collect_voiced_segments(vad, frame, SAMPLE_RATE, BLOCK_SIZE)
            accumulated_sec += dur

            if is_speech:
                buf.append(frame)
                last_voiced_time = time.time()
                if not voiced:
                    voiced = True
                    seg_start_time = time.time()
            else:
                # 非语音帧也拼上少量，避免切得太生硬（可选，不加也行）
                if voiced:
                    buf.append(frame)

            # 触发切段：静音超过阈值 或 单段过长
            should_cut = False
            if voiced:
                if (time.time() - last_voiced_time) >= SILENCE_SEC_TO_CUT:
                    should_cut = True
                if (time.time() - seg_start_time) >= MAX_SEGMENT_SEC:
                    should_cut = True

            if should_cut and buf:
                # 打包当前段，启动转写线程
                segment_i16 = np.concatenate(buf)
                # 转 float32, 归一化到 [-1, 1]
                segment_f32 = segment_i16.astype(np.float32) / 32768.0

                t = threading.Thread(target=transcribe_worker, args=(model, segment_f32, seg_id), daemon=True)
                t.start()
                workers.append(t)

                # 清空，准备下一段
                buf = []
                voiced = False
                seg_id += 1

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # 收尾：缓冲里还有内容也转一下
        if buf:
            segment_i16 = np.concatenate(buf)
            segment_f32 = segment_i16.astype(np.float32) / 32768.0
            t = threading.Thread(target=transcribe_worker, args=(model, segment_f32, seg_id), daemon=True)
            t.start()
            workers.append(t)

        for t in workers:
            t.join()
        print("Done.")


if __name__ == "__main__":
    main()
