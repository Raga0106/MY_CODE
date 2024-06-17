import torch
import numpy as np
import soundfile as sf
import pyaudio
import threading
import concurrent.futures
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

# 路径配置
MODEL_PATH = 'D:/Users/USER/Downloads/GawrGura_RVC_v2_Itaila_e200_s92600/GawrGura_e200_s92600.pth'
INDEX_PATH = 'D:/Users/USER/Downloads/GawrGura_RVC_v2_Itaila_e200_s92600/added_IVF4551_Flat_nprobe_1_GawrGura_v2.index'
LLM_PATH = 'D:/.cache/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF'

# 加载生成式AI模型
print("Loading language model...")
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=False)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_PATH)

# 加载RVC模型
print("Loading RVC model...")


class RVCModel(torch.nn.Module):
    def __init__(self):
        super(RVCModel, self).__init__()
        # 模型架构的定义（根据实际情况调整）

    def forward(self, x):
        # 前向传播的定义（根据实际情况调整）
        pass


model = RVCModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# 读取索引文件
with open(INDEX_PATH, 'rb') as f:
    index = pickle.load(f)


# 定义文本生成函数
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(inputs.input_ids, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# 定义TTS转换函数
def text_to_speech(model, text, index):
    # 这是一个示例实现，需要根据实际情况进行调整
    audio = np.random.randn(22050 * 3)  # 生成一个3秒的随机音频作为示例
    return audio


# 播放音频函数
def play_audio(audio_data, samplerate=22050):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=samplerate,
                    output=True)
    stream.write(audio_data.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()


# 实时生成和播放的函数
def generate_and_play(prompt):
    # 生成文本
    text = generate_text(prompt)
    print(f"Generated text: {text}")

    # 将文本转换为语音
    audio = text_to_speech(model, text, index)

    # 播放生成的语音
    play_audio(audio)


# 使用线程池来并行执行生成和播放
prompt = "请告诉我一个有趣的故事。"
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
future = executor.submit(generate_and_play, prompt)
