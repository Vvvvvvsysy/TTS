import os
import subprocess

from pydub import AudioSegment
import librosa
import soundfile as sf

import pyworld as pw
import numpy as np

def pitch_change(input_path, output_path, rate):
    # 读取音频文件
    input_audio, sr = sf.read(input_path)

    # 提取音频的基频（音调）和谐波信息
    f0, sp, ap = pw.wav2world(input_audio, sr)

    # 将音调降低一个半音

    new_f0 = f0 * 2 ** (rate/12)

    # 使用合成函数将音频合成回去
    synthesized_audio = pw.synthesize(new_f0, sp, ap, sr)

    # 将合成音频保存为文件
    sf.write(output_path, synthesized_audio, sr)


def speed_change(input_path, output_path, speed_factor):
    # 构造调用 sox 的命令
    command = ['sox', input_path, output_path, 'tempo', str(speed_factor)]
    
    # 执行命令
    subprocess.run(command)

base_dir = "/home/wenjun.feng/tts/transfer/child/generated_voice/pos/"
save_dir = "/home/wenjun.feng/tts/transfer/child/voice_change/"

for speech in os.listdir(base_dir):
    speech_path = os.path.join(base_dir, speech)
    pitch_list = [-12, -6, 6, 12]
    speed_list = [0.7, 1.5]
    for pitch in pitch_list:
        output_path_pitch = save_dir + speech.split(".")[0] + f"_{pitch}.wav"
        pitch_change(speech_path, output_path_pitch, pitch)
    for speed in speed_list:
        output_path_speed = save_dir + speech.split(".")[0] + f"_{speed}.wav"
        speed_change(speech_path, output_path_speed, speed)
    


