# TTS Project
TTS(text-to-speech)，文本到语音合成任务，即给定一段文字生成对应的语音。在本项目中主要是针对一些中文指令进行语音合成，如“小晶小晶”、“拨打电话”等。该项目是为了语音识别项目做数据收集，因此需要生成的语音要求多种音色、多种语速语调以及多种风格。下面会介绍3种方法。

## Method1: API
第一种方法是调微软的API，通过切换发音人得到不同的音色以及调整生成的语速语调批量合成语音。该方法中api的调用都基于SSML.xml这个文件，里面是向微软发送请求的数据格式：
```xml
<speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="en-US">
    <voice name="zh-HK-HiuGaaiNeural">
            <prosody rate="0%" pitch="0%">小晶小晶</prosody>
    </voice>
</speak>
```
其中，name代表发音人、rate代表语速、pitch代表语调、“小晶小晶”代表要合成语音的文本。对于语速和语调，这里是从-50~50，间隔10组合取值。对于发音人可以在voice.json中查看，本项目是针对中文文本进行合成，因此只需要中文的发音人：
```c
voice = ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural", "zh-CN-YunxiNeural", "zh-CN-YunxiaNeural", "zh-CN-YunyangNeural", "zh-CN-liaoning-XiaobeiNeural", "zh-TW-HsiaoChenNeural", "zh-TW-YunJheNeural", "zh-TW-HsiaoYuNeural", "zh-CN-shaanxi-XiaoniNeural"]
```

### step1 要批量调用api首先得批量生成对应的xml文件，运行generate_voice_style.py：
```c
// 修改sample变量 这是要生成语音的文本
sample = ["小爱小爱", "小度小度", "小冰小冰","小薇小薇","小青小青","小斌小斌","小晶","语音通话","小布小布","打开屏幕","挂断电话"]
// 修改路径
updated_xml_file_path = f"/home/wenjun.feng/tts/tts_script/python_cli_demo/voice/xml_neg/{voi}_{str(rate)}_{str(pitch)}_0_{str(i)}.xml" 
// xml文件的命名规范 {voi}是发音人index {str(rate)}是语速 {str(pitch)}是语调 _0代表生成的是负样本(生成正样本时去掉_0) {str(i)}代表生成样本的index 
// 生成正样本时，最好句话的样本放在一个文件夹，因为每句话都是一类
```

### step2 批量调用api生成语音，运行tts.py:
```c
// 修改路径即可 xml_path_all是生成的xml文件路径 wav_path_all是生成的语音的存放路径
// 生成正样本时 同样也是每句话的样本放在一个文件夹
xml_path_all = ["/home/wenjun.feng/tts/tts_script/python_cli_demo/voice/xml_neg/"]
wav_path_all = ["/home/wenjun.feng/tts/tts_script/python_cli_demo/voice/voice_neg/"]
```

## Method2 个性化语音合成
第二种方法是微调个性化语音合成模型生成不同音色的语音数据。本项目针对的是中文语音合成，因此预训练模型选用的是modelscope上的个性化语音合成模型SambertHifigan，可以参考：
```c
https://www.modelscope.cn/models/damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k/summary
```
该项目简单好用，生成的语音效果也非常好，对于每个发音人只需要20条左右的音频数据即可完成模型的微调，并生成出非常逼真的语音数据。

### step 1 收集数据
收集单发音人语音数据，每个人大概需要20条3-4秒的语音或者1条2分钟的语音，这里需要把每个人的语音数据放在一个独立的文件夹。数据结构如下：
```c
/data
    /speaker1
        wav1
        wav2
        ...
        wav20
    /speaker2
        wav1
        wav2
        ...
        wav20
```
### step2 微调模型
这一步比较复杂，直接放代码，会在代码里注释。
```c
import os
import time

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import TtsTrainType
from modelscope.tools import run_auto_label
from modelscope.models.audio.tts import SambertHifigan
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

// 预训练模型id---这里不要动
pretrained_model_id = 'damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k'
// 刚才收集数据的文件夹data---需要修改
fine_tune_data_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/split_5/"
// 工作区路径---微调时会有一些中间文件以及微调后的模型
work_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/tts_finetune_data_model_split_5/"
with open("badcase_4.txt","w+") as f:
    // 遍历data 对每个人的数据微调
    for single_person_voice_dir in os.listdir(fine_tune_data_dir):
        try:
            stime = time.time()
            input_wav_dir = os.path.join(fine_tune_data_dir, single_person_voice_dir)
            //  每个发音人一个单独的工作区路径---因为每个发音人会微调得到一个模型
            single_person_work_dir = work_dir + single_person_voice_dir
            if not os.path.exists(single_person_work_dir):
                os.mkdir(single_person_work_dir)
            else:
                print(single_person_voice_dir + "processed over")
                continue
            // 存放数据处理的中间文件
            output_processed_data_dir = os.path.join(single_person_work_dir, "data_processed")
            if not os.path.exists(output_processed_data_dir):
                os.mkdir(output_processed_data_dir)
            // 数据前处理 数据标注等
            ret, report = run_auto_label(input_wav=input_wav_dir, work_dir=output_processed_data_dir, resource_revision="v1.0.7")

            dataset_id = output_processed_data_dir
            // 微调后的模型保存路径
            pretrain_work_dir = os.path.join(single_person_work_dir, "model_finetunes/")

            if not os.path.exists(pretrain_work_dir):
                os.mkdir(pretrain_work_dir)
            
            train_info = {
                TtsTrainType.TRAIN_TYPE_SAMBERT: {  // 配置训练AM（sambert）模型
                    'train_steps': 202,               // 训练多少个step 
                    'save_interval_steps': 200,       // 每训练多少个step保存一次checkpoint
                    'log_interval': 10               // 每训练多少个step打印一次训练日志
                }
            }
            // 配置训练参数，指定数据集，临时工作目录和train_info
            kwargs = dict(
                model=pretrained_model_id,                  // 指定要finetune的模型
                model_revision = "v1.0.6",
                work_dir=pretrain_work_dir,                 // 指定临时工作目录
                train_dataset=dataset_id,                   // 指定数据集id
                train_type=train_info                       // 指定要训练类型及参数
            )

            trainer = build_trainer(Trainers.speech_kantts_trainer,default_args=kwargs)
            
            trainer.train()
            // 至此模型训练完毕
            // 下面开始加载模型 并对正负样本进行生成
            model_dir = os.path.abspath(pretrain_work_dir)
            custom_infer_abs = {
                'voice_name':
                'F7',
                'am_ckpt':
                os.path.join(model_dir, 'tmp_am', 'ckpt'),
                'am_config':
                os.path.join(model_dir, 'tmp_am', 'config.yaml'),
                'voc_ckpt':
                os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan', 'ckpt'),
                'voc_config':
                os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan',
                        'config.yaml'),
                'audio_config':
                os.path.join(model_dir, 'data', 'audio_config.yaml'),
                'se_file':
                os.path.join(model_dir, 'data', 'se', 'se.npy')
            }

            kwargs = {'custom_ckpt': custom_infer_abs}
            // 加载模型
            model_id = SambertHifigan(os.path.join(model_dir, "orig_model"), **kwargs)

            inference = pipeline(task=Tasks.text_to_speech, model=model_id)
            // 语音样本保存地址---save_dir文件夹下要自己创建pos和neg两个文件夹
            save_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/generated_data_train_dev/"
            postive_sample = ["小晶小晶", "拨打电话", "关闭屏幕", "小金小金"]
            negative_sample = ["小爱小爱", "小度小度", "小冰小冰","小薇小薇","小青小青","小斌小斌","小晶","语音通话","小布小布","打开屏幕","挂断电话"]
            for i, sample in enumerate(postive_sample):
                output_voice = inference(input=sample)
                save_path = save_dir + "pos/" + single_person_voice_dir + f"_{i+1}.wav"
                with open(save_path,"wb") as f:
                    f.write(output_voice["output_wav"])
            for i, sample in enumerate(negative_sample):
                output_voice = inference(input=sample)
                save_path = save_dir + "neg/" + single_person_voice_dir + f"_0_{i}.wav"
                with open(save_path,"wb") as f:
                    f.write(output_voice["output_wav"])
            // 语音样本保存地址---save_dir文件夹下要自己创建pos和neg两个文件夹
            save_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/generated_data_train_dev_2/"
            postive_sample = ["我要拍照", "我要录像"]
            negative_sample = ["我要关机","我要打电话","我要录屏","我要开机","打开相机","开始录像","开始拍照","结束录像","结束拍照"]
            for i, sample in enumerate(postive_sample):
                output_voice = inference(input=sample)
                save_path = save_dir + "pos/" + single_person_voice_dir + f"_{i+1}.wav"
                with open(save_path,"wb") as f:
                    f.write(output_voice["output_wav"])
            for i, sample in enumerate(negative_sample):
                output_voice = inference(input=sample)
                save_path = save_dir + "neg/" + single_person_voice_dir + f"_0_{i}.wav"
                with open(save_path,"wb") as f:
                    f.write(output_voice["output_wav"])
            end_time = time.time()
            print("*" * 50)
            print(single_person_voice_dir + "processed over")
            print("time consumed:")
            print(end_time-stime)
            print("*" * 50)
        except:
            // f.write(str(single_person_work_dir)+"\n")    
            continue
```
### step3 生成语音
此前在step2中对模型微调完已经生成一批数据，但是如果之后还想使用微调过的模型，生成新数据，即可参考这一步。
```c
import os
import time

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import TtsTrainType
from modelscope.tools import run_auto_label
from modelscope.models.audio.tts import SambertHifigan
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

// 工作区路径
work_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/tts_finetune_data_model_split_2/"
fine_tune_data_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/split_2/"
with open("badcase_2.txt","w") as f:
    for single_person_voice_dir in os.listdir(fine_tune_data_dir):
        try:
            stime = time.time()
            input_wav_dir = os.path.join(fine_tune_data_dir, single_person_voice_dir)
            single_person_work_dir = work_dir + single_person_voice_dir
            pretrain_work_dir = os.path.join(single_person_work_dir, "model_finetunes/")
            model_dir = os.path.abspath(pretrain_work_dir)
            custom_infer_abs = {
                'voice_name':
                'F7',
                'am_ckpt':
                os.path.join(model_dir, 'tmp_am', 'ckpt'),
                'am_config':
                os.path.join(model_dir, 'tmp_am', 'config.yaml'),
                'voc_ckpt':
                os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan', 'ckpt'),
                'voc_config':
                os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan',
                        'config.yaml'),
                'audio_config':
                os.path.join(model_dir, 'data', 'audio_config.yaml'),
                'se_file':
                os.path.join(model_dir, 'data', 'se', 'se.npy')
            }

            kwargs = {'custom_ckpt': custom_infer_abs}

            model_id = SambertHifigan(os.path.join(model_dir, "orig_model"), **kwargs)

            inference = pipeline(task=Tasks.text_to_speech, model=model_id)
            save_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/generated_data_train_dev/"
            postive_sample = ["小晶小晶", "拨打电话", "关闭屏幕", "小金小金"]
            negative_sample = ["小爱小爱", "小度小度", "小冰小冰","小薇小薇","小青小青","小斌小斌","小晶","语音通话","小布小布","打开屏幕","挂断电话"]
            for i, sample in enumerate(postive_sample):
                save_path = save_dir + "pos/" + single_person_voice_dir + f"_{i+1}.wav"
                if os.path.exists(save_path):
                    continue
                output_voice = inference(input=sample)
                
                print(save_path)
                with open(save_path,"wb") as f:
                    f.write(output_voice["output_wav"])
            for i, sample in enumerate(negative_sample):
                save_path = save_dir + "neg/" + single_person_voice_dir + f"_0_{i}.wav"
                if os.path.exists(save_path):
                    continue
                output_voice = inference(input=sample)
                
                with open(save_path,"wb") as f:
                    f.write(output_voice["output_wav"])
            end_time = time.time()
            print("*" * 50)
            print(single_person_voice_dir + "processed over")
            print("time consumed:")
            print(end_time-stime)
            print("*" * 50)
        except:
            continue
```

## Method3 语音编辑
第三种方法base在方法二上，是一种语音到语音的方法，因此需要已生成的样本，在此基础上对样本进行编辑，也是直接上代码。
```c
import os
import subprocess

from pydub import AudioSegment
import librosa
import soundfile as sf

import pyworld as pw
import numpy as np

// 修改音调
def pitch_change(input_path, output_path, rate):
    // 读取音频文件
    input_audio, sr = sf.read(input_path)
    // 提取音频的基频（音调）和谐波信息
    f0, sp, ap = pw.wav2world(input_audio, sr)
    // 将音调降低一个半音
    new_f0 = f0 * 2 ** (rate/12)
    // 使用合成函数将音频合成回去
    synthesized_audio = pw.synthesize(new_f0, sp, ap, sr)
    // 将合成音频保存为文件
    sf.write(output_path, synthesized_audio, sr)

// 修改语速
def speed_change(input_path, output_path, speed_factor):
    // 构造调用 sox 的命令
    command = ['sox', input_path, output_path, 'tempo', str(speed_factor)]
    // 执行命令
    subprocess.run(command)

// 待编辑数据路径
base_dir = "/home/wenjun.feng/tts/transfer/child/generated_voice/pos/"
// 数据保存地址
save_dir = "/home/wenjun.feng/tts/transfer/child/voice_change/"

// 对语速和语调进行调整 
// 下面的pitch list和speed list是我试过的比较好的参数
// 编辑后数据可以扩展6倍
for speech in os.listdir(base_dir):
    speech_path = os.path.join(base_dir, speech)
    pitch_list = [-12, -6, 6, 12] // 负数表示降调 正数表示升调
    speed_list = [0.7, 1.5] // 调整语速的因子
    for pitch in pitch_list:
        output_path_pitch = save_dir + speech.split(".")[0] + f"_{pitch}.wav"
        pitch_change(speech_path, output_path_pitch, pitch)
    for speed in speed_list:
        output_path_speed = save_dir + speech.split(".")[0] + f"_{speed}.wav"
        speed_change(speech_path, output_path_speed, speed)
```


