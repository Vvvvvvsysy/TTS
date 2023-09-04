import os
import time

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import TtsTrainType
from modelscope.tools import run_auto_label
from modelscope.models.audio.tts import SambertHifigan
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 工作区路径
work_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/tts_finetune_data_model_split_2/"
fine_tune_data_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/split_2/"
# save_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/generated_data_train_dev/"
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
        
            save_dir = "/dataset/fengwj/tts_data/aidatatang_200zh/corpus/generated_data_train_dev_2/"
            postive_sample = ["我要拍照", "我要录像"]
            negative_sample = ["我要关机","我要打电话","我要录屏","我要开机","打开相机","开始录像","开始拍照","结束录像","结束拍照"]
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
            # print(single_person_voice_dir)
            # f.write(str(input_wav_dir.decode())+"\n")
            continue




# model_dir = "/home/wenjun.feng/tts/transfer/child/child_xmly_2_model/"
# custom_infer_abs = {
#     'voice_name':
#     'F7',
#     'am_ckpt':
#     os.path.join(model_dir, 'tmp_am', 'ckpt'),
#     'am_config':
#     os.path.join(model_dir, 'tmp_am', 'config.yaml'),
#     'voc_ckpt':
#     os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan', 'ckpt'),
#     'voc_config':
#     os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan',
#             'config.yaml'),
#     'audio_config':
#     os.path.join(model_dir, 'data', 'audio_config.yaml'),
#     'se_file':
#     os.path.join(model_dir, 'data', 'se', 'se.npy')
# }

# kwargs = {'custom_ckpt': custom_infer_abs}

# model_id = SambertHifigan(os.path.join(model_dir, "orig_model"), **kwargs)

# inference = pipeline(task=Tasks.text_to_speech, model=model_id)

# postive_sample = ["我要拍照", "我要录像"]
# negative_sample = ["我要关机","我要打电话","我要录屏","我要开机","打开相机","开始录像","开始拍照","结束录像","结束拍照"]
# for i, sample in enumerate(postive_sample):
#     output_voice = inference(input=sample)
#     save_path = save_dir + "pos/" + f"_{i+1}.wav"
#     with open(save_path,"wb") as f:
#         f.write(output_voice["output_wav"])
# for i, sample in enumerate(negative_sample):
#     output_voice = inference(input=sample)
#     save_path = save_dir + "neg/"  + f"_0_{i}.wav"
#     with open(save_path,"wb") as f:
#         f.write(output_voice["output_wav"])




