import os
import xml.etree.ElementTree as ET
import random


voice = ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural", "zh-CN-YunxiNeural", "zh-CN-YunxiaNeural", "zh-CN-YunyangNeural", "zh-CN-liaoning-XiaobeiNeural", "zh-TW-HsiaoChenNeural", "zh-TW-YunJheNeural", "zh-TW-HsiaoYuNeural", "zh-CN-shaanxi-XiaoniNeural"]
sample = ["小爱小爱", "小度小度", "小冰小冰","小薇小薇","小青小青","小斌小斌","小晶","语音通话","小布小布","打开屏幕","挂断电话"]
length = len(sample)
count = 0
for voi in voice:
    for rate in range(-50,51,10):
        for pitch in range(-50,51,10):
            for i in range(len(sample)): # 控制正负样本数量
                tree = ET.parse("/home/wenjun.feng/tts/tts_script/python_cli_demo/SSML.xml")
                root = tree.getroot()
                for voice_element in root.findall(".//ns0:voice", namespaces={"ns0": "http://www.w3.org/2001/10/synthesis"}):
                    voice_element.set("name", voi)

                for voice_element in root.findall(".//ns0:voice", namespaces={"ns0": "http://www.w3.org/2001/10/synthesis"}):
                    prosody_element = voice_element.find('{http://www.w3.org/2001/10/synthesis}prosody')
                    if prosody_element is not None:
                        # 修改<prosody>元素的文本内容
                        prosody_element.text = sample[i]
                for prosody_element in root.findall(".//ns0:prosody", namespaces={"ns0": "http://www.w3.org/2001/10/synthesis"}):
                    prosody_element.set("rate", str(rate)+"%")
                    prosody_element.set("pitch", str(pitch)+"%")
                updated_xml_file_path = f"/home/wenjun.feng/tts/tts_script/python_cli_demo/voice/xml_neg/{voi}_{str(rate)}_{str(pitch)}_0_{str(i)}.xml"  # 替换为你想要保存的文件路径
                tree.write(updated_xml_file_path, encoding="utf-8", xml_declaration=True)
                count += 1


