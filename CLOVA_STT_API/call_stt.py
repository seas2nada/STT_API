import os
import re
import argparse
import jiwer
from tqdm import tqdm

import sys
import requests
from tools.kospeech.dataset.kspon.preprocess.preprocess import sentence_filter

client_id = "zswmu1r0s2"
client_secret = "MPpDmbFGLlqIxHUi2l1aGGOGRCYJUDbe5EaNBJA0"
lang = "Kor" # 언어 코드 ( Kor, Jpn, Eng, Chn )
url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang

# BASE="/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlowtelSpeech"
# TEST="/home/ubuntu/Workspace/gradio_asr/datas/AIHUB_TELEPHONE_LOW_QUALITY_test.txt"
BASE="/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech"
TEST="/home/ubuntu/Workspace/gradio_asr/datas/AIHUB_KOREAN_LECTURE_test.txt"

pattern = r'text":"(.*?)"'

# usage: export PYTHONPATH=$PWD:$PYTHONPATH; python CLOVA_STT_API/call_stt.py --log_file_name CLOVA_STT_API/log.txt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file_name", type=str)

    return parser.parse_args()

def calculate_cer(ref, hyp):
    cer = jiwer.cer(ref, hyp)
    return cer * len(ref), len(ref)

args = get_args()

wrongs = 0
lengths = 0
with open(args.log_file_name, 'w') as log_file:
    for file in tqdm(open(TEST, 'r').readlines()):
        file = os.path.join(BASE, file.strip('\n'))

        data = open(file, 'rb')
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Content-Type": "application/octet-stream"
        }
        response = requests.post(url,  data=data, headers=headers)
        rescode = response.status_code
        if(rescode == 200):
            hyp = response.text
            match = re.search(pattern, hyp)
            hyp = match.group(1).strip()
        else:
            raise NotImplementedError("error")

        tname = file.replace(".wav", ".txt")
        if "원천데이터" in file:
            tname = tname.replace("원천데이터", "라벨링데이터")
        with open(tname, 'r') as f:
            ref = f.read()
        ref_spell = sentence_filter(ref.strip('\n'), mode='spelling')
        ref_phone = sentence_filter(ref.strip('\n'), mode='phonetic')

        wrong_spell, length_spell = calculate_cer(ref_spell, hyp)
        spell_cer = wrong_spell / length_spell
        wrong_phone, length_phone = calculate_cer(ref_phone, hyp)
        phone_cer = wrong_phone / length_phone

        if spell_cer < phone_cer:
            wrong = wrong_spell
            length = length_spell
        else:
            wrong = wrong_phone
            length = length_phone

        log_file.write(f"{file}\n")
        log_file.write(f"Reference: {ref}\n")
        log_file.write(f"Prediction: {hyp}\n")
        log_file.write("\n")

        wrongs += wrong
        lengths += length

cer = wrongs / lengths
print(cer)