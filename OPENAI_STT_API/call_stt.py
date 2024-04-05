import os
import argparse
import jiwer
from tqdm import tqdm
 
from openai import OpenAI
client = OpenAI()

BASE="/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlowtelSpeech"
TEST="/home/ubuntu/Workspace/gradio_asr/datas/AIHUB_TELEPHONE_LOW_QUALITY_test.txt"

from tools.kospeech.dataset.kspon.preprocess.preprocess import sentence_filter

# usage: export PYTHONPATH=$PWD:$PYTHONPATH; export OPENAI_API_KEY=${key}; python OPENAI_STT_API/call_stt.py --log_file_name OPENAI_STT_API/log.txt

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
        
        audio_file= open(file, "rb")
        transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        language="ko"
        )

        hyp = transcription.text

        tname = file.replace(".wav", ".txt")
        if "원천데이터" in file:
            tname = tname.replace("원천데이터", "라벨링데이터")
        with open(tname, 'r') as f:
            ref = f.read()
        ref_spell = sentence_filter(ref.strip('\n'), mode='spelling')

        log_file.write(f"{file}\n")
        log_file.write(f"Reference: {ref}\n")
        log_file.write(f"Prediction: {hyp}\n")
        log_file.write("\n")

        wrong, length = calculate_cer(ref_spell, hyp)
        wrongs += wrong
        lengths += length

cer = wrongs / lengths
print(cer)