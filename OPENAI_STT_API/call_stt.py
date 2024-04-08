import os
import argparse
import jiwer
from tqdm import tqdm
 
from openai import OpenAI
client = OpenAI()

BASE="/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlowtelSpeech"
TEST="/home/ubuntu/Workspace/gradio_asr/datas/AIHUB_TELEPHONE_LOW_QUALITY_test.txt"
LONG=True

from tools.kospeech.dataset.kspon.preprocess.preprocess import sentence_filter
from pydub import AudioSegment

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
    # for file in tqdm(open(TEST, 'r').readlines()):
    #     file = os.path.join(BASE, file.strip('\n'))
    for file in ["/home/ubuntu/Workspace/gradio_asr/examples/animals/SLAAO21000001.wav"]:
        if LONG:
            long_audio = AudioSegment.from_wav(file)

            # PyDub handles time in milliseconds
            ten_minutes = 10 * 60 * 1000
            hyps = []
            i = 0
            stop = True
            while stop:
                if ten_minutes * (i + 1) > len(long_audio):
                    audio_file = long_audio[ten_minutes * (i):]
                    stop = False
                else:
                    audio_file = long_audio[ten_minutes * (i):ten_minutes * (i + 1)]
                audio_file.export("temp_seg.wav", format="wav")

                audio_file= open("temp_seg.wav", "rb")
                transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                language="ko"
                )

                hyps.append(transcription.text)
                i += 1
            hyp = " ".join(hyps)

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

        else:
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

        # audio_file= open(file, "rb")
        # transcription = client.audio.transcriptions.create(
        # model="whisper-1", 
        # file=audio_file,
        # language="ko"
        # )

        # hyp = transcription.text

        # tname = file.replace(".wav", ".txt")
        # if "원천데이터" in file:
        #     tname = tname.replace("원천데이터", "라벨링데이터")
        # with open(tname, 'r') as f:
        #     ref = f.read()
        # ref_spell = sentence_filter(ref.strip('\n'), mode='spelling')

        # log_file.write(f"{file}\n")
        # log_file.write(f"Reference: {ref}\n")
        # log_file.write(f"Prediction: {hyp}\n")
        # log_file.write("\n")

        # wrong, length = calculate_cer(ref_spell, hyp)
        # wrongs += wrong
        # lengths += length

cer = wrongs / lengths
print(cer)