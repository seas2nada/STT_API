import json
import requests
import argparse
import jiwer

import os

from tqdm import tqdm

from tools.kospeech.dataset.kspon.preprocess.preprocess import sentence_filter

# usage: export PYTHONPATH=$PWD:$PYTHONPATH; python RTZR_STT_API/call_stt.py --log_file_name 'log.txt'

BASE="/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlowtelSpeech"
TEST="/home/ubuntu/Workspace/gradio_asr/datas/AIHUB_TELEPHONE_LOW_QUALITY_test.txt"
YOUR_CLIENT_ID="HQdMG1JusSjxwZH_0mGh"
YOUR_CLIENT_SECRET="4lyLfZUM7Q40h4BELnc1HCQ62y-2T3WxxbYSJmf4"

def calculate_cer(ref, hyp):
    cer = jiwer.cer(ref, hyp)
    return cer * len(ref), len(ref)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file_name", type=str)

    return parser.parse_args()

args = get_args()

resp = requests.post(
    'https://openapi.vito.ai/v1/authenticate',
    data={'client_id': f'{YOUR_CLIENT_ID}',
          'client_secret': f'{YOUR_CLIENT_SECRET}'}
)
resp.raise_for_status()
token = resp.json()['access_token']

config = {"use_itn": "false", "use_disfluency_filter": "false", "use_profanity_filter": "false"}

wrongs = 0
lengths = 0
with open(args.log_file_name, 'w') as log_file:
    for file in tqdm(open(TEST, 'r').readlines()):
    # for file in ["/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech/Validation/D99/G02/S000014/000051.wav"]:
        file = os.path.join(BASE, file.strip('\n'))

        resp = requests.post(
            'https://openapi.vito.ai/v1/transcribe',
            headers={'Authorization': 'bearer ' + f'{token}'},
            data={'config': json.dumps(config)},
            files={'file': open(file, 'rb')}
        )
        resp.raise_for_status()
        TRANSCRIBE_ID = resp.json()['id']
        YOUR_JWT_TOKEN = token

        result = {'status': 'transcribing'}

        while result['status'] == 'transcribing':
            resp = requests.get(
                'https://openapi.vito.ai/v1/transcribe/'+f'{TRANSCRIBE_ID}',
                headers={'Authorization': 'bearer '+f'{YOUR_JWT_TOKEN}'},
            )
            resp.raise_for_status()
            result = resp.json()
        
        segments = []
        for utt in result['results']['utterances']:
            segments.append(utt['msg'])
        hyp = " ".join(segments)

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
            # ref = ref_spell
        else:
            wrong = wrong_phone
            length = length_phone
            # ref = ref_phone

        wrongs += wrong
        lengths += length

        log_file.write(f"{file}\n")
        log_file.write(f"Reference: {ref}\n")
        log_file.write(f"Prediction: {hyp}\n")
        log_file.write("\n")
    
cer = wrongs / lengths
print(cer)