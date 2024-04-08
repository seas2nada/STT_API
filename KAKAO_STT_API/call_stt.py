import os
import argparse
import jiwer
from tqdm import tqdm

import wave
import asyncio
import datetime
import asyncio
import json
from os.path import exists
# OpenSources, need install
# https://pypi.org/project/websockets/
import websockets
# https://pypi.org/project/aiofile/
from aiofile import AIOFile, Reader

import re

from tools.kospeech.dataset.kspon.preprocess.preprocess import sentence_filter
pattern = r'value":"(.*?)"'

# usage: export PYTHONPATH=$PWD:$PYTHONPATH; python KAKAO_STT_API/call_stt.py --log_file_name KAKAO_STT_API/log.txt

BASE="/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/tvpro_sample"
TEST="/home/ubuntu/Workspace/gradio_asr/datas/AIHUB_TVPRO_test.txt"
# BASE="/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KconfSpeech"
# TEST="/home/ubuntu/Workspace/gradio_asr/datas/AIHUB_CONFERENCE_CALL_test.txt"
# BASE="/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KtelSpeech"
# TEST="/home/ubuntu/Workspace/gradio_asr/datas/AIHUB_COUNSELING_test.txt"
# BASE="/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlowtelSpeech"
# TEST="/home/ubuntu/Workspace/gradio_asr/datas/AIHUB_TELEPHONE_LOW_QUALITY_test.txt"
# BASE="/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech"
# TEST="/home/ubuntu/Workspace/gradio_asr/datas/AIHUB_KOREAN_LECTURE_test.txt"

def remove_non_korean(sentence):
    cleaned_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z ]', '', sentence)
    return cleaned_sentence

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file_name", type=str)

    return parser.parse_args()

def calculate_cer(ref, hyp):
    cer = jiwer.cer(ref, hyp)
    return cer * len(ref), len(ref)

class WebSocketClient():
    # Custom class for handling websocket client
    def __init__(self, url, onStartMessage, bits_per_seconds):
        self.url=url
        # chunk size is depend on sendfile duration, which is now 0.02s(20ms)
        # set chunk size as byte unit
        self.chunksize=bits_per_seconds*0.02/8
        self.onStartMessage = onStartMessage
        pass

    async def connect(self):
        self.connection = await websockets.connect(self.url)
        if self.connection.open:
            await self.connection.send(json.dumps(self.onStartMessage))
            return self.connection

    async def receiveMessage(self, connection, file, ref, log_file):
        while True:
            try:
                message = await connection.recv()
                if message is not None:
                    match = re.search(pattern, message)
                    if match is not None:
                        hyp = match.group(1).strip()
                        
                        log_file.write(f"{file}\n")
                        log_file.write(f"Reference: {ref}\n")
                        log_file.write(f"Prediction: {hyp}\n")
                        log_file.write("\n")

            except websockets.exceptions.ConnectionClosed as e:
                print('Connection with server closed')
                break
            except Exception as e:
                print(e)

    async def sendfile(self, connection, filepath):
        try:
            async with AIOFile(filepath, 'rb') as afp:
                reader = Reader(afp, chunk_size=self.chunksize)
                async for chunk in reader:
                    await connection.send(chunk)
                    await asyncio.sleep(0.02)
        except Exception as e:
            print(e)

def argsChecks(args):
    # Check given arguments are valid
    # Please check guide for more details
    if not exists(args["filepath"]):
        raise "Please give exist filepath in filepath args"
    
    filepath = args["filepath"]
    onStartMessage = {
        "type": "recogStart",
        "service": "DICTATION",
        "requestId": "GNTWSC-{}".format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
        "showFinalOnly": True,
        "showExtraInfo": args["showExtraInfo"],
    }
    if filepath.endswith(".wav"):
        with wave.open(filepath, 'rb') as wf:
            bit_depth = wf.getsampwidth() * 8
            samplerate = wf.getframerate()
            channels = wf.getnchannels()
            onStartMessage["audioFormat"] = "RAWPCM/{bitDepth}/{sampleRate}/{channel}/_/_".format(bitDepth=bit_depth, sampleRate=samplerate, channel=channels)
            bits_per_seconds = bit_depth * samplerate * channels
    elif filepath.endswith(".pcm"):
        # If file is PCM data
        onStartMessage["audioFormat"] = "RAWPCM/16/16000/1/_/_"
        bits_per_seconds = 256000
    elif filepath.endswith(".mp3"):
        # If file is MP3 data
        onStartMessage["audioFormat"] = "MP3/16/16000/1/_/_"
        bits_per_seconds = 256000
    return args["url"], filepath, onStartMessage, bits_per_seconds

if __name__ == '__main__':
    long = "wss://2a5d8510-f29a-4576-ab7b-b9599133bb69.api.kr-central-1.kakaoi.io/ai/speech-to-text/ws/long?signature=50eaf12cd3d6456b847e2667cd487198&x-api-key=4e890a4a7a84dbca02f8d7df4aa56ac6"
    short = "wss://2a5d8510-f29a-4576-ab7b-b9599133bb69.api.kr-central-1.kakaoi.io/ai/speech-to-text/ws?signature=50eaf12cd3d6456b847e2667cd487198&x-api-key=4e890a4a7a84dbca02f8d7df4aa56ac6"

    args = get_args()

    with open(args.log_file_name, 'w') as log_file:
        for file in tqdm(open(TEST, 'r').readlines()):
            file = os.path.join(BASE, file.strip('\n'))
            
            tname = file.replace(".wav", ".txt")
            if "원천데이터" in file:
                tname = tname.replace("원천데이터", "라벨링데이터")
            with open(tname, 'r') as f:
                ref = f.read()
            ref = sentence_filter(ref.strip('\n'), mode='spelling')

            args = {
                "url": short,
                "filepath": file,
                "showFinalOnly": True,
                "showExtraInfo": False,
            }
            
            url, filepath, onStartMessage, bits_per_seconds = argsChecks(args)

            # Creating client object
            client = WebSocketClient(url, onStartMessage, bits_per_seconds)
            loop = asyncio.get_event_loop()
            # Start connecting
            connection = loop.run_until_complete(client.connect())
            # Define async jobs
            tasks = [
                asyncio.ensure_future(client.sendfile(connection, filepath)),
                asyncio.ensure_future(client.receiveMessage(connection, filepath, ref, log_file)),
            ]
            # Run async jobs
            loop.run_until_complete(asyncio.wait(tasks))