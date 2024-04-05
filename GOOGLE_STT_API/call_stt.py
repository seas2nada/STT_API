import os

from google.cloud import speech, speech_v2
from google.cloud.speech_v2.types import cloud_speech


credential_path = "/home/ubuntu/Workspace/STT_API/GOOGLE_STT_API/sttapi_credential.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
project_id = "sttapi-419407"

# encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
config = speech_v2.RecognitionConfig(
    auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
    language_codes="ko-KR",
)
client = speech_v2.SpeechClient()

for file in ["/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech/Validation/D99/G02/S000014/000051.wav"]:
    file = file.strip('\n')
    
    content = open(file, 'rb').read()

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/global/recognizers/_",
        config=config,
        content=content,
    )
    response = client.recognize(request=request)

    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")