curl --location 'https://clovaspeech-gw.ncloud.com/recog/v1/stt?lang=Kor&assessment=false&graph=false' \
--header 'X-CLOVASPEECH-API-KEY: 3c73a73482994fae849d504267a5921d' \
--header 'Content-Type: application/octet-stream' \
--data '/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech/Validation/D99/G02/S000014/000051.wav'

# X-NCP-APIGW-API-KEY-ID="zswmu1r0s2"
# X-NCP-APIGW-API-KEY="MPpDmbFGLlqIxHUi2l1aGGOGRCYJUDbe5EaNBJA0"

# curl --location 'https://clovaspeech-gw.ncloud.com/recog/v1/stt?lang=Eng&assessment=true&graph=true' \
# --header 'X-NCP-APIGW-API-KEY-ID: zswmu1r0s2' \
# --header 'X-NCP-APIGW-API-KEY MPpDmbFGLlqIxHUi2l1aGGOGRCYJUDbe5EaNBJA0' \
# --header 'Content-Type: application/octet-stream' \
# --data '@/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech/Validation/D99/G02/S000014/000051.wav'