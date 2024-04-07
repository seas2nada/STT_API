YOUR_CLIENT_ID="HQdMG1JusSjxwZH_0mGh"
YOUR_CLIENT_SECRET="4lyLfZUM7Q40h4BELnc1HCQ62y-2T3WxxbYSJmf4"

YOUR_JWT_TOKEN=$(curl -X "POST" "https://openapi.vito.ai/v1/authenticate" \
  -H "accept: application/json" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=${YOUR_CLIENT_ID}&client_secret=${YOUR_CLIENT_SECRET}"  | jq -r '.access_token')

RESPONSE=$(curl -X "POST" \
  "https://openapi.vito.ai/v1/transcribe" \
  -H "accept: application/json" \
  -H "Authorization: Bearer ${YOUR_JWT_TOKEN}" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech/Validation/D99/G02/S000014/000051.wav" \
  -F 'config={"use_itn": false}')

echo $RESPONSE