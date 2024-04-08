from pydub import AudioSegment
import os

file = "/home/ubuntu/Workspace/gradio_asr/examples/animals/SLAAO21000001.wav"
seg_dir = "segs"

long_audio = AudioSegment.from_wav(file)

if not os.path.exists(seg_dir):
    os.makedirs(seg_dir)
    
# PyDub handles time in milliseconds
one_minutes = 1 * 60 * 1000
hyps = []
i = 0
stop = True
while stop:
    if one_minutes * (i + 1) > len(long_audio):
        audio_file = long_audio[one_minutes * (i):]
        stop = False
    else:
        audio_file = long_audio[one_minutes * (i):one_minutes * (i + 1)]

    audio_file.export(f"{seg_dir}/seg_{i}.wav", format="wav")
    i += 1