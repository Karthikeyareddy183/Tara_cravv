import os, io, requests
import soundfile as sf
import numpy as np
import wave as w

# Load .env
for l in open('.env').read().splitlines():
    l = l.strip()
    if l and not l.startswith('#') and '=' in l:
        k, v = l.split('=', 1)
        os.environ[k.strip()] = v.strip()

# Read audio
audio, sr = sf.read('assets/denoised_output.wav', dtype='float32')
int16 = (np.clip(audio, -1., 1.) * 32767).astype(np.int16)

# Convert to wav buffer
buf = io.BytesIO()
with w.open(buf, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(int16.tobytes())

# Call Deepgram
resp = requests.post(
    'https://api.deepgram.com/v1/listen',
    params={
        'model': 'nova-2',
        'language': 'en',
        'smart_format': 'true'
    },
    headers={
        'Authorization': f'Token {os.environ["DEEPGRAM_API_KEY"]}',
        'Content-Type': 'audio/wav'
    },
    data=buf.getvalue(),
    timeout=30
)

# Extract words
words = resp.json().get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('words', [])

for wrd in words:
    if abs(wrd.get('start', 0) - 35) < 2:
        print(wrd['word'], round(wrd['start'], 2), round(wrd.get('confidence', 0), 2))