import pickle
import whisperx
import sys

audio_file = sys.argv[1]
input_file = sys.argv[2]
device = 'cuda'

def format_ts(timestamp: float) -> str:
    secs = int(timestamp)
    return "{:0>2}:{:0>2}:{:0>2}".format(secs // 3600, (secs % 3600) // 60, (secs % 3600) % 60)

with open(input_file, 'rb') as f:
    result = pickle.load(f)
    
# load alignment model and metadata
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

# align whisper output
result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file, device)

for segment in result_aligned["segments"]:
  print(format_ts(segment['start']), segment['text'])
