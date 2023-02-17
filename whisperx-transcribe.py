import whisperx
import pickle
import sys

device = "cuda" 
audio_file = sys.argv[1]

# transcribe with original whisper
model = whisperx.load_model("medium.en", device)
result = model.transcribe(audio_file)

temp_file_name = audio_file + '.results.pkl'

with open(temp_file_name, 'wb') as f:
    pickle.dump(result, f)

print('Dumped transcription results to temp pickle file: ', temp_file_name)
