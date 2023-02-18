"""
A memory-optimised version of the stock WhisperX script, which unloads and GCs garbage in memory and in CUDA cache, before attempting alignment.
"""
import whisperx
import sys
import gc
import torch

from typing import Iterator, TextIO

audio_file = sys.argv[1]
device = 'cuda'


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


def write_vtt(transcript: Iterator[dict], file: TextIO):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


whisper_model = whisperx.load_model("medium.en", device)
result = whisper_model.transcribe(audio_file)

# Garbage collect everything before loading alignment model
del whisper_model
gc.collect()
torch.cuda.empty_cache()

input('Press enter to proceed with alignment...')

# load alignment model and metadata
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device)

# align whisper output
result_aligned = whisperx.align(
    result["segments"], model_a, metadata, audio_file, device)

for segment in result_aligned["segments"]:
    print(format_timestamp(segment['start']), segment['text'])

with open(audio_file + ".vtt", 'w', encoding='utf-8') as vtt:
    write_vtt(result_aligned['segments'], file=vtt)

print("wrote VTT sub file to ", audio_file + '.vtt')
