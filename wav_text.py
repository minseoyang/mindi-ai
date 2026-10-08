import os
import torch
import torchaudio
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def transcribe_nested_wav_to_csv(root_dir, output_csv_path, label=1, model_name="openai/whisper-small", language="en"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data = []

    for speaker in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue

        for fname in os.listdir(speaker_path):
            if not fname.endswith(".wav"):
                continue

            wav_path = os.path.join(speaker_path, fname)
            speech_array, sr = torchaudio.load(wav_path)
            speech_array = speech_array.squeeze()  # [1, N] → [N]
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                speech_array = resampler(speech_array)
            speech_array = speech_array.numpy()  # Whisper는 numpy float32 배열을 기대
            inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt").to(device)

            with torch.no_grad():
                predicted_ids = model.generate(inputs["input_features"])
                transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            data.append({
                "speaker": speaker,
                "filename": fname,
                "transcript": transcript,
                "label": label
            })
            print(f"✅ {speaker}/{fname} → {transcript}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📝 저장 완료: {output_csv_path}")


# 실행 예시
transcribe_nested_wav_to_csv(
    root_dir="nodementia",  # 루트 폴더 경로
    output_csv_path="nodementia_transcripts.csv",
    label=0,  # 예: dementia = 1
    language="en"  # Whisper 언어 설정
)
