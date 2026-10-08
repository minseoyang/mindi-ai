import os
import re
import pandas as pd

def clean_text(text):
    """
    텍스트에서 이상한 기호와 숫자를 제거:
    - 한글, 영문, 공백, 기본 구두점(. , ! ?)만 남깁니다.
    - 그 외 모든 숫자와 특수문자는 삭제합니다.
    """
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^ㄱ-ㅎ가-힣A-Za-z\s\.\,\!\?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_cha_participant_only(cha_path, label):
    """
    .cha 파일에서 Participant(P) 발화만 추출하여 DataFrame으로 반환합니다.
    - 순차적인 utterance_id
    - 'text' 컬럼에 정제된 발화 텍스트
    - 'label' 컬럼에 인자로 넘긴 정수 (예: CC → 0, CD → 1)
    """
    records = []
    utterance_id = 0

    with open(cha_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('@'):
                continue

            if line.startswith('*'):
                try:
                    speaker_tag, utterance = line[1:].split(':', 1)
                    speaker_tag = speaker_tag.strip()
                    raw_text = utterance.strip()
                except ValueError:
                    continue

                if speaker_tag.startswith('P'):
                    cleaned = clean_text(raw_text)
                    if cleaned:
                        records.append({
                            'utterance_id': utterance_id,
                            'text': cleaned,
                            'label': label
                        })
                        utterance_id += 1

    return pd.DataFrame(records)

# -----------------------------
# 1) CC 폴더 전체를 하나의 CSV로 합치기
# -----------------------------
cc_folder = 'datasets/ADReSS-IS2020-data/train/transcription/cc'
output_folder = 'transcription_dementia_bank'
os.makedirs(output_folder, exist_ok=True)

cc_records = []

for filename in os.listdir(cc_folder):
    if not filename.endswith('.cha'):
        continue

    cha_path = os.path.join(cc_folder, filename)
    # CC는 정상군 → label=0
    df_cc = parse_cha_participant_only(cha_path, label=0)
    cc_records.append(df_cc)

# 모든 CC 파일 병합
if cc_records:
    df_cc_all = pd.concat(cc_records, ignore_index=True)
    out_cc_csv = os.path.join(output_folder, 'cc_participant_all.csv')
    df_cc_all.to_csv(out_cc_csv, index=False, encoding='utf-8-sig')
    print(f"Saved combined CC utterances: {out_cc_csv}")
else:
    print("CC 폴더에 .cha 파일이 없습니다.")

# -----------------------------
# 2) (선택) CD 폴더 전체를 하나의 CSV로 합치기
# -----------------------------
cd_folder = 'datasets/ADReSS-IS2020-data/train/transcription/cd'
cd_records = []

if os.path.isdir(cd_folder):
    for filename in os.listdir(cd_folder):
        if not filename.endswith('.cha'):
            continue

        cha_path = os.path.join(cd_folder, filename)
        # CD는 치매군 → label=1
        df_cd = parse_cha_participant_only(cha_path, label=1)
        cd_records.append(df_cd)

    if cd_records:
        df_cd_all = pd.concat(cd_records, ignore_index=True)
        out_cd_csv = os.path.join(output_folder, 'cd_participant_all.csv')
        df_cd_all.to_csv(out_cd_csv, index=False, encoding='utf-8-sig')
        print(f"Saved combined CD utterances: {out_cd_csv}")
    else:
        print("CD 폴더에는 .cha 파일이 없습니다.")
else:
    print("CD 폴더가 존재하지 않습니다.")
