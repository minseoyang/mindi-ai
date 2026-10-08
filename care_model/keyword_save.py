import openai
import json
import os

KEYWORD_DIR = "logs/keywords"
os.makedirs(KEYWORD_DIR, exist_ok=True)

def extract_and_save_keywords(text, date_str):
    prompt = f"""
다음 대화에서 중요한 키워드(사람, 장소, 활동 등)를 뽑아주세요.
- 문장은 쓰지 말고, 불릿포인트 목록으로 정리해주세요.
- 중복 없이 명사 위주로 요약해주세요.

{text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 노인과의 대화에서 회상할 중요한 키워드를 추출하는 도우미입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    keywords = [line.strip("- ").strip() for line in response['choices'][0]['message']['content'].split("\n") if line.strip()]
    with open(os.path.join(KEYWORD_DIR, f"{date_str}.json"), "w", encoding="utf-8") as f:
        json.dump({"keywords": keywords}, f, ensure_ascii=False, indent=2)
