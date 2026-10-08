import os
from remind_prompt import *
from daily_prompt import *
from category import *
from utils import *
import openai

# GPT API 키 설정 (환경변수 권장)
openai.api_key = os.getenv(os.environ["OPENAI_API_KEY"])

# 음성파일 폴더
AUDIO_DIR = "data/"

# 대화 이력 저장 폴더
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# GPT 응답 생성 함수
def ask_gpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def get_latest_txt_file():
    txt_files = [
        f for f in os.listdir(LOG_DIR)
        if f.endswith(".txt")
    ]
    if not txt_files:
        return None

    # 수정시간 기준 정렬
    txt_files.sort(key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)), reverse=True)
    return os.path.join(LOG_DIR, txt_files[0])

def main():
    print("\n💬 음성 기반 치매 예방 회상 대화 시작")
    has_previous_log = any(fname.endswith(".txt") for fname in os.listdir(LOG_DIR))

    if has_previous_log:
        latest_path = get_latest_txt_file()
        if latest_path:
            with open(latest_path, "r", encoding="utf-8") as f:
                yesterday_text = f.read()

            prompt = get_remind_prompt(yesterday_text)

            for i in range (1,4):
                remind(prompt)

    # 일상 대화: 항상 진행
    MAX_INDEX = 23  # 최대 파일 수 예상 범위 (필요 시 조정)
    category = pick_category()

    for i in range(4, MAX_INDEX + 1):
        messages = daily(category)

    # 대화 저장
    full_text = "\n".join([m["content"] for m in messages if m["role"] == "user"])
    today = datetime.today().strftime("%Y%m%d")
    filename = f"{category}_{today}.txt"
    path = "logs/"+filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(full_text)

if __name__ == "__main__":
    main()
