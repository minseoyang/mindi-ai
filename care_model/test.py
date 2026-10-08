# 호주 치마 예리 음성 처리 처방: healthcare_model.py

from remind_prompt import *
from daily_prompt import *
from category import *
from utils import *
import openai
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import re

# GPT API 키 설정
openai.api_key = os.getenv(os.environ["OPENAI_API_KEY"])

AUDIO_DIR = "data/"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

category = pick_category()
system_prompt = get_daily_prompt(category)
conversation_log = []

# GPT 응답 생성 함수
def ask_gpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def save_log():
    full_text = "\n".join(conversation_log)
    today = datetime.today().strftime("%Y%m%d")
    filename = f"{category}_{today}.txt"
    with open(os.path.join(LOG_DIR, filename), "w", encoding="utf-8") as f:
        f.write(full_text)


class WavHandler(FileSystemEventHandler):
    def __init__(self):
        self.answer_index = 1

    def on_created(self, event):
        if event.is_directory:
            return

        filename = os.path.basename(event.src_path)
        if not re.match(r"answer\\d+\\.wav", filename):
            return

        time.sleep(0.5)  # 파일 전송 완료 대기
        filepath = event.src_path
        print(f"\n🎧 감지된 파일: {filename}")

        user_text = single_wav_to_text(filepath).strip()
        print(f"🧓 사용자(STT): {user_text}")
        if not user_text:
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        reply = ask_gpt(messages)
        print(f"🤖 GPT: {reply}")

        conversation_log.append(user_text)

# 감지 시작

def main():
    print("👀 실시간 answer*.wav 감지 시작 (Ctrl+C로 종료)")

    event_handler = WavHandler()
    observer = Observer()
    observer.schedule(event_handler, AUDIO_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n📝 대화 로그 저장 중...")
        save_log()
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()