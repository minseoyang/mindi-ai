import os
from language_model.final_BERT import *
from language_model.gpt import *
import openai
import json
import subprocess

openai.api_key=os.environ["OPENAI_API_KEY"]


def count_korean_fruits_vegetables(user_text: str) -> dict:
    system_prompt = (
        "You are a helper that extracts only the names of fruits and vegetables mentioned in Korean sentences.\n"
        "Count each item only once (no duplicates), and respond strictly in the following JSON format:\n"
        '{"count": <number>, "items": ["apple", "carrot", ...]}'
    )

    user_prompt = f'User input: "{user_text}"\n\nPlease extract only the fruits and vegetables mentioned in the sentence.'


    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=150
    )

    reply = response.choices[0].message.content.strip()
    print("[GPT 응답]:", reply)

    reply = reply.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(reply)
        return result
    except Exception as e:
        print(f"JSON 파싱 실패: {e}")
        return {"count": -1, "items": []}


def convert_mp4_to_wav(input_path, output_path, sample_rate=16000):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path
    ]

    subprocess.run(command, check=True)

def evaluate_answer(question: str, answer: str, user_answer: str) -> dict:

    system_prompt = (
        "You are an evaluator who judges whether the user's answer is semantically correct compared to the expected answer.\n"
        "Even if the wording or pronunciation is slightly different, accept it as correct if the meaning is the same.\n"
        "However, if the meaning is incorrect or irrelevant, mark it as wrong.\n"
        "Always respond strictly in the following JSON format:\n\n"
        '{"score": 1}  # If the meaning matches\n'
        '{"score": 0}  # If the meaning does not match\n'
        "Do not say anything else."
    )

    user_prompt = (
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"User's Answer: {user_answer}\n\n"
        "Please respond in JSON format with a score of 1 if the user's answer is semantically correct, or 0 if not."
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=50
    )

    reply = response.choices[0].message.content.strip()
    print("[GPT answer]:", reply)

    try:
        return json.loads(reply)
    except Exception as e:
        print(f"JSON 파싱 실패: {e}")
        return {"score": -1}


def language_model(answer1, answer2, answer3, answer4, answer5, answer7, answer8, answer9,answer10,answer11, answer12,
                   answer13, answer14, answer15, answer16, answer17, answer18, answer19, answer20, answer21):
    score = 0
    Orientation = [
        ("What year is it?", "2025", answer1),
        ("What month is it now?", "June", answer2),
        ("What day of the month is it today?", "four", answer3),
        ("What day of the week is it today?", "wednesday", answer4),
        ("What season is it now?", "summer", answer5),
    ]

    Attention = [
        ("Please repeat the numbers in the exact order I say them: 6 9 7 3", "6 9 7 3", answer7),
        (
            "Please repeat the numbers in the exact order I say them: 5 7 2 8 4", "5 7 2 8 4", answer8),
        ("Please repeat the following word backward: 'apple-honey-jam-berry'", "berry-jam-honey-apple", answer9),
    ]

    Memory = [
        ("What was the person's name I mentioned earlier? 1. Youngsoo 2. Minsoo 3. Jinsoo", "Youngsoo", answer11),
        ("What did he ride? 1. Bus 2. Motorcycle 3. Bicycle", "Bicycle", answer12),
        ("Where did he go? 1. Park 2. Playground 3. Field", "Park", answer13),
        ("What did he do? 1. Basketball 2. Soccer 3. Baseball", "Baseball", answer14),
        ("What time did he start? 1. 10 o'clock 2. 11 o'clock 3. 3 o'clock", "11 o'clock", answer15)
    ]

    Executive_Function = [
        ("1, 2, 3, ?, 5, 6, Look at the number pattern. Say the number that should come next", "four", answer16),
        ("A, B, C, ?, ?, Look at the letter pattern. Say the next two letters that fit the sequence", "C D", answer17)
    ]
    # answer 18
    Count = [
        ("Now, please name as many fruits or vegetables as you can.")
    ]

    for (q, a, u) in Orientation:
        result = evaluate_answer(q, a, u)
        score += result["score"]

    for (q, a, u) in Attention:
        result = evaluate_answer(q, a, u)
        score += result["score"]

    sentence_memory = evaluate_answer("Please repeat the sentence I asked you to memorize a moment ago.",
                       "Minsoo rode his bicycle to the park and played baseball starting at 11 o'clock.", answer10)
    if sentence_memory["score"] == 1:
        score += 10
    else:
        for (q, a, u) in Memory:
            result = evaluate_answer(q, a, u)
            score += result["score"] * 2

    for (q, a, u) in Executive_Function:
        result = evaluate_answer(q, a, u)
        score += result["score"]*2

        # Bert_MLM
        Bert_rst = 0

        # bert 1
        bert_score = Bert_final(answer18)

        # check link (연관성 점수)
        utterance_a = "Does this item bring back any memories? Could you also tell me how you used it?"
        utterance_b = answer18

        link_score = check_link([utterance_a, utterance_b])
        if link_score < 2:
            Bert_rst += bert_score * 0.5
        else:
            Bert_rst += bert_score

        # BERT2
        bert_score = Bert_final(answer19)

        # check link (연관성 점수)
        utterance_a = "Does this item bring back any memories? Could you also tell me how you used it?"
        utterance_b = answer19

        link_score = check_link([utterance_a, utterance_b])
        if link_score < 2:
            Bert_rst += bert_score * 0.5
        else:
            Bert_rst += bert_score

        # Bert3
        bert_score = Bert_final(answer20)

        # check link (연관성 점수)
        utterance_a = "Does this item bring back any memories? Could you also tell me how you used it?"
        utterance_b = answer20

        link_score = check_link([utterance_a, utterance_b])
        if link_score < 2:
            Bert_rst += bert_score * 0.5
        else:
            Bert_rst += bert_score

        result = count_korean_fruits_vegetables(answer21)
        if result["count"] > 10:
            score += 2
        elif result["count"] > 5:
            score += 1
        else:
            score += 0

        return Bert_rst, score