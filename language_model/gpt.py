import os
from openai import OpenAI


def check_link(messages: list[str]) -> int:
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"])

    if len(messages) != 2:
        raise ValueError("messages must contain exactly two utterances: [utterance_A, utterance_B]")

    utterance_a, utterance_b = messages

    system_prompt = (
        "You are an expert evaluator of conversational coherence. "
        "Given two utterances, you will rate how naturally they connect "
        "on a scale from 1 to 5 (1 = completely unrelated, 5 = perfect topic maintenance). "
        "Always respond in the following format:\n\n"
        "Score: <number>\n"
        "Reason: <brief explanation>"
    )

    user_prompt = (
        f"Utterance A: \"{utterance_a}\"\n"
        f"Utterance B: \"{utterance_b}\"\n\n"
        "Please rate this pair on a scale of 1–5.\n\n"
        "Respond only with:\n"
        "Score: <number>\n"
        "Reason: <brief explanation>"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=100
    )

    reply = response.choices[0].message.content.strip()
    print("[GPT 응답]:", reply)

    # Score 추출
    import re
    match = re.search(r"Score:\s*([1-5])", reply)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"GPT 응답에서 점수를 추출할 수 없습니다: {reply}")