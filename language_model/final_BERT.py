import re
import torch
from torch import nn
from transformers import BertTokenizer, BertForMaskedLM

# -----------------------------
# 클린징 함수 (학습 시 사용한 것과 동일)
# -----------------------------
def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^ㄱ-ㅎ가-힣A-Za-z\s\.\,\!\?]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# -----------------------------
# 모델 클래스 정의 (학습 시 사용한 것과 동일)
# -----------------------------
class BERTWithMLMClassifier(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased"):
        super().__init__()
        self.bert_mlm = BertForMaskedLM.from_pretrained(pretrained_model_name)
        hidden_size = self.bert_mlm.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert_mlm.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰 벡터
        logits = self.classifier(cls_output).squeeze(-1)
        return logits


def Bert_final(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BERTWithMLMClassifier(pretrained_model_name="bert-base-uncased")
    model.load_state_dict(torch.load("language_model/final_BERT.pt", map_location=device))
    model.to(device)
    model.eval()

    cleaned = clean_text(text)
    if not cleaned:
        return 0

    encoded = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prob = torch.sigmoid(logits).item()

    return 2 if prob > 0.5 else 0
