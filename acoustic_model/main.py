from acoustic_model.predict_dementia import *
from acoustic_model.get_score import *
import numpy as np
import torch
import joblib
from torchvision import transforms
from transformers import ViTForImageClassification

def load_models(vit_path, lgb_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vit_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    vit_model.load_state_dict(torch.load(vit_path, map_location=device))
    vit_model.to(device)
    vit_model.eval()

    lgbm_model = joblib.load(lgb_path)

    # 이미지 전처리 transform도 같이 반환
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    return vit_model, lgbm_model, device, image_transform


def acoustic_model(audio_paths):
    vit_path = "acoustic_model/vit.pth"
    lgb_path = "acoustic_model/lgbm.pkl"

    vit_model, lgbm_model, device, image_transform = load_models(vit_path, lgb_path)
    probs = []
    for i, audio_path in enumerate(audio_paths):
        prob = predict_dementia(audio_path, vit_model, lgbm_model, device, image_transform)
        probs.append(prob)

    final_prob = round(np.mean(probs), 4)
    score = get_score(final_prob)

    print(f"acoustic Dementia Probability: {final_prob}")

    return score

# def Hwi_test(wav1):
#     vit_path = "vit.pth"
#     lgb_path = "lgbm.pkl"
#
#     audio_paths = [wav1]
#
#     vit_model, lgbm_model, device, image_transform = load_models(vit_path, lgb_path)
#     probs = []
#     for i, audio_path in enumerate(audio_paths):
#         prob = predict_dementia(audio_path, vit_model, lgbm_model, device, image_transform)
#         probs.append(prob)
#
#     final_prob = round(np.mean(probs), 4)
#     score = get_score(final_prob)
#
#     print(f"Dementia Probability: {final_prob}")
#     print(f"Score: {score}")
#     return score
