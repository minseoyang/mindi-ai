import torch
from acoustic_model.remove_silence import *
from acoustic_model.make_spectrogram import *
from acoustic_model.extract_feature import *

def predict_dementia(audio_path, vit_model, lgbm_model, device, image_transform):
    samples, sr = remove_silence(audio_path)
    images = make_spectrogram_chunks(samples, sr, image_transform)
    if not images:
        print("No valid spectrogram chunks for ViT.")
        return None
    vit_inputs = torch.stack(images).to(device)
    with torch.no_grad():
        outputs = vit_model(vit_inputs)
        probs_vit = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
    vit_final = np.mean(probs_vit)

    feats = extract_features(audio_path)
    if feats.size == 0:
        print("No valid acoustic segments for LightGBM.")
        return None
    probs_lgb = lgbm_model.predict_proba(feats)[:, 1]
    lgb_final = np.mean(probs_lgb)

    final_prob = 0.50 * lgb_final + 0.50 * vit_final
    return round(final_prob, 4)