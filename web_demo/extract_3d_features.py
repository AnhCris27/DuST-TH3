import json
import torch
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.video_swin_transformer import SwinTransformer3D


def main():
    scenario_path = 'web_demo/sample_data/scenarios.json'
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load 3D Swin Transformer
    m3d = SwinTransformer3D(
        num_classes=4,
        embed_dim=128,
        mlp_ratio=4.0,
        patch_norm=True,
        patch_size=(2, 4, 4),
        drop_path_rate=0.1,
        num_heads=[4, 8, 16, 32],
        depths=[2, 2, 18, 2],
        pretrained=None,
        pretrained2d=False,
        window_size=(8, 7, 7)
    )
    m3d.load_state_dict(torch.load('saved_models/step_3d/final_3d.pth', map_location=device))
    m3d = m3d.to(device).eval()

    for sc in scenarios:
        frames = []
        for frel in sc['frames']:
            fpath = os.path.join('web_demo', frel.lstrip('/'))
            img = cv2.imread(fpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            frames.append(img.astype(np.float32) / 255.0)

        # Interpolate 4 frames to 77 frames
        sample_indices = np.round(np.linspace(0, len(frames) - 1, 77)).astype(int)
        interpolated = np.array([frames[i] for i in sample_indices])
        vid_tensor = torch.from_numpy(interpolated).permute(3, 0, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = m3d.get_vid_feature(vid_tensor)  # (1, 1024)
            pred_3d = m3d.head(feat)
            prob_3d = torch.softmax(pred_3d, dim=-1)


        sc['video_feature'] = feat.squeeze(0).cpu().numpy().tolist()
        sc['pred_3d_probs'] = prob_3d.squeeze(0).cpu().numpy().tolist()
        print(f"{sc['id']} 3D probs:", [round(p, 4) for p in sc['pred_3d_probs']])

    with open(scenario_path, 'w', encoding='utf-8') as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)

    print("Successfully enriched scenarios with precomputed 3D features!")

if __name__ == '__main__':
    main()
