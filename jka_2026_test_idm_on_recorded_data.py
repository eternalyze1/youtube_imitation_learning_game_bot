# test_idm_on_recorded_data.py
import torch
import numpy as np
import pickle
from jka_2026_train_idm import InverseDynamicsModel
from config import csgo_img_dimension, mouse_x_possibles, mouse_y_possibles

device = 'cuda'
model = InverseDynamicsModel().to(device)
model.load_state_dict(torch.load('idm_best.pth'))
model.eval()

# Load your recorded data (the session with varied mouse movements)
frames = np.load('recorded_data/session_20260403_160752/frames.npz')['frames']
with open('recorded_data/session_20260403_160752/actions.pkl', 'rb') as f:
    actions = pickle.load(f)

# Check diversity of ground truth mouse indices in recorded data
unique_x = set(actions[i]['mouse_x_idx'] for i in range(len(actions)))
unique_y = set(actions[i]['mouse_y_idx'] for i in range(len(actions)))
print(f"Unique mouse X indices in recorded data: {sorted(unique_x)}")
print(f"Unique mouse Y indices in recorded data: {sorted(unique_y)}")
print(f"Number of unique X: {len(unique_x)}, Y: {len(unique_y)}")
print(f"Total frames: {len(actions)}")

# Test IDM on first 100 transitions
print("\nTesting IDM on recorded data (ground truth actions available)")
for i in range(min(100, len(actions)-1)):
    state_t = torch.from_numpy(frames[i]).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
    state_t1 = torch.from_numpy(frames[i+1]).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        out = model(state_t, state_t1)
    pred_x_idx = torch.argmax(out['mouse_x'][0]).item()
    pred_y_idx = torch.argmax(out['mouse_y'][0]).item()
    true_x_idx = actions[i]['mouse_x_idx']
    true_y_idx = actions[i]['mouse_y_idx']
    if i % 10 == 0:   # print every 10 frames to reduce output
        print(f"Frame {i:3d}: Pred mouse ({pred_x_idx:2d},{pred_y_idx:2d}) -> ({mouse_x_possibles[pred_x_idx]:5.1f},{mouse_y_possibles[pred_y_idx]:5.1f}) | True ({true_x_idx:2d},{true_y_idx:2d}) -> ({mouse_x_possibles[true_x_idx]:5.1f},{mouse_y_possibles[true_y_idx]:5.1f})")