import torch
import numpy as np
import pickle
from jka_2026_train_idm import InverseDynamicsModel
from config import mouse_x_possibles, mouse_y_possibles

device = 'cuda'
model = InverseDynamicsModel().to(device)
model.load_state_dict(torch.load('idm_best.pth', map_location=device))
model.eval()

frames = np.load('recorded_data/session_20260404_133108/frames.npz')['frames']
actions = pickle.load(open('recorded_data/session_20260404_133108/actions.pkl', 'rb'))

print(f"Loaded {len(frames)} frames, {len(actions)} actions\n")

# Find frames where the player was ACTUALLY turning
turning_frames = [i for i in range(len(actions)-1) if actions[i]['mouse_x_idx'] != 11]
print(f"Frames with actual X mouse movement: {len(turning_frames)} / {len(actions)}")

print("\n--- IDM predictions on 20 frames where player WAS turning ---")
correct = 0
total = min(20, len(turning_frames))
with torch.no_grad():
    for j in range(total):
        i = turning_frames[j]
        s_t = torch.from_numpy(frames[i]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        s_t1 = torch.from_numpy(frames[i+1]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        out = model(s_t, s_t1)
        pred_x = torch.argmax(out['mouse_x'][0]).item()
        pred_y = torch.argmax(out['mouse_y'][0]).item()
        true_x = actions[i]['mouse_x_idx']
        true_y = actions[i]['mouse_y_idx']
        match = "✅" if pred_x == true_x else "❌"
        if pred_x == true_x:
            correct += 1
        print(f"  Frame {i}: true_x={true_x:2d} pred_x={pred_x:2d} {match}  |  true_y={true_y:2d} pred_y={pred_y:2d}")

print(f"\nAccuracy on turning frames: {correct}/{total}")
print(f"\nPred_x=11 count: {sum(1 for j in range(total) for i in [turning_frames[j]] if True)}")

# Also check: does the IDM EVER predict something other than 11?
print("\n--- IDM predictions on 100 random frames ---")
pred_x_dist = []
with torch.no_grad():
    for i in range(0, min(1000, len(frames)-1), 10):
        s_t = torch.from_numpy(frames[i]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        s_t1 = torch.from_numpy(frames[i+1]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        out = model(s_t, s_t1)
        pred_x = torch.argmax(out['mouse_x'][0]).item()
        pred_x_dist.append(pred_x)

pred_counts = np.bincount(pred_x_dist, minlength=23)
print(f"IDM prediction distribution across 100 frames:")
print(pred_counts)
print(f"Predictions that are index 11: {pred_counts[11]}/{len(pred_x_dist)} ({pred_counts[11]/len(pred_x_dist)*100:.1f}%)")
