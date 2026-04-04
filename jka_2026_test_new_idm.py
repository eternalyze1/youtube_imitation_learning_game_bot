# test_new_idm.py
import torch
import numpy as np
import pickle
from tqdm import tqdm
from jka_2026_train_idm import InverseDynamicsModel

device = 'cuda'
model = InverseDynamicsModel().to(device)
model.load_state_dict(torch.load('idm_best.pth'))
model.eval()

# Load data
frames = np.load('recorded_data/session_20260403_143600/frames.npz')['frames']
with open('recorded_data/session_20260403_143600/actions.pkl', 'rb') as f:
    actions = pickle.load(f)

print(f"Testing on {len(frames)-1} transitions...\n")

correct_attacks = 0
total_attacks = 0
correct_no_attacks = 0
total_no_attacks = 0

batch_size = 64
total_transitions = len(frames) - 1

with torch.no_grad():
    for i in tqdm(range(0, total_transitions, batch_size)):
        end_idx = min(i + batch_size, total_transitions)
        
        batch_t = torch.from_numpy(frames[i:end_idx]).float().permute(0, 3, 1, 2).to(device) / 255.0
        batch_t1 = torch.from_numpy(frames[i+1:end_idx+1]).float().permute(0, 3, 1, 2).to(device) / 255.0
        
        outputs = model(batch_t, batch_t1)
        attack_preds = torch.argmax(outputs['attack'], dim=1).cpu().numpy()
        
        for j, attack_pred in enumerate(attack_preds):
            true_attack = actions[i + j]['attack']
            
            if true_attack == 1:
                total_attacks += 1
                if attack_pred == 1:
                    correct_attacks += 1
            else:
                total_no_attacks += 1
                if attack_pred == 0:
                    correct_no_attacks += 1

print(f"\n{'='*50}")
print(f"RESULTS ON ALL {total_transitions} FRAMES")
print(f"{'='*50}")
print(f"Attack frames: {total_attacks}")
print(f"  Correctly predicted: {correct_attacks}")
if total_attacks > 0:
    print(f"  Attack accuracy: {correct_attacks/total_attacks*100:.1f}%")
print(f"\nNo-attack frames: {total_no_attacks}")
print(f"  Correctly predicted: {correct_no_attacks}")
print(f"  No-attack accuracy: {correct_no_attacks/total_no_attacks*100:.1f}%")
print(f"\nOverall accuracy: {(correct_attacks + correct_no_attacks) / total_transitions * 100:.1f}%")