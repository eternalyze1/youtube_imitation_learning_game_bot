# test_idm_on_single_frame.py
import torch
import numpy as np
import pickle
from jka_2026_train_idm import InverseDynamicsModel
from config import csgo_img_dimension

# Load IDM
device = 'cuda'
model = InverseDynamicsModel().to(device)
model.load_state_dict(torch.load('idm_best.pth'))
model.eval()

# Load one frame pair from your original recorded data
frames = np.load('recorded_data/session_20260403_054610/frames.npz')['frames']
with open('recorded_data/session_20260403_054610/actions.pkl', 'rb') as f:
    actions = pickle.load(f)

# Test on first 10 frames
print("Testing IDM on your original recorded data:")
for i in range(10):
    state_t = torch.from_numpy(frames[i]).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
    state_t1 = torch.from_numpy(frames[i+1]).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        output = model(state_t, state_t1)
    
    # Get predictions
    movement = (output['movement'][0].cpu().numpy() > 0.5).astype(int)
    attack = torch.argmax(output['attack'][0]).item()
    jump = torch.argmax(output['jump'][0]).item()
    
    # Compare with true actions
    true_action = actions[i]
    true_movement = true_action['movement']
    true_attack = true_action['attack']
    
    print(f"Frame {i}:")
    print(f"  Predicted movement: {movement}, True: {true_movement}")
    print(f"  Predicted attack: {attack}, True: {true_attack}")
    print(f"  Attack prob: {output['attack'][0].cpu().numpy()}")
    print()

# Then test on a random frame from YouTube
print("\nTesting IDM on a random YouTube frame:")
youtube_frames = np.load('youtube_frames_60fps.npy')
state_t = torch.from_numpy(youtube_frames[1000]).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
state_t1 = torch.from_numpy(youtube_frames[1001]).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0

with torch.no_grad():
    output = model(state_t, state_t1)

movement = (output['movement'][0].cpu().numpy() > 0.5).astype(int)
attack = torch.argmax(output['attack'][0]).item()
print(f"YouTube frame prediction:")
print(f"  Movement: {movement}")
print(f"  Attack: {attack}")
print(f"  Attack probs: {output['attack'][0].cpu().numpy()}")