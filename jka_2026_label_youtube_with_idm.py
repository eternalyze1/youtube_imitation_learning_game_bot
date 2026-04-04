# jka_2026_label_youtube_with_idm.py
import torch
import numpy as np
import pickle
from jka_2026_train_idm import InverseDynamicsModel
from config import mouse_x_possibles, mouse_y_possibles

def label_frames_with_idm(model, frames, batch_size=64, device='cuda'):
    """Label video frames using trained IDM"""
    model.eval()
    pseudo_actions = []
    
    total_transitions = len(frames) - 1
    print(f"Labeling {total_transitions} transitions...")
    
    with torch.no_grad():
        for i in range(0, total_transitions, batch_size):
            end_idx = min(i + batch_size, total_transitions)
            
            batch_t = frames[i:end_idx]
            batch_t1 = frames[i+1:end_idx+1]
            
            batch_t = torch.from_numpy(batch_t).float().permute(0, 3, 1, 2).to(device) / 255.0
            batch_t1 = torch.from_numpy(batch_t1).float().permute(0, 3, 1, 2).to(device) / 255.0
            
            outputs = model(batch_t, batch_t1)
            
            for j in range(len(batch_t)):
                action = {
                    'movement': (outputs['movement'][j].cpu().numpy() > 0.5).astype(int),
                    'mouse_x_idx': torch.argmax(outputs['mouse_x'][j]).item(),
                    'mouse_y_idx': torch.argmax(outputs['mouse_y'][j]).item(),
                    'attack': torch.argmax(outputs['attack'][j]).item(),
                    'jump': torch.argmax(outputs['jump'][j]).item(),
                    'crouch': torch.argmax(outputs['crouch'][j]).item(),
                    'saber': torch.argmax(outputs['saber'][j]).item(),
                }
                pseudo_actions.append(action)
            
            if len(pseudo_actions) % 10000 == 0:
                print(f"Labeled {len(pseudo_actions)}/{total_transitions} frames ({len(pseudo_actions)/total_transitions*100:.1f}%)")
    
    return pseudo_actions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load IDM
    print("Loading trained IDM...")
    model = InverseDynamicsModel().to(device)
    model.load_state_dict(torch.load('idm_best.pth', map_location=device))
    model.eval()
    print("IDM loaded!")
    
    # Load YouTube frames from compressed .npz file
    print("\nLoading YouTube frames...")
    try:
        frames = np.load('youtube_frames_60fps.npz')['frames']
        print(f"Loaded {len(frames)} frames from youtube_frames_60fps.npz")
    except:
        print("Could not load youtube_frames_60fps.npz")
        print("Trying alternative filename...")
        try:
            frames = np.load('youtube_frames_60fps.npy')
            print(f"Loaded {len(frames)} frames from youtube_frames_60fps.npy")
        except:
            print("ERROR: No YouTube frames file found!")
            return
    
    print(f"Video duration: {len(frames)/60:.2f} seconds ({len(frames)/60/60:.2f} hours)")
    
    # Label frames
    pseudo_actions = label_frames_with_idm(model, frames, batch_size=64, device=device)
    
    # Save only the pseudo-labels (actions)
    print("\nSaving pseudo-labels...")
    with open('youtube_pseudo_actions_60fps.pkl', 'wb') as f:
        pickle.dump(pseudo_actions, f)
    
    print(f"\n✅ Complete!")
    print(f"  Frames file: youtube_frames_60fps.npz (keep this file)")
    print(f"  Actions file: youtube_pseudo_actions_60fps.pkl (~{len(pseudo_actions) * 100 / 1e6:.1f} MB)")
    print(f"  Total transitions: {len(pseudo_actions)}")
    
    # Show sample
    print("\n📊 Sample pseudo-labeled action:")
    sample = pseudo_actions[0]
    print(f"  Movement (WASD): {sample['movement']}")
    print(f"  Attack: {sample['attack']}, Jump: {sample['jump']}")

if __name__ == "__main__":
    main()