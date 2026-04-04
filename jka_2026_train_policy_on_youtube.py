# train_policy_on_youtube.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from jka_2026_controller import Policy
from config import csgo_img_dimension, mouse_x_possibles, mouse_y_possibles

class PolicyDataset(Dataset):
    def __init__(self, frames, actions):
        self.frames = frames
        self.actions = actions
        # Ensure lengths match
        min_len = min(len(frames), len(actions))
        self.frames = self.frames[:min_len]
        self.actions = self.actions[:min_len]
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = torch.from_numpy(self.frames[idx]).float().permute(2, 0, 1) / 255.0
        action = self.actions[idx]
        
        movement = torch.tensor(action['movement'], dtype=torch.float32)
        mouse_x = torch.tensor(action['mouse_x_idx'], dtype=torch.long)
        mouse_y = torch.tensor(action['mouse_y_idx'], dtype=torch.long)
        attack = torch.tensor(action['attack'], dtype=torch.float32)
        jump = torch.tensor(action['jump'], dtype=torch.float32)
        crouch = torch.tensor(action['crouch'], dtype=torch.float32)
        saber = torch.tensor(action['saber'], dtype=torch.float32)
        
        return frame, movement, mouse_x, mouse_y, attack, jump, crouch, saber

def train_policy(model, train_loader, val_loader, train_actions, epochs=30, lr=0.0001, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_frames = len(train_actions)
    
    # --- Movement (WASD) pos_weights ---
    mv = np.array([a['movement'] for a in train_actions])
    movement_pos_weights = []
    for ch, name in enumerate(['W', 'A', 'S', 'D']):
        pos = mv[:, ch].sum()
        neg = total_frames - pos
        pw = neg / pos if pos > 0 else 1.0
        movement_pos_weights.append(pw)
        print(f"Movement {name}: pos_weight={pw:.2f}")
    movement_pos_weight_tensor = torch.tensor(movement_pos_weights, dtype=torch.float32).to(device)
    
    def weighted_movement_bce(pred, target):
        loss_per_el = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        weights = torch.where(target == 1, movement_pos_weight_tensor.unsqueeze(0), torch.ones_like(loss_per_el))
        return (loss_per_el * weights).mean()
    
    # --- Binary action pos_weights ---
    def make_binary_pos_weight(key):
        pos = sum(1 for a in train_actions if a[key] == 1)
        pw = (total_frames - pos) / pos if pos > 0 else 1.0
        print(f"{key}: pos_weight={pw:.2f} ({pos}/{total_frames})")
        return pw
    
    attack_pw = torch.tensor([make_binary_pos_weight('attack')], dtype=torch.float32).to(device)
    jump_pw = torch.tensor([make_binary_pos_weight('jump')], dtype=torch.float32).to(device)
    crouch_pw = torch.tensor([make_binary_pos_weight('crouch')], dtype=torch.float32).to(device)
    saber_pw = torch.tensor([make_binary_pos_weight('saber')], dtype=torch.float32).to(device)
    
    def weighted_binary_bce(pred, target, pos_weight):
        loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        w = torch.where(target == 1, pos_weight, torch.ones_like(loss))
        return (loss * w).mean()
    
    # --- Mouse: inverse frequency weighted CrossEntropy ---
    mouse_x_counts = np.bincount([a['mouse_x_idx'] for a in train_actions], minlength=23)
    mouse_x_weights = torch.tensor([total_frames / (23 * c) if c > 0 else 1.0 for c in mouse_x_counts], dtype=torch.float32).to(device)
    ce_loss_x = nn.CrossEntropyLoss(weight=mouse_x_weights)
    
    mouse_y_counts = np.bincount([a['mouse_y_idx'] for a in train_actions], minlength=15)
    mouse_y_weights = torch.tensor([total_frames / (15 * c) if c > 0 else 1.0 for c in mouse_y_counts], dtype=torch.float32).to(device)
    ce_loss_y = nn.CrossEntropyLoss(weight=mouse_y_weights)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            frame = batch[0].to(device)
            movement_true = batch[1].to(device)
            mouse_x_true = batch[2].to(device)
            mouse_y_true = batch[3].to(device)
            attack_true = batch[4].to(device)
            jump_true = batch[5].to(device)
            crouch_true = batch[6].to(device)
            saber_true = batch[7].to(device)
            
            # Model returns tuple of 7 tensors
            movement_pred, mouse_x_pred, mouse_y_pred, attack_pred, jump_pred, crouch_pred, saber_pred = model(frame)
            
            # Losses — all weighted
            loss_movement = weighted_movement_bce(movement_pred, movement_true)
            loss_mouse_x = ce_loss_x(mouse_x_pred, mouse_x_true)
            loss_mouse_y = ce_loss_y(mouse_y_pred, mouse_y_true)
            loss_attack = weighted_binary_bce(attack_pred, attack_true.unsqueeze(1), attack_pw)
            loss_jump = weighted_binary_bce(jump_pred, jump_true.unsqueeze(1), jump_pw)
            loss_crouch = weighted_binary_bce(crouch_pred, crouch_true.unsqueeze(1), crouch_pw)
            loss_saber = weighted_binary_bce(saber_pred, saber_true.unsqueeze(1), saber_pw)
            
            loss = (loss_movement + loss_mouse_x + loss_mouse_y + 
                   loss_attack + loss_jump + loss_crouch + loss_saber) / 7
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                frame = batch[0].to(device)
                movement_true = batch[1].to(device)
                mouse_x_true = batch[2].to(device)
                mouse_y_true = batch[3].to(device)
                attack_true = batch[4].to(device)
                jump_true = batch[5].to(device)
                crouch_true = batch[6].to(device)
                saber_true = batch[7].to(device)
                
                movement_pred, mouse_x_pred, mouse_y_pred, attack_pred, jump_pred, crouch_pred, saber_pred = model(frame)
                
                loss_movement = weighted_movement_bce(movement_pred, movement_true)
                loss_mouse_x = ce_loss_x(mouse_x_pred, mouse_x_true)
                loss_mouse_y = ce_loss_y(mouse_y_pred, mouse_y_true)
                loss_attack = weighted_binary_bce(attack_pred, attack_true.unsqueeze(1), attack_pw)
                loss_jump = weighted_binary_bce(jump_pred, jump_true.unsqueeze(1), jump_pw)
                loss_crouch = weighted_binary_bce(crouch_pred, crouch_true.unsqueeze(1), crouch_pw)
                loss_saber = weighted_binary_bce(saber_pred, saber_true.unsqueeze(1), saber_pw)
                
                loss = (loss_movement + loss_mouse_x + loss_mouse_y + 
                       loss_attack + loss_jump + loss_crouch + loss_saber) / 7
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'policy_youtube_epoch_{epoch+1}.pth')
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading frames...")
    frames = np.load('youtube_frames_60fps.npz')['frames']
    print(f"Loaded {len(frames)} frames")
    
    print("Loading pseudo-labels...")
    with open('youtube_pseudo_actions_60fps.pkl', 'rb') as f:
        actions = pickle.load(f)
    print(f"Loaded {len(actions)} actions")
    
    # Align lengths
    frames = frames[:len(actions)]
    
    print(f"After alignment: {len(frames)} frames, {len(actions)} actions")
    
    # Shuffle
    indices = np.random.permutation(len(frames))
    frames = frames[indices]
    actions = [actions[i] for i in indices]
    
    # Split
    split_idx = int(len(frames) * 0.8)
    train_frames = frames[:split_idx]
    train_actions = actions[:split_idx]
    val_frames = frames[split_idx:]
    val_actions = actions[split_idx:]
    
    print(f"Train: {len(train_frames)} pairs, Val: {len(val_frames)} pairs")
    
    train_dataset = PolicyDataset(train_frames, train_actions)
    val_dataset = PolicyDataset(val_frames, val_actions)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    policy = Policy().to(device)
    print(f"Policy has {sum(p.numel() for p in policy.parameters()):,} parameters")
    
    print("\nStarting training on YouTube pseudo-labeled data...")
    policy = train_policy(policy, train_loader, val_loader, train_actions, epochs=10, device=device)
    
    torch.save(policy.state_dict(), 'policy_youtube_trained.pth')
    print("\n✅ Training complete! Model saved as 'policy_youtube_trained.pth'")

if __name__ == "__main__":
    main()