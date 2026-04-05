# train_idm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import pickle
import os
import cv2
import win32gui
import time
from config import csgo_img_dimension, mouse_x_possibles, mouse_y_possibles

class InverseDynamicsModel(nn.Module):
    def __init__(self):
        super(InverseDynamicsModel, self).__init__()
        # Pretrained ResNet-18 backbone (frozen early layers)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC layer, keep everything up to avgpool → 512-dim
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze first 6 layers (conv1, bn1, relu, maxpool, layer1, layer2)
        for i, child in enumerate(self.encoder.children()):
            if i < 6:
                for param in child.parameters():
                    param.requires_grad = False
        
        # 512 per frame × 2 frames = 1024
        self.fc1 = nn.Linear(512 * 2, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 4 + 23 + 15 + 8)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"IDM params: {total_params:,} total, {trainable_params:,} trainable")

    def forward(self, state_t, state_t1):
        x_t = self._encode(state_t)
        x_t1 = self._encode(state_t1)
        combined = torch.cat([x_t, x_t1], dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return {
            'movement': torch.sigmoid(x[:, 0:4]),
            'mouse_x': x[:, 4:27],
            'mouse_y': x[:, 27:42],
            'attack': x[:, 42:44],
            'jump': x[:, 44:46],
            'crouch': x[:, 46:48],
            'saber': x[:, 48:50]
        }
    
    def _encode(self, x):
        x = self.encoder(x)
        return x.squeeze(-1).squeeze(-1)  # (batch, 512)


class IDMDataset(Dataset):
    def __init__(self, frames, actions):
        self.frames = frames
        self.actions = actions
        
    def __len__(self):
        return len(self.frames) - 1
    
    def __getitem__(self, idx):
        state_t = torch.from_numpy(self.frames[idx]).float().permute(2, 0, 1) / 255.0
        state_t1 = torch.from_numpy(self.frames[idx + 1]).float().permute(2, 0, 1) / 255.0
        
        action = self.actions[idx]
        
        movement = torch.tensor(action['movement'], dtype=torch.float32)
        mouse_x_idx = torch.tensor(action['mouse_x_idx'], dtype=torch.long)
        mouse_y_idx = torch.tensor(action['mouse_y_idx'], dtype=torch.long)
        attack = torch.tensor(action['attack'], dtype=torch.long)
        jump = torch.tensor(action['jump'], dtype=torch.long)
        crouch = torch.tensor(action['crouch'], dtype=torch.long)
        saber = torch.tensor(action['saber'], dtype=torch.long)
        
        return state_t, state_t1, movement, mouse_x_idx, mouse_y_idx, attack, jump, crouch, saber


def load_session_data(session_dir):
    frames_path = os.path.join(session_dir, 'frames.npz')
    actions_path = os.path.join(session_dir, 'actions.pkl')
    
    frames = np.load(frames_path)['frames']
    with open(actions_path, 'rb') as f:
        actions = pickle.load(f)
    
    min_len = min(len(frames), len(actions))
    frames = frames[:min_len]
    actions = actions[:min_len]
    
    return frames, actions


def collate_fn(batch):
    state_t = torch.stack([item[0] for item in batch])
    state_t1 = torch.stack([item[1] for item in batch])
    movement = torch.stack([item[2] for item in batch])
    mouse_x_idx = torch.stack([item[3] for item in batch])
    mouse_y_idx = torch.stack([item[4] for item in batch])
    attack = torch.stack([item[5] for item in batch])
    jump = torch.stack([item[6] for item in batch])
    crouch = torch.stack([item[7] for item in batch])
    saber = torch.stack([item[8] for item in batch])
    
    return state_t, state_t1, movement, mouse_x_idx, mouse_y_idx, attack, jump, crouch, saber


def train_idm(model, train_loader, val_loader, train_actions, epochs=50, lr=0.001, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_frames = len(train_actions)
    
    # --- Movement (WASD) pos_weights: per-channel inverse frequency ---
    mv = np.array([a['movement'] for a in train_actions])
    movement_pos_weights = []
    for ch, name in enumerate(['W', 'A', 'S', 'D']):
        pos = mv[:, ch].sum()
        neg = total_frames - pos
        pw = neg / pos if pos > 0 else 1.0
        movement_pos_weights.append(pw)
        print(f"Movement {name}: pos_weight={pw:.2f} ({pos:.0f}/{total_frames})")
    movement_pos_weight_tensor = torch.tensor(movement_pos_weights, dtype=torch.float32).to(device)
    
    # Weighted movement loss function
    def weighted_movement_bce(pred, target):
        loss_per_el = nn.functional.binary_cross_entropy(pred, target, reduction='none')  # (batch, 4)
        weights = torch.where(target == 1, movement_pos_weight_tensor.unsqueeze(0), torch.ones_like(loss_per_el))
        return (loss_per_el * weights).mean()
    
    # --- Binary actions: weighted CrossEntropy ---
    attack_frames = sum(1 for a in train_actions if a['attack'] == 1)
    attack_weight = (total_frames - attack_frames) / attack_frames if attack_frames > 0 else 1.0
    attack_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, attack_weight]).to(device))
    print(f"Attack weight: {attack_weight:.2f}")
    
    jump_frames = sum(1 for a in train_actions if a['jump'] == 1)
    jump_weight = (total_frames - jump_frames) / jump_frames if jump_frames > 0 else 1.0
    jump_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, jump_weight]).to(device))
    print(f"Jump weight: {jump_weight:.2f}")
    
    crouch_frames = sum(1 for a in train_actions if a['crouch'] == 1)
    crouch_weight = (total_frames - crouch_frames) / crouch_frames if crouch_frames > 0 else 1.0
    crouch_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, crouch_weight]).to(device))
    print(f"Crouch weight: {crouch_weight:.2f}")
    
    saber_frames = sum(1 for a in train_actions if a['saber'] == 1)
    saber_weight = (total_frames - saber_frames) / saber_frames if saber_frames > 0 else 1.0
    saber_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, saber_weight]).to(device))
    print(f"Saber weight: {saber_weight:.2f}")
    
    # --- Mouse: inverse frequency weighted CrossEntropy ---
    mouse_x_counts = np.bincount([a['mouse_x_idx'] for a in train_actions], minlength=23)
    mouse_x_weights = torch.tensor([total_frames / (23 * c) if c > 0 else 1.0 for c in mouse_x_counts], dtype=torch.float32).to(device)
    mouse_loss_x = nn.CrossEntropyLoss(weight=mouse_x_weights)
    
    mouse_y_counts = np.bincount([a['mouse_y_idx'] for a in train_actions], minlength=15)
    mouse_y_weights = torch.tensor([total_frames / (15 * c) if c > 0 else 1.0 for c in mouse_y_counts], dtype=torch.float32).to(device)
    mouse_loss_y = nn.CrossEntropyLoss(weight=mouse_y_weights)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            state_t, state_t1, movement_true, mouse_x_true, mouse_y_true, attack_true, jump_true, crouch_true, saber_true = batch
            
            state_t = state_t.to(device)
            state_t1 = state_t1.to(device)
            movement_true = movement_true.to(device)
            mouse_x_true = mouse_x_true.to(device)
            mouse_y_true = mouse_y_true.to(device)
            attack_true = attack_true.to(device)
            jump_true = jump_true.to(device)
            crouch_true = crouch_true.to(device)
            saber_true = saber_true.to(device)
            
            outputs = model(state_t, state_t1)
            
            loss_movement = weighted_movement_bce(outputs['movement'], movement_true)
            loss_mouse_x = mouse_loss_x(outputs['mouse_x'], mouse_x_true)
            loss_mouse_y = mouse_loss_y(outputs['mouse_y'], mouse_y_true)
            loss_attack = attack_loss(outputs['attack'], attack_true)
            loss_jump = jump_loss(outputs['jump'], jump_true)
            loss_crouch = crouch_loss(outputs['crouch'], crouch_true)
            loss_saber = saber_loss(outputs['saber'], saber_true)
            
            total_loss = (loss_movement + loss_mouse_x + loss_mouse_y + 
                         loss_attack + loss_jump + loss_crouch + loss_saber) / 7
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                state_t, state_t1, movement_true, mouse_x_true, mouse_y_true, attack_true, jump_true, crouch_true, saber_true = batch
                
                state_t = state_t.to(device)
                state_t1 = state_t1.to(device)
                movement_true = movement_true.to(device)
                mouse_x_true = mouse_x_true.to(device)
                mouse_y_true = mouse_y_true.to(device)
                attack_true = attack_true.to(device)
                jump_true = jump_true.to(device)
                crouch_true = crouch_true.to(device)
                saber_true = saber_true.to(device)
                
                outputs = model(state_t, state_t1)
                
                loss_movement = weighted_movement_bce(outputs['movement'], movement_true)
                loss_mouse_x = mouse_loss_x(outputs['mouse_x'], mouse_x_true)
                loss_mouse_y = mouse_loss_y(outputs['mouse_y'], mouse_y_true)
                loss_attack = attack_loss(outputs['attack'], attack_true)
                loss_jump = jump_loss(outputs['jump'], jump_true)
                loss_crouch = crouch_loss(outputs['crouch'], crouch_true)
                loss_saber = saber_loss(outputs['saber'], saber_true)
                
                total_loss = (loss_movement + loss_mouse_x + loss_mouse_y + 
                             loss_attack + loss_jump + loss_crouch + loss_saber) / 7
                
                val_loss += total_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'idm_best.pth')
            print(f"  -> Saved best model with val loss: {avg_val_loss:.4f}")
    
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # session_dir = 'recorded_data/session_20260404_133108'
    session_dir = 'recorded_data/session_20260405_002501'
    print(f"Loading data from {session_dir}...")
    
    frames, actions = load_session_data(session_dir)
    print(f"Loaded {len(frames)} frames, {len(actions)} actions")
    
    # Shuffle
    indices = np.random.permutation(len(frames))
    frames = frames[indices]
    actions = [actions[i] for i in indices]
    
    # Split
    split_idx = int(len(frames) * 0.8)
    train_frames = frames[:split_idx]
    train_actions = actions[:split_idx]
    val_frames = frames[split_idx:-1]
    val_actions = actions[split_idx:-1]
    
    print(f"Train: {len(train_frames)} frames, Val: {len(val_frames)} frames")
    
    train_dataset = IDMDataset(train_frames, train_actions)
    val_dataset = IDMDataset(val_frames, val_actions)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    model = InverseDynamicsModel().to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\nStarting training...")
    model = train_idm(model, train_loader, val_loader, train_actions, epochs=2, lr=0.001, device=device)
    
    torch.save(model.state_dict(), 'idm_final.pth')
    print("\nTraining complete!")


if __name__ == "__main__":
    main()