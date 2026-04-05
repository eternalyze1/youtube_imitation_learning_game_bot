# train_idm.py – ResNet‑18 frozen backbone + trainable head, with resampling to 16 FPS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
import pickle
import os
import glob
from config import csgo_img_dimension, mouse_x_possibles, mouse_y_possibles

# ---------- Resampling parameters ----------
ORIGINAL_FPS = 60
TARGET_FPS = 16

def resample_to_fps(frames, actions, orig_fps=ORIGINAL_FPS, target_fps=TARGET_FPS):
    """
    Resample frame sequence and corresponding actions to a new frame rate.
    Returns (new_frames, new_actions) where new_actions[i] is the action
    that was taken from new_frames[i] to new_frames[i+1].
    """
    n_frames = len(frames)
    duration = n_frames / orig_fps
    target_frame_count = int(duration * target_fps) + 1  # +1 to keep endpoint
    
    # Indices in original sequence for each target frame (linear mapping)
    indices = np.linspace(0, n_frames - 1, target_frame_count).astype(int)
    new_frames = frames[indices]
    
    # Actions correspond to transitions between consecutive original frames.
    # For each target transition (i -> i+1), we need the action that
    # corresponds to the original transition that best aligns.
    # We'll take the action of the original frame that is closest to the
    # starting point of the target transition.
    new_actions = []
    for i in range(len(indices) - 1):
        # Use the action from the original frame at index `indices[i]`
        # This action was taken to go from frame indices[i] to indices[i]+1.
        # It's an approximation, but works if the subsampling factor is small.
        new_actions.append(actions[indices[i]])
    
    return new_frames, new_actions


# ---------- IDM using frozen ResNet‑18 backbone ----------
class InverseDynamicsModel(nn.Module):
    def __init__(self, input_channels=3, feature_dim=512):
        super().__init__()
        # Pretrained ResNet‑18 (fully frozen)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Trainable head
        self.fc1 = nn.Linear(feature_dim * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(256, 4 + 23 + 15 + 8)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"IDM params: {total_params:,} total, {trainable_params:,} trainable")

    def _encode(self, x):
        if x.shape[-2:] != (224, 224):
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.encoder(x)
        return features.squeeze(-1).squeeze(-1)

    def forward(self, state_t, state_t1):
        x_t = self._encode(state_t)
        x_t1 = self._encode(state_t1)
        combined = torch.cat([x_t, x_t1], dim=1)
        x = torch.relu(self.bn1(self.fc1(combined)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc_out(x)

        return {
            'movement': torch.sigmoid(x[:, 0:4]),
            'mouse_x': x[:, 4:27],
            'mouse_y': x[:, 27:42],
            'attack': x[:, 42:44],
            'jump': x[:, 44:46],
            'crouch': x[:, 46:48],
            'saber': x[:, 48:50]
        }


# ---------- Dataset (unchanged) ----------
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


# ---------- Data loading with resampling ----------
def load_all_sessions(data_root='recorded_data'):
    session_dirs = [d for d in glob.glob(os.path.join(data_root, 'session_*')) if os.path.isdir(d)]
    if not session_dirs:
        raise ValueError(f"No session directories found in {data_root}")

    all_frames = []
    all_actions = []
    for sess_dir in session_dirs:
        frames_path = os.path.join(sess_dir, 'frames.npz')
        actions_path = os.path.join(sess_dir, 'actions.pkl')
        if not (os.path.exists(frames_path) and os.path.exists(actions_path)):
            print(f"Skipping {sess_dir}: missing frames.npz or actions.pkl")
            continue
        frames = np.load(frames_path)['frames']
        with open(actions_path, 'rb') as f:
            actions = pickle.load(f)
        min_len = min(len(frames), len(actions))
        frames = frames[:min_len]
        actions = actions[:min_len]

        # Resample from original FPS to target FPS (16)
        frames, actions = resample_to_fps(frames, actions, ORIGINAL_FPS, TARGET_FPS)
        print(f"Loaded {len(frames)} frames and {len(actions)} actions from {os.path.basename(sess_dir)} (resampled to {TARGET_FPS} FPS)")

        all_frames.append(frames)
        all_actions.extend(actions)

    if not all_frames:
        raise ValueError("No valid sessions found.")
    combined_frames = np.concatenate(all_frames, axis=0)
    print(f"\nTotal: {len(combined_frames)} frames, {len(all_actions)} actions (all at {TARGET_FPS} FPS)")
    return combined_frames, all_actions


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

    # --- Movement (WASD) pos_weights ---
    mv = np.array([a['movement'] for a in train_actions])
    movement_pos_weights = []
    for ch, name in enumerate(['W', 'A', 'S', 'D']):
        pos = mv[:, ch].sum()
        neg = total_frames - pos
        pw = neg / pos if pos > 0 else 1.0
        movement_pos_weights.append(pw)
        print(f"Movement {name}: pos_weight={pw:.2f} ({pos:.0f}/{total_frames})")
    movement_pos_weight_tensor = torch.tensor(movement_pos_weights, dtype=torch.float32).to(device)

    def weighted_movement_bce(pred, target):
        loss_per_el = nn.functional.binary_cross_entropy(pred, target, reduction='none')
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
            state_t, state_t1 = state_t.to(device), state_t1.to(device)
            movement_true = movement_true.to(device)
            mouse_x_true, mouse_y_true = mouse_x_true.to(device), mouse_y_true.to(device)
            attack_true, jump_true, crouch_true, saber_true = attack_true.to(device), jump_true.to(device), crouch_true.to(device), saber_true.to(device)

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
                state_t, state_t1 = state_t.to(device), state_t1.to(device)
                movement_true = movement_true.to(device)
                mouse_x_true, mouse_y_true = mouse_x_true.to(device), mouse_y_true.to(device)
                attack_true, jump_true, crouch_true, saber_true = attack_true.to(device), jump_true.to(device), crouch_true.to(device), saber_true.to(device)

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

    # Load all sessions, automatically resampled to 16 FPS
    frames, actions = load_all_sessions('recorded_data')
    print(f"Total loaded: {len(frames)} frames, {len(actions)} actions (resampled to {TARGET_FPS} FPS)")

    # Align frames to actions+1
    if len(frames) > len(actions) + 1:
        frames_orig_len = len(frames)
        frames = frames[:len(actions)+1]
        print(f"Truncated frames to {len(actions)+1} to match actions (discarded {frames_orig_len - len(frames)} frames).")
    elif len(frames) < len(actions) + 1:
        actions = actions[:len(frames)-1]
        print(f"Truncated actions to {len(frames)-1} to match frames.")

    # Now frames length = N, actions length = N-1
    # Shuffle pairs (frame[i], action[i]) for i in 0..N-2, then append last frame
    indices = np.random.permutation(len(actions))
    shuffled_actions = [actions[i] for i in indices]
    shuffled_frames = frames[indices]   # frames at indices 0..N-2
    # Append the last frame
    shuffled_frames = np.append(shuffled_frames, frames[-1:], axis=0)
    frames = shuffled_frames
    actions = shuffled_actions

    # Split into train/val (80/20)
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

    print("\nStarting training with frozen ResNet‑18 backbone + trainable head (data resampled to 16 FPS)...")
    model = train_idm(model, train_loader, val_loader, train_actions, epochs=50, lr=0.001, device=device)

    torch.save(model.state_dict(), 'idm_final.pth')
    print("\nTraining complete! Models saved as 'idm_best.pth' and 'idm_final.pth'")

if __name__ == "__main__":
    main()