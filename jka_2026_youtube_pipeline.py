# jka_2026_youtube_pipeline.py
# Streaming pipeline: download → stream → label → train → delete, one video at a time.
# Uses decord to stream frames in chunks, never loading the whole video.
# Trains the stateful LSTM policy using Truncated Backpropagation Through Time (TBPTT) with seq_len=96.
# Batched processing of sequences for efficiency.
# All fixes applied: download retries, frames[:-1] alignment, device handling, .reshape(), and decord fallback.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import cv2
import os
import glob
import yt_dlp
import time
import shutil
import tempfile
from decord import VideoReader, cpu
from torchvision import models

# Import config
from config import csgo_img_dimension, mouse_x_possibles, mouse_y_possibles

# ─────────────────────────── CONFIG ───────────────────────────
CHANNEL_URL = "https://www.youtube.com/@JediAcademyLeague/videos"
VIDEO_DIR = "youtube_videos_tmp"
TARGET_FPS = 16
POLICY_EPOCHS_PER_VIDEO = 1
POLICY_LR = 0.0001
BATCH_SIZE = 8                     # Number of sequences processed in parallel
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 96                        # 6 seconds at 16 FPS
POLICY_INPUT_SIZE = (224, 224)
USE_CLASS_WEIGHTS = False           # Keep False for standard losses

# ─────────────────────────── POLICY NETWORK (fixed .reshape) ───────────────────────────
class StatefulLSTMPolicy(nn.Module):
    def __init__(self, lstm_hidden=256, lstm_layers=1):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone = backbone.features
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[-3:].parameters():
            param.requires_grad = True

        self.feature_dim = 1280 * 7 * 7
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True)
        self.movement_head = nn.Sequential(nn.Linear(lstm_hidden, 4), nn.Sigmoid())
        self.attack_head = nn.Sequential(nn.Linear(lstm_hidden, 1), nn.Sigmoid())
        self.jump_head = nn.Sequential(nn.Linear(lstm_hidden, 1), nn.Sigmoid())
        self.crouch_head = nn.Sequential(nn.Linear(lstm_hidden, 1), nn.Sigmoid())
        self.saber_head = nn.Sequential(nn.Linear(lstm_hidden, 1), nn.Sigmoid())
        self.mouse_x_head = nn.Linear(lstm_hidden, len(mouse_x_possibles))
        self.mouse_y_head = nn.Linear(lstm_hidden, len(mouse_y_possibles))

        self.hidden = None

    def reset_state(self):
        self.hidden = None

    def forward(self, x):
        batch_size = x.size(0)
        features = self.backbone(x)
        features = features.reshape(batch_size, -1)   # .reshape instead of .view
        features = features.unsqueeze(1)
        if self.hidden is None:
            h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=x.device)
            c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=x.device)
            self.hidden = (h0, c0)
        lstm_out, self.hidden = self.lstm(features, self.hidden)
        last_out = lstm_out[:, -1, :]
        movement = self.movement_head(last_out)
        attack = self.attack_head(last_out)
        jump = self.jump_head(last_out)
        crouch = self.crouch_head(last_out)
        saber = self.saber_head(last_out)
        mouse_x_logits = self.mouse_x_head(last_out)
        mouse_y_logits = self.mouse_y_head(last_out)
        return movement, mouse_x_logits, mouse_y_logits, attack, jump, crouch, saber

# ─────────────────────────── IDM (imported but we keep it here for clarity) ───────────────────────────
# We'll import the IDM from your existing module; but to avoid circular imports, we'll define a placeholder.
# Actually the pipeline uses from jka_2026_train_idm import InverseDynamicsModel. We'll keep that.
# But to make this file self-contained, we'll import it normally. Ensure jka_2026_train_idm is in the same folder.
from jka_2026_train_idm import InverseDynamicsModel

# ─────────────────────────── STEP 1: GET VIDEO URLS ───────────────────────────
def get_channel_video_urls(channel_url, max_videos=None):
    print(f"\n{'='*60}\nFetching video list from channel...\n{'='*60}")
    ydl_opts = {'quiet': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        entries = info.get('entries', [])
        urls = []
        for entry in entries:
            if entry and entry.get('url'):
                url = entry['url']
                if not url.startswith('http'):
                    url = f"https://www.youtube.com/watch?v={entry.get('id', entry['url'])}"
                urls.append((url, entry.get('title', 'Unknown')))
        if max_videos:
            urls = urls[:max_videos]
        print(f"Found {len(urls)} videos")
        return urls

# ─────────────────────────── STEP 2: DOWNLOAD ONE VIDEO (with retries and temp dir) ───────────────────────────
def download_video(url, title):
    temp_dir = tempfile.mkdtemp(prefix='yt_dlp_')
    ydl_opts = {
        'format': 'worst[ext=mp4][height>=360]/worst[ext=mp4]/worst',
        'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
        'quiet': False,
        'retries': 10,
        'fragment_retries': 10,
        'continuedl': False,
    }
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                temp_filename = ydl.prepare_filename(info)
                for ext in ['.webm', '.mkv']:
                    temp_filename = temp_filename.replace(ext, '.mp4')
                os.makedirs(VIDEO_DIR, exist_ok=True)
                final_filename = os.path.join(VIDEO_DIR, os.path.basename(temp_filename))
                # Move with retries
                for move_attempt in range(5):
                    try:
                        shutil.move(temp_filename, final_filename)
                        break
                    except OSError as e:
                        print(f"Move attempt {move_attempt+1} failed: {e}")
                        time.sleep(2)
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"Downloaded: {final_filename}")
                return final_filename
        except Exception as e:
            print(f"Download attempt {attempt+1} failed: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            time.sleep(5)
    raise Exception("Failed to download after multiple attempts")

# ─────────────────────────── STEP 3: STREAM FRAMES (decord chunked with fallback) ───────────────────────────
def stream_frames_fast(video_path, chunk_size=500, target_fps=TARGET_FPS):
    h, w = csgo_img_dimension
    vr = VideoReader(video_path, ctx=cpu(0))
    src_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / src_fps
    target_count = int(duration * target_fps)
    print(f"  Source: {src_fps:.1f} FPS, {duration:.1f}s, {total_frames} frames")
    print(f"  Target: {target_fps} FPS → {target_count} frames")
    step = src_fps / target_fps
    indices = np.arange(0, target_count, 1, dtype=int)
    src_indices = (indices * step).astype(int)
    src_indices = np.clip(src_indices, 0, total_frames - 1)

    for start in range(0, len(src_indices), chunk_size):
        end = min(start + chunk_size, len(src_indices))
        chunk_src = src_indices[start:end]
        try:
            frames_batch = vr.get_batch(chunk_src).asnumpy()
        except Exception as e:
            print(f"  Decord batch error, falling back to single frame reading for this chunk: {e}")
            frames_batch = []
            for idx in chunk_src:
                frame = vr[idx].asnumpy()
                frames_batch.append(frame)
            frames_batch = np.array(frames_batch)
        resized = np.zeros((len(frames_batch), h, w, 3), dtype=np.uint8)
        for i, f in enumerate(frames_batch):
            resized[i] = cv2.resize(f, (w, h))
        yield resized
    del vr

# ─────────────────────────── STEP 4: LABEL A CHUNK WITH IDM ───────────────────────────
def label_chunk_with_idm(idm, frames, batch_size=64):
    idm.eval()
    actions = []
    total = len(frames) - 1
    if total <= 0:
        return actions
    with torch.no_grad():
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            batch_t = torch.from_numpy(frames[i:end]).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0
            batch_t1 = torch.from_numpy(frames[i+1:end+1]).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0
            out = idm(batch_t, batch_t1)
            for j in range(len(batch_t)):
                actions.append({
                    'movement': (out['movement'][j].cpu().numpy() > 0.5).astype(int),
                    'mouse_x_idx': torch.argmax(out['mouse_x'][j]).item(),
                    'mouse_y_idx': torch.argmax(out['mouse_y'][j]).item(),
                    'attack': torch.argmax(out['attack'][j]).item(),
                    'jump': torch.argmax(out['jump'][j]).item(),
                    'crouch': torch.argmax(out['crouch'][j]).item(),
                    'saber': torch.argmax(out['saber'][j]).item(),
                })
    return actions

# ─────────────────────────── STEP 5: TRAIN POLICY USING BATCHED TBPTT ───────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, sequences_frames, sequences_actions):
        self.sequences_frames = sequences_frames
        self.sequences_actions = sequences_actions
    def __len__(self):
        return len(self.sequences_frames)
    def __getitem__(self, idx):
        return self.sequences_frames[idx], self.sequences_actions[idx]

def collate_sequences(batch):
    batch_frames = []
    batch_actions = []
    for frames_seq, actions_seq in batch:
        frames_tensor = torch.from_numpy(frames_seq).float().permute(0, 3, 1, 2) / 255.0
        if frames_tensor.shape[-2:] != (224, 224):
            frames_tensor = nn.functional.interpolate(frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        batch_frames.append(frames_tensor)
        batch_actions.append(actions_seq)
    batch_frames = torch.stack(batch_frames, dim=0)
    return batch_frames, batch_actions

def train_policy_on_batch(policy, optimizer, frames, actions, seq_len=SEQ_LEN, epochs=1):
    n_frames = len(frames)
    if n_frames < seq_len:
        return
    min_len = min(len(frames), len(actions))
    frames = frames[:min_len]
    actions = actions[:min_len]
    pad_len = (seq_len - min_len % seq_len) % seq_len
    if pad_len > 0:
        frames = np.pad(frames, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
        actions = actions + [actions[-1]] * pad_len
    n_sequences = len(frames) // seq_len
    seq_frames = [frames[i*seq_len:(i+1)*seq_len] for i in range(n_sequences)]
    seq_actions = [actions[i*seq_len:(i+1)*seq_len] for i in range(n_sequences)]
    dataset = SequenceDataset(seq_frames, seq_actions)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sequences, num_workers=0)

    total_loss = 0
    n_batches = 0
    for epoch in range(epochs):
        policy.train()
        policy.reset_state()
        for batch_frames, batch_actions in loader:
            batch_frames = batch_frames.to(DEVICE)
            batch_size, seq_len_f, c, h, w = batch_frames.shape
            if seq_len_f != seq_len:
                continue
            policy.reset_state()
            movements, mx_logits, my_logits, attacks, jumps, crouches, sabers = [], [], [], [], [], [], []
            true_mv, true_mx, true_my, true_atk, true_jmp, true_crc, true_sab = [], [], [], [], [], [], []
            for t in range(seq_len):
                x = batch_frames[:, t, :, :, :]
                mv, mx, my, atk, jmp, crc, sab = policy(x)
                movements.append(mv); mx_logits.append(mx); my_logits.append(my)
                attacks.append(atk); jumps.append(jmp); crouches.append(crc); sabers.append(sab)
                true_mv_t = torch.stack([torch.tensor(act[t]['movement'], dtype=torch.float32) for act in batch_actions], dim=0).to(DEVICE)
                true_mx_t = torch.tensor([act[t]['mouse_x_idx'] for act in batch_actions], dtype=torch.long).to(DEVICE)
                true_my_t = torch.tensor([act[t]['mouse_y_idx'] for act in batch_actions], dtype=torch.long).to(DEVICE)
                true_atk_t = torch.tensor([act[t]['attack'] for act in batch_actions], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                true_jmp_t = torch.tensor([act[t]['jump'] for act in batch_actions], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                true_crc_t = torch.tensor([act[t]['crouch'] for act in batch_actions], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                true_sab_t = torch.tensor([act[t]['saber'] for act in batch_actions], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                true_mv.append(true_mv_t); true_mx.append(true_mx_t); true_my.append(true_my_t)
                true_atk.append(true_atk_t); true_jmp.append(true_jmp_t); true_crc.append(true_crc_t); true_sab.append(true_sab_t)

            movements = torch.stack(movements, dim=0).view(-1, 4)
            mx_logits = torch.stack(mx_logits, dim=0).view(-1, 23)
            my_logits = torch.stack(my_logits, dim=0).view(-1, 15)
            attacks = torch.stack(attacks, dim=0).view(-1, 1)
            jumps = torch.stack(jumps, dim=0).view(-1, 1)
            crouches = torch.stack(crouches, dim=0).view(-1, 1)
            sabers = torch.stack(sabers, dim=0).view(-1, 1)

            true_mv = torch.stack(true_mv, dim=0).view(-1, 4)
            true_mx = torch.stack(true_mx, dim=0).view(-1)
            true_my = torch.stack(true_my, dim=0).view(-1)
            true_atk = torch.stack(true_atk, dim=0).view(-1, 1)
            true_jmp = torch.stack(true_jmp, dim=0).view(-1, 1)
            true_crc = torch.stack(true_crc, dim=0).view(-1, 1)
            true_sab = torch.stack(true_sab, dim=0).view(-1, 1)

            bce = nn.BCELoss(); ce = nn.CrossEntropyLoss()
            loss = (bce(movements, true_mv) + ce(mx_logits, true_mx) + ce(my_logits, true_my) +
                    bce(attacks, true_atk) + bce(jumps, true_jmp) + bce(crouches, true_crc) + bce(sabers, true_sab)) / 7
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1
        if n_batches > 0:
            print(f"      Chunk loss (epoch {epoch+1}): {total_loss/n_batches:.4f}")
    policy.reset_state()

# ─────────────────────────── MAIN PIPELINE ───────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("\nLoading IDM...")
    idm = InverseDynamicsModel().to(DEVICE)
    idm.load_state_dict(torch.load('idm_best.pth', map_location=DEVICE))
    idm.eval()
    print("IDM ready.")

    policy = StatefulLSTMPolicy().to(DEVICE)
    policy_path = 'policy_lstm_trained.pth'
    try:
        policy.load_state_dict(torch.load(policy_path, map_location=DEVICE))
        print(f"Resumed policy from {policy_path}")
    except:
        print("Starting policy from scratch")

    optimizer = optim.Adam(policy.parameters(), lr=POLICY_LR)

    video_urls = get_channel_video_urls(CHANNEL_URL)
    progress_file = 'youtube_pipeline_progress_lstm.pkl'
    completed = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as f:
            completed = pickle.load(f)
        print(f"\nResuming: {len(completed)} videos already processed")

    total_videos = len(video_urls)
    for vid_idx, (url, title) in enumerate(video_urls):
        if url in completed:
            print(f"\n[{vid_idx+1}/{total_videos}] SKIP: {title}")
            continue

        print(f"\n{'='*60}\n[{vid_idx+1}/{total_videos}] {title}\n{'='*60}")
        try:
            print("\n📥 Downloading...")
            video_path = download_video(url, title)
            if not os.path.exists(video_path):
                files = glob.glob(os.path.join(VIDEO_DIR, '*'))
                if files:
                    video_path = files[0]
                else:
                    print("  ❌ Download failed")
                    continue
            file_size_mb = os.path.getsize(video_path) / 1e6
            print(f"  File size: {file_size_mb:.0f} MB")

            print("\n🎬 Streaming frames and training LSTM policy...")
            frame_buffer = []
            total_actions_processed = 0
            chunk_idx = 0
            for chunk in stream_frames_fast(video_path, chunk_size=500, target_fps=TARGET_FPS):
                frame_buffer.extend(chunk)
                if len(frame_buffer) >= 200:
                    chunk_frames = np.array(frame_buffer[:200])
                    actions = label_chunk_with_idm(idm, chunk_frames)
                    if actions:
                        train_policy_on_batch(policy, optimizer, chunk_frames[:-1], actions, seq_len=SEQ_LEN, epochs=POLICY_EPOCHS_PER_VIDEO)
                        total_actions_processed += len(actions)
                    frame_buffer = [frame_buffer[-1]]
                chunk_idx += 1
                if chunk_idx % 10 == 0:
                    print(f"    Processed {chunk_idx} chunks, {total_actions_processed} actions so far")

            if len(frame_buffer) > 1:
                chunk_frames = np.array(frame_buffer)
                actions = label_chunk_with_idm(idm, chunk_frames)
                if actions:
                    train_policy_on_batch(policy, optimizer, chunk_frames[:-1], actions, seq_len=SEQ_LEN, epochs=POLICY_EPOCHS_PER_VIDEO)
                    total_actions_processed += len(actions)

            print(f"  Total actions generated: {total_actions_processed}")
            torch.save(policy.state_dict(), policy_path)
            print(f"  💾 Saved policy checkpoint to {policy_path}")

            os.remove(video_path)
            print(f"  🗑️  Deleted {video_path}")
            completed.add(url)
            with open(progress_file, 'wb') as f:
                pickle.dump(completed, f)
            print(f"\n  ✅ Done! ({len(completed)}/{total_videos} videos processed)")

        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            for f in glob.glob(os.path.join(VIDEO_DIR, '*')):
                try:
                    os.remove(f)
                except:
                    pass
            continue

    print(f"\n{'='*60}\n🎉 PIPELINE COMPLETE!\n  Videos processed: {len(completed)}/{total_videos}\n  Policy saved to: {policy_path}\n{'='*60}")

if __name__ == "__main__":
    main()