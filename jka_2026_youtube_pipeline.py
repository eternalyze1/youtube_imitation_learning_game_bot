# jka_2026_youtube_pipeline.py
# Streaming pipeline: download → stream → label → train → delete, one video at a time.
# Uses decord to stream frames in chunks, never loading the whole video.
# Trains the stateful LSTM policy using Truncated Backpropagation Through Time (TBPTT) with seq_len=96.
# Fixed hidden state size mismatch by dropping incomplete last batch.
# Class weighting is optional (disabled by default).

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
from decord import VideoReader, cpu

from jka_2026_controller import StatefulLSTMPolicy
from jka_2026_train_idm import InverseDynamicsModel
from config import csgo_img_dimension, mouse_x_possibles, mouse_y_possibles

# ─────────────────────────── CONFIG ───────────────────────────
CHANNEL_URL = "https://www.youtube.com/@JediAcademyLeague/videos"
VIDEO_DIR = "youtube_videos_tmp"
TARGET_FPS = 16
POLICY_EPOCHS_PER_VIDEO = 1
POLICY_LR = 0.0001
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 96
POLICY_INPUT_SIZE = (224, 224)

# Class weighting (set to True to enable, False for standard losses)
USE_CLASS_WEIGHTS = False

# ─────────────────────────── STEP 1-4 (unchanged) ───────────────────────────
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

def download_video(url, title):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(VIDEO_DIR, '*')):
        try:
            os.remove(f)
        except:
            pass
    ydl_opts = {
        'format': 'worst[ext=mp4][height>=360]/worst[ext=mp4]/worst',
        'outtmpl': os.path.join(VIDEO_DIR, '%(id)s.%(ext)s'),
        'quiet': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        for ext in ['.webm', '.mkv']:
            filename = filename.replace(ext, '.mp4')
        print(f"Downloaded: {filename}")
        return filename

def stream_frames_fast(video_path, chunk_size=1000, target_fps=TARGET_FPS):
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
        frames_batch = vr.get_batch(chunk_src).asnumpy()
        resized = np.zeros((len(frames_batch), h, w, 3), dtype=np.uint8)
        for i, f in enumerate(frames_batch):
            resized[i] = cv2.resize(f, (w, h))
        yield resized
    del vr

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

# ─────────────────────────── STEP 5: TRAIN POLICY WITH TBPTT ───────────────────────────
class LSTMVideoDataset(Dataset):
    def __init__(self, frames, actions):
        min_len = min(len(frames), len(actions))
        self.frames = frames[:min_len]
        self.actions = actions[:min_len]
        self.target_h, self.target_w = POLICY_INPUT_SIZE
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        frame = self.frames[idx]
        if frame.shape[0] != self.target_h or frame.shape[1] != self.target_w:
            frame = cv2.resize(frame, (self.target_w, self.target_h))
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        a = self.actions[idx]
        return (
            frame_tensor,
            torch.tensor(a['movement'], dtype=torch.float32),
            torch.tensor(a['mouse_x_idx'], dtype=torch.long),
            torch.tensor(a['mouse_y_idx'], dtype=torch.long),
            torch.tensor(a['attack'], dtype=torch.float32),
            torch.tensor(a['jump'], dtype=torch.float32),
            torch.tensor(a['crouch'], dtype=torch.float32),
            torch.tensor(a['saber'], dtype=torch.float32),
        )

def train_policy_on_batch(policy, optimizer, frames, actions, seq_len=SEQ_LEN, epochs=1, use_class_weights=USE_CLASS_WEIGHTS):
    """
    Train LSTM policy using Truncated Backpropagation Through Time (TBPTT).
    Splits data into fixed-length sequences (seq_len) and resets hidden state
    at the start of each sequence. Drops incomplete last batch to keep hidden size consistent.
    If use_class_weights is False, standard (unweighted) losses are used.
    """
    n_frames = len(frames)
    if n_frames < seq_len:
        return

    # Pad to make divisible by seq_len
    pad_len = (seq_len - n_frames % seq_len) % seq_len
    if pad_len > 0:
        frames = np.pad(frames, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
        actions = actions + [actions[-1]] * pad_len

    n_sequences = len(frames) // seq_len
    seq_frames = frames.reshape(n_sequences, seq_len, frames.shape[1], frames.shape[2], frames.shape[3])
    seq_actions = [actions[i*seq_len:(i+1)*seq_len] for i in range(n_sequences)]

    total_loss = 0
    n_batches = 0

    for epoch in range(epochs):
        policy.train()
        policy.reset_state()

        for seq_idx in range(n_sequences):
            seq_frames_chunk = seq_frames[seq_idx]
            seq_actions_chunk = seq_actions[seq_idx]

            dataset = LSTMVideoDataset(seq_frames_chunk, seq_actions_chunk)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0)

            total_frames_seq = len(seq_actions_chunk)

            # Precompute class weights only if enabled
            if use_class_weights:
                mv = np.array([a['movement'] for a in seq_actions_chunk])
                movement_pos_weights = []
                for ch in range(4):
                    pos = mv[:, ch].sum()
                    pw = (total_frames_seq - pos) / pos if pos > 0 else 1.0
                    movement_pos_weights.append(pw)
                mpw = torch.tensor(movement_pos_weights, dtype=torch.float32).to(DEVICE)

                def weighted_movement_bce(pred, target):
                    loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
                    w = torch.where(target == 1, mpw.unsqueeze(0), torch.ones_like(loss))
                    return (loss * w).mean()

                def make_pw(key):
                    pos = sum(1 for a in seq_actions_chunk if a[key] == 1)
                    return torch.tensor([(total_frames_seq - pos) / pos if pos > 0 else 1.0], dtype=torch.float32).to(DEVICE)

                attack_pw = make_pw('attack')
                jump_pw = make_pw('jump')
                crouch_pw = make_pw('crouch')
                saber_pw = make_pw('saber')

                def wbce(pred, target, pw):
                    loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
                    w = torch.where(target == 1, pw, torch.ones_like(loss))
                    return (loss * w).mean()

                mouse_x_counts = np.bincount([a['mouse_x_idx'] for a in seq_actions_chunk], minlength=23)
                mx_w = torch.tensor([total_frames_seq / (23 * c) if c > 0 else 1.0 for c in mouse_x_counts], dtype=torch.float32).to(DEVICE)
                mouse_y_counts = np.bincount([a['mouse_y_idx'] for a in seq_actions_chunk], minlength=15)
                my_w = torch.tensor([total_frames_seq / (15 * c) if c > 0 else 1.0 for c in mouse_y_counts], dtype=torch.float32).to(DEVICE)
                ce_x = nn.CrossEntropyLoss(weight=mx_w)
                ce_y = nn.CrossEntropyLoss(weight=my_w)
            else:
                # Standard losses (no weights)
                bce = nn.BCELoss()
                ce = nn.CrossEntropyLoss()

            for batch in loader:
                # Detach hidden state before forward pass to cut graph
                if policy.hidden is not None:
                    policy.hidden = (policy.hidden[0].detach(), policy.hidden[1].detach())

                frame, mv_true, mx_true, my_true, atk_true, jmp_true, crc_true, sab_true = [b.to(DEVICE) for b in batch]

                mv_pred, mx_pred, my_pred, atk_pred, jmp_pred, crc_pred, sab_pred = policy(frame)

                if use_class_weights:
                    loss = (
                        weighted_movement_bce(mv_pred, mv_true) +
                        ce_x(mx_pred, mx_true) +
                        ce_y(my_pred, my_true) +
                        wbce(atk_pred, atk_true.unsqueeze(1), attack_pw) +
                        wbce(jmp_pred, jmp_true.unsqueeze(1), jump_pw) +
                        wbce(crc_pred, crc_true.unsqueeze(1), crouch_pw) +
                        wbce(sab_pred, sab_true.unsqueeze(1), saber_pw)
                    ) / 7
                else:
                    loss = (
                        bce(mv_pred, mv_true) +
                        ce(mx_pred, mx_true) +
                        ce(my_pred, my_true) +
                        bce(atk_pred, atk_true.unsqueeze(1)) +
                        bce(jmp_pred, jmp_true.unsqueeze(1)) +
                        bce(crc_pred, crc_true.unsqueeze(1)) +
                        bce(sab_pred, sab_true.unsqueeze(1))
                    ) / 7

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

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
                        train_policy_on_batch(policy, optimizer, chunk_frames, actions, seq_len=SEQ_LEN, epochs=POLICY_EPOCHS_PER_VIDEO, use_class_weights=USE_CLASS_WEIGHTS)
                        total_actions_processed += len(actions)
                    frame_buffer = [frame_buffer[-1]]
                chunk_idx += 1
                if chunk_idx % 10 == 0:
                    print(f"    Processed {chunk_idx} chunks, {total_actions_processed} actions so far")
            
            if len(frame_buffer) > 1:
                chunk_frames = np.array(frame_buffer)
                actions = label_chunk_with_idm(idm, chunk_frames)
                if actions:
                    train_policy_on_batch(policy, optimizer, chunk_frames, actions, seq_len=SEQ_LEN, epochs=POLICY_EPOCHS_PER_VIDEO, use_class_weights=USE_CLASS_WEIGHTS)
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