# extract_youtube_frames.py
import cv2
import numpy as np
import os
from config import csgo_img_dimension

def extract_frames_at_60fps(video_path, output_npy='youtube_frames_60fps.npy'):
    """
    Extract frames and force to 60 FPS
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    # Target: 60 FPS
    target_fps = 60
    target_frame_count = int(duration * target_fps)
    
    h, w = csgo_img_dimension  # 150, 280 from your config
    frames = []
    
    print(f"Original video: {original_fps:.2f} FPS, {duration:.2f}s, {total_frames} frames")
    print(f"Target: {target_fps} FPS, {target_frame_count} frames")
    print(f"Resizing frames to: {w}x{h}")
    
    for i in range(target_frame_count):
        # Calculate which source frame to sample
        source_frame_pos = (i / target_frame_count) * total_frames
        source_frame_idx = int(source_frame_pos)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Resize to model input size
            frame = cv2.resize(frame, (w, h))
            frames.append(frame)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Extracted {i+1}/{target_frame_count} frames ({((i+1)/target_frame_count)*100:.1f}%)")
    
    cap.release()
    
    # Save frames
    frames_array = np.array(frames, dtype=np.uint8)
    # np.save(output_npy, frames_array)
    # np.savez_compressed(output_npy.replace('.npy', '.npz'), frames=frames_array)
    
    # Also save as compressed
    np.savez_compressed(output_npy.replace('.npy', '.npz'), frames=frames_array)
    
    print(f"\n✅ Saved {len(frames)} frames to {output_npy}")
    print(f"   File size: {frames_array.nbytes / 1e9:.2f} GB")
    print(f"   Compressed: {output_npy.replace('.npy', '.npz')}")
    
    return frames_array

if __name__ == "__main__":
    video_path = r"youtube_videos\Jedi Knight： Jedi Academy - 3v3 ladder - 2bad vs eXecutors.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        print("Checking alternative paths...")
        # Try with different filename encoding
        import glob
        files = glob.glob("youtube_videos/*.mp4")
        if files:
            video_path = files[0]
            print(f"Found: {video_path}")
        else:
            exit(1)
    
    frames = extract_frames_at_60fps(video_path, 'youtube_frames_60fps.npy')
    
    print(f"\nExtraction complete!")
    print(f"Total frames at 60 FPS: {len(frames)}")
    print(f"Duration at 60 FPS: {len(frames)/60:.2f} seconds")