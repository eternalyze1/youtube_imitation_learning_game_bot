# test_frame_freshness.py
import win32gui
import time
import numpy as np
from screen_input import grab_window
from config import csgo_game_res

hwin = win32gui.FindWindow(None, 'EternalJK')
win32gui.SetForegroundWindow(hwin)
time.sleep(1)

# Capture multiple frames and check if they change
frames = []
for i in range(10):
    # Force window update
    win32gui.InvalidateRect(hwin, None, True)
    win32gui.UpdateWindow(hwin)
    time.sleep(0.05)  # 50ms between captures
    
    img = grab_window(hwin, game_resolution=csgo_game_res, SHOW_IMAGE=False)
    frames.append(img)
    
    if i > 0:
        diff = np.abs(frames[-1].astype(float) - frames[-2].astype(float)).mean()
        print(f"Frame {i}: diff = {diff:.2f}")
        
        if diff < 1.0:
            print("  WARNING: Frame is stale! Very little change.")
    
# Save frames to see if they're identical
for i, frame in enumerate(frames[:3]):
    import cv2
    cv2.imwrite(f"test_frame_{i}.png", frame)
    print(f"Saved test_frame_{i}.png")