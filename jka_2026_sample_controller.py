# jka_2026_sample_controller.py
# Same as jka_2026_controller.py but SAMPLES from distributions
# instead of taking argmax/threshold. This produces more varied,
# human-like behavior at the cost of occasional random actions.

import torch
import torch.nn as nn
import win32gui
import win32api
import win32con
import time
import numpy as np
import cv2
import mss
from config import csgo_game_res, csgo_img_dimension, mouse_x_possibles, mouse_y_possibles
import key_input
import key_output
import pynput

# Import the Policy class from the deterministic controller
from jka_2026_controller import Policy


if __name__ == "__main__":

    key_map = {
        'w': key_output.w_char,
        'a': key_output.a_char,
        's': key_output.s_char,
        'd': key_output.d_char,
        'ctrl': key_output.ctrl_char,
        'space': key_output.space_char,
    }

    current_keys = set()
    quit_flag = False
    frame_count = 0
    prev_img = None

    # FPS reporting
    fps_last_time = time.time()
    fps_frame_count = 0

    # Temperature for mouse sampling (>1 = more random, <1 = more peaked)
    MOUSE_TEMPERATURE = 1.0

    def on_press(key):
        global quit_flag
        try:
            if hasattr(key, 'char') and key.char == 'c':
                print("\n'C' pressed - Exiting...")
                quit_flag = True
                return False
        except:
            pass

    def execute_actions(movement, mouse_delta_x, mouse_delta_y, attack, jump, crouch, saber):
        global current_keys

        # ---- Movement: sample from Bernoulli instead of threshold ----
        keys_to_press = set()
        if np.random.random() < movement[0]: keys_to_press.add('w')
        if np.random.random() < movement[1]: keys_to_press.add('a')
        if np.random.random() < movement[2]: keys_to_press.add('s')
        if np.random.random() < movement[3]: keys_to_press.add('d')
        if np.random.random() < crouch: keys_to_press.add('ctrl')

        # Release keys no longer pressed
        for key in current_keys - keys_to_press:
            key_output.ReleaseKey(key_map[key])
        # Press new keys
        for key in keys_to_press - current_keys:
            key_output.HoldKey(key_map[key])
        current_keys = keys_to_press

        # ---- Mouse movement ----
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouse_delta_x), int(mouse_delta_y), 0, 0)

        # ---- Binary actions: sample from Bernoulli ----
        if np.random.random() < attack:
            key_output.left_click()
        if np.random.random() < jump:
            key_output.right_click()
        if np.random.random() < saber:
            key_output.HoldKey(key_map['space'])
            time.sleep(0.05)
            key_output.ReleaseKey(key_map['space'])

    # Initialize policy
    policy = Policy().cuda()
    try:
        policy.load_state_dict(torch.load('policy_youtube_trained.pth'))
        print("✅ Loaded trained policy from policy_youtube_trained.pth")
    except:
        print("⚠️ No trained weights found, using untrained policy")
    policy.eval()

    # Find game window
    hwin_csgo = win32gui.FindWindow(None, 'EternalJK')
    if not hwin_csgo:
        print("ERROR: Could not find EternalJK window!")
        exit(1)

    win32gui.SetForegroundWindow(hwin_csgo)
    time.sleep(1)

    # Screen capture using mss
    sct = mss.mss()
    h, w = csgo_img_dimension
    print(f"Frame size: {w}x{h}")
    print(f"Mouse temperature: {MOUSE_TEMPERATURE}")
    print("Using SAMPLED actions (Bernoulli for keys, categorical for mouse).")
    print("Press 'C' to quit")

    listener = pynput.keyboard.Listener(on_press=on_press)
    listener.start()

    # Pre-allocate tensors
    input_tensor = torch.zeros(1, 3, h, w, device='cuda')
    frame_buffer = np.zeros((h, w, 3), dtype=np.float32)

    # Warmup
    dummy = torch.randn(1, 3, h, w, device='cuda')
    with torch.no_grad():
        _ = policy(dummy)
    torch.cuda.synchronize()

    target_fps = 60
    frame_time = 1.0 / target_fps

    print("\n🎮 Starting game controller (SAMPLING mode)...\n")

    with torch.no_grad():
        while not quit_flag:
            start = time.time()
            frame_count += 1
            fps_frame_count += 1

            # Capture window
            left, top, right, bottom = win32gui.GetWindowRect(hwin_csgo)
            monitor = {"left": left, "top": top, "width": right - left, "height": bottom - top}
            img = sct.grab(monitor)
            img = np.array(img)
            img = img[:, :, :3]
            img = cv2.resize(img, (w, h))

            # Frame difference debugging (every 60 frames)
            if prev_img is not None and frame_count % 60 == 0:
                diff = np.abs(img.astype(float) - prev_img.astype(float)).mean()
                print(f"Frame diff: {diff:.2f}")
            prev_img = img.copy()

            # Preprocess
            np.copyto(frame_buffer, img.astype(np.float32) / 255.0)
            tensor_cpu = torch.from_numpy(frame_buffer).permute(2, 0, 1).unsqueeze(0)
            input_tensor.copy_(tensor_cpu)

            # Forward pass
            movement, mouse_x, mouse_y, attack, jump, crouch, saber = policy(input_tensor)

            # Mouse: SAMPLE from categorical distribution (with temperature)
            mouse_x_logits = mouse_x[0].cpu()
            mouse_y_logits = mouse_y[0].cpu()
            
            # Apply temperature scaling then softmax to get probabilities
            mouse_x_probs = torch.softmax(mouse_x_logits / MOUSE_TEMPERATURE, dim=0).numpy()
            mouse_y_probs = torch.softmax(mouse_y_logits / MOUSE_TEMPERATURE, dim=0).numpy()
            
            # Sample from the categorical distribution
            mouse_x_idx = np.random.choice(len(mouse_x_possibles), p=mouse_x_probs)
            mouse_y_idx = np.random.choice(len(mouse_y_possibles), p=mouse_y_probs)
            mouse_delta_x = mouse_x_possibles[mouse_x_idx]
            mouse_delta_y = mouse_y_possibles[mouse_y_idx]

            # Get probabilities for binary actions (already sigmoid)
            movement_probs = movement[0].cpu().numpy()
            attack_prob = attack[0].cpu().numpy()[0]
            jump_prob = jump[0].cpu().numpy()[0]
            crouch_prob = crouch[0].cpu().numpy()[0]
            saber_prob = saber[0].cpu().numpy()[0]

            # Debug every 60 frames
            if frame_count % 60 == 0:
                print(f"\n--- Frame {frame_count} ---")
                print(f"Movement probs: {movement_probs.round(2)}")
                print(f"Sampled mouse X: idx={mouse_x_idx}, val={mouse_delta_x:.1f}")
                print(f"Sampled mouse Y: idx={mouse_y_idx}, val={mouse_delta_y:.1f}")
                print(f"Attack: {attack_prob:.3f}, Jump: {jump_prob:.3f}, Crouch: {crouch_prob:.3f}, Saber: {saber_prob:.3f}")

            # Execute actions via sampling
            execute_actions(
                movement_probs,
                mouse_delta_x,
                mouse_delta_y,
                attack_prob,
                jump_prob,
                crouch_prob,
                saber_prob
            )

            # FPS reporting
            now = time.time()
            if now - fps_last_time >= 1.0:
                fps = fps_frame_count / (now - fps_last_time)
                print(f"📊 FPS: {fps:.1f}")
                fps_last_time = now
                fps_frame_count = 0

            # Maintain target FPS
            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    # Cleanup
    print("Releasing all keys...")
    listener.stop()
    for key in list(current_keys):
        key_output.ReleaseKey(key_map[key])
    print("Done")
