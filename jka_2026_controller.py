# jka_2026_controller.py
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

# ----------------------------------------------------------------------
# Policy network with softmax on mouse heads (as in the CSGO paper)
# ----------------------------------------------------------------------
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 3)
        self.conv2 = nn.Conv2d(32, 64, 3, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7680, 128)
        self.fc2 = nn.Linear(128, 4 + 23 + 15 + 5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        movement = torch.sigmoid(x[:, 0:4])
        mouse_x = x[:, 4:27]
        mouse_y = x[:, 27:42]
        attack = torch.sigmoid(x[:, 42:43])
        jump = torch.sigmoid(x[:, 43:44])
        crouch = torch.sigmoid(x[:, 44:45])
        saber = torch.sigmoid(x[:, 45:46])

        return movement, mouse_x, mouse_y, attack, jump, crouch, saber


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
    prev_img = None   # for frame difference debugging

    # FPS reporting
    fps_last_time = time.time()
    fps_frame_count = 0

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

        # ---- Deterministic keys (WASD) - threshold 0.5 ----
        keys_to_press = set()
        if movement[0] > 0.5: keys_to_press.add('w')
        if movement[1] > 0.5: keys_to_press.add('a')
        if movement[2] > 0.5: keys_to_press.add('s')
        if movement[3] > 0.5: keys_to_press.add('d')
        if crouch > 0.5: keys_to_press.add('ctrl')

        # Release keys no longer pressed
        for key in current_keys - keys_to_press:
            key_output.ReleaseKey(key_map[key])
        # Press new keys
        for key in keys_to_press - current_keys:
            key_output.HoldKey(key_map[key])
        current_keys = keys_to_press

        # ---- Mouse movement: use argmax (as in the CSGO paper) ----
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouse_delta_x), int(mouse_delta_y), 0, 0)

        # ---- Binary actions: threshold 0.5 ----
        if attack > 0.5:
            key_output.left_click()
        if jump > 0.5:
            key_output.right_click()
        if saber > 0.5:
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
    print("Using deterministic actions (argmax for mouse, threshold 0.5 for keys).")
    print("Press 'C' to quit")

    listener = pynput.keyboard.Listener(on_press=on_press)
    listener.start()

    # Pre‑allocate tensors
    input_tensor = torch.zeros(1, 3, h, w, device='cuda')
    frame_buffer = np.zeros((h, w, 3), dtype=np.float32)

    # Warmup
    dummy = torch.randn(1, 3, h, w, device='cuda')
    with torch.no_grad():
        _ = policy(dummy)
    torch.cuda.synchronize()

    target_fps = 60
    frame_time = 1.0 / target_fps

    print("\n🎮 Starting game controller...\n")

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

            # Mouse: argmax over probabilities
            mouse_x_probs = mouse_x[0].cpu().numpy()
            mouse_y_probs = mouse_y[0].cpu().numpy()
            mouse_x_idx = np.argmax(mouse_x_probs)
            mouse_y_idx = np.argmax(mouse_y_probs)
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
                print(f"Argmax mouse X: idx={mouse_x_idx}, val={mouse_delta_x:.1f}")
                print(f"Argmax mouse Y: idx={mouse_y_idx}, val={mouse_delta_y:.1f}")
                print(f"Attack: {attack_prob:.3f}, Jump: {jump_prob:.3f}, Crouch: {crouch_prob:.3f}, Saber: {saber_prob:.3f}")

            # Execute actions deterministically
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