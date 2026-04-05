# jka_2026_controller.py (sampling with temperature)
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
from torchvision import models

# ----------------------------------------------------------------------
# Stateful LSTM Policy Network (returns logits for mouse, sigmoid for others)
# ----------------------------------------------------------------------
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
        self.mouse_x_head = nn.Linear(lstm_hidden, len(mouse_x_possibles))   # logits
        self.mouse_y_head = nn.Linear(lstm_hidden, len(mouse_y_possibles))   # logits

        self.hidden = None

    def reset_state(self):
        self.hidden = None

    def forward(self, x):
        batch_size = x.size(0)
        features = self.backbone(x)
        features = features.view(batch_size, -1).unsqueeze(1)
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


if __name__ == "__main__":

    key_map = {
        'w': key_output.w_char,
        'a': key_output.a_char,
        's': key_output.s_char,
        'd': key_output.d_char,
        'ctrl': key_output.ctrl_char,
        'space': key_output.space_char,
        'e': key_output.e_char,
    }

    current_keys = set()
    quit_flag = False
    frame_count = 0
    prev_img = None

    # FPS reporting
    fps_last_time = time.time()
    fps_frame_count = 0

    # Recenter timer
    last_recenter_time = time.time()
    RECENTER_INTERVAL = 0.5

    # Temperature for sampling mouse (lower = sharper, higher = more random)
    TEMPERATURE = 2.0

    def on_press(key):
        global quit_flag
        try:
            if hasattr(key, 'char') and key.char == 'c':
                print("\n'C' pressed - Exiting...")
                quit_flag = True
                return False
        except:
            pass

    def execute_actions(movement_probs, mouse_x_logits, mouse_y_logits,
                        attack_prob, jump_prob, crouch_prob, saber_prob):
        global current_keys

        # ---- Sample keys (WASD) independently ----
        keys_to_press = set()
        if np.random.rand() < movement_probs[0]: keys_to_press.add('w')
        if np.random.rand() < movement_probs[1]: keys_to_press.add('a')
        if np.random.rand() < movement_probs[2]: keys_to_press.add('s')
        if np.random.rand() < movement_probs[3]: keys_to_press.add('d')
        if np.random.rand() < crouch_prob: keys_to_press.add('ctrl')

        for key in current_keys - keys_to_press:
            key_output.ReleaseKey(key_map[key])
        for key in keys_to_press - current_keys:
            key_output.HoldKey(key_map[key])
        current_keys = keys_to_press

        # ---- Sample mouse with temperature ----
        # Apply temperature to logits, then softmax, then sample
        mouse_x_logits_temp = mouse_x_logits / TEMPERATURE
        mouse_y_logits_temp = mouse_y_logits / TEMPERATURE
        # Subtract max for numerical stability (softmax)
        mouse_x_probs = np.exp(mouse_x_logits_temp - np.max(mouse_x_logits_temp))
        mouse_x_probs /= mouse_x_probs.sum()
        mouse_y_probs = np.exp(mouse_y_logits_temp - np.max(mouse_y_logits_temp))
        mouse_y_probs /= mouse_y_probs.sum()

        mouse_x_idx = np.random.choice(len(mouse_x_probs), p=mouse_x_probs)
        mouse_y_idx = np.random.choice(len(mouse_y_probs), p=mouse_y_probs)
        mouse_delta_x = mouse_x_possibles[mouse_x_idx]
        mouse_delta_y = mouse_y_possibles[mouse_y_idx]
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouse_delta_x), int(mouse_delta_y), 0, 0)

        # ---- Sample binary actions ----
        if np.random.rand() < attack_prob:
            key_output.left_click()
        if np.random.rand() < jump_prob:
            key_output.right_click()
        if np.random.rand() < saber_prob:
            key_output.HoldKey(key_map['space'])
            time.sleep(0.05)
            key_output.ReleaseKey(key_map['space'])

    # Initialize policy
    policy = StatefulLSTMPolicy().cuda()
    try:
        policy.load_state_dict(torch.load('policy_lstm_trained.pth'))
        print("✅ Loaded trained policy from policy_lstm_trained.pth")
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

    sct = mss.mss()
    model_input_size = (224, 224)
    print(f"Model input size: {model_input_size[0]}x{model_input_size[1]}")
    print(f"Temperature: {TEMPERATURE} (lower = more deterministic, higher = more random)")
    print("LSTM state persists indefinitely. Press 'C' to quit.")

    listener = pynput.keyboard.Listener(on_press=on_press)
    listener.start()

    input_tensor = torch.zeros(1, 3, model_input_size[0], model_input_size[1], device='cuda')
    frame_buffer = np.zeros((model_input_size[0], model_input_size[1], 3), dtype=np.float32)

    # Warmup
    dummy = torch.randn(1, 3, model_input_size[0], model_input_size[1], device='cuda')
    with torch.no_grad():
        _ = policy(dummy)
    torch.cuda.synchronize()

    target_fps = 16
    frame_time = 1.0 / target_fps

    print("\n🎮 Starting game controller at 16 FPS (sampling actions with temperature)...\n")

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
            img = cv2.resize(img, model_input_size)

            if prev_img is not None and frame_count % 60 == 0:
                diff = np.abs(img.astype(float) - prev_img.astype(float)).mean()
                print(f"Frame diff: {diff:.2f}")
            prev_img = img.copy()

            # Preprocess
            np.copyto(frame_buffer, img.astype(np.float32) / 255.0)
            tensor_cpu = torch.from_numpy(frame_buffer).permute(2, 0, 1).unsqueeze(0)
            input_tensor.copy_(tensor_cpu)

            # Forward pass (returns logits for mouse)
            movement, mouse_x_logits, mouse_y_logits, attack, jump, crouch, saber = policy(input_tensor)

            movement_probs = movement[0].cpu().numpy()
            mouse_x_logits_np = mouse_x_logits[0].cpu().numpy()
            mouse_y_logits_np = mouse_y_logits[0].cpu().numpy()
            attack_prob = attack[0].cpu().numpy()[0]
            jump_prob = jump[0].cpu().numpy()[0]
            crouch_prob = crouch[0].cpu().numpy()[0]
            saber_prob = saber[0].cpu().numpy()[0]

            if frame_count % 60 == 0:
                # For debugging, show softmax probabilities (without temperature)
                probs_x = np.exp(mouse_x_logits_np - np.max(mouse_x_logits_np))
                probs_x /= probs_x.sum()
                probs_y = np.exp(mouse_y_logits_np - np.max(mouse_y_logits_np))
                probs_y /= probs_y.sum()
                print(f"\n--- Frame {frame_count} ---")
                print(f"Movement probs: {movement_probs.round(2)}")
                print(f"Mouse X probs (first 5): {probs_x[:5].round(3)} ... last: {probs_x[-5:].round(3)}")
                print(f"Mouse Y probs (first 5): {probs_y[:5].round(3)} ... last: {probs_y[-5:].round(3)}")
                print(f"Attack: {attack_prob:.3f}, Jump: {jump_prob:.3f}, Crouch: {crouch_prob:.3f}, Saber: {saber_prob:.3f}")

            execute_actions(
                movement_probs,
                mouse_x_logits_np,
                mouse_y_logits_np,
                attack_prob,
                jump_prob,
                crouch_prob,
                saber_prob
            )

            # Periodic recenter
            now = time.time()
            if now - last_recenter_time >= RECENTER_INTERVAL:
                key_output.HoldKey(key_map['e'])
                time.sleep(0.1)
                key_output.ReleaseKey(key_map['e'])
                last_recenter_time = now
                print("  🔄 Recentered view (E)")

            # FPS reporting
            now = time.time()
            if now - fps_last_time >= 1.0:
                fps = fps_frame_count / (now - fps_last_time)
                print(f"📊 FPS: {fps:.1f}")
                fps_last_time = now
                fps_frame_count = 0

            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    print("Releasing all keys...")
    listener.stop()
    for key in list(current_keys):
        key_output.ReleaseKey(key_map[key])
    print("Done")