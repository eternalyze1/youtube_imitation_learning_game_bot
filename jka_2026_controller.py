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
from torchvision import models

# ----------------------------------------------------------------------
# ConvLSTM cell (2D)
# ----------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        height, width = spatial_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

# ----------------------------------------------------------------------
# Stateful ConvLSTM Policy Network (based on CSGO paper)
# ----------------------------------------------------------------------
class StatefulConvLSTMPolicy(nn.Module):
    def __init__(self, lstm_hidden=128, kernel_size=3):
        super().__init__()
        # Vision backbone: EfficientNet-B0 (pretrained)
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone = backbone.features
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[-3:].parameters():
            param.requires_grad = True

        # EfficientNet outputs (1280, 7, 7) for 224x224 input
        self.feature_channels = 1280
        self.feature_h = 7
        self.feature_w = 7

        # ConvLSTM layer
        self.convlstm = ConvLSTMCell(input_dim=self.feature_channels,
                                     hidden_dim=lstm_hidden,
                                     kernel_size=kernel_size)
        self.lstm_hidden = lstm_hidden
        self.spatial_size = (self.feature_h, self.feature_w)

        # After ConvLSTM, we have (lstm_hidden, 7, 7) -> flatten
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(lstm_hidden * self.feature_h * self.feature_w, 256)

        # Action heads
        self.movement_head = nn.Sequential(nn.Linear(256, 4), nn.Sigmoid())
        self.attack_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.jump_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.crouch_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.saber_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.mouse_x_head = nn.Linear(256, len(mouse_x_possibles))
        self.mouse_y_head = nn.Linear(256, len(mouse_y_possibles))

        self.hidden_state = None

    def reset_state(self):
        self.hidden_state = None

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        batch_size = x.size(0)
        features = self.backbone(x)           # (batch, 1280, 7, 7)

        if self.hidden_state is None:
            self.hidden_state = self.convlstm.init_hidden(batch_size, self.spatial_size, x.device)

        h, c = self.convlstm(features, self.hidden_state)
        self.hidden_state = (h, c)            # keep for next frame

        # Flatten and pass through FC
        out = self.flatten(h)                 # (batch, lstm_hidden * 7 * 7)
        out = torch.relu(self.fc(out))

        movement = self.movement_head(out)
        attack = self.attack_head(out)
        jump = self.jump_head(out)
        crouch = self.crouch_head(out)
        saber = self.saber_head(out)
        mouse_x_logits = self.mouse_x_head(out)
        mouse_y_logits = self.mouse_y_head(out)

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

    fps_last_time = time.time()
    fps_frame_count = 0
    last_recenter_time = time.time()
    RECENTER_INTERVAL = 0.5

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

        # Deterministic keys (WASD) – argmax (threshold 0.5)
        keys_to_press = set()
        if movement[0] > 0.5: keys_to_press.add('w')
        if movement[1] > 0.5: keys_to_press.add('a')
        if movement[2] > 0.5: keys_to_press.add('s')
        if movement[3] > 0.5: keys_to_press.add('d')
        if crouch > 0.5: keys_to_press.add('ctrl')

        for key in current_keys - keys_to_press:
            key_output.ReleaseKey(key_map[key])
        for key in keys_to_press - current_keys:
            key_output.HoldKey(key_map[key])
        current_keys = keys_to_press

        # Mouse movement: argmax (no sampling)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouse_delta_x), int(mouse_delta_y), 0, 0)

        # Binary actions: probabilistic (sample) as per paper
        if np.random.rand() < attack:
            key_output.left_click()
        if np.random.rand() < jump:
            key_output.right_click()
        if np.random.rand() < saber:
            key_output.HoldKey(key_map['space'])
            time.sleep(0.05)
            key_output.ReleaseKey(key_map['space'])

    # Initialize policy
    policy = StatefulConvLSTMPolicy().cuda()
    try:
        policy.load_state_dict(torch.load('policy_convlstm_trained.pth'))
        print("✅ Loaded trained policy from policy_convlstm_trained.pth")
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
    print("ConvLSTM state persists indefinitely. Press 'C' to quit.")

    listener = pynput.keyboard.Listener(on_press=on_press)
    listener.start()

    input_tensor = torch.zeros(1, 3, model_input_size[0], model_input_size[1], device='cuda')
    frame_buffer = np.zeros((model_input_size[0], model_input_size[1], 3), dtype=np.float32)

    dummy = torch.randn(1, 3, model_input_size[0], model_input_size[1], device='cuda')
    with torch.no_grad():
        _ = policy(dummy)
    torch.cuda.synchronize()

    target_fps = 16
    frame_time = 1.0 / target_fps

    print("\n🎮 Starting game controller at 16 FPS (ConvLSTM)...\n")

    with torch.no_grad():
        while not quit_flag:
            start = time.time()
            frame_count += 1
            fps_frame_count += 1

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

            np.copyto(frame_buffer, img.astype(np.float32) / 255.0)
            tensor_cpu = torch.from_numpy(frame_buffer).permute(2, 0, 1).unsqueeze(0)
            input_tensor.copy_(tensor_cpu)

            movement, mouse_x_logits, mouse_y_logits, attack, jump, crouch, saber = policy(input_tensor)

            # Argmax for mouse
            mouse_x_probs = torch.softmax(mouse_x_logits[0], dim=0).cpu().numpy()
            mouse_y_probs = torch.softmax(mouse_y_logits[0], dim=0).cpu().numpy()
            mouse_x_idx = np.argmax(mouse_x_probs)
            mouse_y_idx = np.argmax(mouse_y_probs)
            mouse_delta_x = mouse_x_possibles[mouse_x_idx]
            mouse_delta_y = mouse_y_possibles[mouse_y_idx]

            movement_probs = movement[0].cpu().numpy()
            attack_prob = attack[0].cpu().numpy()[0]
            jump_prob = jump[0].cpu().numpy()[0]
            crouch_prob = crouch[0].cpu().numpy()[0]
            saber_prob = saber[0].cpu().numpy()[0]

            if frame_count % 60 == 0:
                print(f"\n--- Frame {frame_count} ---")
                print(f"Movement probs: {movement_probs.round(2)}")
                print(f"Argmax mouse X: idx={mouse_x_idx}, val={mouse_delta_x:.1f}")
                print(f"Argmax mouse Y: idx={mouse_y_idx}, val={mouse_delta_y:.1f}")
                print(f"Attack: {attack_prob:.3f}, Jump: {jump_prob:.3f}, Crouch: {crouch_prob:.3f}, Saber: {saber_prob:.3f}")

            execute_actions(
                movement_probs,
                mouse_delta_x,
                mouse_delta_y,
                attack_prob,
                jump_prob,
                crouch_prob,
                saber_prob
            )

            now = time.time()
            if now - last_recenter_time >= RECENTER_INTERVAL:
                # key_output.HoldKey(key_map['e'])
                # time.sleep(0.1)
                # key_output.ReleaseKey(key_map['e'])
                last_recenter_time = now
                # print("  🔄 Recentered view (E)")

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