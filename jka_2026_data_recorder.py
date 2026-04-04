#!/usr/bin/env python

import ctypes as cts
import ctypes.wintypes as wts
import sys
import time
import os
import pickle

import ctypes_wrappers as cws

import pynput
import win32api
import win32con
import win32gui
import mss
import cv2
import numpy as np
import key_input
import key_output
from config import csgo_img_dimension, mouse_x_possibles, mouse_y_possibles


HWND_MESSAGE = -3
WM_QUIT = 0x0012
WM_INPUT = 0x00FF
WM_KEYUP = 0x0101
WM_CHAR = 0x0102

HID_USAGE_PAGE_GENERIC = 0x01
RIDEV_NOLEGACY = 0x00000030
RIDEV_INPUTSINK = 0x00000100
RIDEV_CAPTUREMOUSE = 0x00000200
RID_HEADER = 0x10000005
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIM_TYPEKEYBOARD = 1
RIM_TYPEHID = 2
PM_NOREMOVE = 0x0000

raw_events = []
recording = False
playback = False
playback_events = []
playback_idx = 0
playback_last_time = 0
session_dir = None

frame_dx = 0
frame_dy = 0
bot_frames = []
bot_actions = []

# FPS tracking
last_fps_time = 0
event_count = 0


def wnd_proc(hwnd, msg, wparam, lparam):
    global raw_events, recording, event_count, last_fps_time, frame_dx, frame_dy
    if msg == WM_INPUT:
        size = wts.UINT(0)
        res = cws.GetRawInputData(cts.cast(lparam, cws.PRAWINPUT), RID_INPUT, None, cts.byref(size), cts.sizeof(cws.RAWINPUTHEADER))
        if res == wts.UINT(-1) or size == 0:
            return 0
        buf = cts.create_string_buffer(size.value)
        res = cws.GetRawInputData(cts.cast(lparam, cws.PRAWINPUT), RID_INPUT, buf, cts.byref(size), cts.sizeof(cws.RAWINPUTHEADER))
        if res != size.value:
            return 0
        ri = cts.cast(buf, cws.PRAWINPUT).contents
        head = ri.header
        if head.dwType == RIM_TYPEMOUSE:
            data = ri.data.mouse
            if recording:
                raw_events.append((data.lLastX, data.lLastY))
                frame_dx += data.lLastX
                frame_dy += data.lLastY
    return cws.DefWindowProc(hwnd, msg, wparam, lparam)


def print_error(code=None, text=None):
    text = text + " - e" if text else "E"
    code = cws.GetLastError() if code is None else code
    print(f"{text}rror code: {code}")


def register_devices(hwnd=None):
    flags = RIDEV_INPUTSINK
    generic_usage_ids = (0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x08)
    devices = (cws.RawInputDevice * len(generic_usage_ids))(
        *(cws.RawInputDevice(HID_USAGE_PAGE_GENERIC, uid, flags, hwnd) for uid in generic_usage_ids)
    )
    if cws.RegisterRawInputDevices(devices, len(generic_usage_ids), cts.sizeof(cws.RawInputDevice)):
        print("Successfully registered input device(s)!")
        return True
    else:
        print_error(text="RegisterRawInputDevices")
        return False


def start_recording():
    global recording, session_dir, raw_events, event_count, last_fps_time, bot_frames, bot_actions, frame_dx, frame_dy
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"recorded_data/session_{timestamp}"
    os.makedirs(session_dir, exist_ok=True)
    raw_events = []
    bot_frames = []
    bot_actions = []
    frame_dx = 0
    frame_dy = 0
    event_count = 0
    last_fps_time = time.time()
    recording = True
    print(f"\n>>> RECORDING to {session_dir} <<<\n")


def stop_recording():
    global recording
    recording = False
    print("\n>>> SAVING DATA (this may take a moment depending on recording length) <<<")
    if raw_events:
        events_path = os.path.join(session_dir, 'raw_events.pkl')
        with open(events_path, 'wb') as f:
            pickle.dump(raw_events, f)
        print(f"Saved {len(raw_events)} raw events to {session_dir}")
        
    # Save behavioral cloning data
    if bot_frames:
        frames_path = os.path.join(session_dir, 'frames.npz')
        np.savez_compressed(frames_path, frames=np.array(bot_frames, dtype=np.uint8))
        actions_path = os.path.join(session_dir, 'actions.pkl')
        with open(actions_path, 'wb') as f:
            pickle.dump(bot_actions, f)
        print(f"Saved {len(bot_frames)} HD frames & actions")
            
    print(f"\n>>> STOPPED <<<\n")


def start_playback():
    global playback, playback_events, playback_idx, playback_last_time, event_count, last_fps_time, current_held_keys
    sessions = [d for d in os.listdir("recorded_data") if d.startswith("session_") and os.path.exists(os.path.join("recorded_data", d, "actions.pkl"))]
    if not sessions:
        print("No recordings found")
        return
    latest = sorted(sessions)[-1]
    events_path = f"recorded_data/{latest}/actions.pkl"
    try:
        with open(events_path, 'rb') as f:
            playback_events = pickle.load(f)
        playback = True
        playback_idx = 0
        playback_last_time = 0
        event_count = 0
        current_held_keys = set()
        last_fps_time = time.time()
        print(f"\n>>> PLAYBACK actions from {latest} ({len(playback_events)} frames) <<<\n")
    except Exception as e:
        print(f"Error: {e}")


def stop_playback():
    global playback, current_held_keys
    playback = False
    
    key_map_dict = {
        'w': key_output.w_char, 'a': key_output.a_char, 's': key_output.s_char, 'd': key_output.d_char,
        'space': key_output.space_char, 'ctrl': key_output.ctrl_char, 'shift': key_output.shift_char
    }
    for k in current_held_keys:
        if k == 'mouse_l': key_output.release_left_click()
        elif k == 'mouse_r': key_output.release_right_click()
        elif k in key_map_dict: key_output.ReleaseKey(key_map_dict[k])
    
    current_held_keys = set()
    print(f"\n>>> PLAYBACK STOPPED <<<\n")


def set_pos(dx, dy):
    extra = cts.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(dx, dy, 0, (0x0001), 0, cts.cast(cts.pointer(extra), cts.c_void_p))
    command = pynput._util.win32.INPUT(cts.c_ulong(0), ii_)
    cts.windll.user32.SendInput(1, cts.pointer(command), cts.sizeof(command))


def playback_event():
    global playback_idx, playback_last_time, event_count, last_fps_time, current_held_keys
    if playback_idx >= len(playback_events):
        stop_playback()
        return
    
    now = time.time()
    
    action = playback_events[playback_idx]
    playback_idx += 1
    playback_last_time = now
    
    # 1) Mouse
    dx = action['raw_dx']
    dy = action['raw_dy']
    set_pos(dx, dy)
    
    # 2) Keys
    target_keys = set()
    if action['movement'][0]: target_keys.add('w')
    if action['movement'][1]: target_keys.add('a')
    if action['movement'][2]: target_keys.add('s')
    if action['movement'][3]: target_keys.add('d')
    if action['crouch']: target_keys.add('ctrl')
    if action['attack']: target_keys.add('mouse_l')
    if action['jump']: target_keys.add('mouse_r')
    if action['saber']: target_keys.add('space')
    
    key_map_dict = {
        'w': key_output.w_char, 'a': key_output.a_char, 's': key_output.s_char, 'd': key_output.d_char,
        'space': key_output.space_char, 'ctrl': key_output.ctrl_char, 'shift': key_output.shift_char
    }
    
    for k in current_held_keys - target_keys:
        if k == 'mouse_l': key_output.release_left_click()
        elif k == 'mouse_r': key_output.release_right_click()
        elif k in key_map_dict: key_output.ReleaseKey(key_map_dict[k])
            
    for k in target_keys - current_held_keys:
        if k == 'mouse_l': key_output.hold_left_click()
        elif k == 'mouse_r': key_output.hold_right_click()
        elif k in key_map_dict: key_output.HoldKey(key_map_dict[k])
            
    current_held_keys = target_keys
    
    event_count += 1
    if now - last_fps_time >= 1.0:
        print(f"PB  FPS: {event_count}")
        event_count = 0
        last_fps_time = now
    
    if dx != 0 or dy != 0:
        print(f"PB {playback_idx:4d}: ({dx:4d}, {dy:4d})")


def main(*argv):
    global recording, playback
    
    wnd_cls = "SO049572093_RawInputWndClass"
    wcx = cws.WNDCLASSEX()
    wcx.cbSize = cts.sizeof(cws.WNDCLASSEX)
    wcx.lpfnWndProc = cws.WNDPROC(wnd_proc)
    wcx.hInstance = cws.GetModuleHandle(None)
    wcx.lpszClassName = wnd_cls
    res = cws.RegisterClassEx(cts.byref(wcx))
    if not res:
        print_error(text="RegisterClass")
        return 0
    hwnd = cws.CreateWindowEx(0, wnd_cls, None, 0, 0, 0, 0, 0, 0, None, wcx.hInstance, None)
    if not hwnd:
        print_error(text="CreateWindowEx")
        return 0
    if not register_devices(hwnd):
        return 0
    
    print("\n" + "="*50)
    print("RAW MOUSE RECORDER (with FPS reporting)")
    print("="*50)
    print("Controls:")
    print("  R - Start recording (press again to stop)")
    print("  P - Playback last recording (press again to stop)")
    print("  Q - Quit")
    print("="*50 + "\n")
    
    msg = wts.MSG()
    pmsg = cts.byref(msg)
    
    last_r = 0
    last_p = 0
    last_q = 0
    debounce = 0.3
    
    sct = mss.mss()
    game_hwnd = win32gui.FindWindow(None, 'EternalJK')
    
    target_fps = 60
    frame_time = 1.0 / target_fps
    next_frame = time.time() + frame_time
    
    global frame_dx, frame_dy

    
    while True:
        # Process ALL pending messages to never lose mouse events
        while cws.PeekMessage(pmsg, 0, 0, 0, 1):
            cws.TranslateMessage(pmsg)
            cws.DispatchMessage(pmsg)
        
        now = time.time()
        if now < next_frame:
            # Busy wait for perfect timing
            continue
        
        next_frame = now + frame_time
        
        if win32api.GetAsyncKeyState(ord('R')) & 0x8000:
            if now - last_r > debounce:
                last_r = now
                if not playback:
                    if not recording:
                        start_recording()
                    else:
                        stop_recording()
        
        if win32api.GetAsyncKeyState(ord('P')) & 0x8000:
            if now - last_p > debounce:
                last_p = now
                if not recording:
                    if not playback:
                        start_playback()
                    else:
                        stop_playback()
        
        if win32api.GetAsyncKeyState(ord('Q')) & 0x8000:
            if now - last_q > debounce:
                last_q = now
                break
        
        if playback:
            playback_event()
        
        # Process bot observation capture
        if recording and game_hwnd:
            left, top, right, bottom = win32gui.GetWindowRect(game_hwnd)
            monitor = {"left": left, "top": top, "width": right - left, "height": bottom - top}
            img = np.array(sct.grab(monitor))[:, :, :3]
            h, w = csgo_img_dimension
            img_small = cv2.resize(img, (w, h))

            mouse_x_idx = np.argmin([abs(x_ - frame_dx) for x_ in mouse_x_possibles])
            mouse_y_idx = np.argmin([abs(y_ - frame_dy) for y_ in mouse_y_possibles])
            
            raw_keys = key_input.key_check()
            keys_lower = [k.lower() for k in raw_keys]
            
            movement = [
                1 if 'w' in keys_lower else 0,
                1 if 'a' in keys_lower else 0,
                1 if 's' in keys_lower else 0,
                1 if 'd' in keys_lower else 0
            ]
            
            crouch = 1 if (win32api.GetAsyncKeyState(win32con.VK_CONTROL) & 0x8000) else 0
            saber = 1 if (win32api.GetAsyncKeyState(win32con.VK_SPACE) & 0x8000) else 0
            attack = 1 if (win32api.GetAsyncKeyState(0x01) & 0x8000) else 0
            jump = 1 if (win32api.GetAsyncKeyState(0x02) & 0x8000) else 0

            bot_frames.append(img_small)
            bot_actions.append({
                "mouse_x_idx": int(mouse_x_idx),
                "mouse_y_idx": int(mouse_y_idx),
                "movement": movement,
                "attack": attack,
                "jump": jump,
                "crouch": crouch,
                "saber": saber,
                "raw_dx": frame_dx,
                "raw_dy": frame_dy
            })
            
            # Reset delta for next frame
            frame_dx = 0
            frame_dy = 0
            
            event_count += 1
            if now - last_fps_time >= 1.0:
                print(f"REC FPS: {event_count}")
                event_count = 0
                last_fps_time = now
            
        # Loop runs exactly at target_fps
    
    if recording:
        stop_recording()
    
    print("\nDone.")
    return 0


if __name__ == "__main__":
    from datetime import datetime
    print("Python {:s} {:03d}bit on {:s}\n".format(" ".join(elem.strip() for elem in sys.version.split("\n")),
                                                    64 if sys.maxsize > 0x100000000 else 32, sys.platform))
    main()