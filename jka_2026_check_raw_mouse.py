# jka_2026_check_raw_mouse.py
import pickle

with open('recorded_data/session_20260403_160752/actions.pkl', 'rb') as f:
    actions = pickle.load(f)

print("Keys in first action:", actions[0].keys())
print("\nFirst 20 mouse indices (x_idx, y_idx):")
for i in range(20):
    print(f"{i:3d}: x_idx={actions[i]['mouse_x_idx']:2d}, y_idx={actions[i]['mouse_y_idx']:2d}")