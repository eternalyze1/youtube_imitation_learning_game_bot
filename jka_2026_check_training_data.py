# check_training_data.py
import pickle
import numpy as np

# Load your recorded actions
with open('recorded_data/session_20260403_054610/actions.pkl', 'rb') as f:
    actions = pickle.load(f)

print(f"Total actions: {len(actions)}")

# Count action types
attacks = sum(a['attack'] for a in actions)
jumps = sum(a['jump'] for a in actions)
crouches = sum(a['crouch'] for a in actions)
sabers = sum(a['saber'] for a in actions)
movements = sum(any(a['movement']) for a in actions)

print(f"Attacks: {attacks}/{len(actions)} ({attacks/len(actions)*100:.1f}%)")
print(f"Jumps: {jumps}/{len(actions)} ({jumps/len(actions)*100:.1f}%)")
print(f"Crouches: {crouches}/{len(actions)} ({crouches/len(actions)*100:.1f}%)")
print(f"Sabers: {sabers}/{len(actions)} ({sabers/len(actions)*100:.1f}%)")
print(f"Movements: {movements}/{len(actions)} ({movements/len(actions)*100:.1f}%)")

# Check first few actions
print("\nFirst 5 actions:")
for i in range(5):
    print(f"  {i}: movement={actions[i]['movement']}, attack={actions[i]['attack']}")