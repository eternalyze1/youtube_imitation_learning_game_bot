import torch
import numpy as np
from jka_2026_controller import Policy

policy = Policy().cuda()
policy.load_state_dict(torch.load('policy_youtube_trained.pth'))
policy.eval()

# Random input
dummy = torch.randn(1, 3, 150, 280).cuda()
with torch.no_grad():
    movement, mouse_x, mouse_y, attack, jump, crouch, saber = policy(dummy)
    print("Mouse X probs:", mouse_x[0].cpu().numpy().round(3))
    print("Mouse Y probs:", mouse_y[0].cpu().numpy().round(3))