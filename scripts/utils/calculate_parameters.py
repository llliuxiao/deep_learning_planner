from transformer_network import RobotTransformer
from thop import profile
import torch

model = RobotTransformer()
laser = torch.randn((1, 6, 1080))
global_path = torch.randn((1, 20, 3))
goal = torch.randn((1, 3))
laser_mask = torch.ones((6, 6), dtype=torch.bool).triu(1)
flops, params = profile(model, inputs=(laser, global_path, goal, laser_mask))
print(f"FLOPs:{flops}")
print(f"Params:{params}")
