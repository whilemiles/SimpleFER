"""trans.py"""
import torch
from ResNet18 import ResNet18

checkpoint = torch.load("saved/ResNet18-lr_best", map_location="cpu")
model = ResNet18()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
sample = torch.randn(1,1,40,40)
trace_model = torch.jit.trace(model, sample)
trace_model.save("saved/ResNet18-lr_best.jit")
