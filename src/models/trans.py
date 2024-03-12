"""trans.py"""
import torch
from ResNet18 import ResNet18

checkpoint = torch.load("saved/acc73", map_location="cpu")
model = ResNet18()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
sample = torch.randn(1,1,40,40)
trace_model = torch.jit.trace(model, sample)
trace_model.save("saved/acc73.jit")
