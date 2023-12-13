"""trans.py"""
import torch
import Emo_CNN

model = Emo_CNN.EmoCNN()
model = torch.load("saved/EmoCNN.pt", map_location="cpu")
model.eval()
sample = torch.randn(1,1,48,48)
trace_model = torch.jit.trace(model, sample)
trace_model.save("saved/EmoCNN.jit")
