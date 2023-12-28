"""trans.py"""
import torch
import Expression_CNN

model = Expression_CNN.ExpressionCNN()
model = torch.load("saved/ExpressionCNN.pt", map_location="cpu")
model.eval()
sample = torch.randn(1,1,48,48)
trace_model = torch.jit.trace(model, sample)
trace_model.save("saved/ExpressionCNN.jit")
