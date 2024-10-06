import torch
import torch.nn as nn
from hourglass_network import lane_detection_network

# model = lane_detection_network()

model = torch.load('savefile/296_tensor(1.6947)_lane_detection_network.pkl')
print("Model >>> ", model.keys())
torch.save(model, 'savefile/pinet_pretrained.pth')

