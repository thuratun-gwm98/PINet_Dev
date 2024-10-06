import torch
import torch.onnx
import onnx
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hourglass_network import lane_detection_network

save_dir = "pretrained_model/relu_replaced_model.onnx"

def export_onnx():

    model = lane_detection_network()

    weight_path = 'pretrained_model/relu_replaced_model.pth'  #'savefile/pinet_pretrained.pth'

    state_dict = torch.load(weight_path)

    model.load_state_dict(state_dict)

    sample_batch_size = 1
    channel = 3
    height = 256
    width = 512

    dummy_input = torch.rand(sample_batch_size, channel, height, width)

    torch.onnx.export(model, 
                      dummy_input, 
                      save_dir, 
                      opset_version=13,
                      do_constant_folding=True)
    print(f"[INFO]: Successfully Exported!")
    
if __name__ == "__main__":
    export_onnx()