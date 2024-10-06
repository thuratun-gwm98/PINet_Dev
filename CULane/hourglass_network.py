#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
from util_hourglass import *

####################################################################
##
## lane_detection_network
##
####################################################################
class lane_detection_network(nn.Module):
    def __init__(self):
        super(lane_detection_network, self).__init__()

        self.resizing = resize_layer(3, 128)

        #feature extraction
        self.layer1 = hourglass_block(128, 128)
        self.layer2 = hourglass_block(128, 128)
        self.layer3 = hourglass_block(128, 128)
        self.layer4 = hourglass_block(128, 128)


    def forward(self, inputs):
        #feature extraction
        out = self.resizing(inputs)
        # print(f"[INFO]: Resize Out >> {out.shape}")
        result1, out, feature1 = self.layer1(out)
        result2, out, feature2 = self.layer2(out)   
        result3, out, feature3 = self.layer3(out)
        result4, out, feature4 = self.layer4(out)

        # return [result1, result2, result3, result4], [feature1, feature2, feature3, feature4]
        return [result4], [feature4]


if __name__ == "__main__":
    model = lane_detection_network()
    x = torch.ones((4, 3, 512, 1024)).float()
    y = model(x)
    print(f"Out >>> {len(y)}")
    print(f"Number of Parameters ---> {sum(p.numel() for p in model.parameters() if p.requires_grad)}")