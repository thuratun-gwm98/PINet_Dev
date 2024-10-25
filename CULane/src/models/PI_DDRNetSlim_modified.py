import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.models.backbones.utils_ddrnetslim_modified import (
    BasicBlock,
    Bottleneck,
    DAPPM,
    BatchNorm2d,
    bn_mom
)
from src.models.backbones.util_hourglass import PIOutput, p

class PI_DDRNetSL(nn.Module):
    def __init__(self, block, layers, planes=64, spp_planes=128, head_planes=128, input_re=True):
        super(PI_DDRNetSL, self).__init__()
        # To do: DDRNet_slim (Backbone + Neck)
        # DDR Net Backbone
        highres_planes = planes * 2
        # self.augment = augment

        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )

        self.down3 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   )

        self.down4 = nn.Sequential(
                                   nn.Conv2d(highres_planes, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 4, momentum=bn_mom),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
                                   BatchNorm2d(planes * 8, momentum=bn_mom),
                                   )

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        # self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)
        self.layer5 =  self._make_layer5(Bottleneck, planes * 8, planes * 8, 1, kernel_size=3, stride=2, padding=1)

        # self.spp = DAPPM(planes * 16, spp_planes, planes * 4)   # 32*4->128, 16*4->64 (16*8)
        self.spp = DAPPM(planes * 16, spp_planes, head_planes)

        self.headIn = nn.Conv2d(head_planes, head_planes, 1, padding=0, stride=1, bias=True, dilation=1)

        self.out_confidence = PIOutput(head_planes, 1)     
        self.out_offset = PIOutput(head_planes, 2)      
        self.out_instance = PIOutput(head_planes, p.feature_size)

        self.input_re = input_re
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)
        self.convout = nn.Conv2d(1, head_planes, 1, padding=0, stride=1, bias=True, dilation=1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)
    
    def _make_layer5(self, block, inplanes, planes, blocks, kernel_size=1, stride=1, padding=0):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=3, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(planes, momentum=bn_mom),
                nn.Conv2d(planes, planes * block.expansion,
                          kernel_size=1, stride=1, padding=(1, 0), bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )       # [4, 512, 16, 32]

        layers = []
        layers.append(block(inplanes, planes, stride, padding=(2, 1), downsample=downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)
    

    def forward(self, inputs):
        
        # TO DO: Here will be DDRNet_slim's backbone + neck
        
        width_output = inputs.shape[-1] // 8
        height_output = inputs.shape[-2] // 8
        # print(f"Scaled Width >>>> {width_output}")
        # print(f"height Output >>>> {height_output}")
        layers = []

        x = self.conv1(inputs)                      # 1, 32, 192, 480

        print(f"After Conv1 >> {x.shape}")

        # x = self.layer1(x)                          # 1, 32, 192, 480
        # layers.append(x)
        # print(f"Layer1 Shape {x.shape}")

        # print(f"After Layer1 >> {x.shape}")

        x = self.layer2(self.relu(x))                   # 1, 64, 96, 240
        layers.append(x)
        print(f"Layer2 Shape {x.shape}")

        # print(f"After layer 2 >> {x.shape}")
  
        x = self.layer3(self.relu(x))
        layers.append(x)
        print(f"Layer 3 shape >> {x.shape}")
        
        x_ = self.layer3_(self.relu(layers[0]))
        print(f"Layer 3 bar shape >>> {x_.shape}")

        # print(f"After layer3, x is >> {x.shape} & x_bar is >> {x_.shape}")

        x = x + self.down3(self.relu(x_))
        # print(f"After Down3 layer3_bar + layer3 out, x is>> {x.shape}")

        print(f"Interpolate 1 >>> {self.compression3(self.relu(layers[1])).shape}")
    
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[1])),
                        scale_factor=(2, 2),
                        mode='bilinear')
        
        # print(f"After inter polation, compression layer3 + previous x_bar of l3, x_bar is {x_.shape} ")
        # if self.augment:
        #     temp = x_

        features = self.layer4(self.relu(x))            # [4, 256, 24, 60]
        layers.append(x)
        print(f"After layer4, Features x is ----->>>> {features.shape} ")

        # return x

        ##### To Neck - Pyramid Sampling with DAPPM ####

        x_ = self.layer4_(self.relu(x_))  # For Feature, We can get from this
        # print(f"After layer4_bar, x_bar is ---->>>> {x_.shape} ")
        print(f"Layer 4_bar shape >>> {x_.shape}")

        x = features + self.down4(self.relu(x_))
        # print(f"After Down 4 >>> {x.shape}")

        
        print(f"Dim to Layer5 ---> {x.shape}")
        # scaled_x = F.interpolate(x, )
        # print(f"Layer 5 info >>>> {self.layer5}")
        print(f"Dim after Layer5 ---> {self.layer5(self.relu(x)).shape}")
        print(f"Dim after SPP ---> {self.spp(self.layer5(self.relu(x))).shape}")
        # print(f"Layer SPP info ----> {self.spp}")
        features_out = F.interpolate(
                        self.spp(self.layer5(self.relu(x))),
                        scale_factor=(2, 2),
                        mode='bilinear')                        # [4, 128, 32, 64]
        
        print(f"Feature Out >>> {features_out.shape}")          # [4, 128, 96, 240]   # [4, 128, 32, 64]

        ### PINet Heads
        # print(f"HeadIn Info : >>> {self.out_confidence}")
        
        # print(f"Outputs Shape ---->>> {outputs.shape}")

        out_confidence = self.out_confidence(features_out)
        out_offset = self.out_offset(features_out)
        out_instance = self.out_instance(features_out)

        print(f"Out Confidence Shape ----->>> {out_confidence.shape}")
        print(f"Out Offset Shapte ----->>> {out_offset.shape}")
        print(f"Out Inshtance ---->>> {out_instance.shape}")

        results = [out_confidence, out_offset, out_instance]

        return [results], [features]
    

def PIDDRNetSlim1(weight, pretrained=False):
    model = PI_DDRNetSL(
        BasicBlock, 
        [2, 2, 2, 2],  
        planes=32, 
        spp_planes=128, 
        head_planes=128, 
        )
    
    if pretrained:
        pretrained_state = torch.load(weight, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_state = {
            k: v
            for k, v in pretrained_state.items()
            if (k in model_dict and v.shape == model_dict[k].shape)
        }
        model_dict.update(pretrained_state)
        print(f"[INFO]: Pretrained weight loaded!")
        model.load_state_dict(model_dict, strict = False)
    return model

def PIDDRNetSlim2(weight, pretrained=False):
    model = PI_DDRNetSL(
        BasicBlock, 
        [2, 2, 2, 2],  
        planes=16, 
        spp_planes=128, 
        head_planes=128, 
        )
    
    if pretrained:
        pretrained_state = torch.load(weight, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_state = {
            k: v
            for k, v in pretrained_state.items()
            if (k in model_dict and v.shape == model_dict[k].shape)
        }
        model_dict.update(pretrained_state)
        print(f"[INFO]: Pretrained weight loaded!")
        model.load_state_dict(model_dict, strict = False)
    return model

def ModelInitializer1():
    model = PI_DDRNetSL(
        BasicBlock, 
        [2, 2, 2, 2],  
        planes=32, 
        spp_planes=128, 
        head_planes=128, 
        )
    return model

def ModelInitializer2():
    model = PI_DDRNetSL(
        BasicBlock, 
        [2, 2, 2, 2],  
        planes=16, 
        spp_planes=128, 
        head_planes=128, 
        )
    return model

if __name__ == "__main__":
    x = torch.ones((4, 3, 768, 1920)).float().cuda()
    weight = "../pretrained_model/DDRNet23s_imagenet.pth"
    model = PI_DDRNetSlim(weight, pretrained=True).cuda()
    out, features = model(x)
    print(f"Number of Parameters ---> {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # print(f"Out Shape >>>> {y.shape}")