import math
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    # nn.Conv2d(inplanes, inplanes, kernel_size=5, stride=2, padding=2, bias=False),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    # nn.Conv2d(inplanes, inplanes, kernel_size=9, stride=4, padding=4, bias=False),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    # nn.Conv2d(inplanes, inplanes, kernel_size=17, stride=8, padding=8, bias=False),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    # nn.Conv2d(inplanes, inplanes, kernel_size=33, stride=16, padding=16, bias=False),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        print(f"In DAPPM >>>>>>>>>")
        print(f"Input Shape >>>> {x.shape}")
        print(f"Scale0 ---> {self.scale0(x).shape}")
        print(f"Scale1 ---> {self.scale1(x).shape}")
        print(f"Scale2 ---> {self.scale2(x).shape}")
        print(f"Scale3 ---> {self.scale3(x).shape}")
        # print(f"Scale4 ---> {self.scale4(x).shape}")

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        # x_list.append(self.process1((F.interpolate(self.scale1(x),
        #                 scale_factor=(2, 2),
        #                 mode='bilinear')+x_list[0])))
        
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        # x_list.append((self.process2((F.interpolate(self.scale2(x),
        #                 scale_factor=(4, 3.75),
        #                 mode='bilinear')+x_list[0]))))
        print(f"Process 2 Size >>> {x_list[2].shape}")
        
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        # x_list.append(self.process3((F.interpolate(self.scale3(x),
        #                 scale_factor=(6, 7.5),
        #                 mode='bilinear')+x_list[2])))
        print(f"Process 3 Size >>> {x_list[3].shape}")
        

        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[3])))
        # x_list.append(self.process4((F.interpolate(self.scale4(x),
        #                 scale_factor=(12, 15),
        #                 mode='bilinear')+x_list[3])))

        print(f"Shortcut X ----> {self.shortcut(x).shape}")
        print(f"Compression -----> {self.compression(torch.cat(x_list, 1)).shape}")
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear')

        return out

class DualResNet_slim(nn.Module):

    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128, augment=False):
        super(DualResNet_slim, self).__init__()

        highres_planes = planes * 2
        self.augment = augment

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

        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        if self.augment:
            self.seghead_extra = segmenthead(highres_planes, head_planes, num_classes)            

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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


    def forward(self, x):

        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        print(f"Input >> {x.shape}")

        x = self.conv1(x)

        print(f"After Conv1 >> {x.shape}")

        # x = self.layer1(x)
        # layers.append(x)

        print(f"After Layer1 >> {x.shape}")

        x = self.layer2(self.relu(x))
        layers.append(x)

        print(f"After layer 2 >> {x.shape}")
  
        x = self.layer3(self.relu(x))
        layers.append(x)
        
        x_ = self.layer3_(self.relu(layers[1]))

        print(f"After layer3, x is >> {x.shape} & x_bar is >> {x_.shape}")

        x = x + self.down3(self.relu(x_))
        print(f"After Down3 layer3_bar + layer3 out, x is>> {x.shape}")
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[1])),
                        size=[height_output, width_output],
                        mode='bilinear')
        print(f"After inter polation, compression layer3 + previous x_bar of l3, x_bar is {x_.shape} ")
        if self.augment:
            temp = x_

        features = self.layer4(self.relu(x))
        layers.append(x)
        print(f"After layer4, x is {features.shape} ")

        x_ = self.layer4_(self.relu(x_))  # For Feature, We can get from this
        print(f"After layer4_bar, x_bar is {x_.shape} ")

        # return x

        ##### To Neck - Pyramid Sampling with DAPPM ####

        x = features + self.down4(self.relu(x_))
        print(f"After Down 4 >>> {x.shape}")

        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear')                    # For Feature, We can get also from this

        # print(f"After compression4, x_bar is {x_.shape}")

        x_ = self.layer5_(self.relu(x_))
        print(f"After Layer 5bar >> {x_.shape}")
        print(f"After Layer 5 shape >>> {self.layer5(self.relu(x)).shape}")

        features_out = F.interpolate(
                        self.spp(self.layer5(self.relu(x))),
                        size=[height_output, width_output],
                        mode='bilinear')

        print(f"Layer 5+ DAPMM interpolation >>> {features_out.shape}")

        # x_ = self.final_layer(x + x_)
        # print(f"Final Layer Out >> {x_.shape}")

        return features_out

        # if self.augment: 
        #     x_extra = self.seghead_extra(temp)
        #     return [x_, x_extra]
        # else:
        #     return x_      

def DualResNet_imagenet_slim(weight, pretrained=False):
    model = DualResNet_slim(
        BasicBlock, [2, 2, 2, 2], 
        num_classes=19, 
        planes=32, 
        spp_planes=128, 
        head_planes=64, 
        augment=False)
    
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

def get_seg_model(cfg, **kwargs):
    model = DualResNet_imagenet_slim(pretrained=False)
    return model


if __name__ == "__main__":
    weight = "pretrained_model/DDRNet23s_imagenet.pth"
    x = torch.ones((4, 3, 256, 512)).float().cuda()
    net = DualResNet_imagenet_slim(weight, pretrained=True).cuda()
    print(f"Net >>> {net}")
    y = net(x)
    print(f"Output Shape::: {y.shape}")