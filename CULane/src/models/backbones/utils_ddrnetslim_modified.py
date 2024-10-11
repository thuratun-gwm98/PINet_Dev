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

# class Bottleneck(nn.Module):
#     expansion = 2

#     def __init__(self, inplanes, planes, stride=1, padding=0, downsample=None, no_relu=True):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
#                                padding=padding, bias=False)
#         self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#         self.no_relu = no_relu

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         print(f"Out >>> {out.shape}")
#         print(f"Residual >>> {residual.shape}")

#         out += residual
#         if self.no_relu:
#             return out
#         else:
#             return self.relu(out)

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(# nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    nn.Conv2d(inplanes, inplanes, kernel_size=5, stride=2, padding=2, bias=False),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=5, stride=3, padding=2, bias=False),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )                                                                     # 1, 2
        
        # self.scale2 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=7, stride=4, padding=(2, 0), bias=False),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )                                                                       # 2, 4
        
        # self.scale3 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=7, stride=(7, 9), padding=2, bias=False),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),          # 1, 2
        #                             )

        # self.scale3 = nn.Sequential(# nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
        #                             nn.Conv2d(inplanes, inplanes, kernel_size=11, stride=3, padding=6, bias=False),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),

        #                             nn.Conv2d(branch_planes, branch_planes, kernel_size=4, stride=3, padding=1, bias=False),
        #                             BatchNorm2d(branch_planes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(branch_planes, branch_planes, kernel_size=1, bias=False),
        #                             )
        
        # self.scale4 = nn.Sequential(# nn.AdaptiveAvgPool2d((1, 1)),
        #                             nn.Conv2d(inplanes, inplanes, kernel_size=11, stride=4, padding=4, bias=False),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),

        #                             nn.Conv2d(branch_planes, branch_planes, kernel_size=4, stride=4, padding=1, bias=False),
        #                             BatchNorm2d(branch_planes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(branch_planes, branch_planes, kernel_size=1, bias=False),
        #                             )

        # self.scale3 = nn.Sequential(# nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
        #                             nn.Conv2d(inplanes, inplanes, kernel_size=17, stride=8, padding=8, bias=False),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )
        # self.scale4 = nn.Sequential(# nn.AdaptiveAvgPool2d((1, 1)),
        #                             nn.Conv2d(inplanes, inplanes, kernel_size=33, stride=16, padding=16, bias=False),
        #                             BatchNorm2d(inplanes, momentum=bn_mom),
        #                             nn.ReLU(inplace=True),
        #                             nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        #                             )
        
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
                                    BatchNorm2d(branch_planes * 3, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 3, outplanes * 2, kernel_size=1, bias=False),        # 640 ---> 512
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes * 2, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        print(f"In DAPPM >>>>>>>>>")
        print(f"Input Shape >>>> {x.shape}")               # 12, 30/32      # 8, 20
        print(f"Scale0 >>> {self.scale0(x).shape}")
        print(f"Scale1 ---> {self.scale1(x).shape}")        # 6, 15/16      # 4, 10
        print(f"Scale2 ---> {self.scale2(x).shape}")        # 3, 8             # 2, 5
        # print(f"Scale3 ---> {self.scale3(x).shape}")        # 2, 4      # 1,1   # 1, 2
        # print(f"Scale4 ---> {self.scale4(x).shape}")        # 1, 2      # 

        #x = self.downsample(x)
        width = x.shape[-1]     # 30 32
        height = x.shape[-2]        # 12
        x_list = []

        x_list.append(self.scale0(x))
        
        print(f"X List 0 >>> {x_list[0].shape}")
        x_list.append(self.process1((F.interpolate(self.scale1(x),   # 6, 15 16  # 4, 10
                        scale_factor=(2, 2),
                        mode='bilinear')+x_list[0])))
        
        print(f"X List 1 >>> {x_list[1].shape}")
        x_list.append(self.process2((F.interpolate(self.scale2(x),    # 3, 8   # 2, 4  / 1, 2
                        scale_factor=(3, 3),
                        mode='bilinear')+x_list[1])))

        # x_list.append(self.process3((F.interpolate(self.scale3(x), # 2, 4   # 1, 2
        #                 scale_factor=(8, 10),
        #                 mode='bilinear')+x_list[2])))
        
        # print(f"Process 3 Size >>> {x_list[3].shape}")
        
        # x_list.append(self.process4((F.interpolate(self.scale4(x),
        #                 size=[height, width],
        #                 mode='bilinear')+x_list[3])))
        # x_list.append(self.process4((F.interpolate(self.scale4(x),   # 1, 2
        #                 scale_factor=(12, 15),
        #                 mode='bilinear')+x_list[3])))
        print(f"X List >>> {len(x_list)}")

        print(f"Shortcut X ----> {self.shortcut(x).shape}")
        print(f"Compression -----> {torch.cat(x_list, 1).shape}")
        # print(f)
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out

