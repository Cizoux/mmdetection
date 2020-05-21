import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
import time
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from ..builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']


class even_Downsample(nn.Module):
    def __init__(self, channels, filt_size=3, stride=2):
        super(even_Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(4*channels, channels, kernel_size=1),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(channels, 4*channels, kernel_size = 1),
            nn.Sigmoid()
        )

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
            
        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((4*channels,1,1,1)))

    def forward(self, x):
        b,c,h,w = x.size()
        part1 = F.pad(x,(1,0,1,0),mode = "reflect")
        part2 = F.pad(x,(1,0,0,1),mode = "reflect")
        part3 = F.pad(x,(0,1,1,0),mode = "reflect")
        part4 = F.pad(x,(0,1,0,1),mode = "reflect")

        x = torch.cat((part1,part2,part3,part4),1)
        x = F.conv2d(x, self.filt, stride=2, groups=4*c) 
        x = x.reshape(b, 4, c, h//2, w//2)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, 4*c, h//2, w//2)
        
        out = self.avgpool(x)   # b 4c  1   1
        out = self.fc1(out)     # b c   1   1
        out = self.fc2(out)     # b 4c  1   1

        x = x * out
        x = x.reshape(b, c, 4, h//2, w//2)
        x = torch.sum(x, 2)

        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if(stride==1):
            self.conv2 = conv3x3(planes,planes)
        else:
            self.conv2 = nn.Sequential(
                even_Downsample(channels=planes, filt_size=3, stride=stride),
                conv3x3(planes, planes),)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, groups) # stride moved
        self.bn2 = norm_layer(planes)
        if(stride==1):
            self.conv3 = conv1x1(planes, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(even_Downsample(channels=planes, filt_size=3, stride=stride),
                conv1x1(planes, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

@BACKBONES.register_module()
class ResNet_ds_ref_blur(nn.Module):

    def __init__(self ,depth, norm_eval=True, frozen_stages=-1, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, filter_size=3, pool_only=True):
        super(ResNet_ds_ref_blur, self).__init__()
        if depth == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif depth == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif depth == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif depth == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        
        self.zero_init_residual = zero_init_residual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]

        if(pool_only):
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)

        if(pool_only):
            # self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1), 
            #     Downsample(filt_size=filter_size, stride=2, channels=planes[0])])
            self.maxpool = even_Downsample(channels = planes[0], filt_size=3, stride=2)
        else:
            self.maxpool = nn.Sequential(*[even_Downsample(filt_size=filter_size, stride=2, channels=planes[0]), 
                nn.MaxPool2d(kernel_size=2, stride=1), 
                even_Downsample(filt_size=filter_size, stride=2, channels=planes[0])])

        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(planes[3] * block.expansion, num_classes)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv1.requires_grad = False
            self.maxpool.requires_grad =False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                        # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    else:
                        print('Not initializing')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, filter_size=1):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride, filter_size=filter_size),
            #     norm_layer(planes * block.expansion),
            # )

            downsample = [even_Downsample(channels = self.inplanes, filt_size=3, stride=stride),] if(stride !=1) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion)]
            # print(downsample)
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer, filter_size=filter_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer, filter_size=filter_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)

        out = [p1,p2,p3,p4]

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return out

    def train(self, mode=True):
        super(ResNet_ds, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
