import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.feature_extractor import feature_extractor
import numpy as np
from models.miniVIT import VIT
from torch.autograd import Variable

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,1,maxdisp,1,1]))/maxdisp
        self.disp = self.disp.expand(1,2,maxdisp,1,1).cuda()

    def forward(self, x):
        out = torch.sum(x * self.disp.data, 2, keepdim=True)
        return out

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x


    def forward(self, x ,presqu, postsqu):

        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out)+pre, inplace=True)

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class depthnet(nn.Module):
    def __init__(self, maxdisp = 192):
        super(depthnet, self).__init__()
        self.maxdisp = int(maxdisp)
        self.feature_extractions = nn.ModuleList([feature_extractor(in_channels=1), feature_extractor(in_channels=3)])
        self.att_blocks = nn.ModuleList([VIT(n_embd=288, n_layer=4, n_head=4, down_ratio=3),
                                         VIT(n_embd=288, n_layer=4, n_head=4, down_ratio=3)])
        self.lastconvs = nn.ModuleList([nn.Sequential(convbn(480, 64, 3, 1, 1, 1),
                                                      nn.ReLU(inplace=True),
                                                      nn.Conv2d(64, 32, kernel_size=1, padding=0, stride = 1, bias=False)),
                                        nn.Sequential(convbn(480, 64, 3, 1, 1, 1),
                                                      nn.ReLU(inplace=True),
                                                      nn.Conv2d(64, 32, kernel_size=1, padding=0, stride = 1, bias=False))])

        self.dres0 = nn.Sequential(convbn_3d(128, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1))
        self.dres2 = hourglass(64)
        self.classif1 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(64, 2, kernel_size=3, padding=1, stride=1,bias=False))

        self.smooth_up = nn.Sequential(convbn(2, 8, 3, 1, 1, 1),
                                       nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def image2feature(self, left, right, idx):
        b, seq_len, c, w, h = left.shape
        left = left.view(b*seq_len, c, w, h).contiguous()
        right = right.view(b*seq_len, c, w, h).contiguous()
        refimg_fea     = self.lastconvs[idx](self.feature_extractions[idx](left))
        targetimg_fea  = self.lastconvs[idx](self.feature_extractions[idx](right))
        s = refimg_fea.shape
        refimg_fea = refimg_fea.view(b, seq_len, s[-3], s[-2], s[-1])
        targetimg_fea = targetimg_fea.view(b, seq_len, s[-3], s[-2], s[-1])
        refimg_fea    = refimg_fea + self.att_blocks[idx](refimg_fea)
        targetimg_fea = targetimg_fea + self.att_blocks[idx](targetimg_fea)
        refimg_fea = refimg_fea.view(b*seq_len, s[-3], s[-2], s[-1])
        targetimg_fea = targetimg_fea.view(b*seq_len, s[-3], s[-2], s[-1])
        return refimg_fea, targetimg_fea

    def forward(self, input):
        # high freq
        left, right = input['left_high'], input['right_high']
        refimg_fea_high, targetimg_fea_high = self.image2feature(left, right, idx=0)
        # all freq
        left, right = input['left_all'], input['right_all']
        refimg_fea_all, targetimg_fea_all = self.image2feature(left, right, idx=1)
        #
        refimg_fea = torch.cat((refimg_fea_high, refimg_fea_all), dim=1)
        targetimg_fea = torch.cat((targetimg_fea_high, targetimg_fea_all), dim=1)
        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, int(self.maxdisp)//4,  refimg_fea.size()[2], refimg_fea.size()[3]).zero_()).to(left.device)
        for i in range(int(self.maxdisp)//4):
            if i > 0 :
                cost[:, :refimg_fea.size()[1], i, :,i:]  = refimg_fea[:,:,:,i:]
                cost[:, refimg_fea.size()[1]:, i, :,i:]  = targetimg_fea[:,:,:,:-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1+cost0
        cost1 = self.classif1(out1)

        # cost1 = F.upsample(cost1, scale_factor=2, mode='trilinear')#!
        b, c, d, w, h = cost1.shape
        cost1 = cost1.permute(0,2,1,3,4).contiguous().view(b*d, c, w, h)
        cost1 = F.upsample(cost1, scale_factor=2, mode='bilinear')
        cost1 = self.smooth_up(cost1)
        cost1 = cost1.view(b, d, cost1.shape[-3], cost1.shape[-2], cost1.shape[-1]).permute(0, 2, 1, 3, 4).contiguous()
        cost1 = F.upsample(cost1, [self.maxdisp, left.size()[-2],left.size()[-1]], mode='trilinear')

        #cost1 = F.upsample(cost1, [self.maxdisp, left.size()[-2],left.size()[-1]], mode='trilinear')#!

        pred1 = F.softmax(cost1,dim=2)
        pred_disp = disparityregression(self.maxdisp)(pred1).squeeze()
        return pred_disp

def test():
    model = depthnet().cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params:', n_parameters)
    x = torch.randn([1, 2, 3, 192, 384]).cuda()
    y = torch.randn([1, 2, 1, 192, 384]).cuda()
    input = {'left_high':y, 'right_high':y, 'left_all':x, 'right_all':x}
    y = model(input)
    return

# test()




