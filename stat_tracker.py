import torch
from torch import nn
from copy import deepcopy


class StatTracker(nn.Module):
    def __init__(self, input_size, input_layer=False):
        super(StatTracker, self).__init__()
        if not input_layer:
            self.bn = nn.BatchNorm2d(input_size[1])
        else:
            self.bn = [nn.BatchNorm2d(input_size[1]).cuda() for _ in range(4)]
        self.input_size = input_size
        self.input_layer = input_layer

    def forward(self, x):
        if not self.input_layer:
            _ = self.bn(x)
        else:
            l = self.input_size[-1]
            l1 = list(range(0, l, 2))
            l2 = list(range(1, l, 2))
            _ = self.bn[0](x[:, :, l1][:, :, :, l1])
            _ = self.bn[1](x[:, :, l1][:, :, :, l2])
            _ = self.bn[2](x[:, :, l2][:, :, :, l1])
            _ = self.bn[3](x[:, :, l2][:, :, :, l2])
        return x

    def getmean(self):
        if not self.input_layer:
            return self.bn.running_mean
        else:
            return torch.stack([_.running_mean for _ in self.bn], dim=0)

    def setmean(self, mean):
        if not self.input_layer:
            self.bn.running_mean = deepcopy(mean.data)
        else:
            for i in range(4):
                self.bn[i].running_mean = deepcopy(mean[i].data)


def get_layer(model):
    layers = []
    for _ in model.children():
        if len(list(_.children())) == 0 or type(_) == StatTracker:
            layers.append(_)
        else:
            layers += get_layer(_)
    return layers


def add_tracker(resnet):
    resnet.conv1 = nn.Sequential(StatTracker([32, 3, 224, 224], True), resnet.conv1)
    resnet.layer1[0].conv1 = nn.Sequential(StatTracker([32, 64, 56, 56]), resnet.layer1[0].conv1)
    resnet.layer1[0].conv2 = nn.Sequential(StatTracker([32, 64, 56, 56]), resnet.layer1[0].conv2)
    resnet.layer1[1].conv1 = nn.Sequential(StatTracker([32, 64, 56, 56]), resnet.layer1[1].conv1)
    resnet.layer1[1].conv2 = nn.Sequential(StatTracker([32, 64, 56, 56]), resnet.layer1[1].conv2)
    resnet.layer2[0].conv1 = nn.Sequential(StatTracker([32, 64, 56, 56], True), resnet.layer2[0].conv1)
    resnet.layer2[0].conv2 = nn.Sequential(StatTracker([32, 128, 28, 28]), resnet.layer2[0].conv2)
    resnet.layer2[0].downsample[0] = nn.Sequential(StatTracker([32, 64, 56, 56], True), resnet.layer2[0].downsample[0])
    resnet.layer2[1].conv1 = nn.Sequential(StatTracker([32, 128, 28, 28]), resnet.layer2[1].conv1)
    resnet.layer2[1].conv2 = nn.Sequential(StatTracker([32, 128, 28, 28]), resnet.layer2[1].conv2)
    resnet.layer3[0].conv1 = nn.Sequential(StatTracker([32, 128, 28, 28], True), resnet.layer3[0].conv1)
    resnet.layer3[0].conv2 = nn.Sequential(StatTracker([32, 256, 14, 14]), resnet.layer3[0].conv2)
    resnet.layer3[0].downsample[0] = nn.Sequential(StatTracker([32, 128, 28, 28], True), resnet.layer3[0].downsample[0])
    resnet.layer3[1].conv1 = nn.Sequential(StatTracker([32, 256, 14, 14]), resnet.layer3[1].conv1)
    resnet.layer3[1].conv2 = nn.Sequential(StatTracker([32, 256, 14, 14]), resnet.layer3[1].conv2)
    resnet.layer4[0].conv1 = nn.Sequential(StatTracker([32, 256, 14, 14], True), resnet.layer4[0].conv1)
    resnet.layer4[0].conv2 = nn.Sequential(StatTracker([32, 512, 7, 7]), resnet.layer4[0].conv2)
    resnet.layer4[0].downsample[0] = nn.Sequential(StatTracker([32, 256, 14, 14], True), resnet.layer4[0].downsample[0])
    resnet.layer4[1].conv1 = nn.Sequential(StatTracker([32, 512, 7, 7]), resnet.layer4[1].conv1)
    resnet.layer4[1].conv2 = nn.Sequential(StatTracker([32, 512, 7, 7]), resnet.layer4[1].conv2)


def correct_shift(resnet):
    convs = [resnet.conv1,
             resnet.layer1[0].conv1,
             resnet.layer1[0].conv2,
             resnet.layer1[1].conv1,
             resnet.layer1[1].conv2,
             resnet.layer2[0].conv1,
             resnet.layer2[0].conv2,
             resnet.layer2[0].downsample[0],
             resnet.layer2[1].conv1,
             resnet.layer2[1].conv2,
             resnet.layer3[0].conv1,
             resnet.layer3[0].conv2,
             resnet.layer3[0].downsample[0],
             resnet.layer3[1].conv1,
             resnet.layer3[1].conv2,
             resnet.layer4[0].conv1,
             resnet.layer4[0].conv2,
             resnet.layer4[0].downsample[0],
             resnet.layer4[1].conv1,
             resnet.layer4[1].conv2,
             ]
    bns = [resnet.bn1,
           resnet.layer1[0].bn1,
           resnet.layer1[0].bn2,
           resnet.layer1[1].bn1,
           resnet.layer1[1].bn2,
           resnet.layer2[0].bn1,
           resnet.layer2[0].bn2,
           resnet.layer2[0].downsample[1],
           resnet.layer2[1].bn1,
           resnet.layer2[1].bn2,
           resnet.layer3[0].bn1,
           resnet.layer3[0].bn2,
           resnet.layer3[0].downsample[1],
           resnet.layer3[1].bn1,
           resnet.layer3[1].bn2,
           resnet.layer4[0].bn1,
           resnet.layer4[0].bn2,
           resnet.layer4[0].downsample[1],
           resnet.layer4[1].bn1,
           resnet.layer4[1].bn2,
           ]
    with torch.no_grad():
        for conv, post_bn in zip(convs, bns):
            pre_bn = conv[0]
            conv = conv[1]
            if not pre_bn.input_layer:
                mean = torch.ones(pre_bn.input_size).cuda()
                mean *= pre_bn.getmean().view(1, -1, 1, 1)
                out = conv(mean).mean([0, 2, 3])
                post_bn.running_mean = out
            else:
                c = torch.ones(pre_bn.input_size).cuda()
                b = torch.zeros(pre_bn.input_size).cuda()
                l1 = list(range(0, pre_bn.input_size[-1], 2))
                l2 = list(range(1, pre_bn.input_size[-1], 2))
                c[:, :, l2] *= 0
                c[:, :, :, l2] *= 0
                b += c * pre_bn.getmean()[0].view(1, -1, 1, 1)
                c = torch.ones(pre_bn.input_size).cuda()
                c[:, :, l2] *= 0
                c[:, :, :, l1] *= 0
                b += c * pre_bn.getmean()[1].view(1, -1, 1, 1)
                c = torch.ones(pre_bn.input_size).cuda()
                c[:, :, l1] *= 0
                c[:, :, :, l2] *= 0
                b += c * pre_bn.getmean()[2].view(1, -1, 1, 1)
                c = torch.ones(pre_bn.input_size).cuda()
                c[:, :, l1] *= 0
                c[:, :, :, l1] *= 0
                b += c * pre_bn.getmean()[3].view(1, -1, 1, 1)
                out = conv(b.cuda()).mean([0, 2, 3])
                post_bn.running_mean = out


