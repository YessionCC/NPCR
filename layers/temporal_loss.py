import torch
#import models_lpf
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from collections import namedtuple
import cv2
import numpy as np


def gradient_1order(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    return xgrad

LossOutput = namedtuple(
    "LossOutput", ["relu1","relu2"])

#LossOutput = namedtuple(
#    "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])


class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features

        '''
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
        '''



        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2"

        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
            if name > '8':
                break
        return LossOutput(**output)




class Temporal_loss(torch.nn.Module):
    def __init__(self):
        super(Temporal_loss, self).__init__()

        #self.model = models_lpf.resnet50(filter_size = 5)
        #self.model.load_state_dict(torch.load('/data/wmy/NR/models/resnet50_lpf5.pth.tar')['state_dict'])
        self.model = LossNetwork()
        self.model.cuda()
        self.model.eval()

        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, x, target):
        x_feature = self.model(x)
        target_feature = self.model(target)

        feature_loss = self.loss(x_feature.relu1,target_feature.relu1) + self.loss(x_feature.relu2,target_feature.relu2)
        feature_loss = feature_loss /2
        return feature_loss, self.loss(x,target)
