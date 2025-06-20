"""
Catherine Breen
cbreen@uw.edu
adapted from: 
https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/

"""

import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


class snowPoleResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(snowPoleResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print("Training intermediate layer parameters...")
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Freezing intermediate layer parameters...")
        # change the final layer
        self.l0 = nn.Linear(
            2048, 4
        )  #### the second value is the number of points you want to predict

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0
