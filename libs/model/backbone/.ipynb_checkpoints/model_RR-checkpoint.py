import torch
import torch.nn as nn
import torchvision.models as models
import math

class bbConv4(torch.nn.Module):
    """A simple 4 layers CNN.
    Used as backbone.
    """
    def __init__(self):
        super(bbConv4, self).__init__()
        self.feature_size = 64
        self.name = "conv4"

        self.layer1 = torch.nn.Sequential(
          torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
          torch.nn.BatchNorm2d(8),
          torch.nn.ReLU(),
          torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
          torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
          torch.nn.BatchNorm2d(16),
          torch.nn.ReLU(),
          torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
          torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
          torch.nn.BatchNorm2d(32),
          torch.nn.ReLU(),
          torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = torch.nn.Sequential(
          torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
          torch.nn.BatchNorm2d(64),
          torch.nn.ReLU(),
          torch.nn.AdaptiveAvgPool2d(1)
        )

        self.flatten = torch.nn.Flatten()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.flatten(h)
        return h

class bbResnet34(torch.nn.Module):
    def __init__(self, path=''):
        super(bbResnet34, self).__init__()

        temp = models.resnet34(pretrained=True)
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        self.fc = temp.fc
        self.features = nn.Sequential(*list(temp.children())[:-1])
        # temp = nn.DataParallel(temp, device_ids=[0], dim=0)
        #
        # if path != '':
        #     ch = torch.load(path)
        #     temp.load_state_dict(ch['state_dict'])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        return self.fc(x), x
    
class bbResnet50(torch.nn.Module):
    def __init__(self, path=''):
        super(bbResnet50, self).__init__()

        temp = models.resnet50(pretrained=True)
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        self.fc = temp.fc
        self.features = nn.Sequential(*list(temp.children())[:-1])
        # temp = nn.DataParallel(temp, device_ids=[0], dim=0)
        #
        # if path != '':
        #     ch = torch.load(path)
        #     temp.load_state_dict(ch['state_dict'])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        return self.fc(x), x