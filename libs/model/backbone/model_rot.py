# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

#collapse-show
class ResnetRot(nn.Module):

    def __init__(self, path=''):
        super(ResnetRot, self).__init__()

        temp = models.resnet34(True)
        temp.fc = nn.Linear(temp.fc.in_features, 8)
        self.fc = temp.fc
        self.features = nn.Sequential(*list(temp.children())[:-1])
        temp = nn.DataParallel(temp, device_ids=[0], dim=0)

        if path != '':
            ch = torch.load(path)
            temp.load_state_dict(ch['state_dict'])

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.fc(x), x