import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

class MIL_model():
    def __init__(self, sp = None):
        self.model = models.resnet34(True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        if sp.mode.upper() != 'LAPTOP':
            self.model.cuda()
    def load_model(self,path):
        ch = torch.load(path)
        self.model.load_state_dict(ch['state_dict'])

class MIL_att_model(nn.Module):
    def __init__(self, path=None, freeze=True):
        super(MIL_att_model, self).__init__()
        self.L = 1000
        self.D = 512
        self.K = 2

        self.model = models.resnet34(True)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(inplace=True),
            nn.Linear(self.D, 2))

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.n_classes = 2

        bag_classifiers = [nn.Linear(self.L, 1) for i in
                           range(self.n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)

        if path != '':
            ch = torch.load(path)
            self.model.load_state_dict(ch['state_dict'])

    def forward(self, x):
        H = self.model(x) # NxL

        A = self.attention(H)       #NxK

        A = torch.transpose(A,1,0)  # KxN
        A = F.softmax(A, dim=1)     # KxN

        M = torch.mm(A,H)       # KxL features

        # out = self.fc(M)

        logits = torch.empty(1, self.n_classes).float()
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        print(logits)
        return logits, M


class ResNetEncoder(nn.Module):

    def __init__(self, path):
        super(ResNetEncoder, self).__init__()

        temp = models.resnet34()
        temp.fc = nn.Linear(temp.fc.in_features, 2)
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

class rnn_single(nn.Module):

    def __init__(self, ndims):
        super(rnn_single, self).__init__()
        self.ndims = ndims

        self.fc1 = nn.Linear(512, ndims)
        self.fc2 = nn.Linear(ndims, ndims)

        self.fc3 = nn.Linear(ndims, 2)

        self.activation = nn.ReLU()

    def forward(self, input, state):
        input = self.fc1(input)
        state = self.fc2(state)
        state = self.activation(state+input)
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)
