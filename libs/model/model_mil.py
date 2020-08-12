import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pdb
from torchsummary import summary
from libs.utils.utils import initialize_weights, print_network
from math import sqrt
import numpy as np


"""
A Modified Implementation of Deep Attention MIL
"""


"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes (experimental usage for multiclass MIL)
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
Complete MIL Network using resnet50 as CNN feature extractor
args:
    gate: whether to use gating in attention network
    pretrained: whether to use weights pretrained on ImageNet for feature network (transfer learning)
    random_init: randomly initialize feature network
    frozen: freeze weights in feature network
    size_args: size config of attention network (refer to self.size_dict below)
    dropout: whether to use dropout in attention network
"""
class MIL_Attention(nn.Module):
    def __init__(self, gate = True, size_arg = "A", dropout = False):
        super(CPC_MIL_Attention, self).__init__()
        self.size_dict = {"A": [1024, 256], "B": [1024, 512]}
        size = self.size_dict[size_arg]

        if gate:
            self.attention_net = Attn_Net_Gated(L = size[0], D = size[1], dropout = dropout, n_classes = 1)

        else:
            self.attention_net = Attn_Net(L = size[0], D = size[1], dropout = dropout, n_classes = 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(size[0], 1),
            nn.Sigmoid()
        )

        initialize_weights(self) # initialize weights before loading the weights for feature network
    
    # move network to appropriate device (e.g. multi-gpu support)           
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)

        
    def forward(self, h):
        A, h = self.attention(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)  # K x L
        Y_prob  = self.classifier(M) # K x 1
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A

"""
Complete MIL Network with additional hidden fc-layer after CNN feature extractor 
args:
    gate: whether to use gating in attention network
    pretrained: whether to use weights pretrained on ImageNet for feature network (transfer learning)
    random_init: randomly initialize feature network
    frozen: freeze weights in feature network
    size_args: size config of attention network (refer to self.size_dict below)
    dropout: whether to use dropout in attention network
"""
class MIL_Attention_fc(nn.Module):
    def __init__(self, gate = True, size_arg = "A", dropout = False):
        super(CPC_MIL_Attention_fc, self).__init__()
        self.size_dict = {"A": [1024, 512, 256], "B": [1024, 512, 512]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]

        if dropout:
            fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Sequential(
            nn.Linear(size[1], 1),
            nn.Sigmoid()
        )

        initialize_weights(self)

                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)

        
    def forward(self, h):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)  # K x L
        Y_prob  = self.classifier(M) # K x 1
        Y_hat = torch.ge(Y_prob, 0.5).float()


        return Y_prob, Y_hat, A

"""
Complete MIL Network using resnet50 as CNN feature extractor but uses a two-way softmax for probability scoring (used for smooth-SVM loss)
args:
    gate: whether to use gating in attention network
    pretrained: whether to use weights pretrained on ImageNet for feature network (transfer learning)
    random_init: randomly initialize feature network
    frozen: freeze weights in feature network
    size_args: size config of attention network (refer to self.size_dict below)
    dropout: whether to use dropout in attention network
"""
class MIL_Attention_Softmax(nn.Module):
    def __init__(self, gate = True, size_arg = "A", dropout = False, n_classes = 2):
        super(MIL_Attention_Softmax, self).__init__()
        self.size_dict = {"A": [1024, 256], "B": [1024, 512]}
        size = self.size_dict[size_arg]

        if gate:
            self.attention_net = Attn_Net_Gated(L = size[0], D = size[1], dropout = dropout, n_classes = 1)

        else:
            self.attention_net = Attn_Net(L = size[0], D = size[1], dropout = dropout, n_classes = 1)
        
        self.classifier = nn.Linear(size[0], n_classes) 

        initialize_weights(self)
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)

        self.classifier = self.classifier.to(device)

        
    def forward(self, h):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)  # K x L
        logits  = self.classifier(M) # K x 1
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        return logits, Y_prob, Y_hat, A

"""
uses a two-way softmax for probability scoring (used for smooth-SVM loss)

additional hidden fc-layer after CNN feature extractor
args:
    gate: whether to use gating in attention network
    pretrained: whether to use weights pretrained on ImageNet for feature network (transfer learning)
    random_init: randomly initialize feature network
    frozen: freeze weights in feature network
    size_args: size config of attention network (refer to self.size_dict below)
    dropout: whether to use dropout in attention network
"""

class MIL_Attention_Softmax_fc(nn.Module):
    def __init__(self, gate = True, size_arg = "A", dropout = False, n_classes = 2):
        super(MIL_Attention_Softmax_fc, self).__init__()
        self.size_dict = {"A": [1024, 512, 256], "B": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)


        self.classifier = self.classifier.to(device)

        
    def forward(self, h, return_features=False):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)  # K x L
        logits  = self.classifier(M) # K x 1
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A, results_dict

class MIL_fc(nn.Module):
    def __init__(self, gate = True, size_arg = "A", dropout = False, n_classes = 2, top_k=1):
        super(MIL_fc, self).__init__()
        assert n_classes == 2
        self.size_dict = {"A": [1000, 512], "B": [512, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[0]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
            
        fc.extend([nn.Linear(size[0], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.append(nn.Linear(size[1], n_classes))
        self.classifier= nn.Sequential(*fc)
        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.classifier = nn.DataParallel(self.classifier, device_ids=device_ids).to('cuda:0')
        else:
            self.classifier.to(device)
    
    def forward(self, h, return_features=False):
        if return_features:
            h = self.classifier.module[:3](h)
            logits = self.classifier.module[3](h)
        else:
            logits  = self.classifier(h) # K x 1
        top_instance_idx = torch.topk(logits[:, 1], self.top_k, dim=0)[1].view(1,)
        
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1)
        # Y_prob = torch.sigmoid(top_instance)
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        y_probs = F.softmax(logits, dim = 1)
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    def __init__(self, gate = True, size_arg = "A", dropout = False, n_classes = 2, top_k=1):
        super(MIL_fc_mc, self).__init__()
        assert n_classes > 2
        self.size_dict = {"A": [1024, 512], "B": [512, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)

        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for i in range(n_classes)])
        initialize_weights(self)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fc = nn.DataParallel(self.fc, device_ids=device_ids).to('cuda:0')
            self.classifiers = nn.DataParallel(self.classifiers, device_ids=device_ids).to('cuda:0')
        else:
            self.fc = self.fc.to(device)
            self.classifiers = self.classifiers.to(device)
    
    def forward(self, h, return_features=False):
        device = h.device
       
        h = self.fc(h)
        logits = torch.empty(h.size(0), self.n_classes).float().to(device)

        for c in range(self.n_classes):
            if isinstance(self.classifiers, nn.DataParallel):
                logits[:, c] = self.classifiers.module[c](h).squeeze(1)
            else:
                logits[:, c] = self.classifiers[c](h).squeeze(1)        

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


        
