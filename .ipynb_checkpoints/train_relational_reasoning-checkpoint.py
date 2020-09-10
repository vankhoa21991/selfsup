
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import glob
import json
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

# lib
import libs.utils.utils_bb as utils_bb
from libs.setting import Train_RR
from libs.model.backbone.model_RR import bbResnet34, bbResnet50
from libs.datagen.datagen_bb import TileDatasetRR

tp = Train_RR()

# Set random seed for reproducibility
np.random.seed(tp.seed)
torch.manual_seed(tp.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(tp.seed)

parser = argparse.ArgumentParser(description="Train backbone")
parser.add_argument('--mode', type=str, choices=['train',  'valid', 'infer'])

args = parser.parse_args()

now = datetime.now()  # current date and time
date = now.strftime("%Y-%m-%d-%H-%M-%S")
result_path = 'results/backbone/' + date + '_{}_{}'.format(args.mode, tp.model_type)
tp.results_dir = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
with open(result_path + '/config.json', 'w') as f:
    json.dump(tp.__dict__, f, indent=4)

class RelationalReasoning(torch.nn.Module):
    """Self-Supervised Relational Reasoning.
    Essential implementation of the method, which uses
    the 'cat' aggregation function (the most effective),
    and can be used with any backbone.
    """

    def __init__(self, backbone, feature_size=64):
        super(RelationalReasoning, self).__init__()
        self.backbone = backbone
        self.backbone = torch.nn.DataParallel(self.backbone)

        self.backbone.cuda()
        self.relation_head = torch.nn.Sequential(
            torch.nn.Linear(feature_size * 2, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1))

        self.relation_head.cuda()
        
        
    def aggregate(self, features, K):
        relation_pairs_list = list()
        targets_list = list()
        size = int(features.shape[0] / K)
        shifts_counter = 1
        for index_1 in range(0, size * K, size):
            for index_2 in range(index_1 + size, size * K, size):
                # Using the 'cat' aggregation function by default
                pos_pair = torch.cat([features[index_1:index_1 + size],
                                      features[index_2:index_2 + size]], 1)
                # Shuffle without collisions by rolling the mini-batch (negatives)
                neg_pair = torch.cat([
                    features[index_1:index_1 + size],
                    torch.roll(features[index_2:index_2 + size],
                               shifts=shifts_counter, dims=0)], 1)
                relation_pairs_list.append(pos_pair)
                relation_pairs_list.append(neg_pair)
                targets_list.append(torch.ones(size, dtype=torch.float32))
                targets_list.append(torch.zeros(size, dtype=torch.float32))
                shifts_counter += 1
                if (shifts_counter >= size):
                    shifts_counter = 1  # avoid identity pairs
        relation_pairs = torch.cat(relation_pairs_list, 0)
        targets = torch.cat(targets_list, 0)
        return relation_pairs, targets

    def train(self, tot_epochs, train_loader):
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': self.relation_head.parameters()}])
        BCE = torch.nn.BCEWithLogitsLoss()
        self.backbone.train()
        self.relation_head.train()
        best_test_loss = 9999
        for epoch in range(tot_epochs):
            # the real target is discarded (unsupervised)
            for i, (data_augmented, _) in tqdm(enumerate(train_loader)):
                K = len(data_augmented)  # tot augmentations
                x = torch.cat(data_augmented, 0).cuda()

                optimizer.zero_grad()
                # forward pass (backbone)
                _, features = self.backbone(x)
                # aggregation function
                relation_pairs, targets = self.aggregate(features, K)
                targets = targets.cuda()
                # forward pass (relation head)

                #print(relation_pairs.size())
                #print(targets.size())

                score = self.relation_head(relation_pairs).squeeze()
                # cross-entropy loss and backward
                loss = BCE(score, targets)
                loss.backward()
                optimizer.step()
                # estimate the accuracy
                predicted = torch.round(torch.sigmoid(score))
                predicted = predicted.cuda()
                correct = predicted.eq(targets.view_as(predicted)).sum()
                accuracy = (100.0 * correct / float(len(targets)))

                if (i % 100 == 0):
                    print('Epoch [{}][{}/{}] loss: {:.5f}; accuracy: {:.2f}%' \
                          .format(epoch + 1, i + 1, len(train_loader) + 1,
                                  loss.item(), accuracy.item()))
                    torch.save(self.backbone.state_dict(), result_path + '/{}_backbone_epoch_{}.pkl'.format(date, epoch))

def select_transform():
    # Those are the transformations used in the paper
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])  

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                          saturation=0.8, hue=0.2)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    rnd_rcrop = transforms.RandomResizedCrop(size=512, scale=(0.08, 1.0),
                                             interpolation=2)
    rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
    train_transform = transforms.Compose([rnd_rcrop, rnd_hflip,
                                          rnd_color_jitter, rnd_gray,
                                          transforms.ToTensor(), normalize])

    return train_transform

if __name__ == '__main__':
    opt = []
    # instantiate the dataset and dataloader
    train_transform = select_transform()
    dataset = TileDatasetRR(K=tp.K, path=tp.TILE_DIR, transform=train_transform)

    trainloader, valloader, test_loader = utils_bb.load_split_train_test(dataset)

    opt.extend([trainloader, valloader])
    # instantiate the dataset and dataloader

    backbone = bbResnet50()
    model = RelationalReasoning(backbone, tp.feature_size)
    

    if args.mode == 'infer':
        inference()
    elif args.mode == 'train':
        model.train(tot_epochs=tp.tot_epochs, train_loader=trainloader)

    elif args.mode == 'valid':
        validate(model, valloader)

    else:
        print('Please select mode!!!!')
        raise NotImplementedError

