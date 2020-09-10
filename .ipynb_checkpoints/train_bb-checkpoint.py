# Generic
import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# lib
import libs.utils.utils_bb as utils_bb
from libs.setting import TrainBBParam
from libs.model.backbone.model_rot import ResnetRot
from libs.model.backbone.model_AE import Autoencoder
from libs.model.backbone.unet import Unet
from libs.model.backbone.model_outpainting import CEImageDataset, CEGenerator, CEDiscriminator
from libs.model.backbone.model_inpainting import _netlocalD, _netG, _netG_resnet34, _netD_resnet, weights_init
from libs.datagen.datagen_bb import tiles_dataset_AE, tiles_dataset_Unet, tiles_dataset_rot

tp = TrainBBParam()

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

# log with tensorboard
writer = SummaryWriter(result_path)

def load_split_train_test(datadir, valid_size = .2):
    # Load data
    transform = transforms.Compose([
#         transforms.Scale(tp.imageSize),
#         transforms.CenterCrop(tp.imageSize),
        transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # instantiate the dataset and dataloader
    if tp.model_type == 'ae':
        train_data = tiles_dataset_AE(tp.TILE_DIR, transform=transform)  # our custom dataset
        test_data = tiles_dataset_AE(tp.TILE_DIR, transform=transform)  # our custom dataset
    elif tp.model_type == 'unet':
        train_data = tiles_dataset_Unet(tp.TILE_DIR, transform=transform)  # our custom dataset
        test_data = tiles_dataset_Unet(tp.TILE_DIR, transform=transform)  # our custom dataset
    elif tp.model_type == 'rot':
        train_data = tiles_dataset_rot(tp.TILE_DIR, transform=transform)  # our custom dataset
        test_data = tiles_dataset_rot(tp.TILE_DIR, transform=transform)  # our custom dataset
    else:
        raise NotImplementedError

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=tp.train_batchsize,num_workers=tp.num_workers)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=tp.val_batchsize, num_workers=tp.num_workers)
    print('Length trainloader: {}'.format(len(trainloader)))
    print('Length testloader: {}'.format(len(testloader)))
    return trainloader, testloader

def train(trainloader, valloader, model, criterion):
    print(model)
    
    if tp.model_path is not None:
        print("Loading checkpoint {}...".format(tp.model_path))
        model.load_state_dict(torch.load(tp.model_path))
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU in order to speed up training.")

    optimizer = torch.optim.Adam(model.parameters(), lr=tp.lr)

    train_losses = []
    valid_losses = []
    best_val_loss = 9999

    for epoch in range(tp.niter):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        start_loaddata = time.time()
        for i, (inputs, targets) in enumerate(trainloader, 0):
            print('Time loading data: {}'.format(time.time() - start_loaddata))
            start_compute = time.time()
            inputs  = utils_bb.get_torch_vars(inputs)
            targets = utils_bb.get_torch_vars(targets)

            # ============ Forward ============
            outputs, encoded = model(inputs)
            # print(inputs.size())
            # print(outputs.size())
            # print(targets.size())
            loss = criterion(outputs, targets)

            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            train_loss += loss.data
            print('[%d, %5d/%5d] loss: %.5f' %(epoch + 1, i + 1,len(trainloader), loss.data))
            
            print('Time computing: {}'.format(time.time() - start_compute))
            start_loaddata = time.time()

        train_loss = train_loss / len(trainloader)
        train_losses.append(train_loss)
        writer.add_scalar('Train loss', train_loss, epoch)

        if valloader and epoch % tp.valid_every == 0:
            # validate-the-model
            model.eval()
            test_acc = 0
            print('Validation model...')
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(valloader, 0):
                    inputs  = utils_bb.get_torch_vars(inputs)
                    targets = utils_bb.get_torch_vars(targets)

                    outputs, encoded = model(inputs)
                    loss = criterion(outputs, targets)

                    # update-average-validation-loss
                    valid_loss += loss.data
                    if tp.model_type == 'rot':
                        _, pred = torch.max(outputs.data, 1)
                        test_acc += torch.sum(pred == targets.data)

            # calculate-average-losses
            test_acc = test_acc / len(valloader)
            valid_loss = valid_loss / len(valloader)
            valid_losses.append(valid_loss)

            print('[%d, %5d/%5d] Validation accuracy: %.5f' % (epoch + 1, i + 1, len(valloader), test_acc))
            print('[%d, %5d/%5d] Validation loss: %.5f' % (epoch + 1, i + 1, len(valloader), valid_loss))
            writer.add_scalar('Val loss', valid_loss, epoch)
            writer.add_scalar('Val accuracy', test_acc, epoch)
            if valid_loss < best_val_loss:
                name = result_path + "/{}_epoch_{}_backbone_{}_{}_{}.pkl".format(date, epoch, tp.model_type,
                                                                                 str(valid_loss.cpu().numpy().round(3)),
                                                                                 str(train_loss.cpu().numpy().round(3)))
                print('Saving Model {}'.format(name))
                torch.save(model.state_dict(), name)
                best_val_loss = valid_loss

    print('Finished Training')



def validate(model,  valloader):
    print("Loading checkpoint {}...".format(tp.model_path))
    model.load_state_dict(torch.load(tp.model_path))
    model.cuda()
    dataiter = iter(valloader)
    images = dataiter.next()[0]
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
    utils_bb.imshow(torchvision.utils.make_grid(images), result_path +"/original_imgs.jpeg")
    images = Variable(images.cuda())

    decoded_imgs, _ = model(images)
    utils_bb.imshow(torchvision.utils.make_grid(decoded_imgs.data), result_path + "/decoded_imgs.jpeg")
    exit(0)

def inference():
    return

if __name__ == '__main__':
    opt = []
    trainloader, valloader = load_split_train_test(tp.TILE_DIR, .2)
    opt.extend([trainloader, valloader])
    # instantiate the dataset and dataloader
    if tp.model_type == 'ae':
        model = Autoencoder()
        criterion = nn.BCELoss()
        opt.extend([model, criterion])

    elif tp.model_type == 'unet':
        model = Unet()
        criterion = nn.MSELoss()
        opt.extend([model, criterion])

    elif tp.model_type == 'rot':
        model = ResnetRot()
        criterion = nn.CrossEntropyLoss()
        opt.extend([model, criterion])
    else:
        raise NotImplementedError

    if args.mode == 'infer':
        inference()
    elif args.mode == 'train':
            train(*opt)
    elif args.mode == 'valid':
        validate(model, valloader)

    else:
        print('Please select mode!!!!')
        raise NotImplementedError
