# Generic
import os
import json
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

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
import libs.utils.utils_outpainting as utils
from libs.setting import Train_outpainting
from libs.datagen.datagen_bb import CEImageDataset
from libs.model.backbone.model_outpainting import CEGenerator, CEDiscriminator, CEGeneratorResnet50

tp = Train_outpainting()

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

def train():
    # instantiate the dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize(tp.output_size),
        transforms.CenterCrop(tp.output_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    dataset = CEImageDataset(tp.TILE_DIR, transform, tp.output_size, tp.input_size, outpaint=True)

    train_loader, val_loader, test_loader = utils_bb.load_split_train_test(dataset)

    # G_net = CEGenerator(extra_upsample=True)
    G_net = CEGeneratorResnet50()
    D_net = CEDiscriminator()
    G_net.apply(utils.weights_init_normal)
    D_net.apply(utils.weights_init_normal)

    device = torch.device('cuda:0')
    # G_net = nn.DataParallel(G_net)
    # D_net = nn.DataParallel(D_net)
    G_net.to(device)
    D_net.to(device)
    print('device:', device)

    # Define losses
    criterion_pxl = nn.L1Loss()
    criterion_D = nn.MSELoss()
    optimizer_G = torch.optim.Adam(G_net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D_net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    criterion_pxl.to(device)
    criterion_D.to(device)

    # Start training
    data_loaders = {'train': train_loader, 'val': val_loader,
                    'test': test_loader}  # NOTE: test is evidently not used by the train method
    n_epochs = 200
    adv_weight = [0.001, 0.005, 0.015, 0.040]  # corresponds to epochs 1-10, 10-30, 30-60, 60-onwards
    hist_loss = train_CE(G_net, D_net, device, criterion_pxl, criterion_D, optimizer_G, optimizer_D,
                         data_loaders, result_path, result_path, n_epochs=n_epochs, outpaint=True, adv_weight=adv_weight)

    # Save loss history and final generator
    pickle.dump(hist_loss, open('hist_loss.p', 'wb'))
    torch.save(G_net.state_dict(), 'generator_final.pt')

def train_CE(G_net, D_net, device, criterion_pxl, criterion_D, optimizer_G, optimizer_D,
             data_loaders, model_save_path, html_save_path, n_epochs=200, start_epoch=0, outpaint=True,
             adv_weight=0.001):
    '''
    Based on Context Encoder implementation in PyTorch.
    '''
    Tensor = torch.cuda.FloatTensor
    hist_loss = defaultdict(list)

    for epoch in range(start_epoch, n_epochs):

        for phase in ['train', 'val']:
            batches_done = 0

            running_loss_pxl = 0.0
            running_loss_adv = 0.0
            running_loss_D = 0.0

            for idx, (imgs, masked_imgs, masked_parts) in enumerate(data_loaders[phase]):
                if phase == 'train':
                    G_net.train()
                    D_net.train()
                else:
                    G_net.eval()
                    D_net.eval()
                torch.set_grad_enabled(phase == 'train')

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], *tp.patch).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(Tensor(imgs.shape[0], *tp.patch).fill_(0.0), requires_grad=False).to(device)
                # Configure input
                imgs = Variable(imgs.type(Tensor)).to(device)
                masked_imgs = Variable(masked_imgs.type(Tensor)).to(device)
                if not (outpaint):
                    masked_parts = Variable(masked_parts.type(Tensor)).to(device)

                # -----------
                #  Generator
                # -----------
                if phase == 'train':
                    optimizer_G.zero_grad()
                # Generate a batch of images
                outputs, _ = G_net(masked_imgs)

                # print(imgs.size())
                # print(masked_imgs.size())
                # print(outputs.size())

                # Adversarial and pixelwise loss
                if not (outpaint):
                    loss_pxl = criterion_pxl(outputs, masked_parts)  # inpaint: compare center part only
                else:
                    loss_pxl = criterion_pxl(outputs, imgs)  # outpaint: compare to full ground truth
                loss_adv = criterion_D(D_net(outputs), valid)
                # Total loss
                cur_adv_weight = utils.get_adv_weight(adv_weight, epoch)
                loss_G = (1 - cur_adv_weight) * loss_pxl + cur_adv_weight * loss_adv
                if phase == 'train':
                    loss_G.backward()
                    optimizer_G.step()

                # ---------------
                #  Discriminator
                # ---------------
                if phase == 'train':
                    optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                if not (outpaint):
                    real_loss = criterion_D(D_net(masked_parts), valid)  # inpaint: check center part only
                else:
                    real_loss = criterion_D(D_net(imgs), valid)  # outpaint: check full ground truth
                fake_loss = criterion_D(D_net(outputs.detach()), fake)
                loss_D = 0.5 * (real_loss + fake_loss)
                if phase == 'train':
                    loss_D.backward()
                    optimizer_D.step()

                # Update & print statistics
                batches_done += 1
                running_loss_pxl += loss_pxl.item()
                running_loss_adv += loss_adv.item()
                running_loss_D += loss_D.item()
                if phase == 'train' and utils.is_power_two(batches_done):
                    print('Batch {:d}/{:d}  loss_pxl {:.4f}  loss_adv {:.4f}  loss_D {:.4f}'.format(
                        batches_done, len(data_loaders[phase]), loss_pxl.item(), loss_adv.item(), loss_D.item()))

            # Store model & visualize examples
            if phase == 'train':
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(G_net.state_dict(), model_save_path + '/G_' + str(epoch) + '.pt')
                torch.save(D_net.state_dict(), model_save_path + '/D_' + str(epoch) + '.pt')
                utils.generate_html(G_net, D_net, device, data_loaders, html_save_path + '/' + str(epoch), outpaint=outpaint)

            # Store & print statistics
            cur_loss_pxl = running_loss_pxl / batches_done
            cur_loss_adv = running_loss_adv / batches_done
            cur_loss_D = running_loss_D / batches_done
            hist_loss[phase + '_pxl'].append(cur_loss_pxl)
            hist_loss[phase + '_adv'].append(cur_loss_adv)
            hist_loss[phase + '_D'].append(cur_loss_D)
            print('Epoch {:d}/{:d}  {:s}  loss_pxl {:.4f}  loss_adv {:.4f}  loss_D {:.4f}'.format(
                epoch + 1, n_epochs, phase, cur_loss_pxl, cur_loss_adv, cur_loss_D))

        print()

    print('Done!')
    return hist_loss

def inference():
    return

def valid_outpainting(model,  loader):
    try:
        os.makedirs(result_path + "/cropped")
        os.makedirs(result_path + "/real")
        os.makedirs(result_path + "/recon")
    except OSError:
        pass

    print("Loading checkpoint {}...".format(tp.netG_path))
    model.load_state_dict(torch.load(tp.netG_path))
    model.cuda()

    dataiter = iter(loader)
    images = dataiter.next()[0]

    utils_bb.imshow(torchvision.utils.make_grid(images), result_path +"/original_imgs.png")
    images = Variable(images.cuda())

    real_cpu = images

    input_real = torch.FloatTensor(tp.val_batchsize, 3, tp.imageSize, tp.imageSize)
    input_cropped = torch.FloatTensor(tp.val_batchsize, 3, tp.imageSize, tp.imageSize)

    real_center = torch.FloatTensor(tp.val_batchsize, 3, int(tp.imageSize / 2), int(tp.imageSize / 2))

    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)

    real_center = Variable(real_center)

    real_center_cpu = real_cpu[:, :, int(tp.imageSize / 4):int(tp.imageSize / 4) + int(tp.imageSize / 2),
                      int(tp.imageSize / 4):int(tp.imageSize / 4) + int(tp.imageSize / 2)]

    with torch.no_grad():
        input_real.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.resize_(real_center_cpu.size()).copy_(real_center_cpu)

    input_cropped = crop_inpainting(input_cropped)

    decoded_imgs, _ = model(input_cropped)

    recon_image = input_cropped.clone()
    recon_image.data[:, :, int(tp.imageSize / 4):int(tp.imageSize / 4 + tp.imageSize / 2),
    int(tp.imageSize / 4):int(tp.imageSize / 4 + tp.imageSize / 2)] = decoded_imgs.data

    utils_bb.imshow(torchvision.utils.make_grid(recon_image.data), result_path + "/decoded_imgs.png")
    exit(0)

if __name__ == '__main__':
    if args.mode == 'infer':
        inference()
    elif args.mode == 'train':
        train()
    elif args.mode == 'valid':
        validate(netG, valloader)

    else:
        print('Please select mode!!!!')
        raise NotImplementedError