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
    elif tp.model_type == 'inpainting':
        transform = transforms.Compose([
                    transforms.Scale(tp.imageSize),
                    transforms.CenterCrop(tp.imageSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_data = dset.ImageFolder(root=tp.TILE_DIR, transform=transform)
        test_data = dset.ImageFolder(root=tp.TILE_DIR, transform=transform)
    elif tp.model_type == 'outpainting':
        input_size = 128
        output_size = 192
        transform = transforms.Compose([
            transforms.Resize(output_size),
            transforms.CenterCrop(output_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        train_data = CEImageDataset(tp.TILE_DIR, transform, output_size, input_size, outpaint=True)
        test_data = CEImageDataset(tp.TILE_DIR, transform, output_size, input_size, outpaint=True)

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

def train_inpainting(trainloader, valloader, netD, netG, criterion, criterionMSE):
    try:
        os.makedirs(result_path + "/cropped")
        os.makedirs(result_path + "/real")
        os.makedirs(result_path + "/recon")
    except OSError:
        pass
    best_val_l2_loss = 9999
    input_real = torch.FloatTensor(tp.train_batchsize, 3, tp.imageSize, tp.imageSize)
    input_cropped = torch.FloatTensor(tp.train_batchsize, 3, tp.imageSize, tp.imageSize)
    label = torch.FloatTensor(tp.train_batchsize)
    real_label = 1
    fake_label = 0

    real_center = torch.FloatTensor(tp.train_batchsize, 3, int(tp.imageSize / 2), int(tp.imageSize / 2))

    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    label = Variable(label)

    real_center = Variable(real_center)
    
    if tp.netD_path is not None and tp.netG_path is not None:
        print("Loading checkpoint {}...".format(tp.netD_path))
        netD.load_state_dict(torch.load(tp.netD_path))
        print("Loading checkpoint {}...".format(tp.netG_path))
        netG.load_state_dict(torch.load(tp.netG_path))
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)

    netD.cuda()
    netG.cuda()
    criterion.cuda()
    # criterionMSE.cuda()
    input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()
    real_center = real_center.cuda()

    # setup optimizer
    optimizerD = torch.optim.Adam(netD.parameters(), lr=tp.lr, betas=(tp.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=tp.lr, betas=(tp.beta1, 0.999))
    resume_epoch = 0

    inx = 0
    for epoch in range(resume_epoch, tp.niter):
        start_loaddata = time.time()
        for i, data in enumerate(trainloader, 0):
            print('Time loading data: {}'.format(time.time() - start_loaddata))
            start_compute = time.time()
            real_cpu, _ = data
            # real_cpu.cuda()
            real_center_cpu = real_cpu[:, :, int(tp.imageSize / 4):int(tp.imageSize / 4) + int(tp.imageSize / 2),
                              int(tp.imageSize / 4):int(tp.imageSize / 4) + int(tp.imageSize / 2)]
            batch_size = real_cpu.size(0)

            with torch.no_grad():
                input_real.resize_(real_cpu.size()).copy_(real_cpu)
                input_cropped.resize_(real_cpu.size()).copy_(real_cpu)
                real_center.resize_(real_center_cpu.size()).copy_(real_center_cpu)

            input_cropped.data[:, 0,
            int(tp.imageSize / 4 + tp.overlapPred):int(tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred),
            int(tp.imageSize / 4 + tp.overlapPred):int(
                tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred)] = 2 * 117.0 / 255.0 - 1.0
            input_cropped.data[:, 1,
            int(tp.imageSize / 4 + tp.overlapPred):int(tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred),
            int(tp.imageSize / 4 + tp.overlapPred):int(
                tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred)] = 2 * 104.0 / 255.0 - 1.0
            input_cropped.data[:, 2,
            int(tp.imageSize / 4 + tp.overlapPred):int(tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred),
            int(tp.imageSize / 4 + tp.overlapPred):int(
                tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred)] = 2 * 123.0 / 255.0 - 1.0


            # train with real
            netD.zero_grad()
            with torch.no_grad():
                label.resize_(batch_size).fill_(real_label)

            output = netD(real_center)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            # noise.data.resize_(batch_size, nz, 1, 1)
            # noise.data.normal_(0, 1)
            fake, _ = netG(input_cropped)
            label.data.fill_(fake_label)
            # print(fake.size())
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG_D = criterion(output, label)
            # errG_D.backward(retain_variables=True)

            # errG_l2 = criterionMSE(fake,real_center)
            wtl2Matrix = real_center.clone()
            wtl2Matrix.data.fill_(tp.wtl2 * tp.overlapL2Weight)
            wtl2Matrix.data[:, :, int(tp.overlapPred):int(tp.imageSize / 2 - tp.overlapPred),
            int(tp.overlapPred):int(tp.imageSize / 2 - tp.overlapPred)] = tp.wtl2

            errG_l2 = (fake - real_center).pow(2)
            errG_l2 = errG_l2 * wtl2Matrix
            errG_l2 = errG_l2.mean()

            errG = (1 - tp.wtl2) * errG_D + tp.wtl2 * errG_l2

            errG.backward()

            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
                  % (epoch, tp.niter, i, len(trainloader),
                     errD.data.cpu().numpy(), errG_D.data.cpu().numpy(), errG_l2.data.cpu().numpy(), D_x, D_G_z1,))

            print('Time computing: {}'.format(time.time() - start_compute))
            start_loaddata = time.time()

            inx = inx + 1

            if inx % 1000 == 0:
                vutils.save_image(real_cpu, result_path +
                                  '/real/real_samples_epoch_%03d.png' % (epoch))
                vutils.save_image(input_cropped.data,result_path +
                                  '/cropped/cropped_samples_epoch_%03d.png' % (epoch))
                recon_image = input_cropped.clone()
                recon_image.data[:, :, int(tp.imageSize / 4):int(tp.imageSize / 4 + tp.imageSize / 2),
                int(tp.imageSize / 4):int(tp.imageSize / 4 + tp.imageSize / 2)] = fake.data
                vutils.save_image(recon_image.data, result_path +
                                  '/recon/recon_center_samples_epoch_%03d.png' % (epoch))

        if errG_l2.data.cpu().numpy() < best_val_l2_loss:
            nameG = result_path + '/{}_netG_epoch_{}_{}_{}.pkl'.format(date, epoch, tp.model_type, str(errG_l2.data.cpu().numpy().round(3)))
            print('Saving Model {}'.format(nameG))

            # do checkpointing
            torch.save(netG.state_dict(), nameG)

            nameD = result_path + '/{}_netD_epoch_{}_{}_{}.pkl'.format(date, epoch, tp.model_type, str(errD.data.cpu().numpy().round(3)))
            print('Saving Model {}'.format(nameD))
            torch.save(netD.state_dict(), nameD)
            best_val_l2_loss = errG_l2.data.cpu().numpy()

    return

def crop_inpainting(tensor_img):
    input_cropped = tensor_img.clone().detach().cuda()

    input_cropped.data[:, 0,
    int(tp.imageSize / 4 + tp.overlapPred):int(tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred),
    int(tp.imageSize / 4 + tp.overlapPred):int(
        tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred)] = 2 * 117.0 / 255.0 - 1.0
    input_cropped.data[:, 1,
    int(tp.imageSize / 4 + tp.overlapPred):int(tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred),
    int(tp.imageSize / 4 + tp.overlapPred):int(
        tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred)] = 2 * 104.0 / 255.0 - 1.0
    input_cropped.data[:, 2,
    int(tp.imageSize / 4 + tp.overlapPred):int(tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred),
    int(tp.imageSize / 4 + tp.overlapPred):int(
        tp.imageSize / 4 + tp.imageSize / 2 - tp.overlapPred)] = 2 * 123.0 / 255.0 - 1.0
    return input_cropped

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

def validate_inpaiting(model,  loader):
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

    elif tp.model_type == 'inpainting':
        netG = _netG_resnet34(tp)
        netD = _netD_resnet(tp)
        # netG = _netG(tp)
        # netD = _netlocalD(tp)
        criterion = nn.BCELoss()
        criterionMSE = nn.MSELoss()
        opt.extend([netD, netG, criterion, criterionMSE])
    elif tp.model_type == 'outpainting':
        netG = CEGenerator(extra_upsample=True)
        print('Please use other repo for training')
    else:
        raise NotImplementedError

    if args.mode == 'infer':
        inference()
    elif args.mode == 'train':
        if tp.model_type == 'inpainting':
            train_inpainting(*opt)
        else:
            train(*opt)
    elif args.mode == 'valid':
        if tp.model_type == 'inpainting':
            validate_inpaiting(netG, valloader)
        elif tp.model_type == 'outpainting':
            validate(netG, valloader)
        else:
            validate(model, valloader)

    else:
        print('Please select mode!!!!')
        raise NotImplementedError
