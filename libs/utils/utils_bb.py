import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.autograd import Variable
import scipy.misc

def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img, name):
    npimg = img.cpu().numpy()   # np array from -1 to 1
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.axis('off')
    plt.imshow(npimg)
    plt.show()

    npimg01 = (npimg - np.min(npimg)) / np.ptp(npimg) # rescale to 0 1

    matplotlib.image.imsave(name, npimg01)

def load_split_train_test(dataset, valid_size = .2, test_size = .2, batchsize=8, num_workers=8):
    num_train = len(dataset)
    indices = list(range(num_train))
    split_val = int(np.floor(valid_size * num_train))
    split_test = int(np.floor((valid_size + test_size) * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, val_idx, test_idx = indices[split_test:], indices[:split_val], indices[split_val:split_test]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)


    train_loader = torch.utils.data.DataLoader(dataset,
                   sampler=train_sampler, batch_size=batchsize,num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             sampler=val_sampler, batch_size=batchsize,
                                             num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset,
                   sampler=test_sampler, batch_size=batchsize, num_workers=num_workers)
    print('Length trainloader: {}'.format(len(train_loader)))
    print('Length valloader: {}'.format(len(val_loader)))
    print('Length testloader: {}'.format(len(test_loader)))
    return train_loader, val_loader, test_loader
