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
