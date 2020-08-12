import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.autograd import Variable

def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img, name):
    npimg = img.cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.axis('off')
    plt.imshow(npimg)
    plt.show()

    im = Image.fromarray((npimg * 255).astype(np.uint8)).convert('RGB')
    im.save(name)