import PIL
import glob
import random
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import matplotlib.pyplot as plt

tensorToImage = torchvision.transforms.ToPILImage()
imageToTensor = torchvision.transforms.ToTensor()

class tiles_dataset_AE(data.Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    A Dataset for Rotation-based Self-Supervision! Images are rotated clockwise.
    - classification - False=Use rotation labels. True=Use original classification labels.
    """
    def __init__(self, path, transform=None):
        self.tile_paths = glob.glob(path + '/**/*.png')
        self.transform = transform

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        img = PIL.Image.open(self.tile_paths[index])
        img = img.resize((512, 512), PIL.Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)

        return img, img
    def __len__(self):
        return len(self.tile_paths)

class tiles_dataset_Unet(data.Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    A Dataset for Rotation-based Self-Supervision! Images are rotated clockwise.
    - classification - False=Use rotation labels. True=Use original classification labels.
    """
    def __init__(self, path, transform=None):
        self.tile_paths = glob.glob(path + '/**/*.png')
        self.transform = transform

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        img = PIL.Image.open(self.tile_paths[index])
        img = img.resize((224, 224), PIL.Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)

        return img, img
    def __len__(self):
        return len(self.tile_paths)

class tiles_dataset_rot(data.Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, path, transform=None):
        self.tile_paths = glob.glob(path + '/**/*.png')
        self.transform = transform

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        img = PIL.Image.open(self.tile_paths[index])
        img = img.resize((512, 512), PIL.Image.ANTIALIAS)

        # 4 classes for rotation
        degrees = [0, 45, 90, 135, 180, 225, 270, 315]
        rand_choice = random.randint(0, len(degrees) - 1)

        npimg = np.asarray(img)
        img = tensorToImage(npimg)
        img = img.rotate(degrees[rand_choice])

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(rand_choice).long()

    def show_batch(self, n=3):
        fig, axs = plt.subplots(n, n)
        fig.tight_layout()
        for i in range(n):
            for j in range(n):
                rand_idx = random.randint(0, len(self) - 1)
                img, label = self.__getitem__(rand_idx)
                axs[i, j].imshow(tensorToImage(img), cmap='gray')
                if self.classification:
                    axs[i, j].set_title('Label: {0} (Digit #{1})'.format(label.item(), label.item()))
                else:
                    axs[i, j].set_title('Label: {0} ({1} Degrees)'.format(label.item(), label.item() * 45))
                axs[i, j].axis('off')

    def __len__(self):
        return len(self.tile_paths)

class tiles_dataset_pt(data.Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, path, transform=None):
        self.tile_paths = glob.glob(path + '/**/*.png')
        self.transform = transform

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        img = PIL.Image.open(self.tile_paths[index])


        img = img.resize((512, 512), PIL.Image.ANTIALIAS)

        # 4 classes for rotation
        degrees = [0, 45, 90, 135, 180, 225, 270, 315]
        rand_choice = random.randint(0, len(degrees) - 1)

        npimg = np.asarray(img)
        img = tensorToImage(npimg)
        img = img.rotate(degrees[rand_choice])

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(rand_choice).long()

    def show_batch(self, n=3):
        fig, axs = plt.subplots(n, n)
        fig.tight_layout()
        for i in range(n):
            for j in range(n):
                rand_idx = random.randint(0, len(self) - 1)
                img, label = self.__getitem__(rand_idx)
                axs[i, j].imshow(tensorToImage(img), cmap='gray')
                if self.classification:
                    axs[i, j].set_title('Label: {0} (Digit #{1})'.format(label.item(), label.item()))
                else:
                    axs[i, j].set_title('Label: {0} ({1} Degrees)'.format(label.item(), label.item() * 45))
                axs[i, j].axis('off')

    def __len__(self):
        return len(self.tile_paths)