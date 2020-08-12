import sys
import os
import numpy as np
import argparse
import random
import PIL.Image as Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL

from . import utils as utils

class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None,
                 sp  = None,
                 tp  = None,
                 tip = None):
        self.tp = tp
        self.tip = tip
        self.sp = sp
        lib = torch.load(libraryfile)

        self.len = len(lib['slides'])
        lib['slides']  = lib['slides'][:self.len]
        lib['grid']    = lib['grid'][:self.len]
        lib['targets'] = lib['targets'][:self.len]
        lib['mpp']     = lib['mpp'][:self.len]
        lib['name']    = lib['name'][:self.len]
        
        empty = [i for i,g in enumerate(lib['grid']) if len(g) == 0]
        print('There are {} files that have no tiles'.format(len(empty)))

        for index in sorted(empty, reverse=True):
            del lib['slides'][index]
            del lib['grid'][index]
            del lib['targets'][index]
            del lib['mpp'][index]
            del lib['name'][index]

        slides = []
        print('There are {} slides in {}'.format(len(lib['slides']), libraryfile))

        for i,name in enumerate(lib['slides'][:self.len]):
            # print(name)
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            # local_name = sp.SRC_TRAIN_DIR + name[52:]
            slide = utils.read_slide(name, self.sp)
            slides.append(slide)

        print('')
        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))
        #print(list(set(slideIDX)))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        #print('Number of slideIDX: {}'.format(len(slideIDX)))
        self.mode = None
        self.save_tiles = self.sp.save_tiles
        self.mpp = lib['mpp']
        self.mult = lib['mult']
        self.size = int(np.round(tip.TILE_SIZE*lib['mult']))
        self.level = lib['level']


        self.inference_size = tp.inference_size

        if transform is None:
#             self.transform = transforms.Compose([
#                 transforms.ColorJitter(hue=.05, saturation=.05),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
#                 transforms.ToTensor(),              # normalize to 0-1
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   # normalize to -1-1
#             ])
            # normalization

            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            self.transform = transform

    def setmode(self,mode):
        self.mode = mode
        if self.mode == 3:
            self.make_inference_data()

    def make_inference_data(self):
        probs = [random.random() for x in range(len(self.slideIDX))]
        # print('probs {}'.format(len(probs)))
        topk = utils.group_argtopk(np.array(self.slideIDX), probs, self.inference_size)
        # print('topk {}'.format(len(topk)))
        self.i_data = [(self.slideIDX[x], self.grid[x], self.targets[self.slideIDX[x]]) for x in topk]
        self.i_data_slideIDX = [self.i_data[i][0] for i in range(len(self.i_data))]
        print('Number of inference tiles before reduction: {}'.format(len(self.slideIDX)))
        print('Number of inference tiles after reduction: {}'.format(len(self.i_data)))

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
        print('Number of top tiles used for training: {}'.format(len(self.t_data)))

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = utils.read_region(self.slides[slideIDX], coord, self.level, self.size, sp=self.sp, save_tiles=self.save_tiles)
 
            if self.mult != 1:
                img = img.resize((self.tip.ROW_TILE_SIZE, self.tip.COL_TILE_SIZE),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, self.targets[self.slideIDX[index]]

        elif self.mode == 2:
            slideIDX, coord, target = self.t_data[index]
            img = utils.read_region(self.slides[slideIDX], coord, self.level, self.size, sp=self.sp, save_tiles=self.save_tiles)
            
            if self.mult != 1:
                img = img.resize((self.tip.ROW_TILE_SIZE,self.tip.COL_TILE_SIZE),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

        elif self.mode == 3:
            slideIDX, coord, target = self.i_data[index]
            img = utils.read_region(self.slides[slideIDX], coord, self.level, self.size, sp=self.sp, save_tiles=self.save_tiles)

            if self.mult != 1:
                img = img.resize((self.tip.ROW_TILE_SIZE, self.tip.COL_TILE_SIZE), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
        elif self.mode == 2:
            return len(self.t_data)
        elif self.mode == 3:
            return len(self.i_data)


class rnndata(data.Dataset):

    def __init__(self, libraryfile, s, shuffle=False, transform=None):
        lib = torch.load(libraryfile)

        self.s = s
        self.transform = transform
        self.slidenames = lib['slides']
        self.targets = lib['targets']
        self.grid = lib['grid']
        self.level = lib['level']
        self.mult = lib['mult']
        self.size = int(tip.TILE_SIZE * lib['mult'])
        self.shuffle = shuffle

        slides = []
        for i, name in enumerate(lib['slides']):
            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i + 1, len(lib['slides'])))
            sys.stdout.flush()

            slides.append(openslide.OpenSlide(name))

        print('')
        self.slides = slides

    def __getitem__(self, index):

        slide = self.slides[index]
        grid = self.grid[index]
        if self.shuffle:
            grid = random.sample(grid, len(grid))

        out = []
        s = min(self.s, len(grid))
        for i in range(s):
            img = slide.read_region(grid[i], self.level, (self.size, self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((tip.ROW_TILE_SIZE, tip.COL_TILE_SIZE), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            out.append(img)

        return out, self.targets[index]

    def __len__(self):

        return len(self.targets)