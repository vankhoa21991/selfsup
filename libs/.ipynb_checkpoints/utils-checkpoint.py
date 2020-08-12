import os
import numpy as np
import torch
import pandas as pd
# import argparse
# import random
# from datetime import datetime
# from utils.wsi.util import vips2numpy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

try:
    import openslide
except Exception:
    print('Unable to import openslide, please check installation!')

try:
    import rasterio
    from rasterio.windows import Window
except Exception:
    print('Unable to import rasterio, please check installation!')

try:
    import pyvips
except Exception:
    print('Unable to import pyvips, please check installation!')
import PIL.Image as Image

def config_env(mode):
    if mode == 'AWS':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
        device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device_ids = [0]
    elif mode == 'GCP':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device_ids = [0]
    elif mode == 'HOMEPC':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device_ids = [0]
    return device_ids

def read_slide(slide_path, sp):
    '''
    read slide into memory
    Args:
        slide_path: slide path
        sp: slide parameters

    Returns:
        slide
    '''

#     print(slide_path)
    if sp.tiff_lib == 'openslide':
        slide = openslide.OpenSlide(slide_path)
    elif sp.tiff_lib == 'pyvips':
        slide = pyvips.Image.openslideload(slide_path, level=0)
        # slide = pyvips.Image.new_from_file(slide_path)
    elif sp.tiff_lib == 'rasterio':
        slide = rasterio.open(slide_path)
    return slide

def read_region(slide, coord, level, size, sp, save_tiles = False):
    '''
    Read slide region using different libraries
    Args:
        slide: slide imported
        coord: coordinates of top left tile of level 0
        level: level to read
        size: tile size
        sp: setting
        save_tiles: save tiles for analysis

    Returns:
        PIL image of tile
    '''
    #print(slide.get('filename'))
    #print(coord)
    #print('column: {}  row: {}  level: {}, size: {}'.format(coord[0],coord[1],level, size))
    #print(level)

    if sp.tiff_lib == 'openslide':
        index = slide.properties['openslide.comment'].find('aperio.MPP')
        word = slide.properties['openslide.comment'][index + 52:index + 58]

        if word == '0.5024':
            img = slide.read_region(coord, level, (size, size)).convert('RGB')
        elif word == '0.2522':
            level = 1
            img = slide.read_region(coord, level, (size, size)).convert('RGB')

        name = slide._filename.split('/')[-1][:-4]

    elif sp.tiff_lib == 'pyvips':
        print(slide.get('filename'))
        print('Width: {}, Height: {}'.format(slide.get('width'), slide.get('height')))

        # img_vips = slide.copy().crop(coord[0], coord[1], size, size)
        # np_3d = vips2numpy(img_vips)

        region = pyvips.Region.new(slide)
        data = region.fetch(coord[0], coord[1], size, size)
        np_3d  = np.ndarray(buffer=data, dtype=np.uint8, shape=[size, size, 4])

        img = Image.fromarray(np_3d).convert("RGB")
        name = slide.get('filename').split('/')[-1][:-4]

    elif sp.tiff_lib == 'rasterio':
        img = slide.read(window=Window(coord[0],coord[1],size,size))
        #print(img.shape)
        img = np.rollaxis(img, 0, 3)
        img = Image.fromarray(img).convert('RGB')
    
    if save_tiles is True:
        tile_path = sp.mil_output_imgs + '/' + name + \
               '_g' + str(coord).replace(', ', '-') + \
               '_l' + str(level) + '_s' + str(size) + '.png'

        img.save(tile_path)
    return img

def group_max(groups, data, nmax):
    '''
    Get max probability of every tiles in each slide, used to evaluate slide
    Args:
        groups: slide indexes of each tile
        data: probabilities of each tile
        nmax: number of slides

    Returns:
        max probabilities of each tiles in each slide
    '''
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return list(out)

def group_argtopk(groups, data,k=1):
    '''
    get index of top k highest probabilities of each slide, used to select tiles for training
    Args:
        groups: slide indexes of each tile
        data: probabilities of each tile
        k: top k

    Returns:
        indexes of top k tiles of each slide
    '''
    order = np.lexsort((data, groups))
    groups = groups[order]
    # data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def calc_err(maxs,real,thres):
    '''
    error calculation
    Args:
        maxs: max slides probabilities
        real: real label
        thres: threshold

    Returns:
        balance error, normal error, FPR, FNR
    '''

    pred = maxs >= thres
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    balance_err = (fnr + fpr) / 2.
    return balance_err, err, fpr, fnr

def adjust_lr(optimizer, epoch, init_lr):
    '''
    adjust learning rate
    Args:
        optimizer: optimizer
        epoch: current epoch
        init_lr: initial learning rate
    '''

    lr = init_lr * (0.1 ** (epoch // 20))
    print('Learning rate: ' + str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model(epoch,model, acc, optimizer, date, best='', path='results'):
    '''
    save torch model
    Args:
        epoch: current epoch
        model: model
        acc: accuracy
        date: training date
        best: best performance or not
        path: save path
    '''
    obj = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc': acc,
        'optimizer': optimizer.state_dict()
    }
    print('Saving new model: {}_mil_epoch_{}_val_{}_{}.pth'.format(date, epoch, round(acc, 2),  best))
    torch.save(obj, os.path.join(path, '{}_mil_val_{}_epoch_{}_{}.pth'
                                 .format(date, round(acc, 2), epoch, best)))

def log(writer, epoch, total_epoch, loss, err, fpr, fnr, balance_err, mode='Train'):
    '''
    log with tensorboard
    Args:
        writer: tensorboard writer
        epoch: current epoch
        total_epoch: total_epoch
        loss: training or validation loss
        err: training or validation error
        fpr: false positive rate
        fnr: false negative rate
        balance_err: balance error
        mode: Train / Validation / Test
    '''

    print('{}\tEpoch: [{}/{}]\tLoss: {}'.format(mode,epoch + 1, total_epoch, loss))
    writer.add_scalar('{} loss'.format(mode), loss, epoch)
    writer.add_scalar('{} accuracy'.format(mode), 1 - err, epoch)
    writer.add_scalar('{} false positive rate'.format(mode), fpr, epoch)
    writer.add_scalar('{} false negative rate'.format(mode), fnr, epoch)
    writer.add_scalar('{} balance accuracy'.format(mode), 1 - balance_err, epoch)

def log_prediction(path, dset, maxs, thres):
    '''
    Log prediction classes, real classes, max probabilities of each slide
    Args:
        path: save path
        dset: dataset
        maxs: max tiles probabilities of each slide
        thres: threshold post processing
    '''

    print('Create prediction result at threshold {} at {}'.format(thres, path))
    fp = open(path, 'w')
    fp.write('file,target,prediction,probability\n')
    for name, target, prob in zip(dset.slidenames, dset.targets, maxs):
        fp.write('{},{},{},{}\n'.format(name, target, int(prob >= thres), prob))
    fp.close()

# helper function
def add_pr_curve_tensorboard(writer, class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''

    classes = ['Negative','Positive']
    test_probs = np.array(list(zip(1 - test_probs, test_probs)))
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)

def plot_roc_auc(targets, maxs, writer = False, epoch=0, date='', err=0, mode='Validation', path = 'results'):
    '''
    Plot ROC curve, and calculate AUC using sklearn library
    Args:
        targets: labels
        maxs: max probabilities of each slide
        writer: writer tensorboard
        epoch: epoch, 0 in test
        err: normal error or balance error
        mode: Train / Validation / Test
        path: path to save image
    '''

    targets = [int(x) for x in targets]
    fpr, tpr, thres = roc_curve(targets, maxs)
    auc = roc_auc_score(targets, maxs)

    fig = plt.figure()
    plt.plot(fpr, tpr, marker='.', label='ROC')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    print('Draw ROC curve at {}'.format(path +'/{}_ROC_{}_epoch_{}_auc_{}.jpg'.format( date,mode, epoch, auc)))
    plt.savefig(path +'/{}_ROC_{}_epoch_{}.jpg'.format( date,mode, epoch))

    if writer:
        writer.add_scalar('{} AUC score'.format(mode), auc, epoch)
    else:
        import json
        print('AUC: {}'.format(auc))
        with open(path + '/auc.json', 'w') as f:
            json.dump(str(auc), f, indent=4)
    return auc