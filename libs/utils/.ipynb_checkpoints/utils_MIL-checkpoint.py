import os
import numpy as np
import torch
import pandas as pd
# import argparse
# import random
# from datetime import datetime
from tqdm import tqdm
import math
# from utils.wsi.util import vips2numpy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from PIL import Image, ImageDraw, ImageFont
import PIL
import pickle
import matplotlib.pyplot as plt

import libs.wsi.util as wsiutil
import libs.wsi.tiles_new as tiles



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
    #print('column: {}  row: {}  level: {}, size: {}'.format(coord[0],coord[1],level, size))
    

    if sp.dataset == 'TG':
        img = slide.read_region(coord, level, (size, size)).convert('RGB')
        return img
    
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
    print('Draw ROC curve at {}'.format(path +'/{}_ROC_{}_epoch_{}_auc_{}.jpg'.format( date,mode, epoch, round(auc,2))))
    plt.savefig(path +'/{}_ROC_{}_epoch_{}_auc_{}.jpg'.format( date,mode, epoch, round(auc,2)))

    if writer:
        writer.add_scalar('{} AUC score'.format(mode), auc, epoch)
    else:
        import json
        print('AUC: {}'.format(auc))
        with open(path + '/auc.json', 'w') as f:
            json.dump(str(auc), f, indent=4)
    return auc

def save_heatmap(slidenames, slideIDX, grids_all, probs_all, result_path, st):
    for idx in tqdm(range(len(slidenames))):
        slide = read_slide(slidenames[idx], st)
        idx_slide = [i for i, j in enumerate(slideIDX) if j == idx]
        grids = grids_all[min(idx_slide):max(idx_slide) + 1]
        probs = probs_all[min(idx_slide):max(idx_slide) + 1]

        print(slide)
        # print(len(grids))
        # print(slide.dimensions)

        large_w, large_h = slide.dimensions
        new_w = math.floor(large_w / st.SCALE_FACTOR)
        new_h = math.floor(large_h / st.SCALE_FACTOR)
        level = slide.get_best_level_for_downsample(st.SCALE_FACTOR)
        whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
        whole_slide_image = whole_slide_image.convert("RGBA")

        pil_img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)

        index = slide.properties['openslide.comment'].find('aperio.MPP')
        mpp = slide.properties['openslide.comment'][index + 52:index + 58]
        # print(mpp)

        z = 0  # height of area at top of summary slide

        rows = new_h
        cols = new_w

        row_tile_size = round(st.ROW_TILE_SIZE / st.SCALE_FACTOR)  # use round?
        col_tile_size = round(st.COL_TILE_SIZE / st.SCALE_FACTOR)  # use round?
        if mpp == '0.2522':
            row_tile_size = round(st.ROW_TILE_SIZE * 2 / st.SCALE_FACTOR)  # use round?
            col_tile_size = round(st.COL_TILE_SIZE * 2 / st.SCALE_FACTOR)  # use round?

        # print('row_tize_size: {}, col_tile_size: {}'.format(row_tile_size, col_tile_size))

        num_row_tiles, num_col_tiles = tiles.get_num_tiles(rows, cols, row_tile_size, col_tile_size)

        np_orig = wsiutil.pil_to_np_rgb(pil_img)

        summary_orig = tiles.create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)

        top_tiles = grids

        poly = Image.new('RGBA', summary_orig.size, (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(poly)

        # print('Min probs: {}'.format(min(probs)))
        normalize_probs = (probs - min(probs)) / (max(probs) - min(probs))

        sorted_norm_probs = sorted(normalize_probs, reverse=True)
        top5 = sorted_norm_probs[int(len(normalize_probs) * 0.05)]
        top20 = sorted_norm_probs[int(len(normalize_probs) * 0.2)]
        top50 = sorted_norm_probs[int(len(normalize_probs) * 0.5)]

        for t in range(len(top_tiles)):
            #     print('Column {}, Row {}'.format(grids[t][0], grids[t][1]))
            border_color = tile_border_color(normalize_probs[t], top5, top20, top50)
            #     print('Color: {}'.format(border_color))
            #     print(normalize_probs[t])
            tiles.tile_border(pdraw, grids[t][1] / st.SCALE_FACTOR + z, grids[t][1] / st.SCALE_FACTOR + row_tile_size + z,
                        grids[t][0] / st.SCALE_FACTOR, grids[t][0] / st.SCALE_FACTOR + col_tile_size,
                        border_color, mode='fill')

        summary_orig.paste(poly, mask=poly)
        # summary_orig.show()
        if not os.path.isdir(result_path + '/heatmap/'): os.makedirs(result_path + '/heatmap/')
        filepath = result_path + '/heatmap/heatmap_' + slidenames[idx].split('/')[-1][:-4] + '.png'
        print('Saving heatmap at {}'.format(filepath))
        summary_orig.save(filepath)

def paste_grid_on_img(pil_img, grids, probs=None, SCALE_FACTOR = 32, plotmode='outline', binary=False, tile_size = 512):
    poly = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(poly)
    
    top_tiles = grids
    
    row_tile_size = round(tile_size / SCALE_FACTOR)  # use round?
    col_tile_size = round(tile_size / SCALE_FACTOR)  # use round?
    
    
    if probs is None:
        print('No probability')
        border_color = (0,255,255, 50)
    elif binary == True:
        print('Binary')       
        thres = 0
        
    else:
        print('Multiple color')
        normalize_probs = (probs - min(probs)) / (max(probs) - min(probs))

        sorted_norm_probs = sorted(normalize_probs, reverse=True)
#         print(sorted_norm_probs)
        top5 = sorted_norm_probs[int(len(normalize_probs) * 0.05)]
        top20 = sorted_norm_probs[int(len(normalize_probs) * 0.2)]
        top50 = sorted_norm_probs[int(len(normalize_probs) * 0.5)]
        print(top5)
        print(top20)
        print(top50)        

    for t in range(len(top_tiles)):
        #     print('Column {}, Row {}'.format(grids[t][0], grids[t][1]))
        if probs is not None and binary is False:
            border_color = tile_border_color(normalize_probs[t], top5, top20, top50)
        elif probs is not None and binary is True:
            border_color = (0,0,255, 50) if probs[t] > thres else (255,255,255, 0)
            
        #     print('Color: {}'.format(border_color))
        #     print(normalize_probs[t])
        
        tiles.tile_border(pdraw, grids[t][1] / SCALE_FACTOR, grids[t][1] / SCALE_FACTOR + row_tile_size,
                    grids[t][0] / SCALE_FACTOR, grids[t][0] / SCALE_FACTOR + col_tile_size,
                    border_color, mode=plotmode)

    pil_img.paste(poly, mask=poly)
    return pil_img

def tile_border_color(tissue_percentage, top5=0.95, top20=0.8, top50=0.5):
    """
    Obtain the corresponding tile border color for a particular tile tissue percentage.

    Args:
    tissue_percentage: The tile tissue percentage

    Returns:
    The tile border color corresponding to the tile tissue percentage.
    """
    if tissue_percentage >= top5:
        border_color = (0,0,255, 50)
    elif (tissue_percentage >= top20) and (tissue_percentage < top5):
        border_color = (0,255,255, 50)
    elif (tissue_percentage >= top50) and (tissue_percentage < top20):
        border_color = (0,255,0, 50)
    elif(tissue_percentage >= 0) and (tissue_percentage < top50):
        border_color = (255,255,0, 50)
    return border_color

def save_inference_results(slidenames, targets, maxs, mpp, slideIDX, grid, probs, path):
    #TODO: add mpp
    infer_result = {
        'slides':slidenames,
        'targets': [int(x) for x in targets],
        'maxs': maxs,
        'mpp' : mpp,
        'slideIDX': slideIDX,
        'grid': grid,
        'probs': probs
    }
    with open(path + '/inference_result.pkl', 'wb') as f:
        pickle.dump(infer_result, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_topk_tiles(dset, probs, path, tp):
    topk = group_argtopk(np.array(dset.slideIDX), probs, tp.mil_k)
    dset.maketraindata(topk)
    if not os.path.isdir(path + '/topk/'): os.makedirs(path + '/topk/')

    for k in tqdm(dset.t_data):
        slideIDX = k[0]
        grid = k[1]
        label = k[2]
        slide = dset.slides[slideIDX]
        tile = read_region(slide, grid, dset.level, dset.size, sp=tp)
        tile_path = path + '/topk/' + '{}_g{}_l{}_s{}_class_{}.png'.format(slide._filename.split('/')[-1][:-4],
                                                           str(grid).replace(', ', '-'),
                                                           str(dset.level), str(dset.size), label
                                                           )
        tile.save(tile_path)
