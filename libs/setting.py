import os
from datetime import datetime
import pandas as pd
import torch


class TileParam():
    def __init__(self):
        self.TISSUE_HIGH_THRESH = 80
        self.TISSUE_LOW_THRESH = 10

        self.TILE_SIZE = 512
        self.ROW_TILE_SIZE = 512
        self.COL_TILE_SIZE = 512
        self.NUM_TOP_TILES = 50

        self.DISPLAY_TILE_SUMMARY_LABELS = False
        self.TILE_LABEL_TEXT_SIZE = 10
        self.LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = False
        self.BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = False

        self.TILE_BORDER_SIZE = 2  # The size of the colored rectangular border around summary tiles.

        self.HIGH_COLOR = (0, 255, 0)
        self.MEDIUM_COLOR = (255, 255, 0)
        self.LOW_COLOR = (255, 165, 0)
        self.NONE_COLOR = (255, 0, 0)

        self.FADED_THRESH_COLOR = (128, 255, 128)
        self.FADED_MEDIUM_COLOR = (255, 255, 128)
        self.FADED_LOW_COLOR = (255, 210, 128)
        self.FADED_NONE_COLOR = (255, 128, 128)

        self.FONT_PATH = "FreeMono.ttf"
        self.SUMMARY_TITLE_FONT_PATH = "FreeMono.ttf"

        self.SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
        self.SUMMARY_TITLE_TEXT_SIZE = 24
        self.SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
        self.TILE_TEXT_COLOR = (0, 0, 0)
        self.TILE_TEXT_SIZE = 36
        self.TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
        self.TILE_TEXT_W_BORDER = 5
        self.TILE_TEXT_H_BORDER = 4

        self.HSV_PURPLE = 270
        self.HSV_PINK = 330

class SlideParam(TileParam):
    def __init__(self, verbose=False):
        super(SlideParam, self).__init__()
        self.mode = 'HOMEPC'

        if self.mode == 'GCP':
            self.dataset = 'PH'
            self.exp = 'L_BL' #   'T_colloid_2186' 'T_hyper_2186' 'L_necrosis_2186' 'L_necrosis_TG' 'L_necrosis_TG-2186' 'L_BL'
            
            if self.dataset == 'CS':
                self.CODE_DIR      = "/home/eljzn_bayer_com/code/MILPath/"
                self.BASE_DIR      = "/home/eljzn_bayer_com/datasets/PD/"
                self.SRC_TRAIN_DIR = os.path.join(self.BASE_DIR, "upload-preprocessed/CS")    # directory of svs/tif
                self.TEMP_DIR      = os.path.join('/home/eljzn_bayer_com/datasets', "temp_pd") #os.path.join(self.BASE_DIR, "temp")
                self.label_file    = self.SRC_TRAIN_DIR + '/labels_o_{}.csv'.format(self.exp)
                
                self.SRC_TRAIN_EXT = "tif"
                
            elif self.dataset == 'PH':
                self.CODE_DIR      = "/home/eljzn_bayer_com/code/MILPath/"
                self.BASE_DIR      = "/home/eljzn_bayer_com/datasets/PD/"
                self.SRC_TRAIN_DIR = os.path.join(self.BASE_DIR, "upload-preprocessed/PH")    # directory of svs/tif
                self.TEMP_DIR      = os.path.join('/home/eljzn_bayer_com/datasets', "temp_pd") #os.path.join(self.BASE_DIR, "temp")
                self.label_file    = self.SRC_TRAIN_DIR + '/labels_o_{}.csv'.format(self.exp)
                
                self.SRC_TRAIN_EXT = "tif"
                
            elif self.dataset == 'TG':
                self.CODE_DIR      = "/home/eljzn_bayer_com/code/MILPath/"
                self.BASE_DIR      = "/home/eljzn_bayer_com/datasets/TG/"
                self.SRC_TRAIN_DIR = os.path.join(self.BASE_DIR, "data/open-tg-gates/images/")    # directory of svs/tif
                self.TEMP_DIR      = os.path.join('/home/eljzn_bayer_com/datasets', "temp_tg")
                self.label_file    = self.BASE_DIR + '/labels_o_L_necrosis_TG.csv'
                
                self.SRC_TRAIN_EXT = "svs"

        elif self.mode == 'HOMEPC':
            self.dataset = 'PD'
            self.exp = 'small'  # 'T_colloid_2186' 'T_hyper_2186' 'L_necrosis_2186' 'L_necrosis_TG'
            self.CODE_DIR = "/home/vankhoa/code/Bayer/MILPath/"
            self.BASE_DIR = "/home/vankhoa/datasets/ImgPath/PD/CS_small"          # main directory of images
            self.SRC_TRAIN_DIR = os.path.join(self.BASE_DIR, "")                  # directory of svs/tif
            self.TEMP_DIR = "/home/vankhoa/datasets/ImgPath/temp_small/"
            self.label_file = self.SRC_TRAIN_DIR + 'labels_o_T_hyper_2186_wgrade.csv'
            self.SRC_TRAIN_EXT = "tif"

        elif self.mode == 'MARCO_DGX':
            self.dataset = 'PD'
            self.exp = 'small'  # 'T_colloid_2186' 'T_hyper_2186' 'L_necrosis_2186' 'L_necrosis_TG'
            self.CODE_DIR = "/home/glsvu/MILPath/"
            self.BASE_DIR = "/home/glsvu/datasets/ImgPath/PD/CS_small"          # main directory of images
            self.SRC_TRAIN_DIR = os.path.join(self.BASE_DIR, "")                # directory of svs/tif
            self.TEMP_DIR = "/home/glsvu/datasets/ImgPath/temp_small/"
            self.label_file = self.SRC_TRAIN_DIR + 'labels_o_T_hyper_2186_wgrade.csv'
            self.SRC_TRAIN_EXT = "svs"

        elif self.mode == 'LAPTOP':
            self.CODE_DIR      = "C:\\Bayer/2_code/MILPath/"
            self.BASE_DIR      = "C:\\Bayer/1_data/realData/uploadData"  # main directory of images
            self.SRC_TRAIN_DIR = os.path.join(self.BASE_DIR, "")  # directory of svs/tif
            self.TEMP_DIR      = "C:\\Bayer/1_data/realData/temp/"

        self.tiff_lib = 'openslide'  # openslide/rasterio/pyvips

        self.TRAIN_PREFIX  = "PD-"

        self.DEST_TRAIN_SUFFIX = ""  # Example: "train-"
        self.DEST_TRAIN_EXT    = "png"

        self.SCALE_FACTOR   = 32
        self.THUMBNAIL_SIZE = 200
        self.THUMBNAIL_EXT  = "jpg"

        self.DEST_TRAIN_THUMBNAIL_DIR = os.path.join(self.TEMP_DIR,
                                       "training_thumbnail_" + self.THUMBNAIL_EXT)

        self.FILTER_SUFFIX          = ""  # Example: "filter-"
        self.FILTER_RESULT_TEXT     = "filtered"
        self.FILTER_PAGINATION_SIZE = 50
        self.FILTER_PAGINATE        = True
        self.FILTER_HTML_DIR        = self.TEMP_DIR

        self.TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(self.TEMP_DIR,
                                                    "tile_summary_on_original_" + self.DEST_TRAIN_EXT)
        self.TILE_SUMMARY_SUFFIX          = "tile_summary"
        self.TILE_SUMMARY_THUMBNAIL_DIR   = os.path.join(self.TEMP_DIR,
                                                    "tile_summary_thumbnail_" + self.THUMBNAIL_EXT)
        self.TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(self.TEMP_DIR,
                                             "tile_summary_on_original_thumbnail_" + self.THUMBNAIL_EXT)
        self.TILE_SUMMARY_PAGINATION_SIZE = 50
        self.TILE_SUMMARY_PAGINATE        = True
        self.TILE_SUMMARY_HTML_DIR        = self.TEMP_DIR

        self.TILE_DATA_SUFFIX = "tile_data"

        self.TOP_TILES_SUFFIX = "top_tile_summary"
        self.TOP_TILES_DIR           = os.path.join(self.TEMP_DIR, self.TOP_TILES_SUFFIX + "_" + self.DEST_TRAIN_EXT)
        self.TOP_TILES_THUMBNAIL_DIR = os.path.join(self.TEMP_DIR,
                                       self.TOP_TILES_SUFFIX + "_thumbnail_" + self.THUMBNAIL_EXT)
        self.TOP_TILES_ON_ORIGINAL_DIR = os.path.join(self.TEMP_DIR,
                                         self.TOP_TILES_SUFFIX + "_on_original_" + self.DEST_TRAIN_EXT)
        self.TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(self.TEMP_DIR,
                                     self.TOP_TILES_SUFFIX + "_on_original_thumbnail_" + self.THUMBNAIL_EXT)

        self.TILE_SUFFIX = "tile"

        self.DEST_TRAIN_DIR       = os.path.join(self.TEMP_DIR, "training_" + self.DEST_TRAIN_EXT) # dir contains reduced size png images (32x)
        self.FILTER_DIR           = os.path.join(self.TEMP_DIR, "filter_" + self.DEST_TRAIN_EXT) # dir filtered png
        self.FILTER_THUMBNAIL_DIR = os.path.join(self.TEMP_DIR, "filter_thumbnail_" + self.THUMBNAIL_EXT)
        self.TILE_SUMMARY_DIR     = os.path.join(self.TEMP_DIR, "tile_summary_" + self.DEST_TRAIN_EXT)
        self.TILE_DATA_DIR        = os.path.join(self.TEMP_DIR, "tile_data")

        self.TILE_DIR     = os.path.join(self.TEMP_DIR, "tiles_" + self.DEST_TRAIN_EXT)
        self.STATS_DIR    = os.path.join(self.TEMP_DIR, "svs_stats")
        self.MAPPING_DIR  = os.path.join(self.CODE_DIR, "mapping")

        self.file_mapping           = self.MAPPING_DIR + '/mapping_{}.json'.format(self.dataset)
        self.file_label_proc        = self.MAPPING_DIR + '/file_label_{}_{}.csv'.format(self.exp, self.TILE_SIZE)
        self.file_label_proc_w_grid = self.MAPPING_DIR + '/file_label_{}_w_grid_{}.pkl'.format(self.exp, self.TILE_SIZE)

        # this lib files are used for conventional MIL code
        self.train_lib    = self.MAPPING_DIR + '/lib_train_{}.json'.format(self.exp)
        self.val_lib      = self.MAPPING_DIR + '/lib_val_{}.json'.format(self.exp)
        self.test_lib     = self.MAPPING_DIR + '/lib_test_{}.json'.format(self.exp)

        self.save_tiles = False     # save tiles to image flag

        self.train_on_all = 0.8     # used to split train/val/test in preprocess.py

        self.mil_output_models = os.path.join(self.CODE_DIR, "models")
        self.mil_output_logs   = os.path.join(self.CODE_DIR, "results")
        self.mil_output_imgs   = os.path.join(self.TEMP_DIR, "tiles")

        if not os.path.isdir(self.mil_output_imgs)  : os.makedirs(self.mil_output_imgs)
        if not os.path.isdir(self.mil_output_logs)  : os.makedirs(self.mil_output_logs)
        if not os.path.isdir(self.mil_output_models): os.makedirs(self.mil_output_models)

        self.rnn_output_models = os.path.join(self.CODE_DIR, "results")
        self.rnn_output_logs   = os.path.join(self.CODE_DIR, "results")

        if not os.path.exists(self.MAPPING_DIR): os.makedirs(self.MAPPING_DIR)
        if not os.path.exists(self.TEMP_DIR)   : os.makedirs(self.TEMP_DIR)

        if verbose:
            self.summary()

    def summary(self):
        print('Base code directory : {}'.format(self.CODE_DIR))
        print('Base data directory : {}'.format(self.BASE_DIR))
        print('Image directory     : {}'.format(self.SRC_TRAIN_DIR))
        print('Mapping directory   : {}'.format(self.MAPPING_DIR))
        print('Temp directory ()   : {}'.format(self.TEMP_DIR))
        print('Label file original : {}'.format(self.label_file))
        print('Label file processed: {}'.format(self.file_label_proc))

        input('Verify the paths and press any key..')

class TrainBBParam(SlideParam):
    def __init__(self):
        super(TrainBBParam, self).__init__()
        self.seed = 21
        
        if self.mode == 'GCP':
            self.TILE_DIR = '/home/eljzn_bayer_com/datasets/tiles_all'
        else:
            self.TILE_DIR = self.TILE_DIR
            

        self.model_type  = 'unet' # 'ae',  'unet', 'rot', inpainting outpainting


        date = '2020-07-16-20-09-02'
        # self.model_path = self.mil_output_logs + '/backbone/{}_train_{}/{}_epoch_29_backbone_{}.pkl'.format(date, self.model_type, date, self.model_type)
        # self.model_path = self.mil_output_logs + '/backbone/{}_train_{}/{}_netG_epoch_220_{}_0.08.pth'.format(date,
        #                                                                                                     self.model_type,
        #                                                                                                     date,
        #                                                                                                     self.model_type)
        self.model_path = None

        # self.model_path = '/home/vankhoa/code/selfsvd/image-outpainting/outpaint_models/G_96.pt'
        self.valid_every = 1
        self.lr = 1e-4

        self.train_batchsize = 8
        self.val_batchsize = 8
        self.num_workers = 8
        self.niter = 3000


class Train_inpainting(SlideParam):
    def __init__(self):
        super(Train_inpainting, self).__init__()
        self.seed = 21

        if self.mode == 'GCP':
            self.TILE_DIR = '/home/eljzn_bayer_com/datasets/tiles_all'
        else:
            self.TILE_DIR = self.TILE_DIR
        self.model_type = 'inpainting'
        self.valid_every = 1
        self.lr = 1e-4

        self.train_batchsize = 8
        self.val_batchsize = 8
        self.num_workers = 8
        self.niter = 3000

        # inpainting
        # self.netD_path = self.mil_output_logs + '/backbone/{}_train_{}/{}_netD_epoch_14_{}_5.78.pkl'.format(date,
        #                                                                                                     self.model_type,
        #                                                                                                     date,
        #                                                                                                     self.model_type)
        # self.netG_path = self.mil_output_logs + '/backbone/{}_train_{}/{}_netG_epoch_14_{}_0.15.pkl'.format(date,
        #                                                                                                     self.model_type,
        #                                                                                                     date,
        #                                                                                                     self.model_type)
        self.netD_path = None
        self.netG_path = None
        self.ngpu = 1

        self.imageSize = 512

        self.ngf = 64
        self.ndf = 64
        self.nc = 3
        self.beta1 = 0.5
        self.netG = ''
        self.netD = ''
        self.nBottleneck = 512
        self.overlapPred = 4
        self.nef = 64
        self.wtl2 = 0.998
        self.wtlD = 0.001
        self.overlapL2Weight = 10


class Train_outpainting(SlideParam):
    def __init__(self):
        super(Train_outpainting, self).__init__()
        self.seed = 21

        if self.mode == 'GCP':
            self.TILE_DIR = '/home/eljzn_bayer_com/datasets/tiles_all'
        else:
            self.TILE_DIR = self.TILE_DIR

        self.model_type = 'outpainting'  # 'ae',  'unet', 'rot', inpainting outpainting

        date = '2020-07-16-20-09-02'

        self.model_path = None

        self.input_size = 128
        self.output_size = 192
        self.expand_size = (self.output_size - self.input_size) // 2
        self.patch_w = self.output_size // 8
        self.patch_h = self.output_size // 8
        self.patch = (1, self.patch_h, self.patch_w)

        # self.model_path = '/home/vankhoa/code/selfsvd/image-outpainting/outpaint_models/G_96.pt'
        self.valid_every = 1
        self.lr = 1e-4

        self.train_batchsize = 8
        self.val_batchsize = 8
        self.num_workers = 8
        self.niter = 3000


