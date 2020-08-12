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
        self.mode = 'GCP'

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

class TrainParam(SlideParam):
    def __init__(self):
        super(TrainParam, self).__init__()
        self.train_lib = self.train_lib

        self.mil_batch_size    = 128
        self.mil_nepochs       = 1000

        self.mil_workers       = 0
        self.mil_test_every    = 2
        self.mil_weights       = self.get_weights_ratio()

        self.mil_k             = 20
        self.mil_lr            = 1e-4
        self.mil_wdecays       = 1e-4
        self.thres_prediction  = 0.98
        self.inference_size    = 200

        self.rnn_lr            = 10e-4
        self.rnn_batch_size    = 2
        self.rnn_nepochs       = 1000
        self.rnn_workers       = 4
        self.rnn_s             = 9
        self.rnn_ndims         = 128
        self.rnn_weights       = 0.7
        self.rnn_shuffle       = True
        
    def get_weights_ratio(self):
        lib_train = torch.load(self.train_lib)
        ratio = lib_train['targets'].value_counts()[0]/(len(lib_train['targets']))
        return ratio
        
class FeatureExtractionParam(SlideParam):
    def __init__(self):
        super(FeatureExtractionParam, self).__init__()
        self.model_type = 'ResNet34'  # ResNet34, AE
        if self.mode == 'GCP':
            self.feature_dir =  '/home/eljzn_bayer_com/datasets/features_dir_{}_{}'.format(self.TILE_SIZE, self.model_type)
        else:
            self.feature_dir = self.BASE_DIR + '/features_dir_{}_{}'.format(self.TILE_SIZE, self.model_type)
        self.label_pkl = self.file_label_proc_w_grid
        self.feature_dim = 1024
        self.batch_size = 128

class SplitParam(FeatureExtractionParam):
    def __init__(self):
        super(SplitParam, self).__init__()
        if self.mode == 'GCP':
            self.split_dir = '/home/eljzn_bayer_com/datasets/split_dir' + '/{}_{}'.format(self.exp, self.TILE_SIZE)  #
        else:
            self.split_dir = self.BASE_DIR + '/split_dir_{}_{}'.format(self.TILE_SIZE, self.model_type)
        self.seed = 21
        self.file_label_proc = self.file_label_proc
        self.label_fracs = 1
        self.label_dict = {0.0: 0, 1.0: 1}
        self.k = 5

class TrainAttParam(SplitParam):
    def __init__(self):
        super(TrainAttParam, self).__init__()
        # PATH
        self.data_root_dir = self.feature_dir           # get from feature extraction class
        self.results_dir   = self.mil_output_logs       # default result dir
        self.split_dir     = self.split_dir             # defined in splitparam class

        # TRAINING
        self.n_classes     = 2
        self.label_dict    = self.label_dict            # defined in splitparam class
        self.encoding_size = 1000                       # resnet 34

        self.max_epochs    = 40
        self.lr            = 1e-5

        self.label_frac    = 1
        self.bag_weight    = 0.8                        # ratio bag loss vs instance loss
        self.reg           = 1e-5
        self.seed          = self.seed                  # defined in splitparam class
        self.k             = self.k                     # k-fold cross validation, adapt from split
        self.k_start       = -1
        self.k_end         = -1
        self.early_stopping = False
        self.opt            = 'adam'  #sgd
        self.drop_out       = True
        self.inst_loss      = None # svm, ce None
        self.bag_loss       = 'ce'  # svm
        self.model_type     = 'clam'  # mil clam

        self.weighted_sample = True
        self.model_size      = 'small'                       # clam:small big  mil: A B

        # LOGGING
        self.log_data  = True
        self.testing   = False
        self.subtyping = False

class TestAttParam(TrainAttParam):
    def __init__(self):
        super(TestAttParam, self).__init__()
        # PATH
        self.data_root_dir = self.feature_dir                                # get from feature extraction class
        self.results_dir   = self.mil_output_logs                            # default result dir
        self.split_dir     = self.split_dir                                  # defined in splitparam class
<<<<<<< Updated upstream
        self.models_dir    = self.results_dir + '/{}/2020-06-22-11-02-04_train_clam_{}_{}'.format(self.exp, self.exp, self.TILE_SIZE)
=======
        self.models_dir    = self.results_dir + '/L_necrosis_TG-2186/2020-06-17-13-27-45_train_clam_L_necrosis_TG-2186_512'.format(self.exp, self.exp, self.TILE_SIZE)
>>>>>>> Stashed changes

        self.exp_code        = ''
        self.model_size      = self.model_size                               # get from TrainAttParam
        self.model_type      = self.model_type
        self.drop_out        = self.drop_out
        self.k               = self.k                                        # k-fold cross validation
        self.k_start         = self.k_start
        self.k_end           = self.k_end
        self.fold            = 0                                           # select fold to evaluate, if -1 run all fold
        self.micro_average   = False
        self.split           = 'test'                                        # 'train', 'val', 'test', 'all'
        self.encoding_size = self.encoding_size

class TrainBBParam(SlideParam):
    def __init__(self):
        super(TrainBBParam, self).__init__()
        self.seed = 21
        
        if self.mode == 'GCP':
            self.TILE_DIR = '/home/eljzn_bayer_com/datasets/tiles_all'
        else:
            self.TILE_DIR = self.TILE_DIR
            

        self.model_type  = 'inpainting' # 'ae',  'unet', 'rot', inpainting


        date = '2020-07-16-20-09-02'
        # self.model_path = self.mil_output_logs + '/backbone/{}_train_{}/{}_epoch_29_backbone_{}.pkl'.format(date, self.model_type, date, self.model_type)
        self.model_path = self.mil_output_logs + '/backbone/{}_train_{}/{}_netG_epoch_220_{}_0.08.pth'.format(date,
                                                                                                            self.model_type,
                                                                                                            date,
                                                                                                            self.model_type)
        self.model_path = None

        self.valid_every = 1
        self.lr = 1e-4

        self.train_batchsize = 32
        self.val_batchsize = 32
        self.num_workers = 8
        self.niter = 3000

        # inpainting
        self.netD_path = self.mil_output_logs + '/backbone/{}_train_{}/{}_netD_epoch_14_{}_5.78.pkl'.format(date,
                                                                                                            self.model_type,
                                                                                                            date,
                                                                                                            self.model_type)
        self.netG_path = self.mil_output_logs + '/backbone/{}_train_{}/{}_netG_epoch_14_{}_0.15.pkl'.format(date,
                                                                                                            self.model_type,
                                                                                                            date,
                                                                                                            self.model_type)
        
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


