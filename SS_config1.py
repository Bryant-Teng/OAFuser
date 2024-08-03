import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse
from SS_config_2 import Y

C = edict()
config = C
cfg = C


C.seed = 3407

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'NYUDepthv2'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'NYUDepthv2')

#Data_RGB
C.rgb_root_folder = Y.rgb_root_folder
C.rgb_format = '.png'

#Data_View
C.rgb_view_path = Y.rgb_view_path
C.rgb_view_format = '.png'
C.view_list = Y.view_list
####
#Data_Label
C.gt_root_folder = Y.gt_root_folder
C.gt_format = '.png'
C.gt_transform = Y.gt_transform
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
#Data_X
C.x_root_folder = Y.x_root_folder
C.x_format = '.png'
C.x_is_single_channel = Y.x_is_single_channel
C.train_source = Y.train_source
C.eval_source = Y.eval_source

C.is_test = False
C.num_train_imgs = Y.num_train_imgs
C.num_eval_imgs = Y.num_eval_imgs
C.num_classes = 14
C.class_names =  ['bike','building','fence','others','person','pole','road','sidewalk','traffic sign','vegetation','vehicle','bridge','rider',
    'sky']

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = Y.backbone # Remember change the path below.
C.pretrained_model = Y.pretrained_model

C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01

C.batch_size = Y.batch_size

C.nepochs = Y.nepochs
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1

##config.batch_size * config.niters_per_epoch  = config.num_train_imgs 
##所以当txt文件中文件编号，小于num_train_imgs时，会重复读取txt文件中的文件编号，直到达到num_train_imgs
##所以当txt文件中文件编号，大于num_train_imgs时，会随机读取txt文件中的文件编号，直到达到num_train_imgs

C.num_workers = Y.num_workers
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] # 
C.eval_flip = False # True # 
C.eval_crop_size = [480, 640] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 100
C.checkpoint_step = 2

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_' + C.dataset_name + '_' + C.backbone)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()