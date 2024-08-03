import cv2
import torch
import numpy as np
from torch.utils import data
import random
from SS_config1 import config
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(List_Img):

    if random.random() >= 0.5:
        List_Img = [cv2.flip(item,1) for item in List_Img]
    return List_Img

def random_scale(List_Img,scales):

    scale = random.choice(scales)

    sh = int(List_Img[0].shape[0] * scale)
    sw = int(List_Img[0].shape[1] * scale)
    List_Img = [cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR) for img in List_Img]

    return List_Img, scale

class TrainPre(object):
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self,List_Img):
        Image_Number = len(List_Img)
        List_Img = random_mirror(List_Img)
        if config.train_scale_array is not None:  #none
            List_Img , scale = random_scale(List_Img, config.train_scale_array)
        
        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(List_Img[0].shape[:2], crop_size)
        
        #rgb
        List_Img[0] = normalize(List_Img[0], self.norm_mean, self.norm_std)
        List_Img[0],_ =  random_crop_pad_to_shape(List_Img[0], crop_pos, crop_size, 0)
        List_Img[0] = List_Img[0].transpose(2, 0, 1)       
       
        ####x,view
        for k in range(Image_Number-2):
           List_Img[k+2] = normalize(List_Img[2+k], self.norm_mean, self.norm_std)
           List_Img[k+2],_ =  random_crop_pad_to_shape(List_Img[k+2], crop_pos, crop_size, 0)
           List_Img[k+2] = List_Img[k+2].transpose(2, 0, 1)
        #####gt
        List_Img[1],_ =  random_crop_pad_to_shape(List_Img[1], crop_pos, crop_size, 255)

        return List_Img

class ValPre(object):
    def __call__(self,List_Img):
       return List_Img

def get_train_loader(engine, dataset):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names,
                   'view_path':config.rgb_view_path,
                   'View_path_format':config.rgb_view_format,
                   'view_list':config.view_list}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=False,
                                   sampler=train_sampler)

    return train_loader, train_sampler