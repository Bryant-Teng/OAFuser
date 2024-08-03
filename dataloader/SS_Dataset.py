import os
from pickletools import uint8
import cv2
import torch
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.view_path = setting['view_path']
        self.view_path_format = setting['View_path_format']
        self.view_list = setting['view_list']
        
    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]
        rgb_path = os.path.join(self._rgb_path, item_name + self._rgb_format)
        x_path = os.path.join(self._x_path, item_name + self._x_format)
        gt_path = os.path.join(self._gt_path, item_name + self._gt_format)

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)

        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._transform_gt:
            gt = self._gt_transform(gt) 

        if self._x_single_channel:
            x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
            x = cv2.merge([x, x, x])
        else:
            x =  self._open_image(x_path, cv2.COLOR_BGR2RGB)

        List_Img = []        
        List_Img.append(rgb)
        List_Img.append(gt)
        List_Img.append(x)
        
        dir_list=[]
        path = self.view_path
        with open(self.view_list,"r") as f:
            c =  f.readlines()
            for dir in c:
                dir = dir.strip('\n')
                dir_list.append(dir)
                path1 = os.path.join(path,dir)
                file_path = os.path.join(path1,item_name)
                file_path = file_path + self.view_path_format
                view = self._open_image(file_path, cv2.COLOR_BGR2RGB)
                List_Img.append(view)
        if self.preprocess is not None:
           List_Img = self.preprocess(List_Img)    #data,Label,model_x,view   
        if self._split_name == 'train':
           List_Img[0] = torch.from_numpy(np.ascontiguousarray(List_Img[0])).float()
           List_Img[1] = torch.from_numpy(np.ascontiguousarray(List_Img[1])).long()
           for k in range(len(List_Img)-2):
               List_Img[k+2] = torch.from_numpy(np.ascontiguousarray(List_Img[k+2])).float()
        Key_List =[]
        Key_List.append('data')
        Key_List.append('label')
        Key_List.append('modal_x')
        Key_List = Key_List + dir_list        
        Key_List.append('fn')
        Key_List.append('n')
        List_Img.append(str(item_name))
        List_Img.append(len(self._file_names))        
        output_dict = dict(zip(Key_List,List_Img))
        #data,Label,model_x,view,fn,n
        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):  # length是 config.batch_size * config.niters_per_epoch
        assert isinstance(length, int)  # length must be an integer
        ## 重复采样
        files_len = len(self._file_names)  ##_file_names是txt文件里面存储的文件编好，这个代码是看一共有多少个文件编好
        ## 重复采样                        
        new_file_names = self._file_names * (length // files_len)   ##原来的文件编号*（需要训练的图片数量/原来文件个数，并且下取整）
        #print('------new_file_names  1------')
        #print(new_file_names)
        ## 随机采样
        rand_indices = torch.randperm(files_len).tolist()   ###将0~n-1（包括0和n-1）随机打乱后获得的数字序列
        new_indices = rand_indices[:length % files_len]     ###取余数个数的随机数

        new_file_names += [self._file_names[i] for i in new_indices]   ###将随机数对应的文件编号加入到new_file_names中
        
        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
