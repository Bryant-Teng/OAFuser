from easydict import EasyDict as edict

Y = edict()

Y.rgb_root_folder = '/root/autodl-tmp/syn/Syn/View/V55'

#Data_View
Y.rgb_view_path = "/root/autodl-tmp/syn/Syn/View"
Y.view_list = "/root/autodl-tmp/SS/syn8/viewlist.txt"

#Data_Label
Y.gt_root_folder = '/root/autodl-tmp/syn/Syn/Label' 
Y.gt_transform = True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
#Data_X
Y.x_root_folder = '/root/autodl-tmp/syn/Syn/View/V11' 
Y.x_is_single_channel = False # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input

Y.train_source = '/root/autodl-tmp/syn/Train_list.txt'
Y.eval_source = '/root/autodl-tmp/syn/Val_list.txt'

Y.num_train_imgs = 172
Y.num_eval_imgs = 28

Y.backbone = 'mit_b0' # Remember change the path below.
Y.pretrained_model = r'/root/autodl-tmp/SS/syn8/mit_b0.pth'

Y.batch_size = 9
Y.nepochs = 900
Y.num_workers = 8