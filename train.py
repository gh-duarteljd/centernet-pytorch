#-------------------------------------#
# train on dataset
#-------------------------------------#
import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
from nets.centernet_training import get_lr_scheduler, set_optimizer_lr
from utils.callbacks import LossHistory
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import CenternetDataset, centernet_dataset_collate
from utils.utils import download_weights, get_classes, show_config
from utils.utils_fit import fit_one_epoch

'''
To train your own target detection model, you must pay attention to the following points:
1. Carefully check whether your format meets the requirements before training. The library requires the data set format to be in VOC format. The content that needs to be prepared includes input pictures and labels
    The input image is a .jpg image, and there is no need to fix the size. It will be automatically resized before being passed in for training.
    The grayscale image will be automatically converted into RGB image for training, no need to modify it yourself.
    If the suffix of the input image is not jpg, you need to convert it into jpg in batches before starting the training.

    The label is in .xml format, and there will be target information to be detected in the file, and the label file corresponds to the input image file.

2. The size of the loss value is used to judge whether it is converged. The more important thing is that there is a trend of convergence, that is, the loss of the verification set keeps decreasing. If the loss of the verification set basically does not change, the model basically converges.
    The specific size of the loss value is meaningless. The big and small values only depend on the calculation method of the loss, and it is not good to be close to 0. If you want to make the loss look better, you can directly add 10000 to the corresponding loss function.
    The loss value during training will be saved in the loss_%Y_%m_%d_%H_%M_%S folder under the logs folder
   
3. The trained weight file is saved in the logs folder. Each training generation (Epoch) contains several training steps (Step), and each training step (Step) performs a gradient descent.
    If you only train a few steps, it will not be saved. The concepts of Epoch and Step need to be clarified.
'''
if __name__ == "__main__":
     #------------------------------------#
     # Cuda Whether to use Cuda
     # No GPU can be set to False
     #------------------------------------#
     Cuda = True
     #------------------------------------------------- --------------------#
     # distributed is used to specify whether to use single-machine multi-card distributed operation
     # Terminal commands only support Ubuntu. CUDA_VISIBLE_DEVICES is used to specify the graphics card under Ubuntu.
     # Under the Windows system, the DP mode is used to call all graphics cards by default, and DDP is not supported.
     # DP mode:
     # set distributed = False
     # Enter CUDA_VISIBLE_DEVICES=0,1 in the terminal python train.py
     # DDP mode:
     # set distributed = True
     # Enter CUDA_VISIBLE_DEVICES=0,1 in the terminal python -m torch.distributed.launch --nproc_per_node=2 train.py
     #------------------------------------------------- --------------------#
     distributed = False
     #------------------------------------------------- --------------------#
     # sync_bn Whether to use sync_bn, DDP mode multi-card available
     #------------------------------------------------- --------------------#
     sync_bn = False
     #------------------------------------------------- --------------------#
     # Whether fp16 uses mixed precision training
     # It can reduce the video memory by about half, and requires pytorch1.7.1 or above
     #------------------------------------------------- --------------------#
     fp16 = False
     #------------------------------------------------- --------------------#
     # classes_path points to the txt under model_data, which is related to the dataset you trained
     # Be sure to modify the classes_path before training to make it correspond to your own data set
     #------------------------------------------------- --------------------#
     classes_path = 'model_data/voc_classes.txt'
     #------------------------------------------------- -------------------------------------------------- --------------------------#
     # For downloading the weight file, please refer to README, which can be downloaded through the network disk. The pre-trained weights of the model are common to different datasets because the features are common.
     # The pre-training weight of the model The more important part is the weight part of the backbone feature extraction network, which is used for feature extraction.
     # Pre-training weights must be used in 99% of cases. If not, the weights of the main part are too random, the effect of feature extraction is not obvious, and the result of network training will not be good
     #
     # If there is an operation to interrupt the training during the training process, you can set the model_path to the weight file under the logs folder, and load a part of the weights that have been trained again.
     # At the same time, modify the parameters of the freezing phase or unfreezing phase below to ensure the continuity of the model epoch.
     #
     # When model_path = '', do not load the weights of the entire model.
     #
     # The weight of the entire model is used here, so it is loaded in train.py, and pretrain does not affect the weight loading here.
     # If you want the model to start training from the pre-trained weights of the backbone, set model_path = '', pretrain = True, and only load the backbone at this time.
     # If you want the model to start training from 0, set model_path = '', pretrain = Fasle, Freeze_Train = Fasle, at this time start training from 0, and there is no process of freezing the backbone.
     #
     # Generally speaking, the training effect of the network starting from 0 will be very poor, because the weights are too random, and the feature extraction effect is not obvious, so it is very, very, very not recommended to start training from 0!
     # If you must start from 0, you can understand the imagenet data set, first train the classification model, obtain the weight of the backbone part of the network, the backbone part of the classification model is common to the model, and train based on this.
     #------------------------------------------------- -------------------------------------------------- --------------------------#
     model_path = 'model_data/centernet_resnet50_voc.pth'
     #------------------------------------------------- -----#
     # input_shape input shape size, a multiple of 32
     #------------------------------------------------- -----#
     input_shape = [512, 512]
     #-------------------------------------------#
     #backbone backbone feature extraction network selection
     # resnet50 and hourglass
     #-------------------------------------------#
     backbone = "resnet50"
     #------------------------------------------------- -------------------------------------------------- --------------------------#
     # pretrained Whether to use the pre-trained weight of the backbone network, here is the weight of the backbone, so it is loaded when the model is built.
     # If model_path is set, the weight of the backbone does not need to be loaded, and the value of pretrained is meaningless.
     # If model_path is not set, pretrained = True, only the backbone is loaded to start training at this time.
     # If model_path is not set, pretrained = False, Freeze_Train = Fasle, the training starts from 0 at this time, and there is no process of freezing the backbone.
     #------------------------------------------------- -------------------------------------------------- --------------------------#
     pretrained = False
    
     #-------------------------------------------------------------------- -------------------------------------------------- ------#
     # The training is divided into two phases, namely the freezing phase and the unfreezing phase. The freezing stage is set to meet the training needs of students with insufficient machine performance.
     # Freeze training requires less video memory, and if the graphics card is very poor, you can set Freeze_Epoch equal to UnFreeze_Epoch, and only perform freeze training at this time.
     #
     # Here are some suggestions for setting parameters, and trainers can adjust them flexibly according to their own needs:
     # (1) Start training from the pre-trained weights of the entire model:
     #Adam:
     # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (freeze)
     # Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (do not freeze)
     # SGD:
     # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 200, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 4e-5. (freeze)
     # Init_Epoch = 0, UnFreeze_Epoch = 200, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 4e-5. (do not freeze)
     # Among them: UnFreeze_Epoch can be adjusted between 100-300.
     # (2) Start training from the pre-trained weights of the backbone network:
     #Adam:
     # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (freeze)
     # Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 5e-4, weight_decay = 0. (do not freeze)
     # SGD:
     # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 200, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 4e-5. (freeze)
     # Init_Epoch = 0, UnFreeze_Epoch = 200, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 4e-5. (do not freeze)
     # Among them: Since the training starts from the pre-trained weights of the backbone network, the weights of the backbone may not be suitable for target detection, and more training is required to jump out of the local optimal solution.
     # UnFreeze_Epoch can be adjusted between 200-300, and 300 is recommended for both YOLOV5 and YOLOX.
     # Adam converges faster than SGD. Therefore, UnFreeze_Epoch can theoretically be smaller, but more Epochs are still recommended.
     # (3) Setting of batch_size:
     # Within the acceptable range of the graphics card, it is better to be large. Insufficient video memory has nothing to do with the size of the data set. If the video memory is insufficient (OOM or CUDA out of memory), please reduce the batch_size.
     # Affected by the BatchNorm layer, the minimum batch_size is 2 and cannot be 1.
     # Under normal circumstances, Freeze_batch_size is recommended to be 1-2 times of Unfreeze_batch_size. It is not recommended to set the gap too large, because it is related to the automatic adjustment of the learning rate.
     #------------------------------------------------- -------------------------------------------------- --------------------------#
     #------------------------------------------------- -----------------#
     # freeze phase training parameters
     # At this time, the backbone of the model is frozen, and the feature extraction network does not change
     # The video memory occupied is small, only fine-tuning the network
     # Init_Epoch The current training generation of the model, its value can be greater than Freeze_Epoch, such as setting:
     # Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
     # will skip the freezing stage, start directly from the 60th generation, and adjust the corresponding learning rate.
     # (used when resuming training from breakpoints)
     # Freeze_Epoch model freezes Freeze_Epoch for training
     # (disabled when Freeze_Train=False)
     # Freeze_batch_size model freezes the batch_size of training
     # (disabled when Freeze_Train=False)
     #------------------------------------------------- -----------------#
     Init_Epoch = 0
     Freeze_Epoch = 50
     Freeze_batch_size = 16
     #------------------------------------------------- -----------------#
     # Unfreezing phase training parameters
     # At this time, the backbone of the model is not frozen, and the feature extraction network will change
     # Occupies a large amount of video memory, and all parameters of the network will change
     # UnFreeze_Epoch model total training epoch
     # SGD takes longer to converge, so set a larger UnFreeze_Epoch
     # Adam can use relatively small UnFreeze_Epoch
     # Unfreeze_batch_size The batch_size of the model after unfreezing
     #------------------------------------------------- -----------------#
     UnFreeze_Epoch = 100
     Unfreeze_batch_size = 8
     #------------------------------------------------- -----------------#
     # Freeze_Train whether to perform frozen training
     # By default, the backbone training is frozen first and then the training is unfreezed.
     #------------------------------------------------- -----------------#
     Freeze_Train = True
    
     #------------------------------------------------- -----------------#
     # Other training parameters: related to learning rate, optimizer, and learning rate drop
     #------------------------------------------------- -----------------#
     #------------------------------------------------- -----------------#
     # Init_lr The maximum learning rate of the model
     # It is recommended to set Init_lr=5e-4 when using Adam optimizer
     # It is recommended to set Init_lr=1e-2 when using SGD optimizer
     # Min_lr The minimum learning rate of the model, the default is 0.01 of the maximum learning rate
     #------------------------------------------------- -----------------#
     Init_lr = 5e-4
     Min_lr = Init_lr * 0.01
     #------------------------------------------------- -----------------#
     # optimizer_type The type of optimizer used, optional adam, sgd
     # It is recommended to set Init_lr=5e-4 when using Adam optimizer
     # It is recommended to set Init_lr=1e-2 when using SGD optimizer
     # momentum The momentum parameter used internally by the optimizer
     # weight_decay weight decay to prevent overfitting
     # adam will cause weight_decay error, it is recommended to set it to 0 when using adam.
     #------------------------------------------------- -----------------#
     optimizer_type = "adam"
     momentum = 0.9
     weight_decay = 0
     #------------------------------------------------- -----------------#
     # lr_decay_type The learning rate drop method used, the options are 'step', 'cos'
     #------------------------------------------------------------------#
     lr_decay_type = 'cos'
     #------------------------------------------------- -----------------#
     # save_period How many epochs save a weight
     #------------------------------------------------- -----------------#
     save_period = 5
     #------------------------------------------------- -----------------#
     # save_dir The folder where weights and log files are saved
     #------------------------------------------------- -----------------#
     save_dir = 'logs'
     #------------------------------------------------- -----------------#
     # eval_flag Whether to evaluate during training, the evaluation object is the verification set
     # After installing the pycocotools library, the evaluation experience is better.
     # eval_period represents how many epochs to evaluate once, frequent evaluation is not recommended
     # Evaluation takes a lot of time, frequent evaluation will lead to very slow training
     # The mAP obtained here will be different from that obtained by get_map.py for two reasons:
     # (1) The mAP obtained here is the mAP of the verification set.
     # (2) The evaluation parameters set here are relatively conservative, the purpose is to speed up the evaluation.
     #------------------------------------------------- -----------------#
     eval_flag = True
     eval_period = 5
     #------------------------------------------------- -----------------#
     # num_workers is used to set whether to use multithreading to read data, 1 means to turn off multithreading
     # After opening, it will speed up the data reading speed, but it will take up more memory
     # Turn on multi-threading when IO is the bottleneck, that is, the GPU computing speed is much faster than the speed of reading pictures.
     #------------------------------------------------- -----------------#
     num_workers = 4

     #------------------------------------------------- -----#
     # train_annotation_path training image path and label
     # val_annotation_path Validate image path and label
     #------------------------------------------------- -----#
     train_annotation_path = '2007_train.txt'
     val_annotation_path = '2007_val.txt'
    
     #------------------------------------------------- -----#
     # Set the graphics card used
     #------------------------------------------------- -----#
     ngpus_per_node = torch.cuda.device_count()
     if distributed:
         dist.init_process_group(backend="nccl")
         local_rank = int(os.environ["LOCAL_RANK"])
         rank = int(os.environ["RANK"])
         device = torch.device("cuda", local_rank)
         if local_rank == 0:
             print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
             print("Gpu Device Count : ", ngpus_per_node)
     else:
         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         local_rank = 0

     #------------------------------------------------- ---#
     # Download pretrained weights
     #------------------------------------------------- ---#
     if pretrained:
         if distributed:
             if local_rank == 0:
                 download_weights(backbone)
             dist. barrier()
         else:
             download_weights(backbone)

     #------------------------------------------------- ---#
     # Get classes
     #------------------------------------------------- ---#
     class_names, num_classes = get_classes(classes_path)

     if backbone == "resnet50":
         model = CenterNet_Resnet50(num_classes, pretrained = pretrained)
     else:
         model = CenterNet_HourglassNet({'hm': num_classes, 'wh': 2, 'reg':2}, pretrained = pretrained)
     if model_path != '':
         #------------------------------------------------- -----#
         # For the weight file, please see README, download from Baidu Netdisk
         #------------------------------------------------- -----#
         if local_rank == 0:
             print('Load weights {}.'. format(model_path))
        
         #------------------------------------------------- -----#
         # Load according to the Key of the pre-trained weight and the Key of the model
         #------------------------------------------------- -----#
         model_dict = model. state_dict()
         pretrained_dict = torch.load(model_path, map_location = device)
         load_key, no_load_key, temp_dict = [], [], {}
         for k, v in pretrained_dict.items():
             if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                 temp_dict[k] = v
                 load_key.append(k)
             else:
                 no_load_key.append(k)
         model_dict. update(temp_dict)
         model.load_state_dict(model_dict)
         #------------------------------------------------- -----#
         # Display no matching Key
         #------------------------------------------------- -----#
         if local_rank == 0:
             print("\nSuccessful Load Key:", str(load_key)[:500], "...\nSuccessful Load Key Num:", len(load_key))
             print("\nFail To Load Key:", str(no_load_key)[:500], "...\nFail To Load Key num:", len(no_load_key))
             print("\n\033[1;33;44m Warm reminder, it is normal that the head part is not loaded, and it is an error that the Backbone part is not loaded.\033[0m")

     #----------------------#
     # Record Loss
     #----------------------#
     if local_rank == 0:
         time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
         log_dir = os.path.join(save_dir, "loss_" + str(time_str))
         loss_history = LossHistory(log_dir, model, input_shape=input_shape)
     else:
         loss_history = None
        
     #------------------------------------------------- -----------------#
     # torch 1.2 does not support amp, it is recommended to use torch 1.7.1 and above to use fp16 correctly
     # So torch1.2 shows "could not be resolve" here
     #------------------------------------------------- -----------------#
     if fp16:
         from torch.cuda.amp import GradScaler as GradScaler
         scaler = GradScaler()
     else:
         scaler = None

     model_train = model. train()
     #----------------------------#
     # Multi-card synchronization Bn
     #----------------------------#
     if sync_bn and ngpus_per_node > 1 and distributed:
         model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
     elif sync_bn:
         print("Sync_bn is not support in one gpu or not distributed.")

     if Cuda:
         if distributed:
             #----------------------------#
             # Multi-card parallel operation
             #----------------------------#
             model_train = model_train.cuda(local_rank)
             model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
         else:
             model_train = torch.nn.DataParallel(model)
             cudnn.benchmark = True
             model_train = model_train.cuda()
    
     #------------------------------#
     # Read the txt corresponding to the dataset
     #------------------------------#
     with open(train_annotation_path) as f:
         train_lines = f. readlines()
     with open(val_annotation_path) as f:
         val_lines = f. readlines()
     num_train = len(train_lines)
     num_val = len(val_lines)
    
     if local_rank == 0:
         show_config(
             classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
             Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
             Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
             save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
         )
         #------------------------------------------------- --------#
         # Total training generations refers to the total number of times to traverse all data
         # The total training step size refers to the total number of gradient descents
         # Each training generation contains several training steps, and each training step performs a gradient descent.
         # Only the minimum training generation is recommended here, there is no upper limit, and only the unfreezing part is considered in the calculation
         #------------------------------------------------- ---------#
         wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
         total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
         if total_step <= wanted_step:
             if num_train // Unfreeze_batch_size == 0:
                 raise ValueError('The data set is too small to train, please expand the data set.')
             wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
             print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step size to above %d.\033[0m"%(optimizer_type, wanted_step))
             print("\033[1;33;44m[Warning] The total amount of training data for this run is %d, the Unfreeze_batch_size is %d, a total of %d Epochs are trained, and the total training step is calculated as %d.\033 [0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
             print("\033[1;33;44m[Warning] Since the total training step size is %d, which is less than the recommended total step size %d, it is recommended to set the total generation as %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

     #------------------------------------------------- -----#
     # Backbone feature extraction network features are common, freezing training can speed up training
     # It can also prevent the weight from being destroyed in the early stage of training.
     # Init_Epoch is the starting generation
     # Freeze_Epoch is the generation of frozen training
     # UnFreeze_Epoch total training generations
     # Prompt OOM or insufficient video memory, please reduce Batch_size
     #------------------------------------------------- -----#
     if True:
         UnFreeze_flag = False
         #---------------------------------------#
         # Freeze a certain part of the training
         #---------------------------------------#
         if Freeze_Train:
             model. freeze_backbone()
                        
         #------------------------------------------------- ------------------#
         # If you don't freeze training, directly set batch_size to Unfreeze_batch_size
         #------------------------------------------------- ------------------#
         batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

         #------------------------------------------------- ------------------#
         # Judge the current batch_size, adaptively adjust the learning rate
         #------------------------------------------------- ------------------#
         nbs = 64
         lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
         lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
         Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
         Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

         #------------------------------------------#
         # Select the optimizer according to optimizer_type
         #------------------------------------------#
         optimizer = {
             'adam' : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
             'sgd' : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
         }[optimizer_type]

         #------------------------------------------#
         # Get the formula for learning rate drop
         #------------------------------------------#
         lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
         #------------------------------------------#
         # Determine the length of each generation
         #------------------------------------------#
         epoch_step = num_train // batch_size
         epoch_step_val = num_val // batch_size

         if epoch_step == 0 or epoch_step_val == 0:
             raise ValueError("The data set is too small to continue training, please expand the data set.")
        
         train_dataset = CenternetDataset(train_lines, input_shape, num_classes, train = True)
         val_dataset = CenternetDataset(val_lines, input_shape, num_classes, train = False)
        
         if distributed:
             train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
             val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
             batch_size = batch_size // ngpus_per_node
             shuffle = False
         else:
             train_sampler = None
             val_sampler = None
             shuffle=True
            
         gen = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                     drop_last=True, collate_fn=centernet_dataset_collate, sampler=train_sampler)
         gen_val = DataLoader(val_dataset , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                     drop_last=True, collate_fn=centernet_dataset_collate, sampler=val_sampler)

         #----------------------#
         # Record the map curve of eval
         #----------------------#
         if local_rank == 0:
             eval_callback = EvalCallback(model, backbone, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                             eval_flag=eval_flag, period=eval_period)
         else:
             eval_callback = None
        
         #------------------------------------------#
         # start model training
         #------------------------------------------#
         for epoch in range(Init_Epoch, UnFreeze_Epoch):
             #------------------------------------------#
             # If the model has a frozen learning part
             # Unfreeze and set parameters
             #------------------------------------------#
             if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                 batch_size = Unfreeze_batch_size

                 #------------------------------------------------- ------------------#
                 # Judge the current batch_size, adaptively adjust the learning rate
                 #------------------------------------------------- ------------------#
                 nbs = 64
                 lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
                 lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
                 Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                 Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                 #------------------------------------------#
                 # Get the formula for learning rate drop
                 #------------------------------------------#
                 lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                 model.unfreeze_backbone()

                 epoch_step = num_train // batch_size
                 epoch_step_val = num_val // batch_size

                 if epoch_step == 0 or epoch_step_val == 0:
                     raise ValueError("The data set is too small to continue training, please expand the data set.")

                 if distributed:
                     batch_size = batch_size // ngpus_per_node
                    
                 gen = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                             drop_last=True, collate_fn=centernet_dataset_collate, sampler=train_sampler)
                 gen_val = DataLoader(val_dataset , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                             drop_last=True, collate_fn=centernet_dataset_collate, sampler=val_sampler)

                 UnFreeze_flag = True

             if distributed:
                 train_sampler.set_epoch(epoch)
                
             set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
             fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                     epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, backbone, save_period, save_dir, local_rank)
            
         if local_rank == 0:
             loss_history.writer.close()