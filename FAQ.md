The blog address of the problem summary is [https://blog.csdn.net/weixin_44791964/article/details/107517428](https://blog.csdn.net/weixin_44791964/article/details/107517428).

# Problem summary
## 1. Download problem
### a. Code download
**Q: Master Up, can you send me a copy of the code, where can I download the code?
Answer: The address on Github is in the video introduction. Just copy it and download it. **

**Q: Master up, why does the code I downloaded indicate that the compressed package is damaged?
Answer: Go to Github to download again. **

**Q: Master up, why is the code I downloaded different from the code you posted on the video and blog?
Answer: I often update the code, and the actual code shall prevail in the end. **

### b. Weight download
**Q: Lord up, why is there no .pth or .h5 file under model_data in the code I downloaded?
Answer: I usually upload the weights to Github and Baidu Netdisk, which can be found in the README of Github. **

### c. Data set download
**Q: Master up, where can I download the XXXX data set?
Answer: I will put the download address of the general data set in the README, basically there are, if not, please contact me to add it in time, just send the issue on github**.

## 2. Environment configuration problem
### a. The environment used in the library now
**The pytorch version corresponding to the pytorch code is 1.2, and the blog address corresponds to**[https://blog.csdn.net/weixin_44791964/article/details/106037141](https://blog.csdn.net/weixin_44791964/article/ details/106037141).

**The tensorflow version corresponding to the keras code is 1.13.2, the keras version is 2.1.5, and the blog address corresponds to**[https://blog.csdn.net/weixin_44791964/article/details/104702142](https://blog .csdn.net/weixin_44791964/article/details/104702142).

**The tensorflow version corresponding to the tf2 code is 2.2.0, no need to install keras, and the blog address corresponds to**[https://blog.csdn.net/weixin_44791964/article/details/109161493](https://blog.csdn. net/weixin_44791964/article/details/109161493).

**Q: Can your code be used in a certain version of tensorflow and pytorch?
Answer: It is best to follow the configuration I recommend, and there are also configuration tutorials! I have not tried other versions! Problems can occur but are generally not major problems. Only a small amount of code needs to be changed. **

### b, 30 series graphics card environment configuration
The 30 series graphics card cannot use the above environment configuration tutorial due to the framework update.
Currently, the 30 video card configurations that I have tested are as follows:
**The pytorch version corresponding to the pytorch code is 1.7.0, cuda is 11.0, and cudnn is 8.0.5**.

**The keras code cannot be configured with cuda11 under win10. You can check it on Baidu under ubuntu. Configure the tensorflow version to 1.15.4, and the keras version to 2.1.5 or 2.3.1 (a small number of function interfaces are different, and the code may need a small amount of adjustment. )**

**The tensorflow version corresponding to the tf2 code is 2.4.0, cuda is 11.0, and cudnn is 8.0.5**.

### c, GPU utilization issues and environmental usage issues
**Q: Why do I install tensorflow-gpu but I don't use the GPU for training?
Answer: Confirm that tensorflow-gpu has been installed, use pip list to check the version of tensorflow, and then check the task manager or use the nvidia command to see if the gpu is used for training, and the task manager depends on the memory usage. **

**Question: Master Up, I don’t seem to be using GPU for training. How can I see if GPU is used for training?
A: To check whether the GPU is used for training, generally use NVIDIA's command line to check. If you want to check the task manager, please check whether the GPU memory is used in the performance section, or check the task manager's Cuda instead of Copy. **
![Insert picture description here](https://img-blog.csdnimg.cn/20201013234241524.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80 NDc5MTk2NA==, size_16, color_FFFFFF, t_70#pic_center )

**Question: Master up, why can’t I use it after configuring according to your environment?
Answer: Please tell me your GPU, CUDA, CUDNN, TF version and PYTORCH version B station private chat. **

**Q: The following error occurs**
```python
Traceback (most recent call last):
   File "C:\Users\focus\Anaconda3\ana\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 58, in <module>
  from tensorflow.python.pywrap_tensorflow_internal import *
File "C:\Users\focus\Anaconda3\ana\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 28, in <module>
pywrap_tensorflow_internal = swig_import_helper()
   File "C:\Users\focus\Anaconda3\ana\envs\tensorflow-gpu\lib\site-packages\tensorflow\python\pywrap_tensorflow_internal.py", line 24, in swig_import_helper
     _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
File "C:\Users\focus\Anaconda3\ana\envs\tensorflow-gpu\lib\imp.py", line 243, in load_module return load_dynamic(name, filename, file)
File "C:\Users\focus\Anaconda3\ana\envs\tensorflow-gpu\lib\imp.py", line 343, in load_dynamic
     return_load(spec)
ImportError: DLL load failed: The specified module could not be found.
```
**Answer: If you have not restarted, restart it, otherwise follow the steps to install again, if you still can't solve it, please tell me your GPU, CUDA, CUDNN, TF version and PYTORCH version in private chat. **

### d, no module problem
**Q: Why does the prompt say no module name utils.utils (no module name nets.yolo, no module name nets.ssd and a series of questions)?
Answer: utils does not need to be installed with pip. It is in the root directory of the warehouse I uploaded. The reason for this problem is that the root directory is wrong. Check the concept of relative directory and root directory. Check it out and basically understand. **

**Q: Why does the prompt say no module name matplotlib (no module name PIL, no module name cv2, etc.)?
Answer: This library is not installed, just open the command line to install it. pip install matplotlib**

**Q: Why do I have installed opencv (pillow, matplotlib, etc.) with pip, or is it prompted no module name cv2?
Answer: There is no activation environment installation, you need to activate the corresponding conda environment for installation before it can be used normally**

**Q: Why does the prompt say No module named 'torch'?
Answer: Actually, I really want to know why there is this problem... What is the situation that pytorch is not installed? Generally, there are two situations, one is that it is not installed, and the other is that it is installed in other environments, and the currently activated environment is not the one installed by yourself. **

**Q: Why does the prompt say No module named 'tensorflow'?
Answer: Same as above. **

### e, cuda installation failure problem
Generally, you need to install Visual Studio before installing cuda, just install a 2017 version.

### f. Ubuntu system problems
**All codes are available under Ubuntu, I have tried both systems. **

### g, VSCODE prompt error problem
**Q: Why are there a lot of errors displayed in VSCODE?
Answer: I also prompt a lot of errors, but it doesn't affect it. It's a problem with VSCODE. If you don't want to see errors, install Pycharm. **

### h. The problem of using cpu for training and prediction
**For keras and tf2 codes, if you want to use cpu for training and prediction, you can directly install the cpu version of tensorflow. **

**For pytorch code, if you want to use cpu for training and prediction, you need to change cuda=True to cuda=False. **

### i, tqdm has no pos parameter problem
**Q: When running the code, it prompts that 'tqdm' object has no attribute 'pos'.
Answer: Reinstall tqdm and change the version. **

### j. Prompt the problem of decode("utf-8")
**Due to the update of the h5py library, the version above h5py=3.0.0 will be automatically installed during the installation process, which will cause a decode("utf-8") error!
Everyone must use the command to install h5py=2.10.0 after installing tensorflow! **
```
pip install h5py==2.10.0
```

### k, Prompt TypeError: __array__() takes 1 positional argument but 2 were given error
You can modify the pillow version to solve it.
```
pip install pillow==8.2.0
```

### l. Other issues
**Q: Why do I get TypeError: cat() got an unexpected keyword argument 'axis', Traceback (most recent call last), AttributeError: 'Tensor' object has no attribute 'bool'?
Answer: This is a version issue, it is recommended to use torch1.2 or above**
** There are many other strange problems, many of which are version problems. It is recommended to install Keras and tensorflow according to my video tutorial. For example, if tensorflow2 is installed, you don’t have to ask me why I can’t run Keras-yolo or something. That must not work. **

## 3. Summary of target detection library problems (face detection and classification libraries can also be referred to)
### a, shape mismatch problem
#### 1), shape mismatch problem during training
**Q: Master up, why does running train.py prompt that the shape does not match?
Answer: In the keras environment, because the type of your training is different from the original type, the network structure will change, so the shape at the end will have a small amount of mismatch. **

#### 2), the shape does not match when predicting
**Q: Why am I running predict.py

It will prompt me that the shape does not match.
In Pytorch it looks like this: **
![Insert picture description here](https://img-blog.csdnimg.cn/20200722171631901.png)
In Keras it looks like this:
![Insert picture description here](https://img-blog.csdnimg.cn/20200722171523380.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80 NDc5MTk2NA==, size_16, color_FFFFFF, t_70)
**Answer: There are three main reasons:
1. In ssd and FasterRCNN, it may be that num_classes in train.py has not been changed.
2. The model_path has not been changed.
3. The classes_path has not been changed.
Please check it out! Make sure that the model_path and classes_path you use correspond to each other! The num_classes or classes_path used during training also needs to be checked! **

### b. Insufficient video memory
**Q: Why does the command line below train.py flash so fast when I run it, and it prompts OOM or something?
Answer: This appears in keras. The video memory has exploded. You can change the batch_size to a smaller size. The memory usage of SSD is the smallest. It is recommended to use SSD;
2G memory: SSD, YOLOV4-TINY
4G memory: YOLOV3
6G video memory: YOLOV4, Retinanet, M2det, Efficientdet, Faster RCNN, etc.
8G+ video memory: choose whatever you want. **
**It should be noted that due to the influence of BatchNorm2d, batch_size cannot be 1, at least 2. **

**Q: Why does it prompt RuntimeError: CUDA out of memory. Tried to allocate 52.00 MiB (GPU 0; 15.90 GiB total capacity; 14.85 GiB already allocated; 51.88 MiB free; 15.07 GiB reserved in total by PyTorch)?
Answer: This is what appeared in pytorch, and the video memory has exploded, as above. **

**Question: Why did I blow up my video memory without using it?
Answer: The video memory is full, so naturally it will not be used, and the model has not started training. **
### c. Training problems (freezing training, LOSS problems, training effect problems, etc.)
**Q: Why freeze training and unfreeze training?
Answer: This is the idea of transfer learning, because the features extracted by the backbone feature extraction part of the neural network are common, and we freeze them for training to speed up the training efficiency and prevent the weights from being destroyed. **
In the freezing phase, the backbone of the model is frozen and the feature extraction network does not change. It takes up less video memory and only fine-tunes the network.
In the unfreezing phase, the backbone of the model is not frozen, and the feature extraction network is changed. It takes up a lot of video memory, and all parameters of the network will change.

**Q: Why is my network not converging, LOSS is XXXX.
Answer: The LOSS of different networks is different. LOSS is just a reference indicator, which is used to check whether the network is converged, not to evaluate whether the network is good or bad. My yolo code is not normalized, so the LOSS value looks relatively high, and the LOSS value is not Important, what is important is whether it is getting smaller and whether the prediction has an effect. **

**Q: Why is my training effect not good? No frame is predicted (frame is not allowed).
answer:**

Consider a few questions:
1. Target information problem, check whether there is target information in the 2007_train.txt file, if not, please modify voc_annotation.py.
2. For the problem of data sets, if the number is less than 500, consider increasing the data set by yourself, and test different models at the same time to confirm that the data set is good.
3. Whether to unfreeze the training, if the distribution of the data set is too far from the regular picture, it is necessary to further unfreeze the training, adjust the backbone, and strengthen the feature extraction ability.
4. Network problems, such as SSD is not suitable for small targets, because the prior frame is fixed.
5. The problem of training time. Some students have only trained for a few generations and said that there is no effect. They have finished training according to the default parameters.
6. Confirm whether you have followed the steps, such as whether the classes in voc_annotation.py have been modified, etc.
7. The LOSS of different networks is different. LOSS is just a reference indicator, which is used to check whether the network is converged, not to evaluate whether the network is good or bad. The value of LOSS is not important, but whether it is converged or not.

**Question: Why did I get a gbk encoding error:**
```python
UnicodeDecodeError: 'gbk' codec can't decode byte 0xa6 in position 446: illegal multibyte sequence
```
**Answer: Do not use Chinese for labels and paths. If you must use Chinese, please pay attention to the encoding problem when processing, and change the encoding method of opening the file to utf-8. **

**Q: My picture is in xxx*xxx resolution, can it be used? **
**Answer: Yes, the code will automatically perform resize or data enhancement. **

**Q: How to conduct multi-GPU training?
Answer: Most of the codes in pytorch can be trained directly using GPU. For keras, you can directly use Baidu. The implementation is not complicated. I don’t have a multi-card and I can’t test it in detail. You need to work hard on your own. **
### d. Grayscale problem
**Q: Can you train grayscale images (predict grayscale images)?
Answer: Most of my libraries convert grayscale images into RGB for training and prediction. If you encounter a situation where the code cannot train or predict grayscale images, you can try to convert the result of Image.open into RGB in get_random_data , try the same when predicting. (for reference only)**

### e. The problem of continuing to practice at breakpoints
**Q: I have already trained several generations, can I continue to train from this foundation
Answer: Yes, you can load the trained weights in the same way as the pre-trained weights before training. Generally, the trained weights will be saved in the logs folder, just modify the model_path to the path of the weights you want to start. **

### f, the problem of pre-training weight
**Q: If I want to train other data sets, what should I do if the pre-training weight is important? **
**Answer: The pre-training weights of the data are common to different data sets, because the features are common, and the pre-training weights must be used for 99% of the cases. If not used, the weights are too random, and the feature extraction effect is not obvious. The results of network training will not be good either. **

**Q: up, I modified the network, can the pre-trained weight still be used?
Answer: If the backbone is modified, if the existing network is not used, basically the pre-training weights cannot be used. Either judge the shape of the convolution kernel in the weight and match it yourself, or you can only pre-train it yourself. ; If the second half is modified, the pre-training weights of the main part of the first half can still be used. If it is a pytorch code, you need to modify the way to load the weights yourself, and load it after judging the shape. If it is a keras code, Just directly by_name=True, skip_mismatch=True. **
The method of weight matching can be referred to as follows:
```python
# Speed up the efficiency of model training
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = model. state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
a = {}
for k, v in pretrained_dict.items():
     try:
         if np.shape(model_dict[k]) == np.shape(v):
             a[k]=v
     except:
         pass
model_dict. update(a)
model.load_state_dict(model_dict)
print('Finished!')
```

**Q: How can I not use pre-trained weights?
Answer: Just comment out the code that loads the pre-trained weights. **

**Q: Why is the effect of not using pre-trained weights so bad?
Answer: Because the randomly initialized weights are not good, the extracted features are not good, which leads to poor model training effects. The effects of voc07+12 and coco+voc07+12 are different, and the pre-training weights are still very important. **

### g. Video detection problem and camera detection problem
**Q: How to use the camera to detect?
Answer: The camera detection can be performed by modifying the parameters of predict.py, and there are also videos explaining the idea of camera detection in detail. **

**Q: How to use video detection?
Answer: Same as above**
### h, start training from 0
**Q: How to start training from 0 on the model?
Answer: It is meaningless to start training from 0 in the case of insufficient computing power and insufficient parameter adjustment ability. The feature extraction ability of the model is very poor in the case of random initialization parameters. Without good parameter adjustment capabilities and computing power, the network cannot converge normally. **
If you must start from 0, please pay attention to a few points when training:
  - Do not load pretrained weights.
  - Don't do frozen training, comment the code of the frozen model.

**Q: Why is the effect of not using pre-trained weights so bad?
Answer: Because the randomly initialized weights are not good, the extracted features are not good, which leads to poor model training effects. The effects of voc07+12 and coco+voc07+12 are different, and the pre-training weights are still very important. **

### i. Save the problem
**Q: How to save the detected pictures?
Answer: Generally, images are used for target detection, so check how to save images in the PIL library. See the comments of the predict.py file for details. **

**Q: How to save video?
Answer: Look at the comments of the predict.py file in detail. **

### j, traversal problem
**Q: How to traverse the pictures in a folder?
Answer: Generally, use os.listdir to find all the pictures in the folder first, and then detect the pictures according to the execution ideas in the predict.py file. See the notes of the predict.py file for details. **

**Q: How to traverse the pictures in a folder? and save.
Answer: For traversal, generally use os.listdir to find all the pictures in the folder first, and then detect the pictures according to the execution ideas in the predict.py file. When saving, the general target detection uses Image, so check how to save the Image in the PIL library. If some libraries use cv2, then check how cv2 saves pictures. See the comments of the predict.py file for details. **

### k. Path problem (No such file or directory)
**Q: Why did I make such a mistake:**
```python
FileNotFoundError: [Errno 2] No such file or directory
…………………………………
…………………………………
```
**Answer: Check the folder path to see if there is a corresponding file; and check 2007_train.txt to see if there is any error in the file path. **
There are several important points about paths:
**There must be no spaces in the folder name.
Pay attention to relative paths and absolute paths.
More Baidu path related knowledge. **

**All path problems are basically root directory problems, check the concept of relative directories carefully! **
### l. Compared with the original version
**Q: How does your code compare with the original version? Can it achieve the effect of the original version?
Answer: Basically, it can be achieved. I have tested it with voc data. I don't have a good graphics card, and I don't have the ability to test and train on coco. **

**Q: Have you implemented all the tricks of yolov4, how far is it from the original version?
Answer: Not all the improvements have been implemented. Since there are too many improvements used by YOLOV4, it is difficult to fully implement and list them. Here are only some of the improvements that I am more interested in and are very effective. The SAM (Attention Mechanism Module) mentioned in the paper is not used by the author's own source code. There are many other tricks, not all tricks have improved, and I can't achieve all tricks. As for the comparison with the original version, I don't have the ability to train the coco data set. According to the feedback of the students who have used it, there is not much difference. **

### m, FPS problem (detection speed problem)
**Q: How much can your FPS reach, can you reach XX FPS?
Answer: The FPS is related to the configuration of the machine. If the configuration is high, it will be fast, and if the configuration is low, it will be slow. **

**Q: Why do I use the server to test the FPS of yolov4 (or others) only a dozen?
Answer: Check if tensorfl is installed correctly

If the gpu version of ow-gpu or pytorch has been installed correctly, you can use the time.time() method to check which piece of code takes longer in the detect_image (not only the network takes a long time, but other processing parts also take time, such as drawing, etc.). **

**Q: Why does the paper say that the speed can reach XX, but there is no such thing here?
Answer: Check whether the tensorflow-gpu or the gpu version of pytorch is installed correctly. If it is installed correctly, you can use the time.time() method to check the detect_image, which piece of code takes longer (not only the network takes a long time, but other The processing part is also time consuming, like plotting etc). Some papers also use multi-batch for prediction, and I did not implement this part. **

### n. The problem of not displaying the predicted picture
**Q: Why doesn't your code show the picture after the prediction is done? Just tell me what the target is on the command line.
Answer: Just install a picture viewer for the system. **

### o. Algorithm evaluation issues (target detection map, PR curve, Recall, Precision, etc.)
**Q: How to calculate the map?
Answer: Watching the map video is a process. **

**Q: When calculating the map, what is the use of MINOVERLAP in get_map.py? Is it iou?
Answer: It is iou. Its function is to judge the degree of coincidence between the predicted frame and the real frame. If the degree of coincidence is greater than MINOVERLAP, the prediction is correct. **

**Q: Why should the self.confidence (self.score) in get_map.py be set so small?
Answer: Look at the principle part of the map video, you need to know all the results and then draw the pr curve. **

**Q: Can you tell me how to draw the PR curve or something.
Answer: You can watch the mAP video, and there is a PR curve in the result. **

**Q: How to calculate Recall and Precision indicators.
Answer: These two indicators should be relative to a specific confidence level, and they will also be obtained when calculating the map. **

### p, coco data set training problem
**Q: How to train the COCO dataset for target detection? .
Answer: The txt file required for coco data training can refer to the yolo3 library of qqwweee, and the format is the same. **

### q, model optimization (model modification) problem
**Q: Up, do you have the code for using Focal LOSS in the YOLO series? Is there any improvement?
Answer: Many people have tried it, but the improvement effect is not great (even lower), and it has its own balance method of positive and negative samples. **

**Q: up, I modified the network, can the pre-trained weight still be used?
Answer: If the backbone is modified, if the existing network is not used, basically the pre-training weights cannot be used. Either judge the shape of the convolution kernel in the weight and match it yourself, or you can only pre-train it yourself. ; If the second half is modified, the pre-training weights of the main part of the first half can still be used. If it is a pytorch code, you need to modify the way to load the weights yourself, and load it after judging the shape. If it is a keras code, Just directly by_name=True, skip_mismatch=True. **
The method of weight matching can be referred to as follows:
```python
# Speed up the efficiency of model training
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = model. state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
a = {}
for k, v in pretrained_dict.items():
     try:
         if np.shape(model_dict[k]) == np.shape(v):
             a[k]=v
     except:
         pass
model_dict. update(a)
model.load_state_dict(model_dict)
print('Finished!')
```

**Q: Up, how to modify the model, I want to send a small paper!
Answer: It is recommended to look at the difference between yolov3 and yolov4, and then look at the paper of yolov4. As a large-scale tuning site, it is very meaningful and uses a lot of tricks. The suggestion I can give is to look at some classic models, and then disassemble the bright structure inside and use them. **

### r. Deployment issues
I haven't deployed it to mobile phones and other devices, so I don't understand many deployment issues...

## 4. Summary of Semantic Segmentation Library Problems
### a, shape mismatch problem
#### 1), shape mismatch problem during training
**Q: Master up, why does running train.py prompt that the shape does not match?
Answer: In the keras environment, because the type of your training is different from the original type, the network structure will change, so the shape at the end will have a small amount of mismatch. **

#### 2), the shape does not match when predicting
**Q: Why does it prompt me that the shape does not match when I run predict.py.
In Pytorch it looks like this: **
![Insert picture description here](https://img-blog.csdnimg.cn/20200722171631901.png)
In Keras it looks like this:
![Insert picture description here](https://img-blog.csdnimg.cn/20200722171523380.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80 NDc5MTk2NA==, size_16, color_FFFFFF, t_70)
**Answer: There are two main reasons:
1. The num_classes in train.py have not been changed.
2. num_classes has not been changed during prediction.
Please check it out! The num_classes used in training and prediction need to be checked! **

### b. Insufficient video memory
**Q: Why does the command line below train.py flash so fast when I run it, and it prompts OOM or something?
Answer: This appears in keras, the memory is bursting, you can change the batch_size to a smaller one. **

**It should be noted that due to the influence of BatchNorm2d, batch_size cannot be 1, at least 2. **

**Q: Why does it prompt RuntimeError: CUDA out of memory. Tried to allocate 52.00 MiB (GPU 0; 15.90 GiB total capacity; 14.85 GiB already allocated; 51.88 MiB free; 15.07 GiB reserved in total by PyTorch)?
Answer: This is what appeared in pytorch, and the video memory has exploded, as above. **

**Question: Why did I blow up my video memory without using it?
Answer: The video memory is full, so naturally it will not be used, and the model has not started training. **

### c. Training problems (freezing training, LOSS problems, training effect problems, etc.)
**Q: Why freeze training and unfreeze training?
Answer: This is the idea of transfer learning, because the features extracted by the backbone feature extraction part of the neural network are common, and we freeze them for training to speed up the training efficiency and prevent the weights from being destroyed. **
**In the freezing stage, the backbone of the model is frozen and the feature extraction network does not change. It takes up less video memory and only fine-tunes the network. **
**During the unfreezing phase, the backbone of the model is not frozen, and the feature extraction network will change. It takes up a lot of video memory, and all parameters of the network will change. **

**Q: Why is my network not converging, LOSS is XXXX.
Answer: The LOSS of different networks is different. LOSS is just a reference indicator, which is used to check whether the network is converged, not to evaluate whether the network is good or bad. My yolo code is not normalized, so the LOSS value looks relatively high, and the LOSS value is not Important, what is important is whether it is getting smaller and whether the prediction has an effect. **

**Q: Why is my training effect not good? No target was predicted, and the result was black.
answer:**
**Consider a few questions:
1. Data set problem, this is the most important problem. If it is less than 500, consider adding a dataset by yourself; be sure to check the label of the dataset. The format of the VOC dataset is analyzed in detail in the video, but it is not enough to have an input image with an output label. It is also necessary to confirm whether each pixel value of the label is for its corresponding type. Many students’ label format is wrong. The most common error format is that the background of the label is black and the target is white. At this time, the pixel value of the target is 255, which cannot be trained normally. The target needs to be 1.
2. Whether to unfreeze the training, if the data set distribution is too far from the regular picture, it is necessary to further unfreeze the training, adjust the backbone, and strengthen the feature extraction ability.
3. Network problem, you can try a different network.
4. The problem of training time. Some students have only trained for a few generations and said that there is no effect. They have finished training according to the default parameters.
5. Confirm whether you have followed the steps.
6. The LOSS of different networks is different. LOSS is just a reference indicator, which is used to check whether the network is converged, not to evaluate whether the network is good or bad. The value of LOSS is not important, but whether it is converged or not. **



**Q: Why is my training effect not good? Inaccurate predictions for small targets.
Answer: For deeplab and pspnet, you can modify the downsample_factor. When the downsample_factor is 16, the downsampling multiple is too much and the effect is not very good. You can modify it to 8. **

**Question: Why did I get a gbk encoding error:**
```python
UnicodeDecodeError: 'gbk' codec can't decode byte 0xa6 in position 446: illegal multibyte sequence
```
**Answer: Do not use Chinese for labels and paths. If you must use Chinese, please pay attention to the encoding problem when processing, and change the encoding method of opening the file to utf-8. **

**Q: My picture is in xxx*xxx resolution, can it be used? **
**Answer: Yes, the code will automatically perform resize or data enhancement. **

**Q: How to conduct multi-GPU training?
Answer: Most of the codes in pytorch can be trained directly using GPU. For keras, you can directly use Baidu. The implementation is not complicated. I don’t have a multi-card and I can’t test it in detail. You need to work hard on your own. **

### d. Grayscale problem
**Q: Can you train grayscale images (predict grayscale images)?
Answer: Most of my libraries convert grayscale images into RGB for training and prediction. If you encounter a situation where the code cannot train or predict grayscale images, you can try to convert the result of Image.open into RGB in get_random_data , try the same when predicting. (for reference only)**

### e. The problem of continuing to practice at breakpoints
**Q: I have already trained several generations, can I continue to train from this foundation
Answer: Yes, you can load the trained weights in the same way as the pre-trained weights before training. Generally, the trained weights will be saved in the logs folder, just modify the model_path to the path of the weights you want to start. **

### f, the problem of pre-training weight

**Q: If I want to train other data sets, what should I do if the pre-training weight is important? **
**Answer: The pre-training weights of the data are common to different data sets, because the features are common, and the pre-training weights must be used for 99% of the cases. If not used, the weights are too random, and the feature extraction effect is not obvious. The results of network training will not be good either. **

**Q: up, I modified the network, can the pre-trained weight still be used?
Answer: If the backbone is modified, if the existing network is not used, basically the pre-training weights cannot be used. Either judge the shape of the convolution kernel in the weight and match it yourself, or you can only pre-train it yourself. ; If the second half is modified, the pre-training weights of the main part of the first half can still be used. If it is a pytorch code, you need to modify the way to load the weights yourself, and load it after judging the shape. If it is a keras code, Just directly by_name=True, skip_mismatch=True. **
The method of weight matching can be referred to as follows:

```python
# Speed up the efficiency of model training
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = model. state_dict()
pretrained_dict = torch.load(model_path, map_location=d 
device)
a = {}
for k, v in pretrained_dict.items():
     try:
         if np.shape(model_dict[k]) == np.shape(v):
             a[k]=v
     except:
         pass
model_dict. update(a)
model.load_state_dict(model_dict)
print('Finished!')
```

**Q: How can I not use pre-trained weights?
Answer: Just comment out the code that loads the pre-trained weights. **

**Q: Why is the effect of not using pre-trained weights so bad?
Answer: Because the randomly initialized weights are not good, the extracted features are not good, which leads to the poor effect of model training, and the pre-training weights are still very important. **

### g. Video detection problem and camera detection problem
**Q: How to use the camera to detect?
Answer: The camera detection can be performed by modifying the parameters of predict.py, and there are also videos explaining the idea of camera detection in detail. **

**Q: How to use video detection?
Answer: Same as above**

### h, start training from 0
**Q: How to start training from 0 on the model?
Answer: It is meaningless to start training from 0 in the case of insufficient computing power and insufficient parameter adjustment ability. The feature extraction ability of the model is very poor in the case of random initialization parameters. Without good parameter adjustment capabilities and computing power, the network cannot converge normally. **
If you must start from 0, please pay attention to a few points when training:
  - Do not load pretrained weights.
  - Don't do frozen training, comment the code of the frozen model.

**Q: Why is the effect of not using pre-trained weights so bad?
Answer: Because the randomly initialized weights are not good, the extracted features are not good, which leads to the poor effect of model training, and the pre-training weights are still very important. **

### i. Save the problem
**Q: How to save the detected pictures?
Answer: Generally, images are used for target detection, so check how to save images in the PIL library. See the comments of the predict.py file for details. **

**Q: How to save video?
Answer: Look at the comments of the predict.py file in detail. **

### j, traversal problem
**Q: How to traverse the pictures in a folder?
Answer: Generally, use os.listdir to find all the pictures in the folder first, and then detect the pictures according to the execution ideas in the predict.py file. See the notes of the predict.py file for details. **

**Q: How to traverse the pictures in a folder? and save.
Answer: For traversal, generally use os.listdir to find all the pictures in the folder first, and then detect the pictures according to the execution ideas in the predict.py file. When saving, the general target detection uses Image, so check how to save the Image in the PIL library. If some libraries use cv2, then check how cv2 saves pictures. See the comments of the predict.py file for details. **

### k. Path problem (No such file or directory)
**Q: Why did I make such a mistake:**
```python
FileNotFoundError: [Errno 2] No such file or directory
…………………………………
…………………………………
```

**Answer: Check the folder path to see if there is a corresponding file; and check 2007_train.txt to see if there is any error in the file path. **
There are several important points about paths:
**There must be no spaces in the folder name.
Pay attention to relative paths and absolute paths.
More Baidu path related knowledge. **

**All path problems are basically root directory problems, check the concept of relative directories carefully! **

### l. FPS problem (detection speed problem)
**Q: How much can your FPS reach, can you reach XX FPS?
Answer: The FPS is related to the configuration of the machine. If the configuration is high, it will be fast, and if the configuration is low, it will be slow. **

**Q: Why does the paper say that the speed can reach XX, but there is no such thing here?
Answer: Check whether the tensorflow-gpu or the gpu version of pytorch is installed correctly. If it is installed correctly, you can use the time.time() method to check the detect_image, which piece of code takes longer (not only the network takes a long time, but other The processing part is also time consuming, like plotting etc). Some papers also use multi-batch for prediction, and I did not implement this part. **

### m. The problem of not displaying the predicted picture
**Q: Why doesn't your code show the picture after the prediction is done? Just tell me what the target is on the command line.
Answer: Just install a picture viewer for the system. **

### n. Algorithm evaluation problem (miou)
**Q: How to calculate miou?
A: Refer to the miou measurement part in the video. **

**Q: How to calculate Recall and Precision indicators.
Answer: The existing code is not available yet. You need to understand the concept of the confusion matrix, and then calculate it yourself. **

### o. Model optimization (model modification) problem
**Q: up, I modified the network, can the pre-trained weight still be used?
Answer: If the backbone is modified, if the existing network is not used, basically the pre-training weights cannot be used. Either judge the shape of the convolution kernel in the weight and match it yourself, or you can only pre-train it yourself. ; If the second half is modified, the pre-training weights of the main part of the first half can still be used. If it is a pytorch code, you need to modify the way to load the weights yourself, and load it after judging the shape. If it is a keras code, Just directly by_name=True, skip_mismatch=True. **
The method of weight matching can be referred to as follows:

```python
# Speed up the efficiency of model training
print('Loading weights into state dict...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict = model. state_dict()
pretrained_dict = torch.load(model_path, map_location=device)
a = {}
for k, v in pretrained_dict.items():
     try:
         if np.shape(model_dict[k]) == np.shape(v):
             a[k]=v
     except:
         pass
model_dict. update(a)
model.load_state_dict(model_dict)
print('Finished!')
```



**Q: Up, how to modify the model, I want to send a small paper!
Answer: It is recommended to read the paper of yolov4 in target detection. As a large-scale tuning site, it is very meaningful and uses a lot of tricks. The suggestion I can give is to look at some classic models, and then disassemble the bright structure inside and use them. You can try commonly used tricks such as attention mechanism. **

### p. Deployment issues
I haven't deployed it to mobile phones and other devices, so I don't understand many deployment issues...

## 5. Communication group issues
**Q: up, is there any QQ group or something?
Answer: No, I don't have time to manage QQ groups...**

## 6. The question of how to study
**Q: Up, how is your study route? I am a novice, how can I learn?
Answer: There are a few points to pay attention to here.
1. I'm not a master, and I don't know many things, and my learning route may not be suitable for everyone.
2. My laboratory does not do in-depth learning, so I learn many things by myself and explore by myself. I don't know whether it is correct or not.
3. I personally feel that learning is more dependent on self-study**
For the learning route, I first learned Mofan’s python tutorial, started with tensorflow, keras, and pytorch, and learned SSD and YOLO after getting started. Then I learned a lot of classic convolutional networks, and then I started to learn a lot of different codes. Yes, my learning method is to read line by line to understand the execution process of the entire code, the shape change of the feature layer, etc. It takes a lot of time and there is no shortcut, it just takes time.