## CenterNet: Objects as Points implementation of target detection model in Pytorch
---

## Table of contents
1. [Top News](#top-news)
2. [Performance](#performance)
3. [Required environment](#required-environment)
4. [Precautions](#precautions)
5. [Download document](#download-document)
6. [Training steps](#training-steps)
7. [Prediction step](#prediction-step)
8. [Evaluation step](#evaluation-step)
9. [Reference](#reference)

## Top News
**`2022-04`**: Substantial updates have been made, supporting step, cos learning rate drop method, adam, sgd optimizer selection, learning rate adaptive adjustment according to batch_size, and new image cropping. Support multi-GPU training, add the calculation of the number of targets of each type, and add heatmap.
The original warehouse address in the BiliBili video is: [https://github.com/xxxx/centernet-pytorch/tree/xxxx](https://github.com/xxxx/centernet-pytorch/tree/xxxx)

**`2021-10`**: Substantial updates have been made, adding a large number of comments, adding a large number of adjustable parameters, modifying the components of the code, adding functions such as fps, video prediction, and batch prediction.

## Performance
| training dataset | weight file name | test dataset | input image size | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12 | [centernet_resnet50_voc.pth](https://github.com/bubbliiiiing/centernet-pytorch/releases/download/v1.0/centernet_resnet50_voc.pth) | VOC-Test07 | 512x512 | - | 77.1 |
| COCO-Train2017 | [centernet_hourglass_coco.pth](https://github.com/bubbliiiiing/centernet-pytorch/releases/download/v1.0/centernet_hourglass_coco.pth) | COCO-Val2017 | 512x512 | 38.4 | 56.8 |


## Required environment
torch==1.2.0

## Precautions
The `centernet_resnet50_voc.pth` in the code is trained using the VOC dataset.
The `centernet_hourglass_coco.pth` in the code is trained using the COCO dataset.
**Be careful not to use Chinese labels, and there should be no spaces in the folder!**
**Before training, it is necessary to create a new txt document under `model_data`, enter the class to be classified in the document, and point `classes_path` to this file in `train.py`.**

## Download document
The weights of `centernet_resnet50_voc.pth`, `centernet_hourglass_coco.pth`, and backbone required for training can be downloaded from Baidu Netdisk.
Link: [https://pan.baidu.com/s/1YOQgpCiXPKiXC9Wgn6Kt0w](https://pan.baidu.com/s/1YOQgpCiXPKiXC9Wgn6Kt0w)
Extraction code: 589g

`centernet_resnet50_voc.pth` is the weight of the VOC dataset.
`centernet_hourglass_coco.pth` is the weight of the COCO dataset.

The download address of the VOC dataset is as follows, which already includes the training set, test set, and verification set (same as the test set), and there is no need to divide it again:
Link: [https://pan.baidu.com/s/1-1Ej6dayrx3g0iAA88uY5A](https://pan.baidu.com/s/1-1Ej6dayrx3g0iAA88uY5A)
Extraction code: ph32

## Training steps
### a. Training VOC07+12 dataset
1. Dataset preparation
**This article uses the VOC format for training. You need to download the VOC07+12 dataset before training and put it in the root directory after decompression.**

2. Dataset Processing
Modify `annotation_mode=2` in `voc_annotation.py`, run `voc_annotation.py` to generate `2007_train.txt` and `2007_val.txt` in the root directory.

3. Start network training
The default parameters of `train.py` are used to train the VOC dataset, and the training can be started by running `train.py` directly.

4. Training result prediction
Two files are required for training result prediction, namely `centernet.py` and `predict.py`. We first need to modify `model_path` and `classes_path` in `centernet.py`. These two parameters must be modified.
**`model_path` points to the trained weight file in the `logs` folder.
`classes_path` points to the txt corresponding to the detection category.**
After completing the modification, you can run `predict.py` for detection. After running, enter the image path to detect.

### b. Train your own dataset
1. Dataset Preparation
**This article uses the VOC format for training, you need to create your own dataset before training.**
Before training, put the label file in the `Annotation` under the `VOC2007` folder under the `VOCdevkit` folder.
Before training, put the picture file in `JPEGImages` under the `VOC2007` folder under the `VOCdevkit` folder.

2. Dataset Processing
After completing the placement of the dataset, we need to use `voc_annotation.py` to obtain `2007_train.txt` and `2007_val.txt` for training.
Modify the parameters in `voc_annotation.py`. The first training can only modify the `classes_path`, which is used to point to the txt corresponding to the detected category.
When training your own dataset, you can create a `cls_classes.txt` by yourself and write the categories you need to distinguish in it.
The content of the `model_data/cls_classes.txt` file is:
```python
cat
dog
...

```
Modify the `classes_path` in `voc_annotation.py` to correspond to `cls_classes.txt` and run `voc_annotation.py`.

3. Start network training
**There are many training parameters, all of which are in `train.py`. You can read the comments carefully after downloading the library. The most important part is still the `classes_path` in `train.py`.**
**`classes_path` is used to point to the txt corresponding to the detection category, which is the same as the txt in `voc_annotation.py`! Training your own data set must be modified!**
After modifying the `classes_path`, you can run `train.py` to start training. After training for multiple epochs, the weights will be generated in the `logs` folder.

4. Training result prediction
Two files are required for training result prediction, namely `centernet.py` and `predict.py`. Modify `model_path` and `classes_path` in `centernet.py`.
**`model_path` points to the trained weight file in the `logs` folder.
`classes_path` points to the txt corresponding to the detection category.**
After completing the modification, you can run `predict.py` for detection. After running, enter the image path to detect.


## Prediction step
### a. Use pre-trained weights
1. After downloading the library, unzip it, download the weight value from Baidu Netdisk, put it into model_data, run predict.py, and enter
```python
img/street.jpg
```
2. Setting in predict.py can perform fps test and video video detection.
### b. Use your own training weights
1. Follow the training steps to train.
2. In the centernet.py file, modify model_path and classes_path in the following parts to correspond to the trained files; **model_path corresponds to the weight file under the logs folder, and classes_path is the class** that model_path corresponds to.
```python
_defaults = {
     #------------------------------------------------- --------------------------#
     # Use your own trained model to make predictions, you must modify model_path and classes_path!
     # model_path points to the weight file under the logs folder, and classes_path points to the txt under model_data
     # If there is a shape mismatch, pay attention to the modification of the model_path and classes_path parameters during training
     #------------------------------------------------- --------------------------#
     "model_path" : 'model_data/centernet_resnet50_voc.pth',
     "classes_path" : 'model_data/voc_classes.txt',
     #------------------------------------------------- --------------------------#
     # Backbone used to select the model to use
     # resnet50, hourglass
     #------------------------------------------------- --------------------------#
     "backbone" : 'resnet50',
     #------------------------------------------------- --------------------------#
     # Enter the size of the image, set it to a multiple of 32
     #------------------------------------------------- --------------------------#
     "input_shape" : [512, 512],
     #------------------------------------------------- --------------------------#
     # Only prediction boxes with scores greater than confidence will be kept
     #------------------------------------------------- --------------------------#
     "confidence" : 0.3,
     #------------------------------------------------- --------------------#
     # nms_iou size used for non-maximum suppression
     #------------------------------------------------- --------------------#
     "nms_iou" : 0.3,
     #------------------------------------------------- --------------------------#
     # Whether to perform non-maximum suppression, you can checkMeasure the effect by yourself
     # When the backbone is resnet50, it is recommended to set it to True, and when the backbone is hourglass, it is recommended to set it to False
     #------------------------------------------------- --------------------------#
     "nms" : True,
     #------------------------------------------------- --------------------#
     # This variable is used to control whether to use letterbox_image to resize the input image without distortion,
     # After many tests, it is found that closing the letterbox_image directly resizes the effect better
     #------------------------------------------------- --------------------#
     "letterbox_image" : False,
     #-------------------------------#
     # Whether to use Cuda
     # No GPU can be set to False
     #-------------------------------#
     "cuda" : True
}
```
3. Run predict.py, enter
```python
img/street.jpg
```
4. Setting in predict.py can perform fps test and video video detection.

## Evaluation Step
### a. Evaluate the test set of VOC07+12
1. This article uses the VOC format for evaluation. VOC07+12 has already divided the test set, so there is no need to use `voc_annotation.py` to generate txt under the ImageSets folder.
2. Modify `model_path` and `classes_path` in `centernet.py`. **`model_path` points to the trained weight file in the `logs` folder. `classes_path` points to the txt corresponding to the detection category.**
3. Run `get_map.py` to get the evaluation result, which will be saved in the `map_out` folder.

### b. Evaluate your own dataset
1. This article uses the VOC format for evaluation.
2. If the `voc_annotation.py` file has been run before training, the code will automatically divide the dataset into a training set, validation set, and test set. If you want to modify the proportion of the test set, you can modify the `trainval_percent` under the `voc_annotation.py` file. `trainval_percent` is used to specify the ratio of (training set + validation set) to the test set, by default (training set + validation set): test set = 9:1. `train_percent` is used to specify the ratio of the training set to the validation set in (training set + validation set), by default training set: validation set = 9:1.
3. After using `voc_annotation.py` to divide the test set, go to the `get_map.py` file to modify the `classes_path`. The `classes_path` is used to point to the txt corresponding to the detection category. This txt is the same as the txt during training. Evaluating your own dataset has to be modified.
4. Modify `model_path` and `classes_path` in `centernet.py`. **`model_path` points to the trained weight file in the `logs` folder. `classes_path` points to the txt corresponding to the detection category.**
5. Run `get_map.py` to get the evaluation result, which will be saved in the `map_out` folder.

## Reference
[https://github.com/xuannianz/keras-CenterNet](https://github.com/xuannianz/keras-CenterNet)

[https://github.com/see--/keras-centernet](https://github.com/see--/keras-centernet)

[https://github.com/xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)
