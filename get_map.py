import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from centernet import CenterNet
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map

if __name__ == "__main__":
    '''
    Unlike AP, Recall and Precision are an area concept. Therefore, when the threshold value (Confidence) is different, the Recall and Precision values ​​of the network are different.
    By default, the Recall and Precision calculated by this code represent the corresponding Recall and Precision values when the threshold (Confidence) is 0.5.

    Limited by the principle of mAP calculation, the network needs to obtain almost all prediction frames when calculating mAP, so that the Recall and Precision values under different threshold conditions can be calculated
    Therefore, the number of txt boxes in map_out/detection-results/ obtained by this code is generally more than that of direct predict. The purpose is to list all possible prediction boxes.
    '''
    #------------------------------------------------- -------------------------------------------------- ---------------#
    # map_mode is used to specify what is calculated when the file is run
    # map_mode is 0 to represent the entire map calculation process, including obtaining prediction results, obtaining real frames, and calculating VOC_map.
    # map_mode is 1 to only get the prediction result.
    # map_mode is 2 to only get the real frame.
    # map_mode is 3 means only calculate VOC_map.
    # map_mode is 4 to use the COCO toolbox to calculate the 0.50:0.95map of the current dataset. You need to get the prediction result, get the real frame and install pycocotools
    #------------------------------------------------- -------------------------------------------------- ----------------#
    map_mode = 0
    #------------------------------------------------- --------------------------------------#
    # The classes_path here is used to specify the category that needs to measure VOC_map
    # Generally, it is consistent with the classes_path used for training and prediction
    #------------------------------------------------- --------------------------------------#
    classes_path = 'model_data/voc_classes.txt'
    #------------------------------------------------- --------------------------------------#
    # MINOVERLAP is used to specify the mAP0.x you want to obtain, what is the meaning of mAP0.x, please Baidu.
    # For example, to calculate mAP0.75, you can set MINOVERLAP = 0.75.
    #
    # When the overlap between a predicted frame and the real frame is greater than MINOVERLAP, the predicted frame is considered a positive sample, otherwise it is a negative sample.
    # Therefore, the larger the value of MINOVERLAP, the more accurate the prediction frame must be predicted to be considered a positive sample. At this time, the calculated mAP value is lower.
    #------------------------------------------------- --------------------------------------#
    MINOVERLAP = 0.5
    #------------------------------------------------- --------------------------------------#
    # Limited by the principle of mAP calculation, the network needs to obtain almost all prediction frames when calculating mAP, so that mAP can be calculated
    # Therefore, the value of confidence should be set as small as possible to obtain all possible prediction frames.
    #
    # This value is generally not adjusted. Because calculating mAP needs to obtain almost all the prediction boxes, the confidence here cannot be changed casually.
    # To obtain the Recall and Precision values under different thresholds, please modify the score_threhold below.
    #------------------------------------------------- --------------------------------------#
    confidence = 0.02
    #------------------------------------------------- --------------------------------------#
    # The size of the non-maximum suppression value used in prediction, the larger the non-maximum suppression, the less strict the non-maximum suppression.
    #
    # This value is generally not adjusted.
    #------------------------------------------------- --------------------------------------#
    nms_iou = 0.5
    #------------------------------------------------- -------------------------------------------------- ------------#
    # Recall and Precision are not an area concept like AP, so when the threshold value is different, the Recall and Precision values of the network are different.
    #
    # By default, the Recall and Precision calculated by this code represent the corresponding Recall and Precision values when the threshold value is 0.5 (here defined as score_threhold).
    # Because the calculation of mAP needs to obtain almost all the prediction boxes, the confidence defined above cannot be changed casually.
    # A score_threhold is specifically defined here to represent the threshold value, and then the Recall and Precision values corresponding to the threshold value are found when calculating mAP.
    #------------------------------------------------- -------------------------------------------------- ------------#
    score_threhold = 0.5
    #------------------------------------------------- ------#
    # map_vis is used to specify whether to enable the visualization of VOC_map calculation
    #------------------------------------------------- ------#
    map_vis = False
    #------------------------------------------------- ------#
    # Point to the folder where the VOC dataset is located
    # By default, it points to the VOC dataset in the root directory
    #------------------------------------------------- ------#
    VOCdevkit_path = 'VOCdevkit'
    #------------------------------------------------- ------#
    # The output folder of the result, the default is map_out
    #------------------------------------------------- ------#
    map_out_path = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        centernet = CenterNet(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            centernet.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj. find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")