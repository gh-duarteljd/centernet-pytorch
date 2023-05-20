import numpy as np
import torch
from torch import nn
from torchvision.ops import nms


def pool_nms(heat, kernel = 3):
     pad = (kernel - 1) // 2

     hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
     keep = (hmax == heat). float()
     return heat * keep

def decode_bbox(pred_hms, pred_whs, pred_offsets, confidence, cuda):
     #------------------------------------------------- ------------------------#
     # When using 512x512x3 pictures for coco data set prediction
     # h = w = 128 num_classes = 80
     # Hot map heat map -> b, 80, 128, 128,
     # Perform non-maximum suppression of the heat map, and use 3x3 convolution to filter the heat map for the maximum value
     # Find the feature point with the highest score in a certain area.
     #------------------------------------------------- ------------------------#
     pred_hms = pool_nms(pred_hms)
    
     b, c, output_h, output_w = pred_hms.shape
     detects = []
     #------------------------------------------------- ------------------------#
     # Only one image is passed in, and the loop is only performed once
     #------------------------------------------------- ------------------------#
     for batch in range(b):
         #------------------------------------------------- ------------------------#
         # heat_map 128*128, num_classes heat map
         # pred_wh 128*128, 2 Predicted width and height of feature points
         # There is a little problem in the pre-processing and post-processing video of the prediction process, either to adjust the parameters, or to adjust the width and height
         # pred_offset 128*128, 2 xy axis offset of feature points
         #------------------------------------------------- ------------------------#
         heat_map = pred_hms[batch].permute(1, 2, 0).view([-1, c])
         pred_wh = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
         pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

         yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
         #------------------------------------------------- ------------------------#
         # xv 128*128, the x-axis coordinates of feature points
         # yv 128*128, the y-axis coordinates of the feature points
         #------------------------------------------------- ------------------------#
         xv, yv = xv.flatten().float(), yv.flatten().float()
         if cuda:
             xv = xv.cuda()
             yv = yv.cuda()

         #------------------------------------------------- ------------------------#
         # class_conf 128*128, the type confidence of feature points
         # class_pred 128*128, the type of feature points
         #------------------------------------------------- ------------------------#
         class_conf, class_pred = torch.max(heat_map, dim = -1)
         mask = class_conf > confidence

         #----------------------------------------#
         # Take out the corresponding result after scoring and filtering
         #----------------------------------------#
         pred_wh_mask = pred_wh[mask]
         pred_offset_mask = pred_offset[mask]
         if len(pred_wh_mask) == 0:
             detects.append([])
             continue

         #----------------------------------------#
         # Calculate the center of the adjusted prediction box
         #----------------------------------------#
         xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
         yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
         #----------------------------------------#
         # Calculate the width and height of the prediction box
         #----------------------------------------#
         half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
         #----------------------------------------#
         # Get the upper left and lower right corners of the prediction box
         #----------------------------------------#
         bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
         bboxes[:, [0, 2]] /= output_w
         bboxes[:, [1, 3]] /= output_h
         detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
         detects.append(detect)

     return detects

def bbox_iou(box1, box2, x1y1x2y2=True):
     """
         Calculating IOUs
     """
     if not x1y1x2y2:
         b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
         b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
         b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
         b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
     else:
         b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
         b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

     inter_rect_x1 = torch.max(b1_x1, b2_x1)
     inter_rect_y1 = torch.max(b1_y1, b2_y1)
     inter_rect_x2 = torch.min(b1_x2, b2_x2)
     inter_rect_y2 = torch.min(b1_y2, b2_y2)

     inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                  torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
                 
     b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
     b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
     iou = inter_area / torch.clamp(b1_area + b2_area - inter_area, min = 1e-6)

     return iou

def centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
     #------------------------------------------------- ----------------#
     # Put the y-axis in front because it is convenient to multiply the width and height of the prediction frame and the image
     #------------------------------------------------- ----------------#
     box_yx = box_xy[..., ::-1]
     box_hw = box_wh[..., ::-1]
     input_shape = np.array(input_shape)
     image_shape = np.array(image_shape)

     if letterbox_image:
         #------------------------------------------------- ----------------#
         # The offset obtained here is the offset of the effective area of the image relative to the upper left corner of the image
         # new_shape refers to the width and height scaling
         #------------------------------------------------- ----------------#
         new_shape = np. round(image_shape * np. min(input_shape/image_shape))
         offset = (input_shape - new_shape)/2./input_shape
         scale = input_shape/new_shape

         box_yx = (box_yx - offset) * scale
         box_hw *= scale

     box_mins = box_yx - (box_hw / 2.)
     box_maxes = box_yx + (box_hw / 2.)
     boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2] ], axis=-1)
     boxes *= np. concatenate([image_shape, image_shape], axis=-1)
     return boxes

def postprocess(prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0.4):
     output = [None for _ in range(len(prediction))]
    
     #------------------------------------------------- ---------#
     # The prediction only uses one image and will only be performed once
     #------------------------------------------------- ---------#
     for i, image_pred in enumerate(prediction):
         detections = prediction[i]
         if len(detections) == 0:
             continue
         #------------------------------------------#
         # Get all types contained in the prediction result
         #------------------------------------------#
         unique_labels = detections[:, -1].cpu().unique()

         if detections.is_cuda:
             unique_labels = unique_labels.cuda()
             detections = detections.cuda()

         for c in unique_labels:
             #------------------------------------------#
             # Obtain all the prediction results after a certain category of score screening
             #------------------------------------------#
             detections_class = detections[detections[:, -1] == c]
             if need_nms:
                 #------------------------------------------#
                 # Using the official non-maximum suppression will be faster!
                 #------------------------------------------#
                 keep = nms(
                     detections_class[:, :4],
                     detections_class[:, 4],
                     nms_thres
                 )
                 max_detections = detections_class[keep]

                 # #---------------------------------------------#
                 # # Sort by confidence of existing objects
                 # #---------------------------------------------#
                 # _, conf_sort_index = torch. sort(detections_class[:, 4], descending=True)
                 # detections_class = detections_class[conf_sort_index]
                 # #---------------------------------------------#
                 # # Perform non-maximum suppression
                 # #---------------------------------------------#
                 # max_detections = []
                 # while detections_class. size(0):
                 # #------------------------------------------------ ---#
                 # # Take out the one with the highest confidence in this category, and judge it step by step.
                 # # Determine whether the degree of overlap is greater than nms_thres, if so, remove it
                 # #------------------------------------------------ ---#
                 # max_detections.append(detections_class[0].unsqueeze(0))
                 # if len(detections_class) == 1:
                 #break
                 # ious = bbox_iou(max_detections[-1], detections_class[1:])
                 # detections_class = detections_class[1:][ious < nms_thres]
                 # #---------------------------------------------#
                 # # stack
                 # #---------------------------------------------#
                 # max_detections = torch.cat(max_detections).data
             else:
                 max_detections = detections_class
            
             output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

         if output[i] is not None:
             output[i] = output[i].cpu().numpy()
             box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][: , 0:2]
             output[i][:, :4] = centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
     return output