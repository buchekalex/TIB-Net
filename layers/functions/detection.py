# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn  # Add this import
from torch.autograd import Function
from ..bbox_utils import decode, nms


class Detect(Function):
    """At test time, Detect is the final layer of SSD. Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    @staticmethod
    def forward(ctx, loc_data, conf_data, prior_data, cfg):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
            cfg: (EasyDict) Configuration dictionary
        """
        num_classes = cfg.NUM_CLASSES
        top_k = cfg.TOP_K
        nms_thresh = cfg.NMS_THRESH
        conf_thresh = cfg.CONF_THRESH
        variance = cfg.VARIANCE
        nms_top_k = cfg.NMS_TOP_K

        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(
            num, num_priors, num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors,
                                       4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4),
                               batch_priors, variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, num_classes, top_k, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(conf_thresh)
                scores = conf_scores[cl][c_mask]

                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                try:
                    ids, count = nms(
                        boxes_, scores, nms_thresh, nms_top_k)
                    count = count if count < top_k else top_k
                    output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                       boxes_[ids[:count]]), 1)
                except Exception as e:
                    print('NMS failed:', e)

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Detect backward is not implemented")
