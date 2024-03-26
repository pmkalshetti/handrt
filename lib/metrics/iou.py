# Ref: https://github.com/korrawe/harp/blob/ea9d46505aa6d0b4a30066f55ddbe2c61e7e23b0/utils/eval_util.py#L41C1-L49C44

import numpy as np
import torch

def sil_iou(ref_masks, pred_masks):
    ref_bools = (ref_masks >= 0.5)
    pred_bools = (pred_masks >= 0.5)
    union = torch.logical_or(ref_bools, pred_bools)
    intersect = torch.logical_and(ref_bools, pred_bools)
    union_sum = union.sum([1,2])
    intersect_sum = intersect.sum([1,2])
    iou = intersect_sum / union_sum
    return torch.mean(iou).detach().numpy()