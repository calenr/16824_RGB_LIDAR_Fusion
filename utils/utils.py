"""
This file contains generic helper functions
"""
from utils.data_structures import Bbox3D

def calc_iou(box1: Bbox3D, box2: Bbox3D) -> float:
    # Use third_party.Objectron.objectron.dataset.iou
    return 0.0


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
