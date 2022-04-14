import torch
from utils.utils import calc_iou
from utils.data_structures import Bbox3D


def calc_accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    :param output: tensor output from model
    :param target: tensor ground truth
    :return: tensor accuracy
    """
    # TODO
    return torch.zeros_like(output)


def calc_ap(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    :param output: tensor output from model
    :param target: tensor ground truth
    :return: tensor ap
    """
    # TODO
    iou = calc_iou(Bbox3D(0), Bbox3D(0))
    return torch.zeros_like(output)


def calc_map(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    :param output: tensor output from model
    :param target: tensor ground truth
    :return: tensor map
    """
    # TODO
    ap = calc_ap(output, target)
    map = torch.mean(ap)
    return map
