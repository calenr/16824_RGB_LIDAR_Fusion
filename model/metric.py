import torch
# from utils.utils import calc_iou
# from utils.data_structures import Bbox3D
from third_party.Objectron.objectron.dataset.box import Box
from third_party.Objectron.objectron.dataset.iou import IoU
from model.loss import convert_yolo_output_to_kitti_labels
from model.loss import KITTILABEL_IDX
import numpy as np
from sklearn.metrics import average_precision_score

def convert_kitti_labels_to_objectron(self, bounding_boxes):
    """
    Convert bounding boxes (stacked along axis=0) from the Kitti
    format to the Objectron format (8 vertices)
    """

    assert isinstance(bounding_boxes, torch.Tensor)
    assert len(bounding_boxes.shape) == 2
    assert bounding_boxes.shape[1] == len(KITTILABEL_IDX)

    objectron_box_list = []

    for box_num in range(bounding_boxes.shape[0]):
        rotation = np.array([0, 0, bounding_boxes[box_num, KITTILABEL_IDX["yaw"]]])
        translation = np.array([
            bounding_boxes[box_num, KITTILABEL_IDX["x"]],
            bounding_boxes[box_num, KITTILABEL_IDX["y"]],
            bounding_boxes[box_num, KITTILABEL_IDX["z"]]])
        scale = np.array([
            bounding_boxes[box_num, KITTILABEL_IDX["h"]],
            bounding_boxes[box_num, KITTILABEL_IDX["w"]],
            bounding_boxes[box_num, KITTILABEL_IDX["l"]]])
        box = Box.from_transformation(rotation, translation, scale)
        objectron_box_list.append(box)

    return objectron_box_list

def NMS(self, flat_example_kitti: torch.Tensor, IoU_th: float=0.5):
    """
    Remove overlapping bounding boxes through non-max suppression
    Input is sorted in descending confidence order
    """

    assert isinstance(flat_example_kitti, torch.Tensor)
    assert len(flat_example_kitti.shape) == 2
    assert flat_example_kitti.shape[1] == self.box_length
    assert isinstance(IoU_th, float)
    assert IoU_th <= 1.0 and IoU_th >= 0.0

    # Convert example tensor to list of box objects
    object_list = convert_kitti_labels_to_objectron(self, flat_example_kitti)
    # Set up indice lists to shuffle bounding boxes
    input_indices = range(flat_example_kitti.shape[0])
    delete_indices = []
    surviving_indices = []
    # NMS algorithm
    while(len(input_indices) > 0):
        cur_indice = input_indices[0]
        # Update output list with the highest confidence box remaining
        surviving_indices.append(cur_indice)
        for ii in range(input_indices):
            # Check for box overlap with remaining boxes
            if ii == 0 or IoU(object_list[cur_indice], object_list[input_indices[ii]]).iou >= IoU_th:
                delete_indices.append(input_indices[ii])
        # Remove current indice and overlapping boxes from possible inputs
        input_indices = [i for j, i in enumerate(input_indices) if j not in delete_indices]
        delete_indices = []
    
    suppressed_flat_example_kitti = flat_example_kitti[surviving_indices, :]

    return suppressed_flat_example_kitti

def calc_accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    :param output: tensor output from model
    :param target: tensor ground truth
    :return: tensor accuracy
    """
    # TODO
    return torch.zeros_like(output)


def calc_ap(output: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    """
    :param output: tensor output from one sample, in the kitti frame, ordered by descending confidence
    :param target: tensor ground truth from one sample
    :param args: input args for model (i.e. threshold values)
    :return: tensor ap
    """
    
    output_box_list = convert_kitti_labels_to_objectron(output)
    target_box_list = convert_kitti_labels_to_objectron(target)
    target_to_output_indices = torch.zeros(target_box_list.shape[0])
    # Match target bounding boxes with corresponding output bounding boxes
    for target_num in range(len(target_box_list)):
        cur_output_indice = -1
        cur_iou = 0
        for output_num in range(len(output_box_list)):
            iou = IoU(target_box_list[target_num], output_box_list[output_num]).iou
            if iou >= args.MAP_overlap_threshold and iou > cur_iou:
                cur_output_indice = output_num
                cur_iou = iou
        target_to_output_indices[target_num] = cur_output_indice

    # Remove target boxes with matching output boxes?

    # Manual method
    fn = torch.sum(target_to_output_indices.clamp(min=0.0)) - torch.sum(target_to_output_indices))
    tp = target.shape[0] - fn
    fp = output.shape[0] - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn) # This is the accuracy metric, I think...
    # How to calculate average precision??

    target_to_output_confidence_scores = torch.zeros(target_box_list.shape[0])
    for ii in range(target_to_output_indices.shape[0]):
        if target_to_output_indices[ii] == -1:
            continue
        else:
            # target_to_output_confidence_scores[ii] = output_box_list[target_to_output_indices[ii]][KITTILABEL_IDX["Conf"]]
            target_to_output_confidence_scores[ii] = 1.0

    ap = average_precision_score(torch.ones(target.shape[0]), target_to_output_confidence_scores)
    return torch.zeros_like(ap)


def calc_map(output_list: list(torch.Tensor), target_list: list(torch.Tensor), calibs_list: list(torch.Tensor), args) -> torch.Tensor:
    """
    :param output_list: list of tensor output from model
    :param target_list: list of tensor ground truth
    :param calibs_list: list of calibration information
    :param args: input args for model (i.e. threshold values)
    :return: tensor map
    """
    # TODO

    ap = torch.zeros(len(output_list))
    for sample_num in range(len(output_list)):
        output_kitti, confidence_list = convert_yolo_output_to_kitti_labels(output_list[sample_num], calibs_list[sample_num], args.confidence_threshold)
        # Sort output_kitti by confidence_list (descending)
        sorted_output_kitti = output_kitti
        suppressed_output = NMS(sorted_output_kitti, args.NMS_overlap_threshold)
        ap[sample_num] = calc_ap(suppressed_output, target_list[sample_num], args)
    map = torch.mean(ap)
    return map
