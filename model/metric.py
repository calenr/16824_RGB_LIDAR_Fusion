import torch
from utils.box import Box
from utils.iou import IoU
from .loss import KITTILABEL_IDX, CLASSNAMES_TO_IDX
import numpy as np
from sklearn.metrics import average_precision_score

def convert_kitti_labels_to_objectron(bounding_boxes: torch.Tensor):
    """
    Convert bounding boxes (stacked along axis=0) from the Kitti
    format to the Objectron format (8 vertices)
    :param bounding_boxes: Tensor N x 16
    :param objectron_box_list: list of objectron class
    """

    assert isinstance(bounding_boxes, torch.Tensor)
    assert len(bounding_boxes.shape) == 2
    assert bounding_boxes.shape[1] == len(KITTILABEL_IDX) or bounding_boxes.shape[1] == len(KITTILABEL_IDX) - 1

    objectron_box_list = []

    for box_num in range(bounding_boxes.shape[0]):
        if bounding_boxes[box_num, KITTILABEL_IDX["class"]] != CLASSNAMES_TO_IDX["Car"]:
            continue
        rotation = np.array([0, 0, bounding_boxes[box_num, KITTILABEL_IDX["yaw"]]])
        translation = np.array([
            bounding_boxes[box_num, KITTILABEL_IDX["x"]],
            bounding_boxes[box_num, KITTILABEL_IDX["y"]],
            bounding_boxes[box_num, KITTILABEL_IDX["z"]]])
        scale = np.array([
            bounding_boxes[box_num, KITTILABEL_IDX["l"]],
            bounding_boxes[box_num, KITTILABEL_IDX["w"]],
            bounding_boxes[box_num, KITTILABEL_IDX["h"]]])
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
    raise NotImplementedError
    return torch.zeros_like(output)


def calc_ap(output: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    """
    :param output: tensor output from one sample, in the kitti frame, ordered by descending confidence
    :param target: tensor ground truth from one sample
    :param args: input args for model (i.e. threshold values)
    :return: tensor ap
    """
    # Remove target boxes with matching output boxes?

    # Manual method
    # fn = torch.sum(target_to_output_indices.clamp(min=0.0)) - torch.sum(target_to_output_indices)
    # tp = target.shape[0] - fn
    # fp = output.shape[0] - tp
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn) # This is the accuracy metric, I think...
    # # How to calculate average precision??

    # target_to_output_confidence_scores = torch.zeros(target_box_list.shape[0])
    # for ii in range(target_to_output_indices.shape[0]):
    #     if target_to_output_indices[ii] == -1:
    #         continue
    #     else:
    #         # target_to_output_confidence_scores[ii] = output_box_list[target_to_output_indices[ii]][KITTILABEL_IDX["Conf"]]
    #         target_to_output_confidence_scores[ii] = 1.0

    # ap = average_precision_score(torch.ones(target.shape[0]), target_to_output_confidence_scores)
    return torch.zeros_like(ap)


def calc_map(output_kitti_list: list[torch.Tensor], target_kitti_list: list[torch.Tensor], MAP_overlap_threshold: float) -> torch.Tensor:
    """
    :param output_kitti: list of tensor output in kitti format (N, 16), each entry of the list is for one example
    :param target_kitti: list of tensor ground truth in kitti format (N, 15), each entry of the list is for one example
    :param MAP_overlap_threshold: minimum iou value to be considered true positive
    :return: tensor map
    """
    assert(isinstance(output_kitti_list, list))
    assert(isinstance(target_kitti_list, list))
    assert(len(output_kitti_list) > 0)
    assert(len(output_kitti_list) == len(target_kitti_list))
    assert(output_kitti_list[0].shape[1] == len(KITTILABEL_IDX))
    assert(target_kitti_list[0].shape[1] == len(KITTILABEL_IDX) - 1)

    # stacked_target = torch.stack(target_kitti_list)
    # N_samples = stacked_target.shape[0]

    pred_for_ap = []
    target_for_ap = []
    # Processing one example from the batch
    for example_idx in range(len(target_kitti_list)):
        output_kitti = output_kitti_list[example_idx]
        target_kitti = target_kitti_list[example_idx]
        # Sort output_kitti by confidence_list (descending)
        _, conf_sort_idx = torch.sort(output_kitti[:, KITTILABEL_IDX["conf"]], dim=0, descending=True)
        sorted_output_kitti = output_kitti[conf_sort_idx]
        # suppressed_output = NMS(sorted_output_kitti, args.NMS_overlap_threshold)

        output_objectron_list = convert_kitti_labels_to_objectron(sorted_output_kitti)
        target_objectron_list = convert_kitti_labels_to_objectron(target_kitti)
        target_to_output_indices = torch.zeros(len(target_objectron_list), dtype=torch.long)
        # Match target bounding boxes with corresponding output bounding boxes
        for target_idx, target_objectron in enumerate(target_objectron_list):
            cur_output_indice = -1
            cur_iou = 0
            for output_idx, output_objectron in enumerate(output_objectron_list):
                iou = IoU(output_objectron, target_objectron).iou()
                if iou >= MAP_overlap_threshold and iou > cur_iou:
                    cur_output_indice = output_idx
                    cur_iou = iou
            target_to_output_indices[target_idx] = cur_output_indice

        # Now iterate through the matched target and output bbox and fill in conf values
        for ii in range(target_to_output_indices.shape[0]):
            if target_to_output_indices[ii] == -1:
                # False negative
                pred_for_ap.append(0)
                target_for_ap.append(1)
            else:
                # True positive
                pred_for_ap.append(sorted_output_kitti[target_to_output_indices[ii], KITTILABEL_IDX["conf"]].item())
                target_for_ap.append(1)

        for ii in range(sorted_output_kitti.shape[0]):
            if ii in target_to_output_indices:
                # This output bbox has been matched to a gt, hence skip
                continue
            else:
                # False positive
                pred_for_ap.append(sorted_output_kitti[ii, KITTILABEL_IDX["conf"]].item())
                target_for_ap.append(0)

    pred_for_ap = np.stack(pred_for_ap)
    target_for_ap = np.stack(target_for_ap)
    ap = average_precision_score(target_for_ap, pred_for_ap)

    # ap[sample_num] = calc_ap(sorted_output_kitti, target_list[sample_num], args)
    # map = torch.mean(ap)
    return ap

def test():
    # Conclusion: you translate then rotate
    # to debug IOU
    # kitti_label_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 0, 0, 0, 0, 0.8])
    # kitti_label_2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3, 1, 1, 0, 1.57, 0.8])
    # kitti_label_1 = torch.from_numpy(kitti_label_1)
    # kitti_label_2 = torch.from_numpy(kitti_label_2)
    # kitti_label_1 = torch.unsqueeze(kitti_label_1, dim=0)
    # kitti_label_2 = torch.unsqueeze(kitti_label_2, dim=0)
    # objectron_1 = convert_kitti_labels_to_objectron(kitti_label_1)
    # objectron_2 = convert_kitti_labels_to_objectron(kitti_label_2)
    # iou = IoU(objectron_1[0], objectron_2[0]).iou()
    # print(iou)

    output_kitti_list = []
    output1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3.01, 5, 0, 0, 0.2, 0.8],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 10, 0.01, 0, 0.3, 0.8],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 20, 0, 0.01, 0.4, 0.8]])
    output1 = torch.from_numpy(output1)
    output2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 0, 10.01, 0, 0.1, 0.8],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 0.01, 20, 0, 0.2, 0.8],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 0, 0, 30.01, 0.3, 0.8]])
    output2 = torch.from_numpy(output2)
    output_kitti_list.append(output1)
    output_kitti_list.append(output2)

    target_kitti_list = []
    target1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3.01, 5.1, 0.2, -.2, 0.1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 10, 0.01, 0.1, 0.2],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 20, 0, 0.01, 0.3]])
    target1 = torch.from_numpy(target1)
    target2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 0, 10.01, 0.2, 0.3],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 0.01, 20, 0.1, 0.1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 0.1, 0.4, 30.01, 0]])
    target2 = torch.from_numpy(target2)
    target_kitti_list.append(target1)
    target_kitti_list.append(target2)

    print(calc_map(output_kitti_list, target_kitti_list, 0.5))
