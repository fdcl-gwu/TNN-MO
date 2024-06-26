# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from typing import Sized
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_keys, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_keypoints: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.num_keys = num_keys
        self.cost_class = cost_class
        self.cost_keypoints = cost_keypoints
        self.cost_giou = cost_giou
        self.cost_bbox = cost_bbox
        self.gamma = 10
        assert cost_class != 0 or cost_keypoints != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, num_boxes):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        C =  self.cost_class * cost_class #+ self.cost_bbox * cost_bbox +  self.cost_giou * cost_giou  #+ self.gamma * L1 #self.cost_class * cost_class + 
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["labels"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def linear_sum_assignment_with_inf(self, cost_matrix):
        cost_matrix = np.asarray(cost_matrix)
        min_inf = np.isneginf(cost_matrix).any()
        max_inf = np.isposinf(cost_matrix).any()
        if min_inf and max_inf:
            raise ValueError("matrix contains both inf and -inf")

        if min_inf or max_inf:
            values = cost_matrix[~np.isinf(cost_matrix)]
            min_values = values.min()
            max_values = values.max()
            m = min(cost_matrix.shape)

            positive = m * (max_values - min_values + np.abs(max_values) + np.abs(min_values) + 1)
            if max_inf:
                place_holder = (max_values + (m - 1) * (max_values - min_values)) + positive
            elif min_inf:
                place_holder = (min_values + (m - 1) * (min_values - max_values)) - positive

            cost_matrix[np.isinf(cost_matrix)] = place_holder
        return linear_sum_assignment(cost_matrix)


def build_matcher(args):
    return HungarianMatcher(num_keys=args.num_keys,cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, cost_keypoints=args.set_cost_keypoints)
