"""
Copyright (C) Zone24x7, Inc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by dulanj <dulanj@zone24x7.com>, 21 August 2021
"""
import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_no_obj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.split_size, self.split_size, self.num_classes + self.num_boxes * 5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        # ===================== #
        #  Calculate box loss   #
        # ===================== #

        box_predictions = exists_box *  (
            (
                    best_box * predictions[..., 26:30]
                    + (1 - best_box) * predictions[..., 21:25]
            )
        )
        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(
            box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # ======================================= #
        #  Calculate object loss if object exists #
        # ======================================= #

        pred_box = (
                best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )

        box_pred_confidence = exists_box * pred_box
        box_targets_confidence = exists_box * target[..., 20:21]

        object_loss = self.mse(
            torch.flatten(box_pred_confidence),
            torch.flatten(box_targets_confidence)
        )
        # ========================== #
        #  Calculate no object loss  #
        # ========================== #

        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ====================== #
        #  Calculate class loss  #
        # ====================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss +
            object_loss +
            self.lambda_no_obj * no_object_loss +
            class_loss
        )

        return loss
