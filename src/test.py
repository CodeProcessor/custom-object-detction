#!/usr/bin/env python
"""
@Filename:    test.py
@Author:      dulanj
@Time:        2021-08-22 15.26
"""
import torch

from src.dataset import WebDataset
from src.model import YoloV1
from src.train import transform
from src.utils import cellboxes_to_boxes, non_max_suppression, plot_image, plot_image_custom
from torch.utils.data import DataLoader

CHANNELS=1
NO_OF_CLASSES = 5
NO_OF_BOXES = 1
SPLIT_SIZE=7
PATH='trained_models/best-model-parameters.pt'

LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = ''
IMG_DIR = 'data/screenshots'
LABEL_DIR = 'data/annotated_data'

def test():

    model = YoloV1(input_channels=CHANNELS, split_size=SPLIT_SIZE, num_boxes=NO_OF_BOXES, num_classes=NO_OF_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(PATH))

    test_dataset = WebDataset(
        csv_file="data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=SPLIT_SIZE, B=NO_OF_BOXES, C=NO_OF_CLASSES
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    for x, y in test_loader:
       x = x.to(DEVICE)
       for idx in range(3):
           bboxes = cellboxes_to_boxes(model(x), C=NO_OF_CLASSES)
           bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
           print(bboxes)
           # plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
           plot_image_custom(x[idx].permute(1,2,0).to("cpu"), bboxes, image_name=f"assets/Predicted_Image_{idx}.jpg")
           print(idx)

       import sys
       sys.exit()

if __name__ == '__main__':
    test()