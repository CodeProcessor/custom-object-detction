"""
Copyright (C) Zone24x7, Inc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by dulanj <dulanj@zone24x7.com>, 21 August 2021
"""
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YoloV1
from dataset import WebDataset
from utils import (
    mean_average_precision,
    get_bboxes
)
from loss import YoloLoss

seed = 28
torch.manual_seed(seed)

CHANNELS=1
NO_OF_CLASSES = 5
NO_OF_BOXES = 1
SPLIT_SIZE=7

LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = ''
IMG_DIR = '/home/dulanj/projects/FaceAuthMe/code/custom-object-detection/data/screenshots'
LABEL_DIR = '/home/dulanj/projects/FaceAuthMe/code/custom-object-detection/data/annotated_data'

torch.autograd.set_detect_anomaly(True)


class Compose(object):
    def __init__(self, _transforms):
        self.transforms = _transforms

    def __call__(self, img, boxes):
        for t in self.transforms:
            img, boxes = t(img), boxes

        return img, boxes


transform = Compose([
    transforms.Resize((448, 448)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
loss_func = YoloLoss(split_size=SPLIT_SIZE, num_boxes=NO_OF_BOXES, num_classes=NO_OF_CLASSES)


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():
    model = YoloV1(input_channels=CHANNELS, split_size=SPLIT_SIZE, num_boxes=NO_OF_BOXES, num_classes=NO_OF_CLASSES).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    train_dataset = WebDataset(
        csv_file="data/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=SPLIT_SIZE, B=NO_OF_BOXES, C=NO_OF_CLASSES
    )

    test_dataset = WebDataset(
        csv_file="data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=SPLIT_SIZE, B=NO_OF_BOXES, C=NO_OF_CLASSES
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE, C=NO_OF_CLASSES
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        train_fn(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_func
        )


if __name__ == '__main__':
    main()
