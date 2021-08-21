"""
Copyright (C) Zone24x7, Inc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by dulanj <dulanj@zone24x7.com>, 21 August 2021
"""

import torch
import torch.nn as nn

"""
Tuples - (kernel size, no of filters, stride , pad)
Lists - [(tuple 1), (tuple 2), repeat times]
String - "MaxPool" - 2x2 Max pool layer with stride 2
"""
object_detection_arch_config = [
    (7, 64, 2, 3),
    "MaxPool",
    (3, 192, 1, 1),
    "MaxPool",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "MaxPool",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "MaxPool",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv_2d = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, bias=False, **kwargs)
        self.batch_normalize = nn.BatchNorm2d(output_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_normalize(self.conv_2d(x)))


class YoloV1(nn.Module):
    def __init__(self, input_channels=3, **kwargs):
        super(YoloV1, self).__init__()
        self.nn_architecture = object_detection_arch_config
        self.input_channels = input_channels
        self.custom_architecture = self._create_convolution_layers(self.nn_architecture)
        self.fully_connected_layers = self._create_fully_connected_layers(**kwargs)

    def forward(self, x):
        x = self.custom_architecture(x)
        return self.fully_connected_layers(torch.flatten(x, start_dim=1))

    def _create_convolution_layers(self, architecture_config):
        layers = []
        input_channels = self.input_channels
        for ele in architecture_config:
            if isinstance(ele, tuple):
                layers += [
                    CNNBlock(
                        input_channels=input_channels,
                        output_channels=ele[1],
                        kernel_size=ele[0],
                        stride=ele[2],
                        padding=ele[3]
                    )
                ]
                input_channels = ele[1]
            elif isinstance(ele, str):
                layers += [
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2
                    )
                ]
            elif isinstance(ele, list):
                for in_ele in ele:
                    in_layers = []
                    if isinstance(in_ele, tuple):
                        in_layers += [
                            CNNBlock(
                                input_channels=input_channels,
                                output_channels=in_ele[1],
                                kernel_size=in_ele[0],
                                stride=in_ele[2],
                                padding=in_ele[3]
                            )
                        ]
                        input_channels = in_ele[1]
                    elif isinstance(in_ele, int):
                        for _ in range(in_ele):
                            layers += in_layers
        return nn.Sequential(*layers)

    def _create_fully_connected_layers(self, split_size, num_boxes, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, split_size * split_size * (num_boxes * 5 + num_classes))
        )


def test():
    model = YoloV1(split_size=7, num_boxes=1, num_classes=5)
    x = torch.randn(2, 3, 448, 448)
    print(model(x).shape)


if __name__ == '__main__':
    test()
