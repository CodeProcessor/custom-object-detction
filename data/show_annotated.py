"""
Copyright (C) Zone24x7, Inc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by dulanj <dulanj@zone24x7.com>, 24 August 2021
"""
import glob
import os.path

import cv2
import numpy as np

IMG_DIR = 'data/screenshots'
LABEL_DIR = 'data/annotated_data'

def get_class_dict():
    file = "data/annotated_data/classes.txt"
    class_dict = {}
    with open(file) as fp:
        for index, class_name in enumerate(fp.readlines()):
            class_dict[f"{index}"] = class_name.replace('\n', '')
    return class_dict


def plot_image_custom(image_path, boxes, image_name="test_image.jpg"):
    """Plots predicted bounding boxes on the image"""
    im = cv2.imread(image_path)
    height, width, _ = im.shape
    class_dict = get_class_dict()
    # Create a Rectangle potch
    for box in boxes:
        obj_class = class_dict[str(int(box[0]))]
        box = box[1:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        bottom_right_x = box[0] + box[2] / 2
        bottom_right_y = box[1] + box[3] / 2
        point1 = (int(upper_left_x * width), int(upper_left_y * height))
        point2 = (int(bottom_right_x * width), int(bottom_right_y * height))
        cv2.rectangle(im, pt1=point1, pt2=point2, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(im, text=obj_class, org=point1, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=2,
                    color=(0, 255, 0))
    cv2.imwrite(image_name, im)


def main():
    class_dict = get_class_dict()
    print(class_dict)
    labled_text = glob.glob(LABEL_DIR + '/*.txt')
    print(labled_text)
    for ele in labled_text:
        base_name = os.path.basename(ele)
        if base_name in ['classes.txt']:
            continue
        _name, _ext = base_name.split('.')

        image_path = os.path.join(IMG_DIR, _name + '.jpg')
        boxes = []
        print(image_path)
        with open(ele) as f:
            for line in f.readlines():
                line.replace('\n', '').strip()
                box = [float(txt.replace('\n', '')) for txt in line.split(' ')]
                box[0] = int(box[0])
                boxes.append(box)

        print(boxes)
        plot_image_custom(image_path=image_path, boxes=boxes, image_name=f"data/tagged/tagged_image_{_name}.jpg")


if __name__ == '__main__':
    main()
