"""
This module contains utility functions for working with Detectron2.

It defines the get_dataset_dicts function.
"""

import os
import cv2

from detectron2.structures import BoxMode

# write a function that loads the dataset into detectron2's standard format
def get_dataset_dicts(data_root, txt_file):
    """
    Load the dataset in the format of the matadero dataset.
    """
    dataset_dicts = []
    filenames = []
    csv_path = os.path.join(data_root, txt_file)
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            filenames.append(line.rstrip())

    for idx, filename in enumerate(filenames):
        record = {}

        image_path = os.path.join(data_root, filename)

        height, width = cv2.imread(image_path).shape[:2] # pylint: disable=no-member

        record['file_name'] = image_path
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width

        image_filename = os.path.basename(filename)
        image_name = os.path.splitext(image_filename)[0]
        annotation_path = os.path.join(data_root, 'labels', '{}.txt'.format(image_name))
        annotation_rows = []

        with open(annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                temp = line.rstrip().split(" ")
                annotation_rows.append(temp)

        objs = []
        for row in annotation_rows:
            xcentre = int(float(row[1])*width)
            ycentre = int(float(row[2])*height)
            bwidth = int(float(row[3])*width)
            bheight = int(float(row[4])*height)

            xmin = int(xcentre - bwidth/2)
            ymin = int(ycentre - bheight/2)
            xmax = xmin  + bwidth
            ymax = ymin + bheight

            obj= {
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_mode': BoxMode.XYXY_ABS,
                # alternatively, we can use bbox_mode = BoxMode.XYWH_ABS
                # 'bbox': [xmin, ymin, bwidth, bheight],
                # 'bbox_mode': BoxMode.XYWH_ABS,
                'category_id': int(row[0]),
                'iscrowd': 0
            }

            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts