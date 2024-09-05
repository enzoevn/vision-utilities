"""
This module helps in creating the structure of the dataset.

It defines the DatasetConfig class which holds the configuration for the dataset,
including paths to annotations, images, and JSON file names.
"""
from dataclasses import dataclass
import os


@dataclass
class DatasetConfig:
    """
    This class holds the configuration for the dataset.
    Attributes:
        annotations_path (str): Path to the annotations directory.
        images_path (str): Path to the images directory.
        json_file_names (list): List of JSON file names(train.json, val.json).
    """
    annotations_path: str
    images_path: str
    json_file_names: list

    def __str__(self):
        return f"DatasetConfig(annotations_path={self.annotations_path}, images_path={self.images_path}, json_file_names={self.json_file_names})"
    
    def dict_labels_path(self):
        """
        Return a dictionary with the paths to the labels.
        labels_path = {
        train: path_to_train_ann,
        val: path_to_val_ann,
        test: path_to_test_ann
        }
        """
        labels_path_dict = {}
        labels_path = os.listdir(self.annotations_path)
        for label_path in labels_path:
            try:
                labels_path_dict[label_path.split(".")[0]] = os.path.join(
                    self.annotations_path, label_path)
            except IndexError:
                labels_path_dict[label_path] = None
        return labels_path_dict
        # for i, key in enumerate(labels_path.keys()):
        #     try:
        #         labels_path[key].split(".")[0] == 'train' 
        #         labels_path[key] = os.path.join(
        #             self.annotations_path, self.json_file_names[i])
        #     except IndexError:
        #         labels_path[key] = None  # or handle the error as needed
        # return labels_path
    

