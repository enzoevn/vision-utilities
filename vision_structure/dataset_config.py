"""
This module helps in creating the structure of the dataset.

It defines the DatasetConfig class which holds the configuration for the dataset,
including paths to annotations, images, and JSON file names.
"""
from dataclasses import dataclass

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
    