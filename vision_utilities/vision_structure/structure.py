"""
This module helps in creating the structure of the dataset.

It defines the Structure class.
"""

import os
import shutil

from pylabel import importer
import pandas as pd
from tqdm import tqdm

from ..vision_structure.images import convert_png_to_jpg
from ..config import dataset_config

pd.set_option('future.no_silent_downcasting', True)


class DataStructure:
    """
    This class helps in creating the structure of the dataset.
    Attributes:
        config (DatasetConfig): Configuration for the dataset.
    """

    def __init__(self, config: dataset_config.DatasetConfig):
        self.config = config

    def detectron2(self, structure_name='detectron2', filter_images=True, classes=None):
        '''
        Create the structure folders to be used with detectron2.
        Ate the moment, only work with coco format input json

        Args:
            filter_images (Bool):Filter images without annotations.
            classes (list): List of classes to be used.
        Example:
            >>> structure = DatasetStructure(config)
            >>> structure.detectron2()
        '''
        base_path = os.getcwd()
        structure_path = os.path.join(base_path, structure_name)
        # path_to_train_ann = os.path.join(
        #     self.config.annotations_path,
        #     self.config.json_file_names[0]
        # )

        labels_path = self.config.dict_labels_path()
        if labels_path['train']:
            train_dataset = importer.ImportCoco(
                path=labels_path['train'], path_to_images=self.config.images_path)
            if filter_images:
                train_dataset.df = train_dataset.df[train_dataset.df['cat_name'] != '']
            if classes is not None:
                train_dataset.df = train_dataset.df[train_dataset.df['cat_name'].isin(
                    classes)]
            train_dataset.export.ExportToYoloV5(output_path=os.path.join(
                structure_path, 'labels'), copy_images=True, cat_id_index=0)
            train_images = train_dataset.df.groupby("img_id").first()
            for img in train_images.get("img_filename").values:
                with open(os.path.join(structure_path, "train.txt"), "a", encoding="utf-8") as f:
                    f.write("images/" + img + "\n")
        if labels_path['val']:
            val_dataset = importer.ImportCoco(
                path=labels_path['val'], path_to_images=self.config.images_path)
            if filter_images:
                val_dataset.df = val_dataset.df[val_dataset.df['cat_name'] != '']
            if classes is not None:
                val_dataset.df = val_dataset.df[val_dataset.df['cat_name'].isin(
                    classes)]
            val_dataset.export.ExportToYoloV5(output_path=os.path.join(
                structure_path, 'labels'), copy_images=True, cat_id_index=0)
            val_images = val_dataset.df.groupby("img_id").first()
            for img in val_images.get("img_filename").values:
                with open(os.path.join(structure_path, "val.txt"), "a", encoding="utf-8") as f:
                    f.write("images/" + img + "\n")
        if labels_path['test']:
            test_dataset = importer.ImportCoco(
                path=labels_path['test'], path_to_images=self.config.images_path)
            if filter_images:
                test_dataset.df = test_dataset.df[test_dataset.df['cat_name'] != '']
            if classes is not None:
                test_dataset.df = test_dataset.df[test_dataset.df['cat_name'].isin(
                    classes)]
            test_dataset.export.ExportToYoloV5(output_path=os.path.join(
                structure_path, 'labels'), copy_images=True, cat_id_index=0)
            test_images = test_dataset.df.groupby("img_id").first()
            for img in test_images.get("img_filename").values:
                with open(os.path.join(structure_path, "test.txt"), "a", encoding="utf-8") as f:
                    f.write("images/" + img + "\n")

        os.remove(os.path.join(structure_path, 'dataset.yaml'))

    def yolo(self, structure_name='yolo', filter_images=True):
        '''
        Create the structure folders to be used with detectron2.
        Ate the moment, only work with coco format input json

        Args:
            filter (Bool):Filter images without annotations.
        Example:
            >>> structure = DatasetStructure(config)
            >>> structure.yolo()
        '''

        base_path = os.getcwd()
        structure_path = os.path.join(base_path, structure_name)
        labels_path = self.config.dict_labels_path()
        if labels_path['train']:
            train_dataset = importer.ImportCoco(
                path=labels_path['train'], path_to_images=self.config.images_path)
            if filter_images:
                train_dataset.df = train_dataset.df[train_dataset.df['cat_name'] != '']
            train_dataset.export.ExportToYoloV5(
                output_path=os.path.join(structure_path,
                                         'train_labels'), copy_images=True, cat_id_index=0)
            shutil.move(os.path.join(structure_path, 'images'),
                        os.path.join(structure_path, 'train'))
        if labels_path['val']:
            val_dataset = importer.ImportCoco(
                path=labels_path['val'], path_to_images=self.config.images_path)
            if filter_images:
                val_dataset.df = val_dataset.df[val_dataset.df['cat_name'] != '']
            val_dataset.export.ExportToYoloV5(output_path=os.path.join(
                structure_path, 'val_labels'), copy_images=True, cat_id_index=0)
            shutil.move(os.path.join(structure_path, 'images'),
                        os.path.join(structure_path, 'val'))
        if labels_path['test']:
            test_dataset = importer.ImportCoco(
                path=labels_path['test'], path_to_images=self.config.images_path)
            if filter_images:
                test_dataset.df = test_dataset.df[test_dataset.df['cat_name'] != '']
            test_dataset.export.ExportToYoloV5(output_path=os.path.join(
                structure_path, 'test_labels'), copy_images=True, cat_id_index=0)
            shutil.move(os.path.join(structure_path, 'images'),
                        os.path.join(structure_path, 'test'))
            
        os.makedirs(os.path.join(structure_path, 'images'))
        shutil.move(os.path.join(structure_path, 'train'),
                    os.path.join(structure_path, 'images'))
        shutil.move(os.path.join(structure_path, 'val'),
                    os.path.join(structure_path, 'images'))
        shutil.move(os.path.join(structure_path, 'test'),
                    os.path.join(structure_path, 'images'))
        
        os.makedirs(os.path.join(structure_path, 'labels'))
        shutil.move(os.path.join(structure_path, 'train_labels'),
                    os.path.join(structure_path, 'train'))
        shutil.move(os.path.join(structure_path, 'val_labels'),
                    os.path.join(structure_path, 'val'))
        shutil.move(os.path.join(structure_path, 'test_labels'),
                    os.path.join(structure_path, 'test'))
        
        shutil.move(os.path.join(structure_path, 'train'),
                    os.path.join(structure_path, 'labels'))
        shutil.move(os.path.join(structure_path, 'val'),
                    os.path.join(structure_path, 'labels'))
        shutil.move(os.path.join(structure_path, 'test'),
                    os.path.join(structure_path, 'labels'))
        

    def tensorflow(self, structure_name='yolo', filter_images=True, classes=None):
        '''
        Create the structure folders to be used with detectron2.
        Ate the moment, only work with coco format input json

        Args:
            filter (Bool):Filter images without annotations.
        Example:
            >>> structure = DatasetStructure(config)
            >>> structure.yolo()
        '''

        base_path = os.getcwd()
        structure_path = os.path.join(base_path, structure_name)
        labels_path = self.config.dict_labels_path()

        if labels_path['train']:
            os.makedirs(os.path.join(structure_path, 'train/images'), exist_ok=True)
        if labels_path['val']:
            os.makedirs(os.path.join(structure_path, 'val/images'), exist_ok=True)
        if labels_path['test']:
            os.makedirs(os.path.join(structure_path, 'test/images'), exist_ok=True)

        if labels_path['train']:
            train_dataset = importer.ImportCoco(
                path=labels_path['train'], path_to_images=self.config.images_path)
            if filter_images:
                train_dataset.df = train_dataset.df[train_dataset.df['cat_name'] != '']
            if classes is not None:
                train_dataset.df = train_dataset.df[train_dataset.df['cat_name'].isin(
                    classes)]
            train_dataset.export.ExportToVoc(
                output_path=os.path.join(structure_path,
                                         'train/labels'))

            train_images = train_dataset.df.groupby("img_id").first()
            for img in tqdm(train_images.get("img_filename").values, desc="Copying train images"):
                shutil.copy2(os.path.join(self.config.images_path, img),
                             os.path.join(structure_path, 'train/images'))
            convert_png_to_jpg(os.path.join(structure_path, 'train/images'))

        if labels_path['val']:
            val_dataset = importer.ImportCoco(
                path=labels_path['val'], path_to_images=self.config.images_path)
            if filter_images:
                val_dataset.df = val_dataset.df[val_dataset.df['cat_name'] != '']
            if classes is not None:
                val_dataset.df = val_dataset.df[val_dataset.df['cat_name'].isin(
                    classes)]
            val_dataset.export.ExportToVoc(output_path=os.path.join(
                structure_path, 'val/labels'))

            val_images = val_dataset.df.groupby("img_id").first()
            for img in tqdm(val_images.get("img_filename").values, desc="Copying val images"):
                shutil.copy2(os.path.join(self.config.images_path, img),
                             os.path.join(structure_path, 'val/images'))
            convert_png_to_jpg(os.path.join(structure_path, 'val/images'))

        if labels_path['test']:
            test_dataset = importer.ImportCoco(
                path=labels_path['test'], path_to_images=self.config.images_path)
            if filter_images:
                test_dataset.df = test_dataset.df[test_dataset.df['cat_name'] != '']
            if classes is not None:
                test_dataset.df = test_dataset.df[test_dataset.df['cat_name'].isin(
                    classes)]
            test_dataset.export.ExportToVoc(output_path=os.path.join(
                structure_path, 'test/labels'))

            test_images = test_dataset.df.groupby("img_id").first()
            for img in tqdm(test_images.get("img_filename").values, desc="Copying test images"):
                shutil.copy2(os.path.join(self.config.images_path, img),
                             os.path.join(structure_path, 'test/images'))
            convert_png_to_jpg(os.path.join(structure_path, 'test/images'))

            
        
