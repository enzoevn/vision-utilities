""" Image processing functions. """

import os
from PIL import Image
from tqdm import tqdm

def convert_png_to_jpg(path):
    """
    Convert all images in a folder to jpg format.
    """
    for filename in tqdm(os.listdir(path), desc="Converting images to jpg"):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(path, filename))
            img = img.convert('RGB')
            img.save(os.path.join(path, filename.replace('.png', '.jpg')), 'JPEG')
            os.remove(os.path.join(path, filename))
        else:
            continue


