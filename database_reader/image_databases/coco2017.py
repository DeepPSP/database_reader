# -*- coding: utf-8 -*-
"""
"""
import os
import numpy as np
import pandas as pd
import json
import cv2
from PIL import Image
from easydict import EasyDict as ED

from database_reader.base import ImageDataBase


__all__ = [
    "COCO"
]


class COCO2017(ImageDataBase):
    """
    """
    def __init__(self, db_path:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_path: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        super().__init__(db_name="COCO2017", db_path=db_path, working_dir=working_dir, verbose=verbose, **kwargs)
        self.image_dirs = ED({
            'train': os.path.join(self.db_path, "train2017"),
            'val': os.path.join(self.db_path, "val2017")
        })
        self.ann_dir = os.path.join(self.db_path, "annotations")
        self.ann_paths = ED({
            'captions': {
                'train': os.path.join(self.ann_dir, 'captions_train2017.json'),
                'val': os.path.join(self.ann_dir, 'captions_val2017.json'),
            },
            'instances': {
                'train': os.path.join(self.ann_dir, 'instances_train2017.json'),
                'val': os.path.join(self.ann_dir, 'instances_val2017.json'),
            },
            'keypoints': {
                'train': os.path.join(self.ann_dir, 'person_keypoints_train2017.json'),
                'val': os.path.join(self.ann_dir, 'person_keypoints_val2017.json'),
            },
        })


    def get_od_ann_csv(self) -> ED:
        """ finished, not checked,
        
        convert the annotations regarding object detection into csv files with 'standard' columns
        """
        cols = ['filename', 'width', 'height', 'iscrowd', 'image_id', 'id', 'xmin', 'ymin', 'xmax', 'ymax', 'box_width', 'box_height', 'box_area', 'category_id', 'category_name', 'supercategory']
        df_ann = ED({
            'train': pd.DataFrame(columns=cols),
            'val': pd.DataFrame(columns=cols),
        })
        for part in ['train', 'val']:
            with open(self.ann_paths.instances[part], 'r') as f:
                content = json.load(f)
                for d in content:
                    iscrowd = d['iscrowd']
                    fn = _image_id_to_filename(d['image_id')
                    category_id = d['category_id']
                    cate = [item for item in content['categories'] if item['id'] == category_id][0]
                    _id = d['id']
                    xmin,ymin,box_width,box_height = list(map(lambda i:int(round(i)), d['bbox']))
                    xmax = xmin+box_width
                    ymax = ymin+box_height
                    area = box_width*box_height
                    img = cv2.imread(os.path.join(self.image_dirs[part], fn))
                    height, width, _ = img.shape
                    vals = {
                        'filename': fn,
                        'width': width,
                        'height': height,
                        'iscrowd': iscrowd,
                        'image_id': d['image_id'],
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax,
                        'box_width': box_width,
                        'box_height': box_height,
                        'box_area': area,
                        'category_id': category_id,
                        'id': _id,
                        'category_name': cate['name'],
                        'supercategory': cate['supercategory'],
                    }
                    df_ann[part] = pd.concat([df_ann[part],pd.DataFrame([vals])],ignore_index=True)
            df_ann[part].to_csv(os.path.join(self.working_dir, f"od_{part}_coco2017.csv"), index=False)
        return df_ann


def _image_id_to_filename(image_id:int) -> str:
    """
    convert image_id to the corresponding filename of the image in COCO2017

    Parameters:
    -----------
    image_id: int,
        the `image_id` of the image, read from the annotation file
    
    Returns:
    --------
    fn: str,
        the filename of the corresponding image
    """
    fn = f"{image_id:012d}.jpg"
    return fn