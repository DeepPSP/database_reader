# -*- coding: utf-8 -*-
"""
Module: utils_image
File: dataset_checker.py
Author: wenhao
remarks: utilities for image and annotation checking
"""
import os
from random import shuffle
import cv2
import matplotlib.pyplot as plt
from typing import Union, List


class ObjectDetectionCheck(object):
    """

    check and label validity of images and bounding box annotations for object detection datasets
    """
    def __init__(self, checker:str, ann_path:str, save_path:str, check_col:str="class", check_cls:Optional[Union[str,List[str]]]=None, **kwargs):
        """

        Parameters:
        -----------
        checker: str,
            indicator for whom checking the dataset
        ann_path: str,
            path to the csv file that stores the annotations,
            the csv file should contain at least the following columns:
            "filename", "xmin", "ymin", "xmax", "ymax",
            "filename" should better be absolute path of the images
        save_path: str,
            path to save the check result, will be a csv file with the following columns:
            "filename", "valid", "checker",
            the column "valid": 1 for valid, 0 for invalid
        check_col: str, default "class",
            the column name of the annotation csv file that stores the class of each bounding box
        check_cls: str, or list of str, optional,
            names of the classes to check,
            images and bounding box annotations with classes belonging to check_cls will be checked,
            if None, all classes of bounding boxes will be checked
        """
        self.checker = checker
        self.check_col = check_col
        self.check_cls = [check_cls] if isinstance(check_cls, str) else check_cls
        self.ann_path = ann_path
        if not self.ann_path.endswith(".csv"):
            raise ValueError("Invalid input of annotation file")
        self.df_ann = pd.read_csv(self.ann_path)
        self.check_cls = self.check_cls or self.df_ann[self.df_ann[self.check_col]].unique().tolist()
        self.save_path = save_path
        
        self.df_check_images = self.df_ann[self.df_ann['filename'].str.contains('|'.join(self.check_cls), case=False)==True].reset_index(drop=True)
        self.check_image_paths = self.df_check_images['filename'].unique().tolist()
        shuffle(self.check_image_paths)
        
        if os.path.isfile(self.save_path):
            self.df_saved = pd.read_csv(self.save_path)
        else:
            self.df_saved = pd.DataFrame(columns=['filename', 'valid', 'checker'])
            self.df_saved.to_csv(self.save_path, index=False)
        self.pending_image_paths = list(set(self.check_image_paths).difference(set(self.df_saved['filename'])))
        
        self.batch_len = kwargs.get("batch_len", 20)
        self.df_saving = pd.DataFrame(columns=['filename', 'valid', 'checker'])
        self.df_saving['filename'] = self.pending_image_paths[:self.batch_len]
        self.df_saving['valid'] = np.nan
        self.df_saving['checker'] = self.checker
        
        self.counter = 0
        self.current_image = None
        
        print(f"total images number is {len(self.check_image_paths)}, pending images number is {len(self.pending_image_paths)}")
        
    def update_save_status(self):
        """
        """
        self.df_saved = pd.read_csv(self.save_path).dropna().reset_index(drop=True)
        self.pending_image_paths = list(set(self.check_image_paths).difference(set(self.df_saved['filename'])))
        print(f"pending images updated, current total number is {len(self.pending_image_paths)}")
        self.df_saving = pd.DataFrame(columns=['filename', 'valid', 'checker'])
        self.df_saving['filename'] = self.pending_image_paths[:self.batch_len]
        self.df_saving['valid'] = np.nan
        self.df_saving['checker'] = self.checker
        self.counter = 0
        
    def save_to_file(self):
        """
        """
        self.df_saved = pd.read_csv(self.save_path).dropna().reset_index(drop=True)
        self.pending_image_paths = set(self.check_image_paths).difference(set(self.df_saved['filename']))
        print(f"pending images updated, current total number is {len(self.pending_image_paths)}")
        self.df_saving = self.df_saving[self.df_saving['filename'].isin(self.pending_image_paths)].dropna()
        self.df_saving.to_csv(self.save_path, index=False, header=False, mode='a')
        
    def __iter__(self):
        """
        """
        return self

    def __next__(self):
        """

        TODO: use tqdm
        """
        if self.counter < len(self.df_saving):
            print(f"{len(self.df_saving)} / {self.counter+1} ...")
            self.current_image = self.df_saving.loc[self.counter, 'filename']
            img = cv2.imread(self.current_image)[...,::-1]
            linewidth = max(1, int(round(max(img.shape[:2])/200)))
            df_img = self.df_check_images[self.df_check_images['filename']==self.current_image]
            display(df_img)
            img_with_boxes = img.copy()
            for _, row in df_img.iterrows():
                cv2.rectangle(img_with_boxes, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), linewidth)
            plt.figure(figsize=(12,12))
            plt.imshow(img_with_boxes)
            plt.show()
        else:
            print("one batch finished, validity labels saved and a new batch loaded")
            self.save_to_file()
            self.update_save_status()
            if len(self.pending_image_paths) == 0:
                raise StopIteration()
            self.__next__()
            
    def keep(self):
        """
        """
        current_valid = self.df_saving[self.df_saving['filename']==self.current_image]['valid'].values[0]
        if np.isnan(current_valid):
            self.df_saving.loc[self.counter, 'valid'] = 1
            self.counter += 1
            print("labelled successfully")
        else:
            self.df_saving.loc[self.counter-1, 'valid'] = 1
            print("re-labelled successfully")
    
    def dismiss(self):
        """
        """
        current_valid = self.df_saving[self.df_saving['filename']==self.current_image]['valid'].values[0]
        if np.isnan(current_valid):
            self.df_saving.loc[self.counter, 'valid'] = 0
            self.counter += 1
            print("labelled successfully")
        else:
            self.df_saving.loc[self.counter-1, 'valid'] = 0
            print("re-labelled successfully")
