"""
remarks: utilities for io of images for tensorflow object detection
"""
import os
import pandas as pd
import tensorflow as tf
import glob
from random import shuffle
from collections import namedtuple, OrderedDict
import xml.etree.ElementTree as ET
from typing import Tuple, Union, Optional, Dict, List


__all__ = [
    "dataset_to_tfrecords",
    "voc_to_yolo",
    "yolo_to_voc",
]


def dataset_to_tfrecords(img_paths:Union[str, List[str]], ann_paths:Union[str, List[str]], tfrecords_save_path:str, pbtxt_dict:Dict[str,int], train_ratio:float=0.8, class_map:Optional[Dict[str,str]]=None, csv_save_path:Optional[str]=None, **kwargs):
    """ finished, checked,

    to tfrecords for object detection training

    Parameters:
    -----------
    img_paths: str, or list of str,
        path(s) for the image files
    ann_paths: str, or list of str,
        path(s) for the bounding box annotation xml files
    tfrecords_save_path: str,
        root path to store the tfrecords for training and test
    pbtxt_dict: dict,
        label map, from class name to class number
    train_ratio: float, default 0.8,
        train test split ratio
    class_map: dict, optional,
        label map, from class names of the annotations to the class names for training
    csv_save_path: str, optional,
        path to store the csv files containing infomation (filename, width, height, class, xmin, ymin, xmax, ymax, subclass, area) of the whole dataset, and the traing set and the test set

    Returns:
    --------
    ret, dict,
        with items "nb_train", "nb_test"
    """
    import time
    if isinstance(img_paths, str) and isinstance(ann_paths, str):
        ip, ap = [img_paths], [ann_paths]
    elif isinstance(img_paths, (list, tuple)) and isinstance(ann_paths, (list, tuple)) and len(img_paths) == len(ann_paths):
        ip, ap = list(img_paths), list(ann_paths)
    else:
        raise ValueError("Invalid input!")

    df_info = pd.DataFrame()
    for i, a in zip(ip, ap):
        df_tmp = voc_to_df(img_path=i, ann_path=a, save_path=None, class_map=class_map, **kwargs)
        df_info = pd.concat([df_info, df_tmp])
    df_info = df_info.reset_index(drop=True)
    
    all_files = df_info['filename'].unique().tolist()
    shuffle(all_files)
    split_idx = int(train_ratio*len(all_files))
    train_files = all_files[: split_idx]
    test_files = all_files[split_idx:]
    df_train = df_info[df_info['filename'].isin(train_files)]
    df_test = df_info[df_info['filename'].isin(test_files)]
    
    save_suffix = int(time.time())
    if csv_save_path is not None:
        if not os.path.exists(csv_save_path):
            os.makedirs(csv_save_path)
        df_info.to_csv(os.path.join(csv_save_path, "all_{}.csv".format(save_suffix)), index=False)
        df_train.to_csv(os.path.join(csv_save_path, "train_{}.csv".format(save_suffix)), index=False)
        df_test.to_csv(os.path.join(csv_save_path, "test_{}.csv".format(save_suffix)), index=False)

    if not os.path.exists(tfrecords_save_path):
        os.makedirs(tfrecords_save_path)
    nb_train = df_to_tfrecord(
        df=df_train,
        save_path=os.path.join(tfrecords_save_path, "train_{}.record".format(save_suffix)),
        pbtxt_dict=pbtxt_dict,
    )
    nb_test = df_to_tfrecord(
        df=df_test,
        save_path=os.path.join(tfrecords_save_path, "test_{}.record".format(save_suffix)),
        pbtxt_dict=pbtxt_dict,
    )
    ret = {"nb_train":nb_train, "nb_test":nb_test}

    return ret


def voc_to_df(img_path:str, ann_path:str, save_path:Optional[str]=None, class_map:Optional[Dict[str,str]]=None, **kwargs) -> pd.DataFrame:
    """ finished, checked,

    pascal voc annotations to one csv file

    Parameters:
    -----------
    img_path: str,
        path for the image files
    ann_path: str,
        path for the bounding box annotation xml files
    save_path: str, optional,
        path to store the csv file
    class_map: dict, optional,
        label map, from class names of the annotations to the class names for training

    Returns:
    --------
    xml_df: DataFrame,
        annotations in one DataFrame
    """
    xml_list = []  
    for xml_file in glob.glob(os.path.join(ann_path, '*.xml')):  
        tree = ET.parse(xml_file)  
        root = tree.getroot() 
#         print(xml_file)
        for member in root.findall('object'):
            fw = int(root.find('size').find('width').text)
            fh = int(root.find('size').find('height').text)
            subcls_name = member.find('name').text
            xmin = int(member.find('bndbox').find('xmin').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            area = (xmax-xmin)*(ymax-ymin)
            value = (os.path.join(img_path, root.find('filename').text),
                     fw,
                     fh,
                     xmin,
                     ymin,
                     xmax,
                     ymax,
                     subcls_name,
                     area,
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'subclass', 'area']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    if class_map is None:
        xml_df['class'] = xml_df['subclass']
    else:
        xml_df['class'] = xml_df['subclass'].apply(lambda sc:class_map[sc])
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'subclass', 'area']
    xml_df = xml_df[column_name]
    if save_path is not None:
        xml_df.to_csv(save_path, index=False)
    return xml_df


def split(df:pd.DataFrame, group) -> List[namedtuple]:
    """
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tfexample(group:namedtuple, pbtxt_dict:Dict[str,int]) -> tf.train.Example:
    """ finished, checked,

    one image with bounding box annotations to one tf Example

    Parameters:
    -----------
    group: namedtuple,
        with "filename" and "data", "data" consisting of bounding boxes and image width, height
    pbtxt_dict: dict,
        label map, from class name to class number

    Returns:
    --------
    tf_example: Example,
    """
    with tf.gfile.GFile(group.filename, 'rb') as fid:
        encoded_jpg = fid.read()
    # encoded_jpg_io = io.BytesIO(encoded_jpg)  
    # image = Image.open(encoded_jpg_io)  
    # width, height = image.size
        
    filename = os.path.basename(group.filename).encode('utf8')
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        width = row['width']
        height = row['height']
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(pbtxt_dict[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(filename),
        'image/source_id': _bytes_feature(filename),
        'image/encoded': _bytes_feature(encoded_jpg),
        'image/format': _bytes_feature(b'jpg'),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/text': _bytes_list_feature(classes_text),
        'image/object/class/label': _int64_list_feature(classes),
    }))
    return tf_example


def df_to_tfrecord(df:pd.DataFrame, save_path:str, pbtxt_dict:Dict[str,int]) -> int:
    """ finished, checked,

    construct tfrecord from information stored in a DataFrame,

    Parameters:
    -----------
    df: DataFrame,
        the DataFrame which stores the information for constructing the tfrecord
    save_path: str,
        path to save the tfrecord
    pbtxt_dict: dict,
        label map, from class name to class number
    
    Returns:
    --------
    nb_samples: int,
        number of samples (images) for the tfrecord
    """
    writer = tf.python_io.TFRecordWriter(save_path)
    nb_samples = 0
    grouped = split(df, 'filename')
    for group in grouped:
        tf_example = create_tfexample(group, pbtxt_dict)
        writer.write(tf_example.SerializeToString())
        nb_samples += 1
    writer.close()
    print('Successfully created the TFRecords: {}'.format(save_path))
    print('nb_samples =', nb_samples)
    return nb_samples


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def yolo_to_voc(yolo_path:str, save_voc_path:str, **kwargs):
    """
    """
    raise NotImplementedError

def voc_to_yolo(voc_path:str, save_yolo_path:str, **kwargs):
    """
    """
    raise NotImplementedError
