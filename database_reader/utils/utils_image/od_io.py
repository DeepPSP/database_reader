"""
remarks: utilities for io of images for tensorflow object detection

voc format: xmin, ymin, xmax, ymax
coco format: xmin, ymin, w, h
yolo format: classIndex xcen ycen w h
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
    "voc_to_df",
    "coco_to_df",
    "yolo_to_df",
]


def dataset_to_tfrecords(img_dirs:Union[str, List[str]], ann_dirs:Union[str, List[str]], ann_fmt:str, tfrecords_save_path:str, pbtxt_dict:Dict[str,int], train_ratio:float=0.8, class_map:Optional[Dict[str,str]]=None, csv_save_path:Optional[str]=None, **kwargs):
    """ finished, checked,

    to tfrecords for object detection training

    Parameters:
    -----------
    img_dirs: str, or list of str,
        directory(s) for the image files
    ann_dirs: str, or list of str,
        directory(s) for the bounding box annotation
    ann_fmt: str,
        format of the bounding box annotations,
        can be one of 'voc', 'coco', 'yolo', case insensitive
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
    if isinstance(img_dirs, str) and isinstance(ann_dirs, str):
        ip, ap = [img_dirs], [ann_dirs]
    elif isinstance(img_dirs, (list, tuple)) and isinstance(ann_dirs, (list, tuple)) and len(img_dirs) == len(ann_dirs):
        ip, ap = list(img_dirs), list(ann_dirs)
    else:
        raise ValueError("Invalid input!")

    df_info = pd.DataFrame()
    for i, a in zip(ip, ap):
        if ann_fmt.lower() == 'voc':
            df_tmp = voc_to_df(img_dir=i, ann_dir=a, save_path=None, class_map=class_map, **kwargs)
        elif ann_fmt.lower() == 'coco':
            df_tmp = coco_to_df(img_dir=i, ann_dir=a, save_path=None, class_map=class_map, **kwargs)
        elif ann_fmt.lower() == 'yolo':
            df_tmp = yolo_to_df(img_dir=i, ann_dir=a, save_path=None, class_map=class_map, **kwargs)
        else:
            raise ValueError("annotation format {} not recognized".format(ann_fmt))
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


def voc_to_df(img_dir:str, ann_dir:str, save_path:Optional[str]=None, class_map:Optional[Dict[str,str]]=None, **kwargs) -> pd.DataFrame:
    """ finished, checked,

    pascal voc annotations to one DataFrame (csv file)

    Parameters:
    -----------
    img_dir: str,
        directory of the image files
    ann_dir: str,
        directory of the bounding box annotation xml files
    save_path: str, optional,
        path to store the csv file
    class_map: dict, optional,
        label map, from class names of the annotations to the class names for training

    Returns:
    --------
    bbox_df: DataFrame,
        annotations in one DataFrame
    """
    xml_list = []
    for xml_file in glob.glob(os.path.join(ann_dir, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if len(root.findall('object')) == 0:
            print('{} has no bounding box annotation'.format(xml_file))
        for member in root.findall('object'):
            fw = int(root.find('size').find('width').text)
            fh = int(root.find('size').find('height').text)
            subcls_name = member.find('name').text
            xmin = int(member.find('bndbox').find('xmin').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            box_width = xmax-xmin
            box_height = ymax-ymin
            area = box_width*box_height
            if area <= 0:
                continue
            values = {
                'filename': root.find('filename').text if root.find('filename') is not None else '',
                'width': fw,
                'height': fh,
                'segmented': root.find('segmented').text if root.find('segmented') is not None else '',
                'subclass': subcls_name,
                'pose': member.find('pose').text if member.find('pose') is not None else '',
                'truncated': member.find('truncated').text if member.find('truncated') is not None else '',
                'difficult': member.find('difficult').text if member.find('difficult') is not None else '',
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'box_width': box_width,
                'box_height': box_height,
                'area': area,
            }
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'segmented', 'pose', 'truncated', 'difficult', 'xmin', 'ymin', 'xmax', 'ymax', 'box_width', 'box_height', 'subclass', 'area']
    bbox_df = pd.DataFrame(xml_list, columns=column_name)
    if class_map is None:
        bbox_df['class'] = bbox_df['subclass']
    else:
        bbox_df['class'] = bbox_df['subclass'].apply(lambda sc:class_map[sc])
    column_name = [
        'filename', 'class', 'subclass',
        'segmented', 'pose', 'truncated', 'difficult',
        'width', 'height',
        'xmin', 'ymin', 'xmax', 'ymax',
        'box_width', 'box_height', 'area',
    ]
    bbox_df = bbox_df[column_name]
    if save_path is not None:
        bbox_df.to_csv(save_path, index=False)
    return bbox_df


def yolo_to_df(img_dir:str, ann_dir:str, save_path:Optional[str]=None, class_map:Optional[Dict[str,str]]=None, **kwargs) -> pd.DataFrame:
    """ finished, checked,

    pascal voc annotations to one csv file

    Parameters:
    -----------
    img_dir: str,
        directory of the image files
    ann_dir: str,
        directory of the bounding box annotation txt files
    save_path: str, optional,
        path to store the csv file
    class_map: dict, optional,
        label map, from class names of the annotations to the class names for training

    Returns:
    --------
    bbox_df: DataFrame,
        annotations in one DataFrame

    NOTE: each line of each file is of the form `classIndex xcen ycen w h`
    """
    ann_list = []
    all_img = os.listdir(img_dir)
    for ann_file in glob.glob(os.path.join(ann_dir, '*.txt')):
        img_file = os.path.splitext(os.path.basename(ann_file))[0]
        img_file = [os.path.join(img_dir, item) for item in all_img if item.startswith(imgfile)]
        if len(img_file) == 0:
            continue
        img_file = img_file[0]
        with tf.gfile.GFile(img_file, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size
        with open(ann_file, 'r') as f:
            for l in f:
                classIndex, xcen, ycen, box_w, box_h = l.strip().split(' ')
                classIndex = int(classIndex)
                box_w, box_h = int(float(box_w)*width), int(float(box_h)*height)
                xcen, ycen = int(float(xcen)*width), int(float(ycen)*height)
                xmin, xmax = xcen - box_w//2, xcen + box_w//2
                ymin, ymax = ycen - box_h//2, ycen + box_h//2
                area = box_w*box_h
                
                # TODO: add to ann_list
    raise NotImplementedError


def coco_to_df(img_dir:str, ann_dir:str, save_path:Optional[str]=None, class_map:Optional[Dict[str,str]]=None, **kwargs) -> pd.DataFrame:
    """ finished, checked,

    pascal voc annotations to one csv file

    Parameters:
    -----------
    img_dir: str,
        directory of the image files
    ann_dir: str,
        directory of the bounding box annotation json file
    save_path: str, optional,
        path to store the csv file
    class_map: dict, optional,
        label map, from class names of the annotations to the class names for training

    Returns:
    --------
    bbox_df: DataFrame,
        annotations in one DataFrame
    """
    raise NotImplementedError


def split(df:pd.DataFrame, group) -> List[namedtuple]:
    """
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tfexample(group:namedtuple, pbtxt_dict:Dict[str,int], ignore_difficult_instances:bool=False) -> tf.train.Example:
    """ finished, checked,

    one image with bounding box annotations to one tf Example

    Parameters:
    -----------
    group: namedtuple,
        with "filename" and "data", "data" consisting of bounding boxes and image width, height
    pbtxt_dict: dict,
        label map, from class name to class number
    ignore_difficult_instances: bool, default False,
        ignore the difficult instances (bounding boxes) or not

    Returns:
    --------
    tf_example: Example,
    """
    with tf.gfile.GFile(group.filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    # width, height = image.size
    key = hashlib.sha256(encoded_jpg).hexdigest()
        
    filename = os.path.basename(group.filename)
    image_id = get_image_id(filename)
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    truncated = []
    poses = []
    difficult_obj = []

    for index, row in group.object.iterrows():
        difficult = bool(int(row['difficult'] or '0'))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))

        width = row['width']
        height = row['height']
        xmins.append(float(row['xmin'] / width))
        xmaxs.append(float(row['xmax'] / width))
        ymins.append(float(row['ymin'] / height))
        ymaxs.append(float(row['ymax'] / height))
        classes_text.append(row['class'].encode('utf8'))
        classes.append(pbtxt_dict[row['class']])
        truncated.append(int(row['truncated']))
        poses.append(row['pose'].encode('utf8'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(filename.encode('utf8')),
        'image/source_id': _bytes_feature(str(image_id).encode('utf8')),
        'image/encoded': _bytes_feature(encoded_jpg),
        'image/format': _bytes_feature(b'jpg'),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/text': _bytes_list_feature(classes_text),
        'image/object/class/label': _int64_list_feature(classes),
        'image/object/difficult': tfrecord_util.int64_list_feature(difficult_obj),
        'image/object/truncated': tfrecord_util.int64_list_feature(truncated),
        'image/object/view': tfrecord_util.bytes_list_feature(poses),
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


GLOBAL_IMG_ID = 0  # global image id.
GLOBAL_ANN_ID = 0  # global annotation id.


def get_image_id(filename):
    """Convert a string to a integer."""
    # Warning: this function is highly specific to pascal filename!!
    # Given filename like '2008_000002', we cannot use id 2008000002 because our
    # code internally will convert the int value to float32 and back to int, which
    # would cause value mismatch int(float32(2008000002)) != int(2008000002).
    # COCO needs int values, here we just use a incremental global_id, but
    # users should customize their own ways to generate filename.
    del filename
    global GLOBAL_IMG_ID
    GLOBAL_IMG_ID += 1
    return GLOBAL_IMG_ID


def get_ann_id():
    """Return unique annotation id across images."""
    global GLOBAL_ANN_ID
    GLOBAL_ANN_ID += 1
    return GLOBAL_ANN_ID
