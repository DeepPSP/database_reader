"""
remarks: utilities for io of images for tensorflow object detection

voc format: xmin, ymin, xmax, ymax
coco format: xmin, ymin, w, h
yolo format: classIndex xcen ycen w h
"""
import os
import io
from PIL import Image
import numpy as np
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

    pascal voc annotations (in xml format) to one DataFrame (csv file)

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
    img_dir_filenames = os.listdir(img_dir)
    for xml_file in glob.glob(os.path.join(ann_dir, '*.xml')):
        tree = ET.parse(xml_file)
        img_file = os.path.splitext(os.path.basename(xml_file))[0]
        img_file = [os.path.join(img_dir, item) for item in img_dir_filenames if item.startswith(img_file)]
        if len(img_file) != 1:
            print(f"number of images corresponding to {os.path.basename(xml_file)} is {len(img_file)}")
            continue
        img_file = img_file[0]
        root = tree.getroot()
        if len(root.findall('object')) == 0:
            print('{} has no bounding box annotation'.format(xml_file))
        for member in root.findall('object'):
            fw = int(root.find('size').find('width').text)
            fh = int(root.find('size').find('height').text)
            # or obtain fw, fh from image read from `img_file`
            subcls_name = member.find('name').text
            xmin = int(member.find('bndbox').find('xmin').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            box_width = xmax-xmin
            box_height = ymax-ymin
            box_area = box_width*box_height
            if box_area <= 0:
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
                'box_area': box_area,
            }
            xml_list.append(values)
    column_names = ['filename', 'width', 'height', 'segmented', 'pose', 'truncated', 'difficult', 'xmin', 'ymin', 'xmax', 'ymax', 'box_width', 'box_height', 'subclass', 'box_area']
    bbox_df = pd.DataFrame(xml_list, columns=column_names)
    if class_map is None:
        bbox_df['class'] = bbox_df['subclass']
    else:
        bbox_df['class'] = bbox_df['subclass'].apply(lambda sc:class_map[sc])
    column_names = [
        'filename', 'class', 'subclass',
        'segmented', 'pose', 'truncated', 'difficult',
        'width', 'height',
        'xmin', 'ymin', 'xmax', 'ymax',
        'box_width', 'box_height', 'box_area',
    ]
    bbox_df = bbox_df[column_names]
    if save_path is not None:
        bbox_df.to_csv(save_path, index=False)
    return bbox_df


def yolo_to_df(img_dir:str, ann_dir:str, save_path:Optional[str]=None, class_map:Optional[Dict[int,str]]=None, **kwargs) -> pd.DataFrame:
    """ finished, checked,

    yolo annotations (in txt format) to one csv file

    Parameters:
    -----------
    img_dir: str,
        directory of the image files
    ann_dir: str,
        directory of the bounding box annotation txt files
    save_path: str, optional,
        path to store the csv file
    class_map: dict, optional,
        label map, from class indices of the annotations to the class names for training

    Returns:
    --------
    bbox_df: DataFrame,
        annotations in one DataFrame

    NOTE: each line of each file is of the form `classIndex xcen ycen w h`
    """
    ann_list = []
    img_dir_filenames = os.listdir(img_dir)
    for ann_file in glob.glob(os.path.join(ann_dir, '*.txt')):
        img_file = os.path.splitext(os.path.basename(ann_file))[0]
        img_file = [os.path.join(img_dir, item) for item in img_dir_filenames if item.startswith(img_file)]
        if len(img_file) != 1:
            print(f"number of images corresponding to {os.path.basename(xml_file)} is {len(img_file)}")
            continue
        img_file = img_file[0]
        with tf.gfile.GFile(img_file, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        fw, fh = image.size
        with open(ann_file, 'r') as f:
            for l in f:
                classIndex, xcen, ycen, box_width, box_height = l.strip().split(' ')
                classIndex = int(classIndex)
                if class_map is not None:
                    classname = class_map[classIndex]
                else:
                    classname = str(classIndex)
                box_width, box_height = int(float(box_width)*fw), int(float(box_height)*fh)
                xcen, ycen = int(float(xcen)*fw), int(float(ycen)*fh)
                xmin, xmax = xcen - box_width//2, xcen + box_width//2
                ymin, ymax = ycen - box_height//2, ycen + box_height//2
                box_area = box_width*box_height
                # TODO: add to ann_list
                if box_area <= 0:
                    continue
                values = {
                    'filename': os.path.basename(img_file),
                    'class': classname,
                    'width': fw,
                    'height': fh,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'box_width': box_width,
                    'box_height': box_height,
                    'box_area': box_area,
                }
                ann_list.append(values)
    column_names = [
        'filename', 'class',
        'width', 'height',
        'xmin', 'ymin', 'xmax', 'ymax',
        'box_width', 'box_height', 'box_area',
    ]
    bbox_df = pd.DataFrame(ann_list, columns=column_names)
    if save_path is not None:
        bbox_df.to_csv(save_path, index=False)
    return bbox_df


def coco_to_df(img_dir:str, ann_dir:str, save_path:Optional[str]=None, class_map:Optional[Dict[str,str]]=None, **kwargs) -> pd.DataFrame:
    """ finished, checked,

    coco annotations (in json format) to one csv file

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


import torch
from torch import Tensor
from packaging import version

if version.parse(torch.__version__) >= version.parse('1.5.0'):
    def _true_divide(dividend, divisor):
        return torch.true_divide(dividend, divisor)
else:
    def _true_divide(dividend, divisor):
        return dividend / divisor

def bboxes_iou_torch(bboxes_a:Tensor, bboxes_b:Tensor, fmt:str='voc', iou_type:str='iou') -> Tensor:
    """ finished, checked,
    
    Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Parameters:
    -----------
    bbox_a (array): An array whose shape is :math:`(N, 4)`.
        :math:`N` is the number of bounding boxes.
        The dtype should be :obj:`numpy.float32`.
    bbox_b (array): An array similar to :obj:`bbox_a`,
        whose shape is :math:`(K, 4)`.
        The dtype should be :obj:`numpy.float32`.
    
    Returns:
    --------
    array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    and https://github.com/Tianxiaomo/pytorch-YOLOv4
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    N, K = bboxes_a.shape[0], bboxes_b.shape[0]

    # top left
    if fmt.lower() == 'voc':  # xmin, ymin, xmax, ymax
        # top left
        tl_intersect = torch.max(bboxes_a[:, np.newaxis, :2], bboxes_b[:, :2]) # of shape `(N,K,2)`
        # bottom right
        br_intersect = torch.min(bboxes_a[:, np.newaxis, 2:], bboxes_b[:, 2:])
        bb_a = bboxes_a[:, 2:] - bboxes_a[:, :2]
        bb_b = bboxes_b[:, 2:] - bboxes_b[:, :2]
        # bb_* can also be seen vectors representing box_width, box_height
    elif fmt.lower() == 'coco':  # xmin, ymin, w, h
        tl_intersect = torch.max((bboxes_a[:, np.newaxis, :2] - bboxes_a[:, np.newaxis, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br_intersect = torch.min((bboxes_a[:, np.newaxis, :2] + bboxes_a[:, np.newaxis, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        bb_a = bboxes_a[:, 2:]
        bb_b = bboxes_b[:, 2:]

    area_a = torch.prod(bb_a, 1)
    area_b = torch.prod(bb_b, 1)
    
    # torch.prod(input, dim, keepdim=False, dtype=None) → Tensor
    # Returns the product of each row of the input tensor in the given dimension dim
    # if tl, br does not form a nondegenerate squre, then the corr. element in the `prod` would be 0
    en = (tl_intersect < br_intersect).type(tl_intersect.type()).prod(dim=2)  # shape `(N,K,2)` ---> shape `(N,K)`

    area_intersect = torch.prod(br_intersect - tl_intersect, 2) * en  # * ((tl < br).all())
    area_union = (area_a[:, np.newaxis] + area_b - area_intersect)

    iou = _true_divide(area_intersect, area_union)

    if iou_type.lower() == 'iou':
        return iou

    if fmt.lower() == 'voc':  # xmin, ymin, xmax, ymax
        # top left
        tl_union = torch.min(bboxes_a[:, np.newaxis, :2], bboxes_b[:, :2]) # of shape `(N,K,2)`
        # bottom right
        br_union = torch.max(bboxes_a[:, np.newaxis, 2:], bboxes_b[:, 2:])
    elif fmt.lower() == 'coco':  # xmin, ymin, w, h
        tl_union = torch.min((bboxes_a[:, np.newaxis, :2] - bboxes_a[:, np.newaxis, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br_union = torch.max((bboxes_a[:, np.newaxis, :2] + bboxes_a[:, np.newaxis, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
    
    # c for covering, of shape `(N,K,2)`
    # the last dim is box width, box hight
    bboxes_c = br_union - tl_union

    area_covering = torch.prod(bboxes_c, 2)  # shape `(N,K)`

    giou = iou - (area_covering - area_union) / area_covering

    if iou_type.lower() == 'giou':
        return giou

    if fmt.lower() == 'voc':  # xmin, ymin, xmax, ymax
        centre_a = (bboxes_a[..., 2 :] + bboxes_a[..., : 2]) / 2
        centre_b = (bboxes_b[..., 2 :] + bboxes_b[..., : 2]) / 2
    elif fmt.lower() == 'coco':  # xmin, ymin, w, h
        centre_a = (bboxes_a[..., : 2] + bboxes_a[..., 2 :]) / 2
        centre_b = (bboxes_b[..., : 2] + bboxes_b[..., 2 :]) / 2

    centre_dist = torch.norm(centre_a[:, np.newaxis] - centre_b, p='fro', dim=2)
    diag_len = torch.norm(bboxes_c, p='fro', dim=2)

    diou = iou - centre_dist.pow(2) / diag_len.pow(2)

    if iou_type.lower() == 'diou':
        return diou

    # bb_a of shape `(N,2)`, bb_b of shape `(K,2)`
    v = torch.einsum('nm,km->nk', bb_a, bb_b)
    v = _true_divide(v, (torch.norm(bb_a, p='fro', dim=1)[:,np.newaxis] * torch.norm(bb_b, p='fro', dim=1)))
    # avoid nan for torch.acos near \pm 1
    # https://github.com/pytorch/pytorch/issues/8069
    eps = 1e-7
    v = torch.clamp(v, -1+eps, 1-eps)
    v = (_true_divide(2*torch.acos(v), np.pi)).pow(2)
    alpha = (_true_divide(v, 1-iou+v))*((iou>=0.5).type(iou.type()))

    ciou = diou - alpha * v

    if iou_type.lower() == 'ciou':
        return ciou


def bboxes_iou_tf(bboxes_a, bboxes_b, fmt='voc', iou_type='iou'):
    """
    """
    raise NotImplementedError


def nms(boxes:np.ndarray, confs:np.ndarray, box_fmt:str='coco', nms_thresh:float=0.5, min_mode:bool=False):
    """
    non-maximum suppression

    Paramters:
    ----------
    boxes: ndarray, of shape (n, 4),
        the bounding boxes
    confs: ndarray, of shape (n,),
        confidence (score) of each bounding box
    box_fmt: str, default 'coco',
        the format of the bounding boxes
    nms_thresh: float, default 0.5,
        threshold for eliminating redundant boxes
    min_mode: bool, default False,
        if True, denominator of computing the intersection ratio will be the area of the smaller box;
        otherwise the area of the union of two boxes

    Returns:
    --------
    ndarray, of shape (m, 4), m <= n

    References:
    -----------
    [1] https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/tool/utils.py
    """
    if box_fmt.lower() == 'coco':
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
    elif box_fmt.lower() == 'voc':
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)
