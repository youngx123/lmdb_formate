# -*- coding: utf-8 -*-
''' 
#Author       : xyoung
#Date         : 2024-08-17 10:11:04
#LastEditors  : kuai le jiu shi hahaha
#LastEditTime : 2024-08-17 13:27:05
'''
import os
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
import random
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
from transform.pipeline import Pipeline


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    A base class of detection dataset. Referring from MMDetection.
    A dataset should have images, annotations and preprocessing pipelines
    NanoDet use [xmin, ymin, xmax, ymax] format for box and
     [[x0,y0], [x1,y1] ... [xn,yn]] format for key points.
    instance masks should decode into binary masks for each instance like
    {
        'bbox': [xmin,ymin,xmax,ymax],
        'mask': mask
     }
    segmentation mask should decode into binary masks for each class.
    Args:
        img_path (str): image data folder
        ann_path (str): annotation file path or folder
        use_instance_mask (bool): load instance segmentation data
        use_seg_mask (bool): load semantic segmentation data
        use_keypoint (bool): load pose keypoint data
        load_mosaic (bool): using mosaic data augmentation from yolov4
        mode (str): 'train' or 'val' or 'test'
        multi_scale (Tuple[float, float]): Multi-scale factor range.
    """

    def __init__(
            self,
            img_path: str,
            ann_path: str,
            input_size: Tuple[int, int],
            pipeline: Dict = None,
            keep_ratio: bool = True,
            use_instance_mask: bool = False,
            use_seg_mask: bool = False,
            use_keypoint: bool = False,
            load_mosaic: bool = False,
            mode: str = "train",
            multi_scale: Optional[Tuple[float, float]] = None,
    ):
        assert mode in ["train", "val", "test"]
        self.img_path = img_path
        self.ann_path = ann_path
        self.input_size = input_size
        self.pipeline = Pipeline(pipeline, keep_ratio)
        self.keep_ratio = keep_ratio
        self.use_instance_mask = use_instance_mask
        self.use_seg_mask = use_seg_mask
        self.use_keypoint = use_keypoint
        self.load_mosaic = load_mosaic
        self.multi_scale = multi_scale
        self.mode = mode

        self.data_info = self.get_data_info(ann_path)

    def __len__(self):
        # return 40000
        if len(self.data_info) > 100000:
            return len(self.data_info)
        else:
            return len(self.data_info)

    def __getitem__(self, idx):
        if self.mode == "val" or self.mode == "test":
            return self.get_val_data(idx)
        else:
            while True:
                data = self.get_train_data(idx)
                if data is None:
                    idx = self.get_another_id()
                    continue
                return data

    @staticmethod
    def get_random_size(
            scale_range: Tuple[float, float], image_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Get random image shape by multi-scale factor and image_size.
        Args:
            scale_range (Tuple[float, float]): Multi-scale factor range.
                Format in [(width, height), (width, height)]
            image_size (Tuple[int, int]): Image size. Format in (width, height).

        Returns:
            Tuple[int, int]
        """
        assert len(scale_range) == 2
        scale_factor = random.uniform(*scale_range)
        width = int(image_size[0] * scale_factor)
        height = int(image_size[1] * scale_factor)
        return width, height

    @abstractmethod
    def get_data_info(self, ann_path):
        pass

    @abstractmethod
    def get_train_data(self, idx):
        pass

    @abstractmethod
    def get_val_data(self, idx):
        pass

    def get_another_id(self):
        return np.random.random_integers(0, len(self.data_info) - 1)


#  CocoDataset
class CocoDataset(BaseDataset):
    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
        'file_name': '000000000139.jpg',
        'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
        'height': 426,
        'width': 640,
        'date_captured': '2013-11-21 01:34:01',
        'flickr_url':
            'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
        'id': 139},
        ...
        ]
        """
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.class_names = [cat["name"] for cat in self.cats]
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        id = img_info["id"]
        if not isinstance(id, int):
            raise TypeError("Image id must be int.")
        info = {"file_name": file_name, "height": height, "width": width, "id": id}
        return info

    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []
        for ann in anns:
            x1, y1, w, h = ann["bbox"]
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False) or ann.get("ignore", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann["keypoints"])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore
        )
        if self.use_instance_mask:
            annotation["masks"] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation["keypoints"] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation["keypoints"] = np.zeros((0, 51), dtype=np.float32)
        return annotation

    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """
        img_info = self.get_per_img_info(idx)
        file_name = img_info["file_name"]
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            print("image {} read failed.".format(image_path))
            raise FileNotFoundError("Cant load image! Please check image path!")
        ann = self.get_img_annotation(idx)
        meta = dict(
            img=img,
            img_info=img_info,
            gt_bboxes=ann["bboxes"],
            gt_labels=ann["labels"],
            gt_bboxes_ignore=ann["bboxes_ignore"],
        )
        if self.use_instance_mask:
            meta["gt_masks"] = ann["masks"]
        if self.use_keypoint:
            meta["gt_keypoints"] = ann["keypoints"]

        input_size = self.input_size
        if self.multi_scale:
            input_size = self.get_random_size(self.multi_scale, input_size)

        meta = self.pipeline(self, meta, input_size)
        meta["img"] = meta["img"]
        return meta

    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)
