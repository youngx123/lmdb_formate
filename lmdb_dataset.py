# -*- coding: utf-8 -*-
''' 
#Author       : xyoung
#Date         : 2024-08-16 18:57:31
#LastEditors  : kuai le jiu shi hahaha
#LastEditTime : 2024-08-16 19:04:09
'''


from http.client import NON_AUTHORITATIVE_INFORMATION
import os
from tkinter import NO
from PIL import Image
import numpy as np
import cv2
import io
from torch.utils.data import Dataset
import six
import lmdb
import pickle
import torch
# from xml_dataset import XMLDataset
from .base import BaseDataset


BASE_MAP_SIZE= 1099511627776  # 1T size


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


''' 
#description: load lmdb 
#return {*}
'''
class LMDBDataset(BaseDataset):
    def __init__(self,db_path, **kwargs):
        super(LMDBDataset, self).__init__(**kwargs)

        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                            readonly=True, lock=False,
                            readahead=False, meminit=False)
        
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        input_size = self.input_size
        meta = loads_data(byteflow)
        if self.transform is not None:
            meta = self.pipeline(self, meta, input_size)
        meta["img"] = torch.from_numpy(meta["img"])
        return meta


    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'