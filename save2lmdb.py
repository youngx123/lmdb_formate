# -*- coding: utf-8 -*-
''' 
#Author       : xyoung
#Date         : 2024-08-16 17:36:47
#LastEditors  : kuai le jiu shi hahaha
#LastEditTime : 2024-08-17 13:31:51
'''

import os
import cv2
import lmdb
import pickle
import torch
from tqdm import tqdm
from xml_dataset import XMLDataset


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
#description:  可视化标注框
#param {*} image
#param {*} bbox
#return {*}
'''
def show_bboxes(image, bbox):
    if len(bbox) == 0:
        return image
    for box in bbox:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return image


''' 
#description: 
#param {*} dataloader
#param {*} lmdb_path
#param {*} write_frequency
#return {*}
'''
def savelmdb(dataloader, lmdb_path, write_frequency=1000):
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)
    isdir = os.path.isdir(lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                map_size=BASE_MAP_SIZE, readonly=False,
                meminit=False, map_async=True)
    
    txn = db.begin(write=True)
    for i in tqdm(range(len(dataloader))):
        meta = dataloader.__getitem__(i)
        # print(meta)
        # print(meta["img"].shape)
        # img = show_bboxes(meta["img"], meta["gt_bboxes"])
        # cv2.imwrite(f"{str(i)}.jpg", meta["img"]*255)
        txn.put(u'{}'.format(i).encode('ascii'), dumps_data(meta))

        if i % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(i + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))
    
    print("Flushing database ...")
    db.sync()
    db.close()
    return lmdb_path


''' 
#description: load lmdb 
#return {*}
'''
class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                            readonly=True, lock=False,
                            readahead=False, meminit=False)
        
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        # imgbuf = unpacked[0]
        # buf = six.BytesIO()
        # buf.write(imgbuf)
        # buf.seek(0)
        # img = Image.open(buf).convert('RGB')

        # # load label
        # target = unpacked[1]

        # if self.transform is not None:
        #     img = self.transform(img)

        # im2arr = np.array(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # return img, target
        return unpacked

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


''' 
#description: get dataset
#param {*} class_name
#param {*} img_path
#param {*} ann_path
#param {*} mode "train" / "test" /"val"
#return {*}
'''
def get_dataset(class_name, img_path, ann_path,img_size, mode):
    parames = {
        "img_path": img_path,
        "ann_path": ann_path,
        "input_size": img_size,
        "class_names": class_name,
        "keep_ratio": True,
        "pipeline":{
            "perspective": 0.0,
            "scale": (1, 1),
            # "normalize": [[123.675, 103.53, 116.28], [58.395, 57.375, 57.12]]
            "normalize": [[0, 0, 0], [1, 1, 1]]
        }
    }
    return XMLDataset(mode=mode, **parames)
    

if __name__ == '__main__':
    class_name = ["car", "bus", "truck", "person", 
                "bicycle", "tricycle", "motorbike"]
    img_path= "/data1/share2/xiangyang/D320/train/image"
    ann_path= "/data1/share2/xiangyang/D320/train/xml"

    dataset = get_dataset(class_name, img_path, ann_path,[416, 416], "train")
    dataset_lenth = len(dataset)
    print("num : ", dataset_lenth)
    savelmdb(dataset, "/data1/share2/xyang_workSpace/D320_train_lmdb")

    # lmdb_dataset = LMDBDataset("/mnt/data2/ProjectDemo/BSD_datasets/D320_train_lmdb")
    # print(lmdb_dataset.__getitem__(0))
    # meta = lmdb_dataset.__getitem__(1)
    # for i in range(0, lmdb_dataset.length):
    #     meta = lmdb_dataset.__getitem__(i )
    #     print("num : ", meta["img"].shape)
