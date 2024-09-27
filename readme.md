xml标注格式图片保存为 lmdb数据集并在pytorch中

```python
调用接口
class_name = ["car", "bus", "truck", "person", 
			"bicycle", "tricycle", "motorbike"]
img_path= "/data1/share2/xiangyang/D320/train/image"
ann_path= "/data1/share2/xiangyang/D320/train/xml"

dataset = get_dataset(class_name, img_path, ann_path,[416, 416], "train")
dataset_lenth = len(dataset)
print("num : ", dataset_lenth)
savelmdb(dataset, "/data1/share2/xyang_workSpace/D320_train_lmdb")
```


```python
保存的lmddb进行数据加载
lmdb_dataset = LMDBDataset("/mnt/data2/ProjectDemo/BSD_datasets/D320_train_lmdb")
print(lmdb_dataset.__getitem__(0))
meta = lmdb_dataset.__getitem__(1)
for i in range(0, lmdb_dataset.length):
	meta = lmdb_dataset.__getitem__(i )
	print("num : ", meta["img"].shape)
```