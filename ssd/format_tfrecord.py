import os
import sys
import tensorflow as tf
import json
import cv2
import numpy as np
from object_detection.utils import dataset_util
from tqdm import tqdm

sys.path.append(r"D:\venv sa python 3.9\SSD\models\research\object_detection")
sys.path.append(r"D:\venv sa python 3.9\SSD\models\research")

DATASETS = {
    "train": {"json_path": "train.json", "record_path": "train.record", "image_dir": "D:/venv sa python 3.9/rmvid_data/images/train"},
    "val": {"json_path": "val.json", "record_path": "val.record", "image_dir": "D:/venv sa python 3.9/rmvid_data/images/val"}
}

LABEL_MAP_PATH = "label_map.pbtxt"

def create_tf_example(image_info, annotations, image_dir, label_map):
    img_path = os.path.join(image_dir, image_info["file_name"])
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"Ne mogu učitati {img_path}, skip")
        return None

    height, width, _ = image.shape
    
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_image = fid.read()
    
    filename = image_info["file_name"].encode('utf8')
    image_format = b'jpg'
    
    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []
    for ann in annotations:
        if ann["image_id"] != image_info["id"]:
            continue
        
        x, y, w, h = ann["bbox"]
        xmins.append(x / width)
        xmaxs.append((x + w) / width)
        ymins.append(y / height)
        ymaxs.append((y + h) / height)
        classes_text.append(label_map[ann["category_id"]].encode('utf8'))
        classes.append(ann["category_id"])
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def convert_coco_to_tfrecord(json_path, record_path, image_dir, label_map):
    with open(json_path, "r") as f:
        coco_data = json.load(f)
    
    writer = tf.io.TFRecordWriter(record_path)
    
    for image_info in tqdm(coco_data["images"], desc=f"Generiranje {record_path}"):
        tf_example = create_tf_example(image_info, coco_data["annotations"], image_dir, label_map)
        
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())
    
    writer.close()
    print(f"TFRecord datoteka spremljena: {record_path}")

def create_label_map(label_map_path, categories):
    with open(label_map_path, "w") as f:
        for category in categories:
            f.write(f"item {{\n  id: {category['id']}\n  name: '{category['name']}'\n}}\n")
    print(f"Label map spremljen: {label_map_path}")

if __name__ == "__main__":
    with open(DATASETS["train"]["json_path"], "r") as f:
        categories = json.load(f)["categories"]
    
    label_map = {category["id"]: category["name"] for category in categories}
    create_label_map(LABEL_MAP_PATH, categories)
    
    for dataset in DATASETS.values():
        convert_coco_to_tfrecord(dataset["json_path"], dataset["record_path"], dataset["image_dir"], label_map)
