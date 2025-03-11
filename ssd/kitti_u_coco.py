import os
import json
import cv2
import glob
from tqdm import tqdm

DATASETS = {
    "train": {"image_dir": r"D:\venv sa python 3.9\rmvid_data\images\train", "label_dir": r"D:\venv sa python 3.9\rmvid_data\labels\train", "output_json": "train.json"},
    "val": {"image_dir": r"D:\venv sa python 3.9\rmvid_data\images\val", "label_dir": r"D:\venv sa python 3.9\rmvid_data\labels\val", "output_json": "val.json"},
    "test": {"image_dir": r"D:\venv sa python 3.9\rmvid_data\images\test", "label_dir": r"D:\venv sa python 3.9\rmvid_data\labels\test", "output_json": "test.json"}
}

LABEL_MAP = {
    0: "Car",
    1: "Van",
    2: "Truck",
    3: "Pedestrian",
    4: "Person_sitting",
    5: "Cyclist",
    6: "Tram"
}

def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    return [x_min, y_min, width, height]

def convert_yolo_to_coco(image_dir, label_dir, output_json):
    coco_data = {
        "info": {"description": "YOLO to COCO Converted Dataset", "version": "1.0", "year": 2024},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    for id, name in LABEL_MAP.items():
        coco_data["categories"].append({"id": id + 1, "name": name})
    
    annotation_id = 1
    image_id = 1
    
    for label_file in tqdm(glob.glob(os.path.join(label_dir, "*.txt")), desc=f"Konverzija {output_json}"):
        image_file = os.path.join(image_dir, os.path.basename(label_file).replace(".txt", ".png"))
        
        if not os.path.exists(image_file):
            print(f"Preskačem {image_file}, slika ne postoji.")
            continue
        
        image = cv2.imread(image_file)
        img_height, img_width, _ = image.shape
        
        coco_data["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_file),
            "width": img_width,
            "height": img_height
        })
        
        with open(label_file, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            coco_bbox = yolo_to_coco_bbox(bbox, img_width, img_height)
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": coco_bbox,
                "area": coco_bbox[2] * coco_bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1
        
        image_id += 1
    
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)
    print(f"Konverzija završena, COCO anotacije spremljene u {output_json}")

for dataset in DATASETS.values():
    convert_yolo_to_coco(dataset["image_dir"], dataset["label_dir"], dataset["output_json"])
