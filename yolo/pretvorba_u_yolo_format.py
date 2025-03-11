import os

class_map = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6
}

def convert_kitti_to_yolo(kitti_label_dir, yolo_label_dir, img_width=1242, img_height=375):
    if not os.path.exists(yolo_label_dir):
        os.makedirs(yolo_label_dir)

    for file_name in os.listdir(kitti_label_dir):
        if not file_name.endswith(".txt"):
            continue
        
        kitti_file = os.path.join(kitti_label_dir, file_name)
        yolo_file = os.path.join(yolo_label_dir, file_name)
        
        with open(kitti_file, "r") as f_in, open(yolo_file, "w") as f_out:
            for line in f_in:
                parts = line.strip().split()
                class_name = parts[0]
                if class_name == "DontCare" or class_name not in class_map:
                    continue
                
                class_id = class_map[class_name]
                x1, y1, x2, y2 = map(float, parts[4:8])

                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"Pretvoreno: {file_name}")
        
kitti_label_dir = r"D:\venv sa python 3.9\rmvid_data\data_object_label_2\training\label_2"
yolo_label_dir = r"D:\venv sa python 3.9\rmvid_data\yolo_labels"
convert_kitti_to_yolo(kitti_label_dir, yolo_label_dir)
