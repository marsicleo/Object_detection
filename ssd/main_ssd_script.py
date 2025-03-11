import os
import sys
import tensorflow as tf
import cv2
import numpy as np

sys.path.append(r"D:\venv sa python 3.9\SSD\models\research\object_detection")
sys.path.append(r"D:\venv sa python 3.9\SSD\models\research")
sys.path.append(r"D:\venv sa python 3.9\SSD\models\research\slim")

test_dir = r"D:/venv sa python 3.9/rmvid_data/images/test"
output_dir = r"D:/venv sa python 3.9/SSD/ssd_results"
no_bb_dir = r"D:\venv sa python 3.9\SSD\ssd_results\no_box" 
bb_dir = r"D:\venv sa python 3.9\SSD\ssd_results\with_box"

if __name__ == "__main__":
    print("Početak SSD treniranja...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("TensorFlow koristi GPU:", gpus)
        except RuntimeError as e:
            print("Greška pri postavljanju TensorFlow GPU konfiguracije:", e)

    pipeline_config =r"D:\venv sa python 3.9\SSD\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\pipeline.config"
    model_dir = r"D:\venv sa python 3.9\SSD\trained_model"
    
    print(os.path.isfile(pipeline_config))
    train_dataset = "train.record"
    val_dataset = "val.record"
    label_map = "label_map.pbtxt"

    print("Početak treninga SSD modela...")
    
    os.system(f"""
        python models/research/object_detection/model_main_tf2.py ^
            --pipeline_config_path="{pipeline_config}" ^
            --model_dir="{model_dir}" ^
            --alsologtostderr ^
            --train_steps=2000 ^
            --sample_1_of_n_eval_examples=1 ^
            --num_train_steps=2000
    """)
    print("Trening završen.")
    export_dir = r"D:\venv sa python 3.9\SSD\ssd_runs"
    
    model_path = os.path.join(export_dir, "saved_model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Exportani model nije pronađen na {model_path}!")
    print("Model uspješno exportan i spremljen u:", model_path)
    
    print("Izvozim model u SavedModel format...")
        os.system(f"""
            python models/research/object_detection/exporter_main_v2.py ^
                --input_type=image_tensor ^
                --pipeline_config_path={pipeline_config} ^
                --trained_checkpoint_dir=r"D:\venv sa python 3.9\SSD\trained_model" ^
                --output_directory={export_dir} ^
                --alsologtostderr
        """)
        
    print("Učitavam exportani TF2 model (SavedModel)...")
    detect_fn = tf.saved_model.load(model_path)

    test_dir = "D:/venv sa python 3.9/rmvid_data/images/test/"
    output_dir = r"D:/venv sa python 3.9/SSD/ssd_results"
    os.makedirs(output_dir, exist_ok=True)

    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not test_images:
        raise FileNotFoundError(f"Nema testnih slika u {test_dir}")
    print(f"Pronađeno {len(test_images)} testnih slika.")
    all_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    test_images = all_images[:10]
    print(f"Uzet ću {len(test_images)} slika za prikaz.")
    
    for img_name in test_images:
        img_path = f"{test_dir}/{img_name}"
        print(f"Obrada: {img_path}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"Ne mogu učitati {img_path}")
            continue

        input_tensor = tf.convert_to_tensor([image], dtype=tf.uint8)
        detections = detect_fn(input_tensor)
        no_bb_path = f"{no_bb_dir}/{img_name}"
        cv2.imwrite(no_bb_path, image)
        boxed_image = image.copy()

        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()
        num_det = int(detections['num_detections'][0])

        height, width, _ = boxed_image.shape
        score_thresh = 0.5

        for i in range(num_det):
            if scores[i] < score_thresh:
                continue
            (ymin, xmin, ymax, xmax) = boxes[i]
            (startX, startY, endX, endY) = (
                int(xmin*width), int(ymin*height),
                int(xmax*width), int(ymax*height)
            )

            cv2.rectangle(boxed_image, (startX, startY), (endX, endY), (0,255,0), 2)
            class_id = int(classes[i])
            label_text = f"ID:{class_id} {scores[i]:.2f}"
            cv2.putText(
                boxed_image,
                label_text,
                (startX, max(startY-10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0), 1
            )
            
        bb_path = f"{bb_dir}/{img_name}"
        cv2.imwrite(bb_path, boxed_image)

        print(f" => Bez boxeva : {no_bb_path}")
        print(f" => S boxevima : {bb_path}")
    print("Obrada završena!")
    
    print("Pokrećem evaluaciju modela na TESTNOM skupu...")
    os.system(f"""
        python models/research/object_detection/model_main_tf2.py ^
            --pipeline_config_path={pipeline_config} ^
            --model_dir={model_dir} ^
            --checkpoint_dir={model_dir} ^
            --alsologtostderr
    """)
    print("Evaluacija završena! ")
