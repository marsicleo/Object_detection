import os
import sys
import tensorflow as tf
import cv2
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from collections import defaultdict
import re

sys.path.append(r"D:\venv sa python 3.9\SSD\models\research\object_detection")
sys.path.append(r"D:\venv sa python 3.9\SSD\models\research")
sys.path.append(r"D:\venv sa python 3.9\SSD\models\research\slim")

test_dir = r"D:/venv sa python 3.9/rmvid_data/images/test"
output_dir = r"D:/venv sa python 3.9/SSD/ssd_results"
no_bb_dir = r"D:\venv sa python 3.9\SSD\ssd_results\no_box" 
bb_dir = r"D:\venv sa python 3.9\SSD\ssd_results\with_box"

if __name__ == "__main__":
    
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
    export_dir = r"D:\venv sa python 3.9\SSD\ssd_runs"
    model_path = os.path.join(export_dir, "saved_model")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Exportani model nije pronađen na {model_path}!")
    print("Model uspješno exportan i spremljen u:", model_path)
    
    print("Učitavam exportani TF2 model")
    detect_fn = tf.saved_model.load(model_path)

    test_dir = "D:/venv sa python 3.9/rmvid_data/images/test/"
    output_dir = r"D:/venv sa python 3.9/SSD/ssd_results"
    
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not test_images:
        raise FileNotFoundError(f"Nema testnih slika u {test_dir}")
    print(f"Pronađeno {len(test_images)} testnih slika.")
    
    all_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    test_images = all_images[:800]
    print(f"Uzet ću {len(test_images)} slika za prikaz.")

    def load_label_map(label_map_path):
        label_map = {}
        with open(label_map_path, 'r') as file:
            content = file.read()
            items = re.findall(r'item.*?\{.*?id:\s*(\d+).*?name:\s*[\'\"](.*?)[\'\"]', content, re.S)
            for item_id, name in items:
                label_map[int(item_id)] = name
        return label_map

    label_map = load_label_map(r'D:\venv sa python 3.9\SSD\label_map.pbtxt')

    confidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'confidence')
    detection_quality = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'detection_quality')
    confidence.automf(3, names=['low', 'medium', 'high'])
    detection_quality['low'] = fuzz.trimf(detection_quality.universe, [0, 0, 0.5])
    detection_quality['medium'] = fuzz.trimf(detection_quality.universe, [0.3, 0.5, 0.7])
    detection_quality['high'] = fuzz.trimf(detection_quality.universe, [0.6, 1, 1])

    rule1 = ctrl.Rule(confidence['low'], detection_quality['low'])
    rule2 = ctrl.Rule(confidence['medium'], detection_quality['medium'])
    rule3 = ctrl.Rule(confidence['high'], detection_quality['high'])
    quality_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    quality_simulation = ctrl.ControlSystemSimulation(quality_ctrl)

    detection_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    log_file = os.path.join('detection_log.txt')

    with open(log_file, 'w') as log:
        def get_confidence_label(score):
            return 'low' if score < 0.4 else 'medium' if score < 0.7 else 'high'

        for img_name in test_images:
            img_path = os.path.join(test_dir, img_name)
            image = cv2.imread(img_path)
            input_tensor = tf.convert_to_tensor([image], dtype=tf.uint8)
            detections = detect_fn(input_tensor)

            scores = detections['detection_scores'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(int)
            num_det = int(detections['num_detections'][0])

            detected = False
            for i in range(num_det):
                quality_simulation.input['confidence'] = scores[i]
                quality_simulation.compute()
                adaptive_thresh = 0.1 + (quality_simulation.output['detection_quality'] * 0.5)

                if scores[i] >= adaptive_thresh:
                    detected = True
                    conf_label = get_confidence_label(scores[i])
                    class_name = label_map.get(classes[i], f'Unknown ({classes[i]})')
                    detection_stats[class_name][conf_label]['count'] += 1

            if detected:
                log.write(f"Detekcija na slici: {img_name}\n")
                print(f"Detekcija na slici: {img_name}")

        log.write("\n--- Statistika detekcija ---\n")
        print("\n--- Statistika detekcija ---")
        for class_name, confs in detection_stats.items():
            log.write(f"Klasa {class_name}:\n")
            print(f"Klasa {class_name}:")
            for conf_label, data in confs.items():
                line = f"  Confidence {conf_label}: {data['count']} detekcija\n"
                log.write(line)
                print(line)
    print(f"Log spremljen u: {log_file}")

