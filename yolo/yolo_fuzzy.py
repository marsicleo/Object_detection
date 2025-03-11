import os
import tensorflow as tf
import cv2
from ultralytics import YOLO
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from collections import defaultdict

if __name__ == "__main__":
    print(" Početak YOLO treniranja...")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(" TensorFlow koristi GPU:", gpus)
        except RuntimeError as e:
            print(" Greška pri postavljanju TensorFlow GPU konfiguracije:", e)

    model_path = r"D:\venv sa python 3.9\runs\train_kitti46\weights\best.pt"
    if not os.path.exists(model_path):
        model_path = r"D:\venv sa python 3.9\runs\train_kitti46\weights\last.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Model nije pronađen na {model_path}!")

    model = YOLO(model_path)
    print(f" Model učitan: {model_path}")

    test_dir = "D:/venv sa python 3.9/rmvid_data/images/test/"
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not test_images:
        raise FileNotFoundError(f" Nema testnih slika u {test_dir}")

    print(f" Pronađeno {len(test_images)} testnih slika.")

    confidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'confidence')
    size = ctrl.Antecedent(np.arange(0, 101, 10), 'size')
    decision = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'decision')
    confidence.automf(3)
    size.automf(3)
    
    decision['low'] = fuzz.trimf(decision.universe, [0, 0, 0.5])
    decision['medium'] = fuzz.trimf(decision.universe, [0.3, 0.5, 0.7])
    decision['high'] = fuzz.trimf(decision.universe, [0.5, 1, 1])

    rule1 = ctrl.Rule(confidence['poor'] | size['poor'], decision['low'])
    rule2 = ctrl.Rule(confidence['average'] & size['average'], decision['medium'])
    rule3 = ctrl.Rule(confidence['good'] | size['good'], decision['high'])

    decision_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    decision_sim = ctrl.ControlSystemSimulation(decision_ctrl)
    detection_summary = defaultdict(lambda: defaultdict(int))

    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        print(f" Obrada: {img_path}")

        results = model(img_path, save=True, conf=0.3)
        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                xyxy = box.xyxy[0]
                area = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) / (416 * 416) * 100

                decision_sim.input['confidence'] = conf
                decision_sim.input['size'] = area
                decision_sim.compute()
                decision_value = decision_sim.output['decision']
                decision_label = 'Nisko' if decision_value < 0.4 else 'Srednje' if decision_value < 0.7 else 'Visoko'

                class_id = int(box.cls[0]) if box.cls is not None else 0
                class_name = r.names.get(class_id, f"Class_{class_id}")

                if conf >= 0.7:
                    confidence_level = "Visoko"
                elif conf >= 0.4:
                    confidence_level = "Srednje"
                else:
                    confidence_level = "Nisko"

                detection_summary[class_name][confidence_level] += 1

    output_dir = r"D:\venv sa python 3.9\yolo_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "detection_summary.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        for cls, conf_dict in detection_summary.items():
            f.write(f"Klasa: {cls}\n")
            for conf_level, count in conf_dict.items():
                f.write(f"  {conf_level} povjerenje: {count} objekata\n")
            f.write("\n")

    print(" Testiranje završeno! Rezultati po klasama zapisani u:", output_path)
