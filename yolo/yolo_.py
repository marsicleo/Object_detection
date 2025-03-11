import os
import tensorflow as tf
import cv2
from ultralytics import YOLO

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

    model = YOLO("yolov8n.pt")  
    model.train(
        data="kitti.yaml",
        epochs=10,
        batch=16,
        imgsz=416,
        device="cuda",
        project="D:/yolo_runs",
        name="train_kitti",
        cache=False,
        workers=2,
        single_cls=False
    )
    print(" Trening završen. Učitavam najbolji model...")
    
    model_path = r"D:\yolo_runs\train_kitti46\weights\best.pt"
    if not os.path.exists(model_path):
        model_path = r"D:\yolo_runs\train_kitti46\weights\last.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Model nije pronađen na {model_path}!")
    
    model = YOLO(model_path)
    print(f" Model učitan: {model_path}")

    test_dir = "D:/venv sa python 3.9/rmvid_data/images/test/"
    output_dir = r"D:\venv sa python 3.9\yolo_results"

    os.makedirs(output_dir, exist_ok=True)

    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not test_images:
        raise FileNotFoundError(f" Nema testnih slika u {test_dir}")

    print(f" Pronađeno {len(test_images)} testnih slika.")

    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        print(f" Obrada: {img_path}")

        results = model(img_path, save=True, conf=0.3)
        
        detected_img_path = os.path.join("runs/detect/predict", img_name)
        if os.path.exists(detected_img_path):
            os.rename(detected_img_path, os.path.join(output_dir, img_name))
            print(f" Spremio rezultat: {output_dir}{img_name}")

    print(" Testiranje završeno! Rezultati su spremljeni u:", output_dir)
    print(" Pokrećem evaluaciju modela na TESTNOM skupu...")

    metrics = model.val(
        data="kitti.yaml",
        split="test",
        batch=16,
        imgsz=416,
        conf=0.3,
        device="cuda"
    )
    print(" Evaluacija završena!")

    print(f" mAP@50: {metrics.box.map50:.4f}")
    print(f" mAP@50-95: {metrics.box.map:.4f}")
    print(f" Precision: {metrics.box.mp:.4f}")
    print(f" Recall: {metrics.box.mr:.4f}")
    print(f" F1 Score: {metrics.box.f1.mean():.4f}")
    print(f" Inference time: {metrics.speed['inference']:.2f}ms per image")


