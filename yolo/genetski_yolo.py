import os
import random
import tensorflow as tf
from ultralytics import YOLO

HYPERPARAM_RANGES = {
    "batch": [4, 8, 16, 32],
    "imgsz": [256, 320, 416, 512, 640],
    "learning_rate": (1e-5, 1e-2),
    "optimizer": ["SGD", "Adam", "AdamW", "RMSprop"]
}

def generate_random_params():
    return {
        "batch": random.choice(HYPERPARAM_RANGES["batch"]),
        "imgsz": random.choice(HYPERPARAM_RANGES["imgsz"]),
        "learning_rate": 10 ** random.uniform(-5, -2),
        "optimizer": random.choice(HYPERPARAM_RANGES["optimizer"])
    }

def evaluate_model(params):
    print(f"\n🔍 Treniranje s parametrima: {params}")

    model = YOLO("yolov8n.pt")
    
    model.train(
        data="kitti.yaml",
        epochs=1,
        batch=params["batch"],
        imgsz=params["imgsz"],
        device="cuda",
        project="D:/yolo_runs",
        name="train_kitti",
        lr0=params["learning_rate"],
        optimizer=params["optimizer"],
        cache=True,
        workers=2,
        single_cls=False
    )

    model_path = r"D:\yolo_runs\train_kitti\weights\best.pt"
    if not os.path.exists(model_path):
        return 0

    model = YOLO(model_path)
    metrics = model.val(data="kitti.yaml", split="test", device="cuda")
    return metrics.box.map50

def genetic_algorithm(pop_size=5, generations=5, mutation_rate=0.2):
    population = [generate_random_params() for _ in range(pop_size)]
    for gen in range(generations):
        print(f"\n===  Generacija {gen+1} ===")

        scores = [(params, evaluate_model(params)) for params in population]
        scores.sort(key=lambda x: x[1], reverse=True)
        print(f" Najbolji model u generaciji {gen+1}: {scores[0]}")

        top_models = scores[:pop_size // 2]
        next_gen = []
        for _ in range(pop_size // 2):
            p1, p2 = random.sample(top_models, 2)
            child_params = {key: random.choice([p1[0][key], p2[0][key]]) for key in HYPERPARAM_RANGES}
            
            if random.random() < mutation_rate:
                mutation_key = random.choice(list(HYPERPARAM_RANGES.keys()))
                if isinstance(HYPERPARAM_RANGES[mutation_key], list):
                    child_params[mutation_key] = random.choice(HYPERPARAM_RANGES[mutation_key])
                else:
                    child_params[mutation_key] = generate_random_params()[mutation_key]
            
            next_gen.append(child_params)

        population = [p[0] for p in top_models] + next_gen
    print("\n Genetska optimizacija završena!")
    print(" Najbolji pronađeni parametri su:", scores[0][0])

if __name__ == "__main__":
    genetic_algorithm()
