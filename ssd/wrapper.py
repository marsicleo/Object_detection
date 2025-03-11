import os
import subprocess
import random
import copy
import time
import shutil
import glob
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

NUM_GENERATIONS = 5
POPULATION_SIZE = 4
TRAINING_STEPS = 1000

BASE_TEMPLATE_PATH = r"D:\venv sa python 3.9\SSD\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\pipeline_template.config"
BASE_MODEL_DIR = r"D:\venv sa python 3.9\SSD\trained_model"

param_ranges = {
    "l2_regularizer_weight": (1e-5, 1e-4),
    "gamma": (1.5, 3.0),
    "alpha": (0.25, 0.75),
    "learning_rate_base": (0.05, 0.1),
    "total_steps": (2000, 3000),
    "warmup_learning_rate": (0.01, 0.05),
    "warmup_steps": (500, 1500),
    "momentum_optimizer_value": (0.8, 0.95)
}

def random_individual():
    individual = {}
    for key, (low, high) in param_ranges.items():
        if key in ["min_depth", "total_steps", "warmup_steps"]:
            individual[key] = random.randint(int(low), int(high))
        else:
            individual[key] = random.uniform(low, high)
    return individual

def generate_pipeline_config(template_path, output_path, params):
    with open(template_path, "r") as f:
        template = f.read()
    
    config_str = template.format(**params)
    with open(output_path, "w") as f:
        f.write(config_str)
    
    print(f"Generirana pipeline config datoteka: {output_path}")

def get_map_from_event(model_dir):
    event_files = glob.glob(os.path.join(model_dir, '**', 'events.out.tfevents.*'), recursive=True)
    if not event_files:
        eval_dir = r"D:\venv sa python 3.9\SSD\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\checkpoint\eval"
        event_files = glob.glob(os.path.join(eval_dir, '**', 'events.out.tfevents.*'), recursive=True)
    
    if not event_files:
        print("Nisu pronađeni event file-ovi za evaluaciju!")
        return 0.0

    event_file = max(event_files, key=os.path.getmtime)
    print(f"Pronađen event file: {event_file}")
    
    time.sleep(10)
    event_acc = EventAccumulator(r"D:\venv sa python 3.9\SSD\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\checkpoint\eval")
    event_acc.Reload()
    print(event_acc.Tags())

    tensor_events = event_acc.Tensors('DetectionBoxes_Precision/mAP')
    
    for i, event in enumerate(tensor_events):
        val = tf.make_ndarray(event.tensor_proto)
    return val

def train_and_evaluate(pipeline_config_path, model_dir):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    train_cmd = (
        f'python "D:/venv sa python 3.9/SSD/train_ssd.py" '
        f'--pipeline_config_path="{pipeline_config_path}" '
        f'--model_dir="{model_dir}" '
        f'--alsologtostderr '
        f'--train_steps={TRAINING_STEPS} '
        f'--sample_1_of_n_eval_examples=1 '
        f'--num_train_steps={TRAINING_STEPS}'
    )
    print("Pokrećem trening s naredbom:")
    print(train_cmd)
    
    train_result = subprocess.run(train_cmd, shell=True)
    if train_result.returncode != 0:
        print("Trening je završio s greškom. Dodjeljujem fitness 0.")
        return 0.0
    
    eval_cmd = (
        f'python "D:/venv sa python 3.9/SSD/models/research/object_detection/model_main_tf2.py" '
        f'--pipeline_config_path="{pipeline_config_path}" '
        f'--model_dir="{model_dir}" '
        f'--checkpoint_dir="{model_dir}" '
        f'--run_once=True '
        f'--eval_only'
    )
    print("Pokrećem evaluaciju s naredbom:")
    print(eval_cmd)
    
    eval_result = subprocess.run(eval_cmd, shell=True)
    if eval_result.returncode != 0:
        print("Evaluacija je završila s greškom. Dodjeljujem fitness 0.")
        return 0.0
    
    time.sleep(5)
    
    map_value = get_map_from_event(model_dir)
    print(f"Stvarni mAP dohvaćen: {map_value:.4f}")
    return map_value

def selection(population):
    sorted_population = sorted(population, key=lambda ind: ind["fitness"], reverse=True)
    num_selected = max(1, len(sorted_population) // 2)
    return sorted_population[:num_selected]

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual, mutation_rate=0.1):
    mutant = copy.deepcopy(individual)
    for key, (low, high) in param_ranges.items():
        if random.random() < mutation_rate:
            if key in ["min_depth", "total_steps", "warmup_steps"]:
                mutant[key] = random.randint(int(low), int(high))
            else:
                mutant[key] = random.uniform(low, high)
            print(f"Mutacija: {key} promijenjen u {mutant[key]:.4f}")
    return mutant

def genetic_algorithm():
    population = [random_individual() for _ in range(POPULATION_SIZE)]
    
    for gen in range(NUM_GENERATIONS):
        print(f"\n=== Generacija {gen+1} od {NUM_GENERATIONS} ===")
        print("\nPočetna populacija generacije:")
        for idx, individual in enumerate(population):
            print(f"Jedinka {idx+1}: {individual}")

        for idx, individual in enumerate(population):
            print(f"\nEvaluacija jedinke {idx+1}/{len(population)} s parametrima:")
            for key, value in individual.items():
                print(f"  {key}: {value}")
            
            individual_model_dir = os.path.join(BASE_MODEL_DIR, f"gen{gen+1}_ind{idx+1}")
            pipeline_config_path = os.path.join(os.path.dirname(BASE_TEMPLATE_PATH), "pipeline.config")
            generate_pipeline_config(BASE_TEMPLATE_PATH, pipeline_config_path, individual)
            
            fitness = train_and_evaluate(pipeline_config_path, individual_model_dir)
            individual["fitness"] = fitness
            print(f"Evaluacija završena - Jedinka {idx+1} fitness (mAP): {fitness:.4f}")
            
            time.sleep(2)
            
        selected = selection(population)
        print("\nOdabrane jedinke (fitness):", [f"{ind['fitness']:.4f}" for ind in selected])

        new_population = copy.deepcopy(selected)
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=0.1)
            new_population.append(child)
        
        print("\nNova populacija nakon crossovera i mutacije:")
        for idx, ind in enumerate(new_population):
            print(f"Jedinka {idx+1}: {ind}")

        population = new_population
        print(f"Generacija {gen+1} završena.\n")
    
    best = max(population, key=lambda ind: ind["fitness"])
    print("\n Najbolja jedinka nakon GA optimizacije")
    for key, value in best.items():
        if key != "fitness":
            print(f"{key}: {value}")
    print(f"Fitness (mAP): {best['fitness']:.4f}")
    
    return best

if __name__ == "__main__":
    best_individual = genetic_algorithm()
    print("\nOptimizirani parametri:", best_individual)
