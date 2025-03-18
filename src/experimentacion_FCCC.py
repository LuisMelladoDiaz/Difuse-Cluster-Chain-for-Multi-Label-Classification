from models.FCCC import FCCC
import pandas as pd

# CONFIGURACIÓN
MULTILABEL_DATASETS_DIR = "datasets/multi_etiqueta/"
DATASETS = {"Birds": 19, "Emotions": 6, "FoodTruck": 12}
UMBRAL = 0
NUM_CLUSTERS = 3
SEEDS = [42, 123, 7, 13, 2023, 1995, 101, 555, 999, 314, 69, 1001, 21, 33, 404, 777, 666, 1234, 4321, 9876, 2468, 1357, 1111, 2222, 3333, 4444, 5555, 8888, 9999, 10000]
NUM_EXPERIMENTOS = 10


resultados = {}

# EXPERIMENTACIÓN
for dataset in DATASETS:
    resultados[dataset] = {}
    for i in range(NUM_EXPERIMENTOS):

        train_file_path = f"{MULTILABEL_DATASETS_DIR}{dataset}/{dataset.lower()}-train.arff"
        num_labels = DATASETS[dataset]
        seed = SEEDS[i]

        print(f"Ejecutando experimento {i+1} para el dataset {dataset}...")
        
        prediction, metrics = FCCC(train_file_path, num_labels, False, NUM_CLUSTERS, seed)
        resultados[dataset][i+1] = metrics


# RESULTADOS
all_metrics = []
for dataset, experimentos in resultados.items():
    df = pd.DataFrame.from_dict(experimentos, orient='index')
    df.index.name = "Experimento"
    df.loc['Promedio'] = df.mean()
    df.to_csv(f"experimentos/FCCC/{dataset}_metrics.csv")
    all_metrics.append(df.loc['Promedio'])

# Crear tabla final con la media de todas las métricas
final_metrics = pd.DataFrame(all_metrics).mean().to_frame().T
final_metrics.index = ["Global"]
final_metrics.to_csv("experimentos/FCCC/global_metrics.csv")

# Mostrar los resultados
print("Resultados guardados en CSVs.")
