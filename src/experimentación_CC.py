import pandas as pd

from models.CC import train_CC
from utils.preprocessing import load_multilabel_dataset

# CONFIGURACIÓN
MULTILABEL_DATASETS_DIR = "datasets/multi_etiqueta/"
DATASETS = {"Birds": 19, "Emotions": 6}
SEEDS = [42, 123, 7, 13, 2023, 1995, 101, 555, 999, 314, 69, 1001, 21, 33, 404, 777, 666, 1234, 4321, 9876, 2468]
NUM_EXPERIMENTOS = 10

resultados = {}

# EXPERIMENTACIÓN
for dataset in DATASETS:
    resultados[dataset] = {}

    file_path = f"{MULTILABEL_DATASETS_DIR}{dataset}/{dataset.lower()}"
    train_file_path = f"{file_path}-train.arff"
    test_file_path = f"{file_path}-test.arff"

    # Cargar datos
    X_train, y_train, _, _ = load_multilabel_dataset(train_file_path, DATASETS[dataset])
    X_test, y_test, _, _ = load_multilabel_dataset(test_file_path, DATASETS[dataset])
    
    for i in range(NUM_EXPERIMENTOS):
        
        seed = SEEDS[i]
        print(f"Ejecutando experimento {i+1} para el dataset {dataset}...")

        # Entrenar CC
        Y_pred_chain, metrics = train_CC(X_train, y_train, X_test, y_test, random_state=seed)
        
        # Calcular métricas
        resultados[dataset][i+1] = metrics

# GUARDAR RESULTADOS
all_metrics = []
for dataset, experimentos in resultados.items():
    df = pd.DataFrame.from_dict(experimentos, orient='index')
    df.index.name = "Experimento"
    df.loc['Promedio'] = df.mean()
    df.to_csv(f"experimentos/CC/{dataset}_metrics.csv")
    all_metrics.append(df.loc['Promedio'])

# Crear tabla final con la media de todas las métricas
final_metrics = pd.DataFrame(all_metrics).mean().to_frame().T
final_metrics.index = ["Global"]
final_metrics.to_csv("experimentos/CC/global_metrics.csv")

# Mostrar los resultados
print("Resultados guardados en CSVs.")
