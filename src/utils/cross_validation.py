import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.io import arff
import os

def load_dataset(filepath):
    if filepath.endswith(".arff"):
        data = arff.loadarff(filepath)
        df = pd.DataFrame(data[0])
    else:
        df = pd.read_csv(filepath)
    return df

def save_arff(df, filepath):
    with open(filepath, "w") as f:
        f.write("@RELATION dataset\n\n")
        for column in df.columns:
            dtype = "NUMERIC" if np.issubdtype(df[column].dtype, np.number) else "STRING"
            f.write(f"@ATTRIBUTE {column} {dtype}\n")
        f.write("\n@DATA\n")
        df.to_csv(f, index=False, header=False, sep=",", mode="a", lineterminator="\n")
    
    # Eliminar líneas vacías en el archivo guardado
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line for line in f.readlines() if line.strip()]
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)

def kfold_split_and_save(df, output_dir, k=5, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    for i, (train_index, test_index) in enumerate(kf.split(df)):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        
        train_filepath = os.path.join(output_dir, f"train_fold{i+1}.arff")
        test_filepath = os.path.join(output_dir, f"test_fold{i+1}.arff")
        
        save_arff(train_data, train_filepath)
        save_arff(test_data, test_filepath)
        print(f"Saved {train_filepath} and {test_filepath}")

def process_all_datasets(base_dir="datasets/multi_etiqueta/"):
    for dataset_name in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_name)
        if os.path.isdir(dataset_path):  # Asegurar que es una carpeta
            for file in os.listdir(dataset_path):
                if file.endswith("-full.arff"):  # Filtrar archivos que terminan en -full.arff
                    full_path = os.path.join(dataset_path, file)
                    print(f"Processing {full_path}")
                    df = load_dataset(full_path)
                    kfold_split_and_save(df, dataset_path)

# Ejecutar procesamiento en todos los datasets
process_all_datasets()
