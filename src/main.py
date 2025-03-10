import tkinter as tk
from tkinter import messagebox, filedialog
import os
from models.FCCC import FCCC  # Usando la función FCCC
from utils.preprocessing import load_csv_and_train_test_split


# Función para ejecutar FCCC
def run_FCCC(file, q_labels, experiment_name):
    y_pred = FCCC(file, q_labels, experiment_name=experiment_name)  # Usando la función FCCC con el nombre del experimento
    messagebox.showinfo("Resultado", f"Las predicciones con FCCC se han realizado exitosamente.\nEl experimento se guardó como {experiment_name}.")


# Función para obtener los archivos de dataset disponibles en la carpeta datasets
def get_datasets_from_folder():
    dataset_folder = 'datasets'
    datasets = []
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.csv') or file.endswith('.arff'):
                datasets.append(os.path.join(root, file))
    return datasets


# Función para abrir un archivo CSV
def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return file_path


# Función para abrir un archivo ARFF
def open_arff_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("ARFF files", "*.arff")])
    return file_path


# Función principal de la interfaz gráfica
def run_gui():
    root = tk.Tk()
    root.title("FCCC - Seleccionar Dataset y Nombre de Experimento")

    # Obtener los archivos de datasets disponibles
    datasets = get_datasets_from_folder()

    # Dataset predeterminado (birds-train.arff)
    default_dataset = 'datasets/multi_etiqueta/Birds/birds-train.arff'
    if default_dataset not in datasets:
        datasets.insert(0, default_dataset)  # Asegúrate de que esté en la lista de datasets

    # Variable para almacenar la opción seleccionada
    dataset_var = tk.StringVar()
    dataset_var.set(default_dataset)  # Selección predeterminada

    # Menú desplegable para seleccionar el dataset
    dataset_label = tk.Label(root, text="Selecciona el dataset:")
    dataset_label.pack(pady=10)

    dataset_menu = tk.OptionMenu(root, dataset_var, *datasets)
    dataset_menu.pack(pady=10)

    # Campo de entrada para el número de etiquetas
    label_entry_label = tk.Label(root, text="Número de etiquetas (por defecto 19):")
    label_entry_label.pack(pady=5)
    label_entry = tk.Entry(root)
    label_entry.insert(0, "19")  # Número de etiquetas predeterminado
    label_entry.pack(pady=5)

    # Campo de entrada para el nombre del experimento
    experiment_label = tk.Label(root, text="Nombre del experimento:")
    experiment_label.pack(pady=5)
    experiment_entry = tk.Entry(root)
    experiment_entry.insert(0, "experimento")  # Nombre de experimento predeterminado
    experiment_entry.pack(pady=5)

    # Función para ejecutar el algoritmo FCCC
    def execute_algorithm():
        dataset_file = dataset_var.get()
        experiment_name = experiment_entry.get()
        q_labels = int(label_entry.get()) if label_entry.get().isdigit() else 19  # Número de etiquetas, por defecto 19

        # Verificar que se haya seleccionado un dataset
        if not dataset_file:
            messagebox.showerror("Error", "Por favor, selecciona un dataset.")
            return

        # Confirmar que el nombre del experimento no está vacío
        if not experiment_name:
            messagebox.showerror("Error", "Por favor, ingresa un nombre para el experimento.")
            return

        # Ejecutar FCCC
        run_FCCC(dataset_file, q_labels, f"experimentos/FCCC/{experiment_name}_results.csv")

    # Botón para ejecutar el algoritmo
    execute_button = tk.Button(root, text="Ejecutar FCCC", command=execute_algorithm)
    execute_button.pack(pady=20)

    # Iniciar la interfaz
    root.mainloop()


# Ejecutar la interfaz gráfica
if __name__ == "__main__":
    run_gui()
