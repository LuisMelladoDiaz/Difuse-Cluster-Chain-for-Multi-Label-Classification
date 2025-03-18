import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from models.CC import train_CC
from models.CC import train_ECC
from models.FCCC import FCCC
from experimentation_framework import ExperimentationFramework


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


class ExperimentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Label Classification Experimentation")
        self.root.geometry("800x600")
        
        # Configuración de estilos
        self.style = ttk.Style()
        self.style.configure('TFrame', padding=10)
        self.style.configure('TLabel', padding=5)
        self.style.configure('TButton', padding=5)
        
        # Crear contenedor principal
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Variables
        self.model_var = tk.StringVar(value="FCCC")
        self.dataset_var = tk.StringVar(value="Birds")
        self.num_experiments_var = tk.StringVar(value="10")
        self.num_clusters_var = tk.StringVar(value="3")
        self.threshold_var = tk.StringVar(value="0.1")
        self.num_chains_var = tk.StringVar(value="10")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Sección de Modelo
        model_frame = ttk.LabelFrame(self.main_frame, text="Modelo", padding="5")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(model_frame, text="Seleccionar modelo:").grid(row=0, column=0, padx=5)
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                 values=["CC", "ECC", "FCCC"], state="readonly")
        model_combo.grid(row=0, column=1, padx=5)
        
        # Sección de Dataset
        dataset_frame = ttk.LabelFrame(self.main_frame, text="Dataset", padding="5")
        dataset_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(dataset_frame, text="Seleccionar dataset:").grid(row=0, column=0, padx=5)
        dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.dataset_var,
                                   values=["Birds", "Emotions"], state="readonly")
        dataset_combo.grid(row=0, column=1, padx=5)
        
        # Sección de Parámetros
        params_frame = ttk.LabelFrame(self.main_frame, text="Parámetros", padding="5")
        params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Número de experimentos
        ttk.Label(params_frame, text="Número de experimentos:").grid(row=0, column=0, padx=5)
        ttk.Entry(params_frame, textvariable=self.num_experiments_var, width=10).grid(row=0, column=1, padx=5)
        
        # Parámetros específicos de FCCC
        self.fccc_frame = ttk.Frame(params_frame)
        self.fccc_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(self.fccc_frame, text="Número de clusters:").grid(row=0, column=0, padx=5)
        ttk.Entry(self.fccc_frame, textvariable=self.num_clusters_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(self.fccc_frame, text="Umbral:").grid(row=1, column=0, padx=5)
        ttk.Entry(self.fccc_frame, textvariable=self.threshold_var, width=10).grid(row=1, column=1, padx=5)
        
        # Parámetros específicos de ECC
        self.ecc_frame = ttk.Frame(params_frame)
        self.ecc_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(self.ecc_frame, text="Número de cadenas:").grid(row=0, column=0, padx=5)
        ttk.Entry(self.ecc_frame, textvariable=self.num_chains_var, width=10).grid(row=0, column=1, padx=5)
        
        # Botón de ejecución
        ttk.Button(self.main_frame, text="Ejecutar Experimentos", 
                  command=self.run_experiments).grid(row=3, column=0, columnspan=2, pady=20)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.main_frame, length=300, 
                                          mode='determinate', variable=self.progress_var)
        self.progress_bar.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Etiqueta de estado
        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(self.main_frame, textvariable=self.status_var).grid(row=5, column=0, columnspan=2)
        
        # Configurar visibilidad de frames específicos
        self.update_parameter_frames()
        
        # Vincular evento de cambio de modelo
        model_combo.bind('<<ComboboxSelected>>', self.update_parameter_frames)
        
    def update_parameter_frames(self, event=None):
        """Actualiza la visibilidad de los frames de parámetros según el modelo seleccionado."""
        model = self.model_var.get()
        
        if model == "FCCC":
            self.fccc_frame.grid()
            self.ecc_frame.grid_remove()
        elif model == "ECC":
            self.fccc_frame.grid_remove()
            self.ecc_frame.grid()
        else:  # CC
            self.fccc_frame.grid_remove()
            self.ecc_frame.grid_remove()
    
    def run_experiments(self):
        """Ejecuta los experimentos según la configuración seleccionada."""
        try:
            # Obtener parámetros
            model_name = self.model_var.get()
            dataset_name = self.dataset_var.get()
            num_experiments = int(self.num_experiments_var.get())
            
            # Configurar diccionario de datasets
            datasets = {"Birds": 19, "Emotions": 6}
            
            # Seleccionar función del modelo
            if model_name == "CC":
                model_function = train_CC
                output_dir = "experimentos/CC"
                model_params = {}
            elif model_name == "ECC":
                model_function = train_ECC
                output_dir = "experimentos/ECC"
                model_params = {"number_of_chains": int(self.num_chains_var.get())}
            else:  # FCCC
                model_function = FCCC
                output_dir = "experimentos/FCCC"
                model_params = {
                    "num_clusters": int(self.num_clusters_var.get()),
                    "threshold": float(self.threshold_var.get())
                }
            
            # Crear directorio de salida si no existe
            os.makedirs(output_dir, exist_ok=True)
            
            # Inicializar y ejecutar framework
            self.status_var.set("Ejecutando experimentos...")
            self.progress_var.set(0)
            self.root.update()
            
            framework = ExperimentationFramework(
                model_name=model_name,
                model_function=model_function,
                output_dir=output_dir,
                datasets=datasets,
                num_experiments=num_experiments,
                **model_params
            )
            
            framework.run_experiments()
            framework.save_results()
            
            self.progress_var.set(100)
            self.status_var.set("¡Experimentos completados!")
            messagebox.showinfo("Éxito", "Los experimentos se han completado correctamente.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar los experimentos: {str(e)}")
            self.status_var.set("Error en la ejecución")

def main():
    root = tk.Tk()
    app = ExperimentationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
