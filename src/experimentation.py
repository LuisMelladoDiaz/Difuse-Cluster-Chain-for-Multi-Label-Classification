from experimentation_framework.experimentation_CC import run_cc_experiments
from experimentation_framework.experimentation_ECC import run_ecc_experiments
from experimentation_framework.experimentation_FCCC import run_fcc_experiments
from experimentation_framework.experimetantion_LCC_MLC import run_lcc_mlc_experiments

## CONFIGURATION ###################################################################################################

DATASETS = {"Emotions": 6, "CHD_49": 6, "Foodtruck": 12,  "Water-quality": 14, "Birds": 19}
DATASETS_SPARSE = {"VirusGo": 6, "Yahoo_Business": 30, "Yahoo_Arts": 26}
DATASETS_BIG = {"Mediamill": 101}
DATASETS_BIG_SPARSE = {"Bibtex": 159}

NUM_EXPERIMENTS = 1
NUM_FOLDS = 1
NUM_CHAINS = 2
THRESHOLD = 0.05
NUM_CLUSTERS = 3

## RUN EXPERIMENTS #################################################################################################

def run_all_experiments(datasets, sparse=False, label=""):
    print(f"Ejecutando experimentos con datasets: {label.upper()} | Sparse: {sparse}\n")

    print(f"→ Ejecutando con modelo: CC ({label})")
    run_cc_experiments(datasets, NUM_EXPERIMENTS, NUM_FOLDS, sparse=sparse)

    print(f"→ Ejecutando con modelo: ECC ({label})")
    run_ecc_experiments(datasets, NUM_EXPERIMENTS, NUM_CHAINS, NUM_FOLDS, sparse=sparse)

    print(f"→ Ejecutando con modelo: FCCC ({label})")
    run_fcc_experiments(datasets, NUM_EXPERIMENTS, NUM_FOLDS, NUM_CLUSTERS, THRESHOLD, sparse=sparse)

    print(f"→ Ejecutando con modelo: LCC_MLC ({label})")
    run_lcc_mlc_experiments(datasets, NUM_EXPERIMENTS, NUM_FOLDS, sparse=sparse)


##run_all_experiments(DATASETS, sparse=False, label="small-medium datasets") # EJECUTADOS CON EXITO
##run_all_experiments(DATASETS_SPARSE, sparse=True, label="sparse datasets")# Ejecutados con exito para cc y ecc, para fccc y lcc mlc fallo en yahoo arts fold2
##run_all_experiments(DATASETS_BIG, sparse=False, label="big datasets") # EJECUTADOS CON EXITO
##run_all_experiments(DATASETS_BIG_SPARSE, sparse=True, label="big sparse datasets") # EJECUTADOS CON EXITO

run_fcc_experiments(DATASETS, NUM_EXPERIMENTS, NUM_FOLDS, NUM_CLUSTERS, THRESHOLD, sparse=False)