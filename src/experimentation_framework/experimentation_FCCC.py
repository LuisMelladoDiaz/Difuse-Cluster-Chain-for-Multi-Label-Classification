from models.FCCC import FCCC
from experimentation_framework.experimentation_framework import ExperimentationFramework

def run_fcc_experiments(DATASETS, NUM_EXPERIMENTS, NUM_FOLDS, NUM_CLUSTERS, THRESHOLD, sparse):
    framework = ExperimentationFramework(
        model_name="FCCC",
        model_function=FCCC,
        output_dir="experimentos/FCCC",
        datasets=DATASETS,
        num_experiments=NUM_EXPERIMENTS,
        num_folds=NUM_FOLDS,
        threshold=THRESHOLD,
        num_clusters = NUM_CLUSTERS,
        sparse=sparse
    )

    framework.run_experiments()
    framework.save_results()
