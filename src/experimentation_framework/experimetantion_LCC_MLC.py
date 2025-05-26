from experimentation_framework.experimentation_framework import ExperimentationFramework
from models.LCC_MLC import LCC_MLC


def run_lcc_mlc_experiments(DATASETS, NUM_EXPERIMENTS, NUM_FOLDS, sparse):
    framework = ExperimentationFramework(
        model_name="LCC-MLC",
        model_function=LCC_MLC,
        output_dir="experimentos/LCC-MLC",
        datasets=DATASETS,
        num_experiments=NUM_EXPERIMENTS,
        num_folds=NUM_FOLDS,
        sparse=sparse
    )

    framework.run_experiments()
    framework.save_results()
