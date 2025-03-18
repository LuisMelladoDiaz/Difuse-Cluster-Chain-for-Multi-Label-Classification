from models.FCCC import FCCC

## CONFIGURACIÓN #####################################################################################################################################################################################################

MULTILABEL_DATASETS_DIR =  "datasets/multi_etiqueta/"
DATASETS = {"Birds":19, "Emotions":6, "FoodTruck":12}
UMBRAL = 0
NUM_CLUSERS = 2
NUM_EXPERIMENTOS = 3
SEEDS= [42, 123, 7, 13, 2023, 1995, 101, 555, 999, 314, 69, 1001, 21, 33, 404, 777, 666, 1234, 4321, 9876, 2468, 1357, 1111, 2222, 3333, 4444, 5555, 8888, 9999, 10000]


## EXPERIMENTACIÓN #####################################################################################################################################################################################################

for dataset in DATASETS:
    for i in range(NUM_EXPERIMENTOS):

        train_file_path = MULTILABEL_DATASETS_DIR + dataset + "/" + str(dataset).lower() + "-train.arff"
        num_labels = DATASETS[dataset]
        experiment_name  = "experimentos/FCCC/" + dataset + "_results_" + str(i) + ".csv"
        seed = SEEDS[i]

        print("Experimento " + dataset + " " + str(i))

        FCCC(train_file_path, num_labels, False, NUM_CLUSERS, seed, experiment_name)

