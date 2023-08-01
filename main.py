from evaluation import evaluation
from parameters import datasets

for dataset, pooling_sizes in zip(datasets.keys(), datasets.values()):
    # evaluate PPLP
    evaluation(
        dataset=dataset,
        pooling_sizes=pooling_sizes,
        classifier='GBDT',
        params=None,
        fold=10,
        times=10
    )

    # evaluate GNN models with different readout functions
    # evaluation_gnn(
    #     gnn_model='GraphSAGE',
    #     readout='pplp',
    #     dataset=dataset,
    #     pooling_sizes=pooling_sizes,
    #     fold=10,
    #     times=10
    # )
