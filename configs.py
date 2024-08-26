cora = {
    # Data parameters
    'name': 'cora',
    'input_dim': 1433,  # 4 feature columns
    'num_nodes': 2708,
    'num_classes': 7,

    # Training parameters
    "epochs": 50,
    "lr": 0.1,
    "weight_decay": 0.00001,
    "optimizer": "sgd",
}


def load_dataset_configs(dataset_name):
    if dataset_name == 'cora':
        return cora
    else:
        raise ValueError(f"Unknown or unimplemented dataset: {dataset_name}")
