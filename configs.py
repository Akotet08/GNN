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

citeseer = {
    # Data parameters
    'name': 'citeseer',
    'input_dim': 3703,  # 4 feature columns
    'num_nodes': 3327,
    'num_classes': 6,

    # Training parameters
    "epochs": 50,
    "lr": 0.1,
    "weight_decay": 0.00001,
    "optimizer": "sgd",
}

pubmed = {
    # Data parameters
    'name': 'pubmed',
    'input_dim': 500,  # 4 feature columns
    'num_nodes': 19717,
    'num_classes': 3,

    # Training parameters
    "epochs": 50,
    "lr": 0.1,
    "weight_decay": 0.00001,
    "optimizer": "sgd",
}

movielense = {
    # Data parameters
    'name': 'movielense',
    'input_dim': 0,  # 4 feature columns
    'num_users': 610,
    'num_items': 9742,

    # Training parameters
    "epochs": 50,
    "lr": 0.1,
    "weight_decay": 0.00001,
    "optimizer": "sgd",
}


def load_dataset_configs(dataset_name):
    if dataset_name == 'cora':
        return cora
    elif dataset_name == 'citeseer':
        return citeseer
    elif dataset_name == 'pubmed':
        return pubmed
    elif dataset_name == 'movielense':
        return movielense
    else:
        raise ValueError(f"Unknown or unimplemented dataset: {dataset_name}")
