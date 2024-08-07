book_crossing = {
    # Data parameters
    'name': 'book_crossing',
    'input_dim': 4,  # 4 feature columns
    'num_users': 92107,
    'num_items': 271379,

    # Training parameters
    "batch_size": 1024,
    "epochs": 50,
    "lr": 0.1,
    "weight_decay": 0.00001,
    "optimizer": "sgd",
}

movielense = {
    # Data parameters
    'name': 'movielense',
    'input_dim': 21,  # 4 feature columns
    'num_users': 6040,
    'num_items': 3883,

    # Training parameters
    "batch_size": 1024,
    "epochs": 50,
    "lr": 0.1,
    "weight_decay": 0.00001,
    "optimizer": "sgd",
}


def load_dataset_configs(dataset_name):
    if dataset_name == 'book_crossing':
        return book_crossing
    elif dataset_name == 'movielense':
        return movielense
    else:
        raise ValueError(f"Unknown or unimplemented dataset: {dataset_name}")
