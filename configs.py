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
    'num_users': 6040, # FUll: 6040,
    'num_items':  3883, # Full: 3883,

    # Training parameters
    "batch_size": 1024,
    "epochs": 50,
    "lr": 0.1,
    "weight_decay": 0.00001,
    "optimizer": "sgd",
}

movielense_small = {
    # Data parameters
    'name': 'movielense_small',
    'input_dim': 404,
    'num_users': 610,
    'num_items':  9742,

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
    elif dataset_name == 'movielense_small':
        return movielense_small
    else:
        raise ValueError(f"Unknown or unimplemented dataset: {dataset_name}")
