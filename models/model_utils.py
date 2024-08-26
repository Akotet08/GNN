from models.GCNConv import Net


def get_model(args, dataset_config, device):
    method_name = args.method
    num_features = dataset_config['input_dim']
    num_classes = dataset_config['num_classes']

    if method_name == 'gcn':
        return Net(args, num_features, num_classes)
    else:
        raise NotImplementedError

