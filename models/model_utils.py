from models.SimpleGCNconv import SimpleGNN
from models.SimpleLightGCN import LightGCN


def get_model(args, dataset_config, device):
    method_name = args.method
    output_dim = args.output_dim

    num_users = dataset_config['num_users']
    num_items = dataset_config['num_items']

    input_features = dataset_config['input_dim']

    if method_name == 'gcn':
        return SimpleGNN(input_features, output_dim)
    if method_name == 'lightgcn':
        return LightGCN(num_users, num_items, args, device)
    else:
        raise NotImplementedError

