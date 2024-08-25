from models.GCNConv import GCN
from models.SimpleLightGCN import LightGCN
from models.NGCF import NGCF


def get_model(args, dataset_config, device):
    method_name = args.method
    output_dim = args.output_dim

    num_users = dataset_config['num_users']
    num_items = dataset_config['num_items']

    input_features = dataset_config['input_dim']

    if method_name == 'gcn':
        return GCN(args, num_users, num_items)
    elif method_name == 'lightgcn':
        return LightGCN(num_users, num_items, args, device)
    elif method_name == 'ngcf':
        return NGCF(num_users, num_items, args, device)
    else:
        raise NotImplementedError

