import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='GNN Experiments')
    parser.add_argument('--dataset', type=str, default='book_crossing', choices=['book_crossing', 'movielense'], help='Dataset name')
    parser.add_argument('--seed', type=int, default=100, help='Seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=2, help='GPU ID')

    parser.add_argument('--method', type=str, default='lightgcn', help='gnn method')

    parser.add_argument('--hps', action='store_true', help='Search for hyperparameters')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=1024, help='Local batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of local epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Local learning rate')
    parser.add_argument('--wd', type=float, default=0.0001, help='Local weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer')

    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--output_dim', type=int, default=512, help='output dimension')
    parser.add_argument('--k', type=int, default=10, help='kth rank for metric computations')
    parser.add_argument('--separate_rate', type=float, default=0.2, help='rate to define head and tail items')

    parser.add_argument('--note', type=str, default='', help='Note for the experiment')

    args = parser.parse_args()
    return args
