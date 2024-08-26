import torch
import wandb
import numpy as np
import os.path as osp
import utils.pretty_print as pp
import torch.nn.functional as F
from utils.utils import set_seed
from arg_parser import parse_arguments
import torch_geometric.transforms as T
from models.model_utils import get_model
from configs import load_dataset_configs
from torch_geometric.datasets import Planetoid
from metrics import dirichlet_energy, mean_average_distance

print("GPU available: ", torch.cuda.is_available())


def init_wandb(args, dataset_configs):
    wandb.init(
        project="GNN-preliminary",
        config={
            "dataset": args.dataset,
            "seed": args.seed,
            "num_layers": args.num_layers,
            "method": args.method,
            "note": args.note,
            "dropout": args.dropout,
            "random_features": args.random_features,

            "epochs": dataset_configs['epochs'],
            "lr": dataset_configs['lr'],
            "weight_decay": dataset_configs['weight_decay'],
            "optimizer": dataset_configs['optimizer'],
        }
    )

def get_random_features(data):
    n, m = data.x.shape
    features = []
    for _ in range(n):
        features.append(torch.randn((1, 64)))

    return torch.cat(features, dim=0)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    embedding = model(data)
    loss = F.nll_loss(embedding[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item(), embedding


@torch.no_grad()
def test(model, data):
    model.eval()
    log_probs, accs = model(data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def main():
    args = parse_arguments()
    set_seed(args.seed)

    dataset_name = args.dataset
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else "cpu"
    dataset_configs = load_dataset_configs(dataset_name)
    # Hps
    if args.hps:
        dataset_configs['epochs'] = args.epochs
        dataset_configs['lr'] = args.lr
        dataset_configs['weight_decay'] = args.wd
        dataset_configs['optimizer'] = args.optimizer

    dataset = args.dataset
    transform = T.Compose([
        T.RandomNodeSplit(num_val=500, num_test=500),
        T.TargetIndegree(),
    ])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, transform=transform)
    data = dataset[0]

    if args.random_features:
        features = get_random_features(data)
        data.x = features

        dataset_configs['input_dim'] = 64

    model = get_model(args, dataset_configs, device)
    init_wandb(args, dataset_configs)

    edge_index = data.edge_index
    adj_dict = {}
    for i, j in edge_index.t().tolist():
        if i not in adj_dict:
            adj_dict[i] = []
        adj_dict[i].append(j)

    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    epoch_loss_list = []
    epoch_train_acc_list = []
    epoch_test_acc_list = []
    log_every = 1
    for epoch in range(1, args.epochs + 1):
        epoch_loss, embedding = train(model, data, optimizer)
        epoch_loss_list.append(epoch_loss)

        train_acc, test_acc = test(model, data)
        epoch_train_acc_list.append(train_acc)
        epoch_test_acc_list.append(test_acc)

        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

        avg_epoch_loss = np.mean(epoch_loss_list)
        avg_epoch_acc = np.mean(epoch_train_acc_list)
        avg_epoch_test_acc = np.mean(epoch_test_acc_list)
        if epoch % log_every == 0:
            pass
            wandb.log({'epoch': epoch + 1,
                       'train_loss': avg_epoch_loss,
                       'train_acc': avg_epoch_acc,
                       'test_acc': avg_epoch_test_acc})

    de = dirichlet_energy(embedding, adj_dict=adj_dict)
    mad = mean_average_distance(embedding, adj_dict=adj_dict)
    wandb.log({'mad': mad,
               'dirichlet energy': de})


if __name__ == '__main__':
    pp.print_string(' == GNN == ')
    main()
