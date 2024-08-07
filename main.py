import torch
import utils.pretty_print as pp
from utils.utils import set_seed
from arg_parser import parse_arguments
from models.model_utils import get_model
from configs import load_dataset_configs
from server import Server

print("GPU available: ", torch.cuda.is_available())


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
        dataset_configs['batch_size'] = args.batch_size
        dataset_configs['weight_decay'] = args.wd
        dataset_configs['optimizer'] = args.optimizer

    model = get_model(args, dataset_configs, device)
    server = Server(model, dataset_configs, device, args)
    server.run()


if __name__ == '__main__':
    pp.print_string(' == GNN == ')
    main()
