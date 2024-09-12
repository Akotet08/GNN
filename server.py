import torch
import wandb
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader.bookcrossing_dataset import Book_crossing_Dataset
from dataloader.movielense_dataset import MovieLense_Dataset
from dataloader.movielense_small_dataset import MovieLenseSmallDataset
from metrics import dirichlet_energy, mean_average_distance
from metrics import ranking_measure_test_set, ranking_measure_degree_test_set
from metrics import compare_head_tail_rec_percentage
from utils.pretty_print import print_header
from collections import defaultdict
from sklearn.manifold import TSNE


class Server:
    def __init__(self, model, dataset_configs, device, args):
        self.dataset = None
        self.model = model
        self.args = args
        self.dataset_configs = dataset_configs
        self.device = device

        self.init_server()
        self.init_wandb()

        self.train_loader = DataLoader(self.dataset.train_dataset, batch_size=dataset_configs['batch_size'],
                                       shuffle=True)
        self.test_loader = DataLoader(self.dataset.test_dataset, batch_size=dataset_configs['batch_size'],
                                      shuffle=False)

    def init_server(self):
        if self.dataset_configs['name'] == 'book_crossing':
            self.dataset = Book_crossing_Dataset()
        elif self.dataset_configs['name'] == 'movielense':
            self.dataset = MovieLense_Dataset()
        elif self.dataset_configs['name'] == 'movielense_small':
            self.dataset = MovieLenseSmallDataset()
        else:
            raise NotImplementedError("Dataset not implemented")
        self.model.to(self.device)

    def init_wandb(self):
        wandb.init(
            project="GNN-preliminary",
            config={
                "dataset": self.args.dataset,
                "seed": self.args.seed,
                "num_layers": self.args.num_layers,
                "K": self.args.k,
                "method": self.args.method,
                "note": self.args.note,
                "dropout": self.args.dropout,

                "batch_size": self.dataset_configs['batch_size'],
                "epochs": self.dataset_configs['epochs'],
                "lr": self.dataset_configs['lr'],
                "weight_decay": self.dataset_configs['weight_decay'],
                "optimizer": self.dataset_configs['optimizer'],
            }
        )

    def run(self):
        print_header('Server is running...')
        self.train()

        print_header('Testing...')
        self.evaluate(silent=False)

    def train(self):
        epochs = self.dataset_configs['epochs']
        self.model.train()

        log_every = 1
        print_every = 1
        optimizer = self.get_optimizer(self.model)

        self.model.log_embedding_tse('initial', self.dataset.joint_adjacency_matrix_normal_spatial.to(self.device))
        for epoch in range(epochs):
            epoch_loss_list = []
            for batch_idx, (user_idx, pos_idx, neg_idx) in enumerate(tqdm(self.train_loader)):
                user_idx, pos_idx, neg_idx = user_idx.to(self.device), pos_idx.to(self.device), neg_idx.to(self.device)
                optimizer.zero_grad()

                loss, embd = self.model(user_idx, pos_idx, neg_idx,
                                        self.dataset.joint_adjacency_matrix_normal_spatial.to(self.device))

                loss.backward()
                optimizer.step()
                epoch_loss_list.append(loss.item())

            avg_epoch_loss = np.mean(epoch_loss_list)
            if (epoch + 1) % print_every == 0:
                print(f'epoch:{epoch + 1}/{epochs}. Loss: {avg_epoch_loss:.3f}')

            if (epoch + 1) % log_every == 0:
                wandb.log({'epoch': epoch + 1, 'train_loss': avg_epoch_loss})

        user_emb, item_emb = torch.split(embd, [self.dataset_configs['num_users'],
                                                self.dataset_configs['num_items']])

        user_mask = np.ones((self.dataset_configs['num_users'], 1))
        item_mask = np.zeros((self.dataset_configs['num_items'], 1))
        mask = np.concatenate([user_mask, item_mask], axis=0)

        edge_index = self.dataset.joint_adjacency_matrix_normal_spatial.coalesce().indices()
        adj_dict = {}
        for i, j in edge_index.t().tolist():
            if i not in adj_dict:
                adj_dict[i] = []
            adj_dict[i].append(j)

        de_first_hop = dirichlet_energy(embd, adj_dict=adj_dict)
        mad_first_hop = mean_average_distance(embd, adj_dict=adj_dict)

        edge_index = self.dataset.second_hop_adjacency_matrix.coalesce().indices()
        adj_dict = defaultdict(list)
        for idx in tqdm(range(edge_index.size(1))):
            i = edge_index[0, idx].item()
            j = edge_index[1, idx].item()
            adj_dict[i].append(j)

        adj_dict = dict(adj_dict)

        de_second_hop = dirichlet_energy(embd, adj_dict=adj_dict)
        mad_second_hop = mean_average_distance(embd, adj_dict=adj_dict)

        # differnt measures
        ones_all = torch.ones((len(embd), 1), device=embd.device)
        ones_user = torch.ones((len(user_emb), 1), device=user_emb.device)
        ones_item = torch.ones((len(item_emb), 1), device=item_emb.device)

        node_similarity_all = torch.linalg.matrix_norm(embd - (torch.matmul(ones_all.t(), embd) / len(embd)))
        node_similarity_user = torch.linalg.matrix_norm(user_emb - (torch.matmul(ones_user.t(), user_emb) / len(user_emb)))
        node_similarity_item = torch.linalg.matrix_norm(item_emb - (torch.matmul(ones_item.t(), item_emb) / len(item_emb)))

        wandb.log(
            {
                'mad_first_hop': mad_first_hop,
                   'dirichlet energy_first_hop': de_first_hop,
                   'mad_second_hop': mad_second_hop,
                   'dirichlet energy_second_hop': de_second_hop,
                   'node similarity all': node_similarity_all.item(),
                   'node similarity user': node_similarity_user.item(),
                   'node similarity item': node_similarity_item.item(),
                   })
        
        self.model.log_embedding_tse('final', self.dataset.joint_adjacency_matrix_normal_spatial.to(self.device))

    def evaluate(self, silent=False):
        self.model.eval()
        with ((torch.no_grad())):
            ndcg, recall, mrr = [], [], []
            head_ndcg_batch, tail_ndcg_batch = [], []
            head_recall_batch, tail_recall_batch = [], []
            body_ndcg_batch, body_recall_batch = [], []
            head_per_batch, tail_per_batch, body_per_batch = [], [], []
            test_items = list(self.dataset.test_item_set.keys())
            for (user_id, ground_truth) in tqdm(self.test_loader):
                user_id = user_id.to(self.device)
                ground_truth = ground_truth.to(self.device)

                predicted_score = self.model.predict(user_id,
                                                     self.dataset.joint_adjacency_matrix_normal_spatial.to(self.device))

                user_historical_mask = torch.ones((len(user_id), self.dataset.item_num), device=self.device)
                user_id_list = user_id.cpu().numpy().tolist()

                for idx, user in enumerate(user_id_list):
                    common_items = list(self.dataset.train_user_set[user].keys())
                    user_historical_mask[idx, common_items] = 0  # Mask out items the user has interacted with

                # Apply the mask to the predicted scores
                predicted_score = torch.mul(user_historical_mask, predicted_score)

                ndcg_batch, recall_batch, mrr_batch = ranking_measure_test_set(
                    predicted_score,
                    ground_truth,
                    self.args.k,
                    test_items)
                head_ndcg, head_recall, tail_ndcg, tail_recall, body_ndcg, body_recall = ranking_measure_degree_test_set(
                    predicted_score,
                    ground_truth,
                    self.args.k,
                    self.dataset.item_degrees,
                    self.args.separate_rate,
                    test_items)

                head_per, tail_per, body_per = compare_head_tail_rec_percentage(
                    predicted_score,
                    test_items,
                    self.dataset.item_degrees, self.args.separate_rate, self.args.k)

                ndcg.append(ndcg_batch)
                recall.append(recall_batch)
                mrr.append(mrr_batch)

                head_ndcg_batch.append(head_ndcg)
                tail_ndcg_batch.append(tail_ndcg)
                body_ndcg_batch.append(body_ndcg)
                body_recall_batch.append(body_recall)
                head_recall_batch.append(head_recall)
                tail_recall_batch.append(tail_recall)

                head_per_batch.append(head_per)
                tail_per_batch.append(tail_per)
                body_per_batch.append(body_per)

            if not silent:
                print(f' NDCG@{self.args.k}: {np.mean(ndcg):.6f} \n',
                      f'Recall@{self.args.k}: {np.mean(recall):.6f} \n',
                      f'MRR@{self.args.k}: {np.mean(mrr):.6f} \n',
                      f'Head NDCG@{self.args.k}: {np.mean(head_ndcg):.6f} \n',
                      f'Head Recall@{self.args.k}: {np.mean(head_recall_batch):.6f} \n',
                      f'Tail NDCG@{self.args.k}: {np.mean(tail_ndcg_batch):.6f} \n',
                      f'Tail Recall@{self.args.k}: {np.mean(tail_recall_batch):.6f} \n',
                      f'Body Recall@{self.args.k}: {np.mean(body_recall_batch):.6f} \n',
                      f'Body NDCG@{self.args.k}: {np.mean(body_ndcg_batch):.6f} \n',
                      f'Head Percentage@{self.args.k}: {np.mean(head_per_batch):.6f} \n',
                      f'Tail Percentage@{self.args.k}: {np.mean(tail_per_batch):.6f} \n',
                      f'Body Percentage@{self.args.k}: {np.mean(body_per_batch):.6f} \n', )

            wandb.log({f'NDCG@{self.args.k}': np.mean(ndcg),
                       f'Recall@{self.args.k}': np.mean(recall),
                       f'MRR@{self.args.k}': np.mean(mrr),
                       f'Head NDCG@{self.args.k}': np.mean(head_ndcg_batch),
                       f'Head Recall@{self.args.k}': np.mean(head_recall_batch),
                       f'Tail NDCG@{self.args.k}': np.mean(tail_ndcg_batch),
                       f'Tail Recall@{self.args.k}': np.mean(tail_recall_batch),
                       f'Body NDCG@{self.args.k}': np.mean(body_ndcg_batch),
                       f'Body Recall@{self.args.k}': np.mean(body_recall_batch),
                       f'Head Percentage@{self.args.k}': np.mean(head_per_batch),
                       f'Tail Percentage@{self.args.k}': np.mean(tail_per_batch),
                       f'Body Percentage@{self.args.k}': np.mean(body_per_batch), })

    def get_optimizer(self, model):
        optimizer_name = self.dataset_configs['optimizer']
        lr = self.dataset_configs['lr']
        wd = self.dataset_configs['weight_decay']
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f'Invalid optimizer {optimizer_name}')
        return optimizer
