import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, args, device):
        super(LightGCN, self).__init__()

        self.name = "Light GCN"

        self.user_num = num_users
        self.item_num = num_items

        self.user_id_Embeddings = nn.Embedding(self.user_num, args.output_dim)
        self.item_id_Embeddings = nn.Embedding(self.item_num, args.output_dim)

        self.device = device
        self.L = args.num_layers
        self.dropout = args.dropout

        self.to(self.device)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.user_id_Embeddings.weight)
        nn.init.xavier_uniform_(self.item_id_Embeddings.weight)

    def forward(self, user_id, pos_item, neg_item, edge_index):
        if self.dropout > 0:
            edge_index_dropped, _ = dropout_edge(copy.deepcopy(edge_index.indices()), self.dropout)
            original_indices = edge_index.indices()
            original_values = edge_index.values()

            # Find the edges that were kept after dropout
            dropped_edges_mask = torch.zeros(original_indices.size(1), dtype=torch.bool)
            dropped_edges_mask[torch.unique(edge_index_dropped, dim=1, return_inverse=True)[1]] = True

            # Select the values corresponding to the dropped edges
            new_values = original_values[dropped_edges_mask]

            # Reconstruct the sparse tensor
            edge_index = torch.sparse_coo_tensor(
                indices=edge_index_dropped,
                values=new_values,
                size=edge_index.size(),
                device=edge_index.device
            )

        cur_embedding = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)

        all_embeddings = [cur_embedding]
        for i in range(self.L):
            cur_embedding = torch.mm(edge_index, cur_embedding)
            cur_embedding = F.normalize(cur_embedding, p=2, dim=1)
            all_embeddings.append(cur_embedding)

        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num])

        user_embedded = user_embeddings[user_id]
        pos_item_embedded = item_embeddings[pos_item]
        neg_item_embedded = item_embeddings[neg_item]

        pos_score = torch.mul(user_embedded, pos_item_embedded).sum(dim=-1, keepdim=False)
        neg_score = torch.mul(user_embedded, neg_item_embedded).sum(dim=-1, keepdim=False)

        rec_loss = -(pos_score - neg_score).sigmoid().log().mean()

        # if rec_loss is nan
        if torch.isnan(rec_loss).any():
            print("rec loss is nan")
        return rec_loss, all_embeddings

    def predict(self, user_id, edge_index):
        cur_embedding = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)

        all_embeddings = [cur_embedding]
        for i in range(self.L):
            cur_embedding = torch.mm(edge_index, cur_embedding)
            all_embeddings.append(cur_embedding)

        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.user_num, self.item_num])

        user_embedded = user_embeddings[user_id]
        score = torch.mm(user_embedded, item_embeddings.t())

        return score
