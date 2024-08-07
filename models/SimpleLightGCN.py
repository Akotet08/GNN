import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, args, device):
        super(LightGCN, self).__init__()

        self.name = "Light GCN"

        self.user_num = num_users
        self.item_num = num_items

        self.user_id_Embeddings = nn.Embedding(self.user_num, args.output_dim)
        self.item_id_Embeddings = nn.Embedding(self.item_num, args.output_dim)

        nn.init.xavier_uniform_(self.user_id_Embeddings.weight)
        nn.init.xavier_uniform_(self.item_id_Embeddings.weight)

        self.device = device
        self.L = args.num_layers

        self.to(self.device)

    def forward(self, user_id, pos_item, neg_item, edge_index):
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
