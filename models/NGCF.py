import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, num_users, num_items, args, device):
        super().__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.embedding_dim = args.output_dim
        self.weight_size = [64] * args.num_layers
        self.n_layers = args.num_layers
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.device = device

        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
            # self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        self._init_weight_()

        self.to(self.device)

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_id, pos_item, neg_item, edge_index):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(edge_index, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            # ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings = norm_embeddings

        # all_embeddings = torch.cat(all_embeddings, dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])

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
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            side_embeddings = torch.sparse.mm(edge_index, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            # ego_embeddings = self.dropout_list[i](ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings = norm_embeddings

        # all_embeddings = torch.cat(all_embeddings, dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])

        user_embedded = user_embeddings[user_id]
        score = torch.mm(user_embedded, item_embeddings.t())

        return score

