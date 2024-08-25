import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, args, user_num, item_num,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.user_num = user_num
        self.item_num = item_num

        self.user_id_Embeddings = nn.Embedding(self.user_num, args.output_dim)
        self.item_id_Embeddings = nn.Embedding(self.item_num, args.output_dim)

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(args.output_dim, args.hidden_dim, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(args.hidden_dim))
        for _ in range(args.num_layers - 2):
            self.convs.append(
                GCNConv(args.hidden_dim, args.hidden_dim, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(args.hidden_dim))
        self.convs.append(
            GCNConv(args.hidden_dim, args.output_dim, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.user_id_Embeddings.reset_parameters()
        self.item_id_Embeddings.reset_parameters()

    def forward(self, user_id, pos_item, neg_item, edge_index):
        x = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = F.normalize(x, p=2, dim=-1)
        x = self.convs[-1](x, edge_index)

        user_embeddings, item_embeddings = torch.split(x, [self.user_num, self.item_num])

        user_embedded = user_embeddings[user_id]
        pos_item_embedded = item_embeddings[pos_item]
        neg_item_embedded = item_embeddings[neg_item]

        pos_score = torch.mul(user_embedded, pos_item_embedded).sum(dim=-1, keepdim=False)
        neg_score = torch.mul(user_embedded, neg_item_embedded).sum(dim=-1, keepdim=False)

        rec_loss = -(pos_score - neg_score).sigmoid().log().mean()

        # if rec_loss is nan
        if torch.isnan(rec_loss).any():
            print("rec loss is nan")
        return rec_loss, x

    def predict(self, user_id, edge_index):
        x = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        user_embeddings, item_embeddings = torch.split(x, [self.user_num, self.item_num])
        user_embedded = user_embeddings[user_id]
        score = torch.mm(user_embedded, item_embeddings.t())

        return score
