import copy

import numpy as np
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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

    @torch.no_grad()
    def log_embedding_tse(self, stage, edge_index):
        # projection_tse = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10)
        projection_pca = PCA(n_components=2)

        cur_embedding = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)
        all_embeddings = [cur_embedding]
        for i in range(self.L):
            cur_embedding = torch.sparse.mm(edge_index, cur_embedding)
            cur_embedding = F.normalize(cur_embedding, p=2, dim=1)
            all_embeddings.append(cur_embedding)

        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)
        user_embedding, item_embedding = torch.split(all_embeddings, [self.user_num, self.item_num])

        user_projected = projection_pca.fit_transform(user_embedding.cpu().detach().numpy())
        user_projected_variance = np.linalg.norm(projection_pca.explained_variance_)

        item_projected = projection_pca.fit_transform(item_embedding.cpu().detach().numpy())
        item_projected_variance = np.linalg.norm(projection_pca.explained_variance_)

        all_embeddings_projected = projection_pca.fit_transform(all_embeddings.cpu().detach().numpy())
        all_embeddings_projected_variance = np.linalg.norm(projection_pca.explained_variance_)

        user_projected_with_label = [[x, y, f"{self.L}"] for x, y in user_projected.tolist()]
        item_projected_with_label = [[x, y, f"{self.L}"] for x, y in item_projected.tolist()]
        all_embeddings_with_label = [[x, y, f"{self.L}"] for x, y in all_embeddings_projected.tolist()]

        user_table = wandb.Table(data=user_projected_with_label, columns=["x", "y", "num_layers"])
        item_table = wandb.Table(data=item_projected_with_label, columns=["x", "y", "num_layers"])
        all_embeddings_table = wandb.Table(data=all_embeddings_with_label, columns=["x", "y", "num_layers"])

        user_title = 'user embedding ' + stage
        item_title = 'item embedding ' + stage
        all_embeddings_title = 'all embedding ' + stage

        wandb.log({
            f'user embedding_{stage}': wandb.plot.scatter(user_table, "x", "y",
                                                          title=user_title),
            f'item embedding_{stage}': wandb.plot.scatter(item_table, "x", "y",
                                                          title=item_title),
            f'all embedding_{stage}': wandb.plot.scatter(all_embeddings_table, "x", "y",
                                                         title=all_embeddings_title),
            f'user variance {stage}': user_projected_variance,
            f'item variance {stage}': item_projected_variance,
            f'all embedding variance': all_embeddings_projected_variance
        })

    def propagate(self, edge_index, cur_embedding):
        """
        Propagate embeddings through the graph for one layer using the adjacency matrix.
        edge_index: sparse adjacency matrix.
        cur_embedding: concatenated user and item embeddings.
        """
        return torch.sparse.mm(edge_index, cur_embedding)

    def forward(self, user_id, pos_item, neg_item, edge_index):
        """
        Forward pass for LightGCN with dropout on edges.
        user_id: Tensor of user IDs.
        pos_item: Tensor of positive item IDs (items the users interacted with).
        neg_item: Tensor of negative item IDs (items not interacted with).
        edge_index: Sparse adjacency matrix.
        """
        # Apply edge dropout if specified
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
            edge_index_dropped = torch.sparse_coo_tensor(
                indices=edge_index_dropped,
                values=new_values,
                size=edge_index.size(),
                device=edge_index.device
            )
        else:
            edge_index_dropped = edge_index

        # Concatenate user and item embeddings
        cur_embedding = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)

        # Store all embeddings from each layer, including initial embeddings
        all_embeddings = [cur_embedding]

        # Propagate embeddings through L layers
        for _ in range(self.L):
            cur_embedding = self.propagate(edge_index_dropped, cur_embedding)
            cur_embedding = F.normalize(cur_embedding, p=2, dim=1)
            all_embeddings.append(cur_embedding)

        # Stack and average embeddings from all layers
        all_embeddings = torch.stack(all_embeddings, dim=0)
        final_embedding = torch.mean(all_embeddings, dim=0)

        # Split final embeddings back into user and item embeddings
        user_embeddings, item_embeddings = torch.split(final_embedding, [self.user_num, self.item_num])

        # Get embeddings for the specific users and items (positive and negative)
        user_embedded = user_embeddings[user_id]
        pos_item_embedded = item_embeddings[pos_item]
        neg_item_embedded = item_embeddings[neg_item]

        # Calculate scores (dot product of user and item embeddings)
        pos_score = torch.mul(user_embedded, pos_item_embedded).sum(dim=-1)
        neg_score = torch.mul(user_embedded, neg_item_embedded).sum(dim=-1)

        # Compute BPR loss (Bayesian Personalized Ranking)
        rec_loss = -(pos_score - neg_score).sigmoid().log().mean()

        # Handle potential NaNs in the loss
        if torch.isnan(rec_loss).any():
            print("Warning: rec_loss contains NaN values")

        return rec_loss, final_embedding

    def predict(self, user_id, edge_index):
        """
        Prediction function to get recommendation scores for all items for a given user.
        user_id: Tensor of user IDs.
        edge_index: Sparse adjacency matrix.
        """
        # Concatenate user and item embeddings
        cur_embedding = torch.cat([self.user_id_Embeddings.weight, self.item_id_Embeddings.weight], dim=0)

        # Store all embeddings from each layer, including initial embeddings
        all_embeddings = [cur_embedding]

        # Propagate embeddings through L layers
        for _ in range(self.L):
            cur_embedding = self.propagate(edge_index, cur_embedding)
            cur_embedding = F.normalize(cur_embedding, p=2, dim=1)
            all_embeddings.append(cur_embedding)

        # Stack and average embeddings from all layers
        all_embeddings = torch.stack(all_embeddings, dim=0)
        final_embedding = torch.mean(all_embeddings, dim=0)

        # Split final embeddings back into user and item embeddings
        user_embeddings, item_embeddings = torch.split(final_embedding, [self.user_num, self.item_num])

        # Get the embedding for the specific user
        user_embedded = user_embeddings[user_id]

        # Compute recommendation scores for all items (dot product of user and item embeddings)
        score = torch.mm(user_embedded, item_embeddings.t())

        return score
