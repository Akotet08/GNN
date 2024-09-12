import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, args, device):
        super(MatrixFactorization, self).__init__()

        self.name = "Matrix Factorization"

        self.user_num = num_users
        self.item_num = num_items

        self.user_id_Embeddings = nn.Embedding(self.user_num, args.output_dim)
        self.item_id_Embeddings = nn.Embedding(self.item_num, args.output_dim)

        self.device = device
        self.dropout = args.dropout

        self.to(self.device)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.user_id_Embeddings.weight)
        nn.init.xavier_uniform_(self.item_id_Embeddings.weight)

    @torch.no_grad()
    def log_embedding_tse(self, stage, edge_index=None):
        # Use PCA to project embeddings to 2D space for visualization
        projection_pca = PCA(n_components=2)

        user_embedding = self.user_id_Embeddings.weight
        item_embedding = self.item_id_Embeddings.weight

        user_projected = projection_pca.fit_transform(user_embedding.cpu().detach().numpy())
        user_projected_variance = np.linalg.norm(projection_pca.explained_variance_)

        item_projected = projection_pca.fit_transform(item_embedding.cpu().detach().numpy())
        item_projected_variance = np.linalg.norm(projection_pca.explained_variance_)

        user_projected_with_label = [[x, y, "User"] for x, y in user_projected.tolist()]
        item_projected_with_label = [[x, y, "Item"] for x, y in item_projected.tolist()]

        user_table = wandb.Table(data=user_projected_with_label, columns=["x", "y", "type"])
        item_table = wandb.Table(data=item_projected_with_label, columns=["x", "y", "type"])

        user_title = 'User Embedding ' + stage
        item_title = 'Item Embedding ' + stage

        wandb.log({
            f'user embedding_{stage}': wandb.plot.scatter(user_table, "x", "y", title=user_title),
            f'item embedding_{stage}': wandb.plot.scatter(item_table, "x", "y", title=item_title),
            f'user variance {stage}': user_projected_variance,
            f'item variance {stage}': item_projected_variance
        })

    def forward(self, user_id, pos_item, neg_item, edge_index=None):
        """
        Forward pass for Matrix Factorization.
        user_id: Tensor of user IDs.
        pos_item: Tensor of positive item IDs (items the users interacted with).
        neg_item: Tensor of negative item IDs (items not interacted with), not used in MF.
        """
        # Get embeddings for users and items
        user_embedded = self.user_id_Embeddings(user_id)
        pos_item_embedded = self.item_id_Embeddings(pos_item)

        # normalize user and item embeddings
        user_embedded = F.normalize(user_embedded, p=2, dim=1)
        pos_item_embedded = F.normalize(pos_item_embedded, p=2, dim=1)

        # Calculate positive scores (dot product of user and item embeddings)
        pos_score = torch.mul(user_embedded, pos_item_embedded).sum(dim=-1)

        # If negative item is provided, compute negative score (optional)
        if neg_item is not None:
            neg_item_embedded = self.item_id_Embeddings(neg_item)
            neg_item_embedded = F.normalize(neg_item_embedded, p=2, dim=1)

            neg_score = torch.mul(user_embedded, neg_item_embedded).sum(dim=-1)
            # Compute BPR loss (Bayesian Personalized Ranking)
            rec_loss = -(pos_score - neg_score).sigmoid().log().mean()
        else:
            rec_loss = -pos_score.sigmoid().log().mean()

        # Handle potential NaNs in the loss
        if torch.isnan(rec_loss).any():
            print("Warning: rec_loss contains NaN values")

        return rec_loss, torch.ones_like(user_embedded)

    def predict(self, user_id, edge_index=None):
        """
        Prediction function to get recommendation scores for all items for a given user.
        user_id: Tensor of user IDs.
        """
        # Get the embedding for the specific user
        user_embedded = self.user_id_Embeddings(user_id)

        # Get embeddings for all items
        item_embeddings = self.item_id_Embeddings.weight

        # Compute recommendation scores for all items (dot product of user and item embeddings)
        score = torch.mm(user_embedded, item_embeddings.t())

        return score
