import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb


class Net(torch.nn.Module):
    def __init__(self, args, num_nodes, num_features, num_classes):
        super().__init__()
        self.args = args

        self.conv1 = GCNConv(num_features, args.hidden_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(args.num_layers -  1):
            self.convs.append(GCNConv(args.hidden_dim, args.hidden_dim))

        self.conv2 = GCNConv(args.hidden_dim, num_classes)
        # self.embedding = torch.nn.Embedding(num_nodes, num_features)

    @torch.no_grad()
    def log_embedding_tse(self, stage, data):
        # projection_tse = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10)
        projection_pca = PCA(n_components=2)

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        labels = data.y.cpu().detach().numpy().reshape(-1, 1)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)

        x_projected = projection_pca.fit_transform(x.cpu().detach().numpy())
        x_projected = np.concatenate([x_projected, labels], axis=1)
        x_table = wandb.Table(data=x_projected.tolist(), columns=["x", "y", 'class'])

        x_title = 'embedding ' + stage
        wandb.log({
            f'embedding_{stage}': wandb.plot.scatter(x_table, "x", "y",
                                                          title=x_title),
        })
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


