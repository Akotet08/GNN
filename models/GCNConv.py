import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


