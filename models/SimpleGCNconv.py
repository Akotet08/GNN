import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.sigmoid(x)
        x = self.conv2(x, edge_index)
        return x
