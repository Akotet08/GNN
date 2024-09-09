import torch
import numpy as np
from random import choice
from torch.utils.data import Dataset
from torch_geometric.datasets import MovieLens
from dataloader.base_dataset import BaseDataset, TestDataset
from torch_geometric.utils import to_scipy_sparse_matrix


def sample_edges_for_test(data, test_ratio=0.4):
    edge_index = data[('user', 'rates', 'movie')].edge_index

    num_edges = edge_index.size(1)
    num_test = int(num_edges * test_ratio)

    perm = torch.randperm(num_edges)
    test_edge_idx = perm[:num_test]

    # Create test and train masks for edges
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask[test_edge_idx] = True
    train_mask = ~test_mask

    # Split the edges into test and train based on the mask
    test_edges = edge_index[:, test_mask]
    train_edges = edge_index[:, train_mask]

    # Update data with train edges and store test edges separately
    data[('user', 'rates', 'movie')].train_edge_index = train_edges
    data[('user', 'rates', 'movie')].test_edges = test_edges  # Store this if you want to use later

    return data, test_edges


class MovieLenseSmallDataset(BaseDataset):
    def __init__(self):
        super(MovieLenseSmallDataset, self).__init__()

        dataset = MovieLens('/home/akotet/datasets/movielense_small/raw')
        data = dataset[0]

        num_edges = data[('user', 'rates', 'movie')].edge_index.size(1)
        data[('user', 'rates', 'movie')].edge_attr = torch.ones((num_edges, 1), dtype=torch.float)

        self.user_num = data['user'].num_nodes
        self.item_num = data['movie'].num_nodes

        data = self.connected_components(data)

        self.data, self.test_edges = sample_edges_for_test(data)

        self.item_list = list(range(self.item_num))
        self.user_list = list(range(self.user_num))

        self.item_feature_matrix = data['movie'].x

        self._generate_set()
        self._count_degrees()
        self._create_sparse_adjacency()

        self.train_dataset = TrainDataset(self.data, self.item_list, self.train_user_set)
        self.test_dataset = TestDataset(self.test_user_set, self.item_num)

    def connected_components(self, data, component_idx=1):
        import scipy.sparse as sp

        def to_bipartite_adj(data):
            edge_index = data[('user', 'rates', 'movie')].edge_index
            num_users = data['user'].num_nodes
            num_movies = data['movie'].num_nodes

            # Shift movie indices by the number of users to avoid collisions
            edge_index_adjusted = edge_index.clone()
            edge_index_adjusted[1] += num_users

            # Create adjacency matrix (no edge weights)
            adj = to_scipy_sparse_matrix(edge_index_adjusted, num_nodes=num_users + num_movies)
            return adj, num_users, num_movies

        # Create bipartite adjacency matrix
        adj, num_users, num_movies = to_bipartite_adj(data)

        # Step 2: Use scipy to find connected components
        num_components, component_labels = sp.csgraph.connected_components(adj)

        # Step 3: Select the desired component (use component_idx to select)
        unique_components, count = np.unique(component_labels, return_counts=True)
        if component_idx >= num_components:
            raise ValueError(f"Component {component_idx} out of bounds. Only {num_components} components found.")

        # Select the desired component (default is largest, but can be changed via component_idx)
        desired_component = unique_components[
            count.argsort()[-1]]  # Select the `component_idx`-th largest component
        subset_np = component_labels == desired_component  # Create a mask for the selected component

        # Step 4: Convert to torch and match device
        user_mask = subset_np[:num_users]
        movie_mask = subset_np[num_users:]

        # Step 6: Convert masks to tensors
        user_subset = torch.from_numpy(user_mask).to(data[('user', 'rates', 'movie')].edge_index.device, torch.bool)
        movie_subset = torch.from_numpy(movie_mask).to(data[('user', 'rates', 'movie')].edge_index.device, torch.bool)

        # Step 7: Create the subset_dict for subgraph
        subset_dict = {
            'user': user_subset,
            'movie': movie_subset
        }

        # Step 8: Call the `subgraph` method to get the induced subgraph
        subgraph = data.subgraph(subset_dict)

        return subgraph

    def _generate_set(self):
        user_to_item_edge_type = ('user', 'rates', 'movie')
        if user_to_item_edge_type in self.data.edge_types:
            edge_index = self.data[user_to_item_edge_type].train_edge_index

            for i, (user_id, item_id) in enumerate(edge_index.t()):
                self.train_user_set[user_id.item()][item_id.item()] = 1
                self.train_item_set[item_id.item()][user_id.item()] = 1

            test_edge_index = self.data[user_to_item_edge_type].test_edges
            for i, (user_id, item_id) in enumerate(test_edge_index.t()):
                self.test_user_set[user_id.item()][item_id.item()] = 1
                self.test_item_set[item_id.item()][user_id.item()] = 1

    def _create_sparse_adjacency(self):
        # Extract edge_index for user-item interactions
        edge_index = self.data[('user', 'rates', 'movie')].train_edge_index

        # Extract user_ids and item_ids from edge_index
        user_ids = edge_index[0].tolist()
        item_ids = edge_index[1].tolist()

        # Interaction matrix
        value = [1.0] * len(user_ids)
        self.interact_matrix = torch.sparse_coo_tensor([user_ids, item_ids], value, (self.user_num, self.item_num))

        # Create a joint adjacency matrix with extended item indices
        tmp_index = [user_ids, [item_id + self.user_num for item_id in item_ids]]
        tmp_adj = torch.sparse_coo_tensor(tmp_index, value,
                                          (self.user_num + self.item_num, self.user_num + self.item_num))

        # Create the symmetric joint adjacency matrix
        joint_adjacency_matrix = (tmp_adj + tmp_adj.t()).coalesce()

        # Degree calculation and normalization
        row_indices = joint_adjacency_matrix.indices()[0]
        col_indices = joint_adjacency_matrix.indices()[1]
        joint_adjacency_matrix_values = joint_adjacency_matrix.values()

        degree = torch.sparse.sum(joint_adjacency_matrix, dim=1).to_dense()
        inv_sqrt_degree = torch.pow(degree, -0.5)
        inv_sqrt_degree[torch.isinf(inv_sqrt_degree)] = 0

        # Normalize adjacency values
        joint_adjacency_matrix_normal_value = inv_sqrt_degree[row_indices] * joint_adjacency_matrix_values * \
                                              inv_sqrt_degree[col_indices]

        self.joint_adjacency_matrix = joint_adjacency_matrix

        self.joint_adjacency_matrix_normal_spatial = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices], dim=0),
            joint_adjacency_matrix_normal_value,
            (self.user_num + self.item_num, self.user_num + self.item_num)
        ).coalesce()

        # Second-hop adjacency matrix
        self.second_hop_adjacency_matrix = self.calculate_second_hop(self.joint_adjacency_matrix_normal_spatial)

    def calculate_second_hop(self, adjacency_matrix):
        try:
            second_hop = torch.sparse.mm(adjacency_matrix, adjacency_matrix)
            return self._remove_self_loops(second_hop)
        except MemoryError:
            print("Memory Error encountered with second-hop calculation")
            return None

    def _remove_self_loops(self, sparse_matrix):
        indices = sparse_matrix.indices()
        mask = indices[0] != indices[1]
        return torch.sparse_coo_tensor(indices[:, mask], sparse_matrix.values()[mask], sparse_matrix.size())


class TrainDataset(Dataset):
    def __init__(self, data, item_list, train_user_set):
        super(TrainDataset, self).__init__()
        # Assuming 'data' is a HeteroData object with edge_index for 'user' to 'movie' interactions
        self.edge_index = data[('user', 'rates', 'movie')].train_edge_index
        self.num_edges = self.edge_index.size(1)
        self.item_list = item_list
        self.train_user_set = train_user_set

    def __len__(self):
        return self.num_edges  # The number of interactions (edges)

    def __getitem__(self, idx):
        user_id = self.edge_index[0][idx].item()  # User node index
        item_id = self.edge_index[1][idx].item()  # Item node index (positive item)

        # Generate negative item
        neg_item = choice(self.item_list)
        while neg_item in self.train_user_set[user_id]:
            neg_item = choice(self.item_list)

        return user_id, item_id, neg_item


if __name__ == '__main__':
    dataset = MovieLenseSmallDataset()
