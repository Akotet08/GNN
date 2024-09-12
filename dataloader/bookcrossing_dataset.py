import os
import json
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import to_scipy_sparse_matrix
from dataloader.base_dataset import BaseDataset, TrainDataset, TestDataset


SAMPLING_RATIO = 0.4


def load_data():
    path_config = 'data_preprocessor/dataset_path.json'
    with open(path_config, 'r') as f:
        paths = json.load(f)

    path_to_dataset = paths['book_crossing']['processed']
    train_interaction_filename = f'book_crossing_train_{SAMPLING_RATIO}.csv'
    test_interaction_filename = f'book_crossing_test_{SAMPLING_RATIO}.csv'
    features_filename = 'book_crossing_item_features.csv'

    train_interaction_data_path = os.path.join(path_to_dataset, train_interaction_filename)
    test_interaction_data_path = os.path.join(path_to_dataset, test_interaction_filename)
    feature_data_path = os.path.join(path_to_dataset, features_filename)

    train_interaction_df = pd.read_csv(train_interaction_data_path)
    test_interaction_df = pd.read_csv(test_interaction_data_path)

    item_features_df = pd.read_csv(feature_data_path)

    config_path = os.path.join(path_to_dataset, 'book_crossing_config.json')
    with open(config_path, 'r') as f:
        configs = json.load(f)

    print('loaded dataset and configs...')
    return train_interaction_df, test_interaction_df, item_features_df, configs


class Book_crossing_Dataset(BaseDataset):
    def __init__(self):
        super(Book_crossing_Dataset, self).__init__()

        self.train_interaction, self.test_interaction, self.item_features, configs = load_data()

        self.user_num = configs['num_users']
        self.item_num = configs['num_items']

        self.item_list = list(range(self.item_num))
        self.user_list = list(range(self.user_num))

        self.item_feature_matrix = self.item_features.drop('item_id', axis=1).to_numpy()

        self._generate_set()
        self._count_degrees()
        self.create_sparse_adjacency()

        self.train_dataset = TrainDataset(self.train_interaction, self.item_list, self.train_user_set)
        self.test_dataset = TestDataset(self.test_user_set, self.item_num)

        # self.connected_components()

    def connected_components(self, component_idx=1):
        import scipy.sparse as sp

        def to_bipartite_adj():
            edge_index = torch.Tensor([self.train_interaction['user_id'].tolist(), self.train_interaction['item_id'].tolist()])
            num_users = self.user_num
            num_movies = self.item_num

            # Shift movie indices by the number of users to avoid collisions
            edge_index[1] += num_users

            # Create adjacency matrix (no edge weights)
            adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_users + num_movies)
            return adj, num_users, num_movies

        # Create bipartite adjacency matrix
        adj, num_users, num_movies = to_bipartite_adj()

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



if __name__ == '__main__':
    dataset = Book_crossing_Dataset()
