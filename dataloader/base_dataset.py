import os
import json
import torch
import pandas as pd
from random import choice
from collections import defaultdict
from torch.utils.data import Dataset


class BaseDataset:
    def __init__(self, verbose=False):
        self.joint_adjacency_matrix_normal_spatial = None
        self.joint_adjacency_matrix = None
        self.interact_matrix = None
        self.verbose = verbose

        self.train_user_set = defaultdict(dict)
        self.train_item_set = defaultdict(dict)
        self.test_user_set = defaultdict(dict)
        self.test_item_set = defaultdict(dict)

        self.user_degrees = {}
        self.item_degrees = {}

    def _generate_set(self):
        for row in self.train_interaction.itertuples(index=False):
            user_id = row.user_id
            item_id = row.item_id
            rating = row.rating
            self.train_user_set[user_id][item_id] = rating
            self.train_item_set[item_id][user_id] = rating

        for row in self.test_interaction.itertuples(index=False):
            user_id = row.user_id
            item_id = row.item_id
            rating = row.rating
            self.test_user_set[user_id][item_id] = rating
            self.test_item_set[item_id][user_id] = rating

    def _count_degrees(self):
        for user in self.user_list:
            self.user_degrees[user] = len(list(self.train_user_set[user].keys()))

        for item in self.item_list:
            self.item_degrees[item] = len(list(self.train_item_set[item].keys()))

    def create_sparse_adjacency(self):
        index = [self.train_interaction['user_id'].tolist(), self.train_interaction['item_id'].tolist()]
        value = [1.0] * len(self.train_interaction)

        self.interact_matrix = torch.sparse_coo_tensor(index, value, (self.user_num, self.item_num))

        tmp_index = [self.train_interaction['user_id'].tolist(),
                     (self.train_interaction['item_id'] + self.user_num).tolist()]
        tmp_adj = torch.sparse_coo_tensor(tmp_index, value,
                                          (self.user_num + self.item_num, self.user_num + self.item_num))

        joint_adjacency_matrix = (tmp_adj + tmp_adj.t()).coalesce()

        row_indices = joint_adjacency_matrix.indices()[0]
        col_indices = joint_adjacency_matrix.indices()[1]
        joint_adjacency_matrix_value = joint_adjacency_matrix.values()

        degree = torch.sparse.sum(joint_adjacency_matrix, dim=1).to_dense()
        degree = torch.pow(degree, -0.5)
        degree[torch.isinf(degree)] = 0

        self.joint_adjacency_matrix = joint_adjacency_matrix

        joint_adjacency_matrix_normal_value = degree[row_indices] * joint_adjacency_matrix_value * degree[col_indices]
        self.joint_adjacency_matrix_normal_spatial = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices], dim=0), joint_adjacency_matrix_normal_value,
            (self.user_num + self.item_num, self.user_num + self.item_num)).coalesce()


class TrainDataset(Dataset):
    def __init__(self, interaction, item_list, train_user_set):
        super(TrainDataset, self).__init__()
        self.interaction = interaction
        self.item_list = item_list
        self.train_user_set = train_user_set

    def __len__(self):
        return len(self.interaction)

    def __getitem__(self, idx):
        entry = self.interaction.iloc[idx]
        user = entry.user_id
        pos_item = entry.item_id
        neg_item = choice(self.item_list)

        while neg_item in self.train_user_set[user]:
            neg_item = choice(self.item_list)

        return user, pos_item, neg_item


class TestDataset(Dataset):
    def __init__(self, test_user_set, item_num):
        super(TestDataset, self).__init__()
        self.user_set = test_user_set

        self.user_list = list(test_user_set.keys())
        self.item_num = item_num

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user_id = self.user_list[idx]

        item_list = torch.tensor(list(self.user_set[user_id].keys()), dtype=torch.long)
        tensor = torch.zeros(self.item_num, dtype=torch.float32).scatter(0, item_list, 1)

        return user_id, tensor