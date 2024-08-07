import os
import json
import pandas as pd
from dataloader.base_dataset import BaseDataset, TrainDataset, TestDataset

SAMPLING_RATIO = 1.0

def load_data():
    path_config = 'data_preprocessor/dataset_path.json'
    with open(path_config, 'r') as f:
        paths = json.load(f)

    path_to_dataset = paths['movielense']['processed']
    train_interaction_filename = f'movielense_train_{SAMPLING_RATIO}.csv'
    test_interaction_filename = f'movielense_test_{SAMPLING_RATIO}.csv'
    features_filename = 'movielense_item_feature_df.csv'

    train_interaction_data_path = os.path.join(path_to_dataset, train_interaction_filename)
    test_interaction_data_path = os.path.join(path_to_dataset, test_interaction_filename)
    feature_data_path = os.path.join(path_to_dataset, features_filename)

    train_interaction_df = pd.read_csv(train_interaction_data_path)
    test_interaction_df = pd.read_csv(test_interaction_data_path)

    item_features_df = pd.read_csv(feature_data_path)

    config_path = os.path.join(path_to_dataset, 'movielense_config.json')
    with open(config_path, 'r') as f:
        configs = json.load(f)

    print('loaded dataset and configs...')
    return train_interaction_df, test_interaction_df, item_features_df, configs


class MovieLense_Dataset(BaseDataset):
    def __init__(self):
        super(MovieLense_Dataset, self).__init__()

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


if __name__ == '__main__':
    dataset = MovieLense_Dataset()