import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp
from utils import rand_train_test_idx, class_rand_splits

class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        split_type: 'random' for random splitting, 'class' for splitting with equal node num per class
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        label_num_per_class: num of nodes per class
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_recommendation_dataset(dataset_name, social_data= False, test_dataset= True, bottom=0, cv =None, split=None, user_fre_threshold = None, item_fre_threshold = None):
    save_dir = "dataset/" + dataset_name
    if not os.path.exists(save_dir):
        print("dataset is not exist!!!!")
        return None

    if os.path.exists(save_dir + '/encoded_user_feature.pkl'):
        user_feature = pd.read_pickle(save_dir + '/encoded_user_feature.pkl')
    else:
        user_feature = None


    if os.path.exists(save_dir + '/encoded_item_feature.pkl'):
        item_feature = pd.read_pickle(save_dir + '/encoded_item_feature.pkl')
    else:
        item_feature = None

    social = None

    if test_dataset == True:
        interact_train = pd.read_pickle(save_dir + '/interact_train.pkl')
        interact_test = pd.read_pickle(save_dir + '/interact_test.pkl')
        if social_data == True:
            social = pd.read_pickle(save_dir + '/social.pkl')
        item_encoder_map = pd.read_csv(save_dir + '/item_encoder_map.csv')
        item_num = len(item_encoder_map)
        user_encoder_map = pd.read_csv(save_dir + '/user_encoder_map.csv')
        user_num = len(user_encoder_map)

        if bottom != None:
            interact_train = interact_train[interact_train['score'] > bottom]
            interact_test = interact_test[interact_test['score'] > bottom]
        
        if user_feature is not None:
            user_feature = torch.tensor(user_feature.values, dtype=torch.float64)
        else:
            user_num = None
        
        if item_feature is not None:
            # endcode item feature
            item_feature.drop(['item', 'encoded'], axis=1, inplace=True)
            for col in item_feature.columns:
                item_feature[col] = LabelEncoder().fit_transform(item_feature[col])
            
            item_feature = torch.tensor(item_feature.values, dtype=torch.float64)
        else:
            item_feature = None
        
        # select edges randomly for training 400 edges
        np.random.seed(0)
        idx = np.random.choice(len(interact_train), 400, replace=False)
        interact_train = interact_train.iloc[idx]
        print(f'interact_train shape: {len(interact_train)}')
        
        G = nx.Graph(bipartite=True)
        # G.add_nodes_from(interact_train['userid'], label='user', bipartite=0, feature=user_feature)
        for index, row in interact_train.iterrows():
            G.add_node(f'usr_{row["userid"]}', label='user', bipartite=0, feature=[1,1,1,1])
            G.add_node(f'itm_{row["itemid"]}', label='item', bipartite=1, feature=item_feature[int(row['itemid'])])
            G.add_edge(f'usr_{row["userid"]}', f'itm_{row["itemid"]}', score=row['score'])

        # get tail and head items. tail-items: items with degree < 20% of the max degree
        # head-items: items with degree > 80% of the max degree 

        # for index, row in interact_test.iterrows():
        #     G.add_node(f'usr_{row["userid"]}', label='user', bipartite=0, feature=[1,1,1,1])
        #     G.add_node(f'itm_{row["itemid"]}', label='item', bipartite=1, feature=item_feature[int(row['itemid'])])
        #     G.add_edge(f'usr_{row["userid"]}', f'itm_{row["itemid"]}', score=row['score'])
        
        adjacency_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()))
        adjacency_matrix = sp.coo_matrix(adjacency_matrix)
        adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])
        adjacency_matrix = adjacency_matrix.tocoo().astype(np.float32)

        edge_index = torch.from_numpy( np.vstack((adjacency_matrix.row, adjacency_matrix.col)).astype(np.int64))
        edge_attr = None
        
        node_feat = torch.tensor([G.nodes[node]['feature'] for node in G.nodes()], dtype=torch.float64)
        
        # get usr and item indexes in G
        usr_nodes = [idx for idx, node in enumerate(G.nodes()) if G.nodes[node]['label'] == 'user']
        itm_nodes = [idx for idx, node in enumerate(G.nodes()) if G.nodes[node]['label'] == 'item']

        # negative samples per user. it is a negative sample if the user has not interacted with the item
        neg_sample_indexes = {}
        neg_sample = 10
        for user in usr_nodes:
            interacted_items = [edge[1] for edge in G.edges() if edge[0] == user]
            for _ in range(neg_sample):
                item = np.random.choice(itm_nodes)
                while item in interacted_items:
                    item = np.random.choice(itm_nodes)
                if user not in neg_sample_indexes:
                    neg_sample_indexes[user] = [item]
                else:
                    neg_sample_indexes[user].append(item)

        dataset = NCDataset(dataset_name)
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': edge_attr,
                         'node_feat': node_feat.to(torch.float32),
                         'num_nodes': len(G.nodes()),
                        'user_num': user_num,
                         'item_num': item_num, 
                         'usr_nodes_idx': usr_nodes,
                         'itm_nodes_idx': itm_nodes,
                         'graph': G,
                         'neg_sample_indexes': neg_sample_indexes}
        
        edge_label = torch.tensor([G.edges[edge]['score'] for edge in G.edges()], dtype=torch.float64)
        dataset.label = edge_label

        return dataset

# Test load_recommendation_dataset
if __name__ == '__main__':
    dataset_name = 'book_crossing'
    social_data = False
    test_dataset = True
    bottom = 0
    cv = None
    split = None
    user_fre_threshold = None
    item_fre_threshold = None
    dataset = load_recommendation_dataset(dataset_name, social_data, test_dataset, bottom, cv, split, user_fre_threshold, item_fre_threshold)