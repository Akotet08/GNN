import torch
import numpy as np
from torch import Tensor
import torch.linalg as TLA
from typing import Optional

'''
This metrics are based on 
https://github.com/makgyver/rectorch/blob/master/rectorch/metrics.py
'''


class Metrics:
    @staticmethod
    def ndcg_at_k(pred_scores, ground_truth, k):
        assert pred_scores.shape == ground_truth.shape, "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(pred_scores.shape[1], k)
        n_users = pred_scores.shape[0]

        # Get the indices of the top k predicted scores
        idx_topk_part = torch.topk(pred_scores, k, dim=1, largest=True, sorted=False).indices
        topk_part = pred_scores.gather(1, idx_topk_part)

        # Sort the top k scores and their corresponding indices
        idx_part = torch.argsort(topk_part, dim=1, descending=True)
        idx_topk = idx_topk_part.gather(1, idx_part)

        tp = 1. / torch.log2(torch.arange(2, k + 2, device=pred_scores.device, dtype=pred_scores.dtype))
        dcg = (ground_truth.gather(1, idx_topk) * tp).sum(dim=1)

        sorted_ground_truth, _ = torch.sort(ground_truth, dim=1, descending=True)
        idcg = torch.tensor([(tp[:min(int(n), k)]).sum() for n in sorted_ground_truth.sum(dim=1)],
                            device=pred_scores.device)

        ndcg = dcg / idcg
        return ndcg.cpu().numpy()

    @staticmethod
    def recall_at_k(pred_scores, ground_truth, k):
        assert pred_scores.shape == ground_truth.shape, "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(pred_scores.shape[1], k)

        # Get the indices of the top k predicted scores
        topk_indices = torch.topk(pred_scores, k, dim=1, largest=True, sorted=False).indices

        # Create a binary tensor of the same shape as pred_scores with True at the top k indices
        pred_scores_binary = torch.zeros_like(pred_scores, dtype=torch.bool)
        pred_scores_binary.scatter_(1, topk_indices, True)

        x_true_binary = (ground_truth > 0)

        # Compute the number of true positives
        num_true_positives = (x_true_binary & pred_scores_binary).sum(dim=1).float()

        # Compute the recall
        recall = num_true_positives / torch.min(torch.tensor(k, dtype=torch.float32, device=ground_truth.device),
                                                x_true_binary.sum(dim=1).float())

        return recall.cpu().numpy()

    @staticmethod
    def mrr_at_k(pred_scores, ground_truth, k):
        assert pred_scores.shape == ground_truth.shape, "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(pred_scores.shape[1], k)

        # Get the indices of the top k predicted scores
        _, indices = torch.sort(pred_scores, descending=True)

        # Get the top k indices
        topk_indices = indices[:, :k]

        # Gather the ground truth values at the top k indices
        hits = ground_truth.gather(1, topk_indices)

        # Get the ranks of the hits
        hits_nonzero = (hits > 0).nonzero(as_tuple=False)

        # Initialize MRR values
        mrr = torch.zeros(ground_truth.shape[0], device=pred_scores.device)

        # Compute reciprocal ranks
        if hits_nonzero.size(0) > 0:
            reranks = hits_nonzero[:, 0]
            cranks = hits_nonzero[:, 1]
            unique_reranks = reranks.unique()

            for r in unique_reranks:
                indices = (reranks == r).nonzero(as_tuple=False).view(-1)
                first_hit = cranks[indices].min().item()
                mrr[r] = 1. / (1 + first_hit)

        return mrr.cpu().numpy()


def build_adj_dict(num_nodes: int, edge_index: Tensor) -> dict[int, list[int]]:
    # Initialize adjacency dict with empty neighborhoods for all nodes
    adj_dict: dict[int, list[int]] = {node_id: [] for node_id in range(num_nodes)}

    # Extract the source and destination indices from the sparse tensor
    source_indices = edge_index.indices()[0].long().tolist()
    destination_indices = edge_index.indices()[1].long().tolist()

    # Iterate through all edges and add head nodes to adjacency list of tail nodes
    for ctail, c_head in zip(source_indices, destination_indices):
        if c_head not in adj_dict[ctail]:
            adj_dict[ctail].append(c_head)

    return adj_dict


@torch.no_grad
def dirichlet_energy(feat_matrix: Tensor, edge_index: Optional[Tensor] = None, adj_dict: Optional[dict] = None,
                     p: Optional[int | float] = 2, ) -> float:
    if (edge_index is None) and (adj_dict is None):
        raise ValueError("Neither 'edge_index' nor 'adj_dict' was provided for Dirichlet energy computation")
    if (edge_index is not None) and (adj_dict is not None):
        raise ValueError(
            "Both 'edge_index' and 'adj_dict' were provided. Only one should be passed."
        )

    num_nodes: int = feat_matrix.shape[0]
    de: Tensor = 0

    if adj_dict is None:
        adj_dict = build_adj_dict(num_nodes=num_nodes, edge_index=edge_index)

    def inner(x_i: Tensor, x_js: Tensor) -> Tensor:
        return TLA.vector_norm(x_i - x_js, ord=p, dim=1).square().sum()

    for node_index in range(num_nodes):
        own_feat_vector = feat_matrix[[node_index], :]
        nbh_feat_matrix = feat_matrix[adj_dict[node_index], :]

        de += inner(own_feat_vector, nbh_feat_matrix)

    return torch.sqrt(de / num_nodes).item()


@torch.no_grad
def mean_average_distance(feat_matrix: Tensor, edge_index: Optional[Tensor] = None,
                          adj_dict: Optional[dict] = None, ) -> float:
    if (edge_index is None) and (adj_dict is None):
        raise ValueError("Neither 'edge_index' nor 'adj_dict' was provided for MAD calculation")
    if (edge_index is not None) and (adj_dict is not None):
        raise ValueError(
            "Both 'edge_index' and 'adj_dict' were provided. Only one should be passed."
        )

    num_nodes: int = feat_matrix.shape[0]
    mad: Tensor = 0

    if adj_dict is None:
        adj_dict = build_adj_dict(num_nodes=num_nodes, edge_index=edge_index)

    def inner(x_i: Tensor, x_js: Tensor) -> Tensor:
        return (
                1
                - (x_i @ x_js.t())
                / (TLA.vector_norm(x_i, ord=2) * TLA.vector_norm(x_js, ord=2, dim=1))
        ).sum()

    for node_index in range(num_nodes):
        own_feat_vector = feat_matrix[[node_index], :]
        nbh_feat_matrix = feat_matrix[adj_dict[node_index], :]

        mad += inner(own_feat_vector, nbh_feat_matrix)

    return (mad / num_nodes).item()


def ranking_measure_test_set(pred_scores, ground_truth, k, test_item):
    pred_scores = pred_scores[:, test_item]
    ground_truth = ground_truth[:, test_item]

    # user_num
    ndcg_list = Metrics.ndcg_at_k(pred_scores, ground_truth, k).tolist()
    recall_list = Metrics.recall_at_k(pred_scores, ground_truth, k).tolist()
    mrr_list = Metrics.mrr_at_k(pred_scores, ground_truth, k).tolist()

    return np.mean(ndcg_list), np.mean(recall_list), np.mean(mrr_list)


def ranking_measure_degree_test_set(pred_scores, ground_truth, k, item_degrees, separate_rate, test_item):
    sorted_item_degrees = sorted(item_degrees.items(), key=lambda x: x[1])
    item_list_sorted, _ = zip(*sorted_item_degrees)
    body_length = int(len(item_list_sorted) * (1 - separate_rate))
    tail_length = int(len(item_list_sorted) * separate_rate)
    head_length = int(len(item_list_sorted) * separate_rate)

    head_item = list(set(item_list_sorted[-head_length:]).intersection(set(test_item)))
    tail_item = list(set(item_list_sorted[:tail_length]).intersection(set(test_item)))
    body_item = list(set(item_list_sorted[:body_length]).intersection(set(test_item)))

    head_ndcg_list = np.nan_to_num(Metrics.ndcg_at_k(pred_scores[:, head_item], ground_truth[:, head_item], k)).tolist()
    head_recall_list = np.nan_to_num(
        Metrics.recall_at_k(pred_scores[:, head_item], ground_truth[:, head_item], k)).tolist()

    tail_ndcg_list = np.nan_to_num(Metrics.ndcg_at_k(pred_scores[:, tail_item], ground_truth[:, tail_item], k)).tolist()
    tail_recall_list = np.nan_to_num(
        Metrics.recall_at_k(pred_scores[:, tail_item], ground_truth[:, tail_item], k)).tolist()

    body_ndcg_list = np.nan_to_num(Metrics.ndcg_at_k(pred_scores[:, body_item], ground_truth[:, body_item], k)).tolist()
    body_recall_list = np.nan_to_num(
        Metrics.recall_at_k(pred_scores[:, body_item], ground_truth[:, body_item], k)).tolist()

    return np.mean(head_ndcg_list), np.mean(head_recall_list), np.mean(tail_ndcg_list), np.mean(
        tail_recall_list), np.mean(body_ndcg_list), np.mean(body_recall_list)


def compare_head_tail_rec_percentage(pred_scores, test_item, item_degrees, separate_rate, k):
    sorted_items = sorted(item_degrees.items(), key=lambda x: x[1])
    item_list_sorted, _ = zip(*sorted_items)
    body_length = int(len(item_list_sorted) * (1 - separate_rate))
    head_length = int(len(item_list_sorted) * separate_rate)
    tail_length = int(len(item_list_sorted) * separate_rate)

    tail_item_indices = list(set(item_list_sorted[:tail_length]).intersection(set(test_item)))
    head_item_indices = list(set(item_list_sorted[-head_length:]).intersection(set(test_item)))
    body_item_indices = list(set(item_list_sorted[:body_length]).intersection(set(test_item)))

    tail_item_indices_tensor = torch.tensor(tail_item_indices, device=pred_scores.device)
    head_item_indices_tensor = torch.tensor(head_item_indices, device=pred_scores.device)
    body_item_indices_tensor = torch.tensor(body_item_indices, device=pred_scores.device)

    idx_topk_part = torch.topk(pred_scores, k, dim=1, largest=True, sorted=False).indices

    tail_matches = idx_topk_part.unsqueeze(2) == tail_item_indices_tensor.unsqueeze(0).unsqueeze(1)
    head_matches = idx_topk_part.unsqueeze(2) == head_item_indices_tensor.unsqueeze(0).unsqueeze(1)
    body_matches = idx_topk_part.unsqueeze(2) == body_item_indices_tensor.unsqueeze(0).unsqueeze(1)

    tail_overlap_counts = tail_matches.any(dim=2).sum(dim=1)
    head_overlap_counts = head_matches.any(dim=2).sum(dim=1)
    body_overlap_counts = body_matches.any(dim=2).sum(dim=1)

    tail_percentage = (tail_overlap_counts / k).mean().cpu().numpy()
    head_percentage = (head_overlap_counts / k).mean().cpu().numpy()
    body_percentage = (body_overlap_counts / k).mean().cpu().numpy()

    return head_percentage, tail_percentage, body_percentage



