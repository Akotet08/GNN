import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from rectorch.metrics import Metrics

def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list) / len(acc_list)


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion=None, args=None):
    model.eval()
    if args.method == 'nodeformer':
        out, _ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau)
    else:
        out = model(dataset)

    if dataset.name == 'book_crossing':
        # recommendation so metric is ndcg
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        train_idx = train_idx.to(out.device)
        valid_idx = valid_idx.to(out.device)
        test_idx = test_idx.to(out.device)

        usr_index = dataset.graph['usr_nodes_idx']
        item_index = dataset.graph['itm_nodes_idx']

        predicted_scores = {}
        for usr in usr_index:
            usr_emb = out[usr]
            for itm in item_index:
                item_emb = out[itm]
                pos_score = torch.sum(usr_emb * item_emb)

                # if edge exists in dataset.graph['graph'], append the score
                grph = dataset.graph['graph']

                if usr in predicted_scores:
                    predicted_scores[usr].append(pos_score.item())
                else:
                    predicted_scores[usr] = [pos_score.item()]

        true_label = {}
        for usr in usr_index:
            true_label[usr] = []
            for itm in item_index:
                item_node = list(grph.nodes())[itm]
                user_node = list(grph.nodes())[usr]
                if dataset.graph['graph'].has_edge(user_node, item_node):
                    true_label[usr].append(1)
                else:
                    true_label[usr].append(0)
        
        true_label = np.array([true_label[usr] for usr in usr_index])
        predicted_scores = np.array([predicted_scores[usr] for usr in usr_index])
           
        # ndcg_at_5 = np.mean(Metrics.ndcg_at_k(predicted_scores, true_label, k=5).tolist())
        ndcg_at_10 = np.mean(Metrics.ndcg_at_k(predicted_scores, true_label, k=10).tolist())
        ndcg_at_20 = np.mean(Metrics.ndcg_at_k(predicted_scores, true_label, k=50).tolist())

        # head and tail items
        degree_list = []
        for itemidx in item_index:
            item_node = list(grph.nodes())[itemidx]
            x = len(degree_list)
            degree_list.append((x, grph.degree(item_node)))
        
        degree_list = sorted(degree_list, key=lambda x: x[1], reverse=True)

        head_items = [x[0] for x in degree_list[:69]]
        tail_items = [x[0] for x in degree_list[-200:]]

        head_predicted_scores = predicted_scores[:, head_items]
        head_true_label = true_label[:, head_items]

        tail_predicted_scores = predicted_scores[:, tail_items]
        tail_true_label = true_label[:, tail_items]

        k = 10

        head_ndcg_at_k = Metrics.ndcg_at_k(head_predicted_scores, head_true_label, k=10) 
        head_ndcg_at_k[np.isnan(head_ndcg_at_k)] = 0
        head_ndcg_at_k = np.mean(head_ndcg_at_k.tolist())

        tail_ndcg_at_k = Metrics.ndcg_at_k(tail_predicted_scores, tail_true_label, k=10)
        tail_ndcg_at_k[np.isnan(tail_ndcg_at_k)] = 0
        tail_ndcg_at_k = np.mean(tail_ndcg_at_k.tolist())

        return 1, ndcg_at_10, head_ndcg_at_k, tail_ndcg_at_k, out
                
    else:
        train_acc = eval_func(
            dataset.label[split_idx['train']], out[split_idx['train']])
        valid_acc = eval_func(
            dataset.label[split_idx['valid']], out[split_idx['valid']])
        test_acc = eval_func(
            dataset.label[split_idx['test']], out[split_idx['test']])

    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))

    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out


@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, result=None):
    model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    adjs_, x = dataset.graph['adjs'], dataset.graph['node_feat']
    adjs = []
    adjs.append(adjs_[0])
    for k in range(args.rb_order - 1):
        adjs.append(adjs_[k + 1])
    out, _ = model(x, adjs)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out