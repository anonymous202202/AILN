import torch

def get_recall(indices, targets):
    targets = targets.expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = hits.size(0)
    recall = float(n_hits)
    return recall


def get_mrr(indices, targets):
    targets = targets.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data
    return mrr.item()

def evaluate(indices, targets, k):
    _, indices1 = torch.topk(indices, k, -1)
    recall = get_recall(indices1, targets)
    mrr = get_mrr(indices1, targets)
    return recall, mrr
