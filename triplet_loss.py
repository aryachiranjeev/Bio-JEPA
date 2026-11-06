import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist

def hard_example_mining_used(dist_mat, labels, return_inds=False): ##chiranjeev - updated
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # 1. Create Identity Masks
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # 2. Find hardest positive and negative
    dist_ap, dist_an = [], []
    p_inds, n_inds = [], []
    for i in range(N):
        # Hardest Positive: Max distance where is_pos is True
        pos_indices = torch.where(is_pos[i])[0]
        pos_dists = dist_mat[i][pos_indices]
        # print("i:", i, "dist_mat[i]:",dist_mat[i], "is_pos[i]:",is_pos[i], "pos_indices:",pos_indices, "pos_dists:",pos_dists)
        max_p_dist, relative_p_idx = torch.max(pos_dists, 0)
        dist_ap.append(max_p_dist.unsqueeze(0))
        p_inds.append(pos_indices[relative_p_idx].unsqueeze(0))

        # Hardest Negative: Min distance where is_neg is True
        neg_indices = torch.where(is_neg[i])[0]
        neg_dists = dist_mat[i][neg_indices]
        # print("i:", i, "dist_mat[i]:",dist_mat[i], "is_neg[i]:",is_neg[i], "neg_indices:",neg_indices, "neg_dists:",neg_dists)
        min_n_dist, relative_n_idx = torch.min(neg_dists, 0)
        dist_an.append(min_n_dist.unsqueeze(0))
        n_inds.append(neg_indices[relative_n_idx].unsqueeze(0))

    # Stack results back into tensors
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    
    # print("dist_ap:",dist_ap.shape, "dist_an:",dist_an.shape)
    # print("return_inds:",return_inds)
    
    if return_inds:
        p_inds = torch.cat(p_inds)
        n_inds = torch.cat(n_inds)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def hard_example_mining_old(dist_mat, labels, return_inds=False): ##chiranjeev - updated
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # 1. Create Identity Masks
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # 2. Find hardest positive and negative
    dist_ap, dist_an = [], []
    p_inds, n_inds = [], []
    for i in range(N):
        # Hardest Positive: Max distance where is_pos is True
        pos_indices = torch.where(is_pos[i])[0]
        pos_dists = dist_mat[i][pos_indices]
        # print("i:", i, "dist_mat[i]:",dist_mat[i], "is_pos[i]:",is_pos[i], "pos_indices:",pos_indices, "pos_dists:",pos_dists)
        max_p_dist, relative_p_idx = torch.max(pos_dists, 0)
        dist_ap.append(max_p_dist.unsqueeze(0))
        p_inds.append(pos_indices[relative_p_idx].unsqueeze(0))

        # Hardest Negative: Min distance where is_neg is True
        neg_indices = torch.where(is_neg[i])[0]
        neg_dists = dist_mat[i][neg_indices]
        # print("i:", i, "dist_mat[i]:",dist_mat[i], "is_neg[i]:",is_neg[i], "neg_indices:",neg_indices, "neg_dists:",neg_dists)
        min_n_dist, relative_n_idx = torch.min(neg_dists, 0)
        dist_an.append(min_n_dist.unsqueeze(0))
        n_inds.append(neg_indices[relative_n_idx].unsqueeze(0))

    # Stack results back into tensors
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)
    
    # print("dist_ap:",dist_ap.shape, "dist_an:",dist_an.shape)
    # print("return_inds:",return_inds)
    
    if return_inds:
        p_inds = torch.cat(p_inds)
        n_inds = torch.cat(n_inds)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an
  
def hard_example_mining_7_july(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # Use masked fill instead of view(N,-1) to handle unequal positives per identity
    dist_ap = (dist_mat * is_pos.float() - (~is_pos).float() * 1e9).max(dim=1, keepdim=True)[0]
    relative_p_inds = (dist_mat * is_pos.float() - (~is_pos).float() * 1e9).max(dim=1, keepdim=True)[1]
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an = (dist_mat * is_neg.float() + (~is_neg).float() * 1e9).min(dim=1, keepdim=True)[0]
    relative_n_inds = (dist_mat * is_neg.float() + (~is_neg).float() * 1e9).min(dim=1, keepdim=True)[1]
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an
