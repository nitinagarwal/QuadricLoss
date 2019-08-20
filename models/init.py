# Copyright (c) 2019 Nitin Agarwal (agarwal@uci.edu)


# borrowed from DGCNN (https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py)
from __future__ import print_function
import torch

def pairwise_distance(point_cloud):
    """
    Input: point_cloud: tensor (B, N, D)
    Output: pairwise distance: (B, N, N)
    """

    batch_size = point_cloud.size()[0]
    point_cloud = torch.squeeze(point_cloud)
    if batch_size==1:
        point_cloud = point_cloud.unsqueeze(0)
    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    point_cloud_inner = -2*torch.bmm(point_cloud, point_cloud_transpose)
    point_cloud_square = (point_cloud**2).sum(dim=-1, keepdim=True)
    point_cloud_square_transpose = point_cloud_square.permute(0, 2, 1)
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose


def knn(dist_mat, k=20):
    """
    Input: pairwise distance: (B, N, N)
           k: int
    Output: nearest neighbors: (B, N, k)
    """

    _, nn_idx = torch.topk(dist_mat, k=k, largest=False, sorted=False)
    return nn_idx


def get_edge_feature(point_cloud, nn_idx, k=20, is_cuda=True):
    """
    Input: point_cloud (B, N, D)
           pairwise distance: (B, N, N)
           k: int
    Output: edge features: (B, N, k, 2*D)
    """

    batch_size = point_cloud.size()[0]
    point_cloud = torch.squeeze(point_cloud)

    if batch_size==1:
        point_cloud = point_cloud.unsqueeze(0)

    _,num_points,num_dims = point_cloud.size()

    idx_ = torch.arange(batch_size) * num_points
    idx_ = idx_.view(batch_size, 1, 1).long()

    if is_cuda:
        idx_ = idx_.cuda()

    # BxNxkxD (get all neighbours)
    point_cloud_flat = point_cloud.contiguous().view(-1, num_dims)
    point_cloud_nbrs = torch.index_select(point_cloud_flat, dim=0, index=(nn_idx+idx_).view(-1, 1).squeeze())
    point_cloud_nbrs = point_cloud_nbrs.view(batch_size,num_points,k,-1)
    # print(point_cloud_nbrs.size())

    point_cloud_central = point_cloud.unsqueeze(-2)
    point_cloud_central = point_cloud_central.expand(-1,-1,k,-1)

    edge_feature = torch.cat((point_cloud_central, point_cloud_nbrs-point_cloud_central), dim=-1)
    return edge_feature


if __name__ == "__main__":

    B, N, D = 2, 100, 3
    points = torch.rand(B,N,D)
    points = points.cuda()
    
    out = pairwise_distance(points)
    out = knn(out)
    out = get_edge_feature( points, out )
    print(points.size(), out.size())


