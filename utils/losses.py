# Copyright (c) 2019 Nitin Agarwal (agarwal@uci.edu)


import numpy as np
import sys
import torch

sys.path.append('./utils')
from pc_utils import *


def chamferLoss(V1, V2, average=True):
    """ Chamfer loss between two 3D point sets
    Input: Vin = input vertices = batchSize x N1 x 3
           Vout = recon vertices = batchSize x N2 x 3
    Output: Loss: chamfer loss. (sum of losses from both pointsets)
            indices: indices corresponding to Vin which minimize chamfer distance
    """
    
    x,y = V1, V2
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)
  
    dis1, idx1 = P.min(2)
    dis2, idx2 = P.min(1)
    # dis2, idx2 = P.min(2)

    if average:
        # average across all points and batches
        Loss = (torch.mean(dis1) + torch.mean(dis2))
    else:
        # average across all points only 
        dis1 = torch.sum(dis1, 1)
        dis2 = torch.sum(dis2, 1)
        Loss = (torch.mean(dis1 + dis2))

    return Loss, idx1, idx2


def l1_loss(V1, V2, average=True):
    """ Standard L1 loss 
    Input : V1, V2 = Batchsize x N x Dimension
    average = True (mean) False(sum over all pts)
    Output : Loss = L1 loss
    """

    Loss = torch.abs(V1-V2)
    Loss = torch.sum(Loss, 2)  # sum error in the last dimension
    
    if average:
        Loss = torch.mean(Loss, 1)
    else:
        Loss = torch.sum(Loss, 1)  # sum the error in all points
        
    # computing the mean across batches
    return Loss.mean()


def surface_normal_loss(ver, adj, corres, normal):
    """ Computes the surface normal loss according to paper Pixel2Mesh (ECCV 2018) 
    Input: ver = BxNx3 (input mesh vertices)
           adj = BxNxK (input mesh adjacency matrix, K=max degree)
           corres = BxNx3 (reconstructed points corresponding to input vertices)
           normal = BxNx3 (input mesh vertex normals)
    """
    
    B = ver.size()[0]
    N = ver.size()[1]
    K = adj.size()[-1]

    # ver - B x (N+1) x 3
    x = torch.zeros(B, 1, 3).cuda()
    ver = torch.cat((x, ver), dim=1)

    # Adj -  B x N*K 
    adj = adj.view(B, -1).long()
    
    # compute the mask for K = 0
    mask = adj.gt(0).float()

    # edges - Bx N x K x 3 (check this should it be ver or corres)
    edges = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(ver, adj) ])
    edges = edges.view(B, N, -1, 3)

    corres = torch.unsqueeze(corres, 2)
    corres = corres.repeat(1,1,K,1)
    
    normal = torch.unsqueeze(normal, 2)
    normal = normal.repeat(1,1,K,1)

    # edges - Bx N x K x 3
    edges = corres - edges
    # normalize the edges
    mag = torch.sqrt(torch.sum(edges*edges,-1))
    mag = mag.view(B, N, K, 1)
    edges = edges/mag

    # inner product (BxNxK)
    loss = torch.matmul(edges.view(B,N,K,1,3), normal.view(B,N,K,3,1))
    loss = torch.squeeze(loss, -1)
    loss = torch.squeeze(loss, -1)

    # loss -  B x N*K 
    loss = loss.view(B, -1)
    loss = loss*mask       # make loss=0 for K=0

    loss = torch.abs(loss)
    loss = torch.sum(loss, -1)

    # computing the mean across batches
    return loss.mean()


def quadric_loss(Q, V, average=False):
    """ computing the quadric error loss
    error = V_t * Q * V

    Input: Q = BxNx16 (Quadric matrices of input mesh)
           V = BxNx3 (reconstructed vertices)
    Output: quad loss
    """
    
    B = V.size()[0]
    N = V.size()[1]

    # V = BxNx4
    x = torch.ones(B, N, 1).cuda()
    V = torch.cat((V, x), 2)

    # V = BxNx1x4
    V = torch.unsqueeze(V, 2)
    
    # V_trans = BxNx4x1
    V_trans = torch.transpose(V, 3, 2).contiguous()

    # BxNx4x4
    V_new = torch.matmul(V_trans, V)

    # BxNx16
    V_new = V_new.view(B, N, -1)

    Loss = (V_new*Q).sum(-1)
    if average:
        Loss = Loss.mean(-1)
    else:
        Loss = Loss.sum(-1)

    # computing the mean across batches
    return torch.mean(Loss)


def normal_loss(N1, N2):
    """ computing the cosine normal loss
    error = 1 - abs(dot(N1, N2)) 
    all normal vectors are normalized

    Input: N1 = BxNx3 (ground truth normals)
           N2 = BxNx3 
    Output: normal loss
    """

    loss = N1*N2
    loss = loss.sum(-1)
    loss = torch.abs(loss)
    loss = 1-loss
    loss = torch.squeeze(loss)

    return torch.max(loss), torch.mean(loss), torch.min(loss)



def where(cond, x_1, x_2):
    cond = cond.type(x_1.dtype)
    return (cond*x_1) + ((1-cond)*x_2)


def surfaceLoss(points, faces): 
    """
    Input - points (BxNx3)
            faces (BxNxKx9)

    Ouput - min distances for each query point (BxN)
    """

    batchSize = points.size()[0]
    N = points.size()[1]
    K = faces.size()[2]

    # points BxNxKx3
    points = torch.unsqueeze(points, 2)
    points = points.repeat(1, 1, K, 1)

    B = faces[:,:,:,0:3]
    E0 = faces[:,:,:,3:6] - B
    E1 = faces[:,:,:,6:9] - B

    D = B - points
    a = torch.sum(E0*E0, dim=-1) + 1e-12
    b = torch.sum(E0*E1, dim=-1) + 1e-12
    c = torch.sum(E1*E1, dim=-1) + 1e-12
    d = torch.sum(E0*D, dim=-1) + 1e-12
    e = torch.sum(E1*D, dim=-1) + 1e-12
    f = torch.sum(D*D, dim=-1) + 1e-12

    det = a*c - b*b
    s = b*e - c*d
    t = b*d - a*e

    #region 4
    dist41  = where(-d>=a, a+2*d+f, -d*d/a+f)
    dist422 = where(-e>=c, c+2*e+f,-e*e/c+f)
    dist42 = where(e >= 0, f, dist422)
    dist4 = where(d < 0, dist41, dist42)

    #region 3
    dist3 = where(e>=0,f, dist422)

    #region 5
    dist5 = where(d>=0,f, dist41)

    #region 0
    ss = s/(det+1e-12)
    tt = t/(det+1e-12)
    dist0 = ss*(a*ss+b*tt + 2*d)+tt*(b*ss+c*tt+2*e)+f

    #region 2
    temp0 = b+d
    temp1 = c+e
    numer = temp1 -temp0
    denom = a - 2*b +c
    ss = numer/(denom+1e-12)
    tt = 1 - ss
    dist212 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist21 = where(numer>=denom,a + 2*d +f,dist212)
    dist22 = where(temp1<=0,c+2*e+f, where(e>=0,f, -e*e/c+f))
    dist2 = where(temp1>temp0,dist21,dist22)

    #region 6
    temp0 = b + e
    temp1 = a + d
    numer = temp1 -temp0
    denom = a-2*b+c
    tt = numer/(denom+1e-12)
    ss = 1 -tt
    dist612 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist61 = where(numer>=denom,c+2*e+f,dist612)
    dist62 = where(temp1<=0,a+2*d+f, where(d>=0,f,-d*d/a+f))
    dist6 = where(temp1>temp0, dist61,dist62)

    #region 1
    numer = c+e-b-d
    denom = a -2*b + c
    ss = numer/(denom+1e-12)
    tt = 1 - ss
    dist122 = ss*(a*ss+b*tt+2*d)+tt*(b*ss+c*tt+2*e)+f
    dist12 = where(numer>denom, a + 2*d+f, dist122)
    dist1 = where(numer<=0, c+2*e+f,dist12)


    # dist BxNxK
    dista = where(s<0, where(t<0,dist4,dist3), where(t<0,dist5,dist0))
    distb = where(s<0, dist2, where(t<0,dist6,dist1))
    dist = where(s+t<=det, dista, distb)

    # dist BxN
    # finding min among the neighbours
    dist, _ = torch.min(dist, -1)

    # dist BxN
    # removing any -ve dist
    dist = torch.max(dist,torch.zeros_like(dist))

    dist = torch.sum(dist, -1)
    return torch.mean(dist)



if __name__ == "__main__":

    meshA = '../data/sample.obj'
    V, F = load_obj_data(meshA)
    adj = get_adjacency_matrix(V, F, K_max=271)
    Nin = compute_vertex_normals(V, F)
    Q = compute_Q_matrix(V, F)
    face_cords = get_face_coordinates(V, F, K_max=271)
    
    V = torch.from_numpy(V.astype(np.float32))
    adj = torch.from_numpy(adj.astype(np.float32))
    Nin = torch.from_numpy(Nin.astype(np.float32))
    Q = torch.from_numpy(Q.astype(np.float32))
    Q = Q.view(V.shape[0], -1)
    face_cords = torch.from_numpy(face_cords.astype(np.float32))
    
    V.requires_grad_()
    adj.requires_grad_()
    Nin.requires_grad_()
    Q.requires_grad_()
    face_cords.requires_grad_()
    
    V = V.to('cuda')
    adj = adj.to('cuda')
    Nin = Nin.to('cuda')
    Q = Q.to('cuda')
    face_cords = face_cords.to('cuda')
    
    V = torch.unsqueeze(V, 0)
    adj = torch.unsqueeze(adj, 0)
    Nin = torch.unsqueeze(Nin, 0)
    Q = Q.unsqueeze(0)
    face_cords = face_cords.unsqueeze(0)

    V_query = torch.rand(1, V.size()[1], 3, dtype=torch.float)
    V_query.requires_grad_()
    V_query = V_query.to('cuda')

    
    """testing various loss functions"""
    loss, _, _ = chamferLoss(V, V_query)
    loss.backward()
    print('Chamfer Loss %f' %(loss.item()))
   
    loss = surface_normal_loss(V, adj, V_query, Nin)
    loss.backward()
    print('Normal Loss %f' %(loss.item()))

    loss = quadric_loss(Q, V_query)
    loss.backward()
    print('Quadric Loss %f' %(loss.item()))

    loss = surfaceLoss(V_query, face_cords)
    loss.backward()
    print('Surface Loss %f' %(loss.item()))
   







