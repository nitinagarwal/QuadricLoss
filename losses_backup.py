import numpy as np
import sys
import torch
import argparse
import subprocess
import re

from torch.autograd import Variable

sys.path.append('./utils')
from baseline_utils import *
# from provider import *

sys.path.append('../AtlasNet/nndistance/')
from modules.nnd import *
distChamfer = NNDModule()

def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


# distChamfer = ext.chamferDist()
# def distChamfer(a,b):
#     x,y = a,b
#     bs, num_points, points_dim = x.size()
#     xx = torch.bmm(x, x.transpose(2,1))
#     yy = torch.bmm(y, y.transpose(2,1))
#     zz = torch.bmm(x, y.transpose(2,1))
#     diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
#     rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
#     ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
#     P = (rx.transpose(2,1) + ry - 2*zz)
#     return P.min(1)[0], P.min(2)[0]

def chamferLoss(Vin, Vout, average=True):
    """ Chamfer loss between two 3D point sets
    Input: Vin = input vertices = batchSize x N x 3
           Vout = recon vertices = batchSize x N x 3
    Output: Loss: chamfer loss. (sum of losses from both pointsets)
            indices: indices corresponding to Vin which minimize chamfer distance
    """

    dis1, dis2, idx1, idx2 = distChamfer(Vin, Vout)

    if average:
        # average across all points and batches
        Loss = (torch.mean(dis1) + torch.mean(dis2))
    else:
        dis1 = torch.sum(dis1, 1)
        dis2 = torch.sum(dis2, 1)
        Loss = (torch.mean(dis1 + dis2))

    return Loss, idx1, idx2


def l1_loss(V1, V2, average=True, dim=False):
    """ Standard L1 loss 
    Input : V1, V2 = Batchsize x N x Dimension
    average = True (mean) False(sum over all pts)
    dim = True (mean over dim) False (take sum)
    Output : Loss = L1 loss
    """

    Loss = torch.abs(V1-V2)
    
    if dim:
        Loss = torch.mean(Loss, 2)   # computing the mean in the last dimension 
    else:
        Loss = torch.sum(Loss, 2)  # sum error in the last dimension

    if average:
        return torch.mean(Loss)
    else:
        Loss = torch.sum(Loss, 1)  # sum the error in all points
        
    # computing the mean across batches
    return Loss.mean()

def surface_normal_loss(ver, adj, corres, normal):
    """ Computes the surface normal loss according to paper Pixel2Mesh (ECCV 2018) 
    Input: ver = BxNx3
           adj = BxNxK
           corres = BxNx3
           normal = BxNx3 (vertex normals)
    """
    B = ver.size()[0]
    N = ver.size()[1]
    K = adj.size()[-1]

    # V - B x (N+1) x 3
    x = Variable(torch.zeros(B, 1, 3)).cuda()
    ver = torch.cat((x, ver), dim=1)

    # Adj -  B x N*K 
    adj = adj.view(B, -1).long()
    # print(Adj)
    
    # compute the mask for K = 0
    mask = adj.gt(0).float()

    # ver - Bx N*K x 3
    edges = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(ver, adj) ])
    # ver - Bx N x K x 3
    edges = edges.view(B, N, -1, 3)
    # print(edges)

    corres = torch.unsqueeze(corres, 2)
    corres = corres.repeat(1,1,K,1)
    # print(corres)
    
    normal = torch.unsqueeze(normal, 2)
    normal = normal.repeat(1,1,K,1)

    # edges - Bx N x K x 3
    edges = corres - edges
    # normalize the edges
    mag = torch.sqrt(torch.sum(edges*edges,-1))
    mag = mag.view(B, N, K, 1)
    edges = edges/mag
    # print(edges)

    # inner product (BxNxK)
    loss = torch.matmul(edges.view(B,N,K,1,3), normal.view(B,N,K,3,1))
    loss = torch.squeeze(loss, -1)
    loss = torch.squeeze(loss, -1)
    # print(loss)

    # loss -  B x N*K 
    loss = loss.view(B, -1)
    loss = loss*mask       # make loss=0 for K=0

    loss = torch.abs(loss)
    loss = torch.sum(loss, -1)

    # computing the mean across batches
    return loss.mean()


def quadratic_error_loss(Q, V, average=False):
    """ computing the quadratic error loss
    error = V_t * Q * V

    Input: Q = BxNx16 
           V = BxNx3 
    Output: quad loss
    """
    B = V.size()[0]
    N = V.size()[1]

    # V = BxNx4
    x = Variable(torch.ones(B, N, 1)).cuda()
    V = torch.cat((V, x), 2)

    # V = BxNx1x4
    V = torch.unsqueeze(V, 2)
    
    # V_trans = BxNx4x1
    V_trans = torch.transpose(V, 3, 2).contiguous()

    # BxNx4x4
    V_new = torch.matmul(V_trans, V)

    # BxNx16
    V_new = V_new.view(B, N, -1)

    if average:
        Loss = torch.cat([torch.mean(torch.abs(torch.bmm(a.view(N,1,-1), i.view(N, -1, 1)))).unsqueeze(0) for a, i in zip(V_new, Q) ])
    else:
        Loss = torch.cat([torch.sum(torch.abs(torch.bmm(a.view(N,1,-1), i.view(N, -1, 1)))).unsqueeze(0) for a, i in zip(V_new, Q) ])

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
    loss = torch.sum(loss, -1)
    loss = torch.abs(loss)
    loss = 1-loss
    loss = torch.squeeze(loss)

    return torch.max(loss), torch.mean(loss), torch.min(loss)

def read_mesh(filename):
    
    if(filename.endswith('ply')):
        vertices, faces = load_ply_data(filename)

    elif(filename.endswith('obj')):
         vertices, faces = load_obj_data(filename)

    vertices = normalize_shape(vertices)
    return vertices, faces


def where(cond, x_1, x_2):
    cond = cond.float()
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

    # print(points.size(), faces.size())
    # points BxNxKx3
    points = torch.unsqueeze(points, 2)
    points = points.repeat(1, 1, K, 1)
    # print(points.size(), faces.size())

    B = faces[:,:,:,0:3]
    E0 = faces[:,:,:,3:6] - B
    E1 = faces[:,:,:,6:9] - B

    # print(type(B.data), type(points.data))
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
    # print(s.size(), t.size())

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

    """testing surface_normal_loss"""
    # meshA = '/home/minions/Dropbox/GraphicsLab/Projects/3D_Content_Creation/data/all_data/abc_2.5k/0030000_partstudio_00_model_ste_00_split_0.obj'

    # V, F = load_obj_data(meshA)
    # adj = get_adjacency_matrix(V, F, 50)
    # Nin = compute_vertex_normals(V, F)

    # V = torch.from_numpy(V.astype(np.float32))
    # adj = torch.from_numpy(adj.astype(np.float32))
    # Nin = torch.from_numpy(Nin.astype(np.float32))
    
    # V = Variable(V, requires_grad=True).unsqueeze(0)
    # adj = Variable(adj, requires_grad=True).unsqueeze(0)
    # Nin = Variable(Nin, requires_grad=True).unsqueeze(0)
    # corres = Variable(torch.rand(1,2500,3), requires_grad=True)
    
    # V = V.cuda()
    # adj = adj.cuda()
    # Nin = Nin.cuda()
    # corres = corres.cuda()
    # print(V.size(), adj.size(), Nin.size(), corres.size())

    # loss = surface_normal_loss(V, adj, corres, Nin)
    # loss.backward()
    # print(loss)


    """testing surfaceLoss"""
    # meshA = '/home/minions/Dropbox/Officelinux/00_dump/temp/chair_0005.ply'
    # V, F = load_ply_data(meshA)
    # face_cords = get_face_coordinates(V, F)
    # # print(face_cords[0,:,:])
 
    # V = torch.from_numpy(V.astype(np.float))
    # face_cords = torch.from_numpy(face_cords.astype(np.float))
    # query = torch.from_numpy(np.array([0.25, 0.5, 0.5]).astype(np.float))
    # # query = torch.Tensor([0.25,0.5,0.5])
    # query = query.unsqueeze(0)
    # query = query.repeat(V.size()[0], 1)

    # V = Variable(query, requires_grad=True).unsqueeze(0)
    # face_cords = Variable(face_cords, requires_grad=True).unsqueeze(0)
    # # query = Variable(query, requires_grad=True)

    # print(V.size(), face_cords.size(), type(face_cords))
    # loss = surfaceLoss(V, face_cords)
    # # loss.backward()
    # print(loss.data[0])
    # # print(loss.data[0,1000,:])


    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default = 'None',  help='folder for all mesh files')
    # parser.add_argument('--file', type=str, default = 'None',  help='list of mesh files')
    parser.add_argument('--cls', nargs="+", type=str, help='which category')
    parser.add_argument('--typ', type=str, default='test', help='train/test')

    parser.add_argument('--error', type=str, default = 'None',  help='quadric/normal/metro')
    opt = parser.parse_args()

    # catfile = os.path.join(opt.path, opt.file)

    # getting all the mesh file names
    mesh_files = [] 
    shape_cls = [] 
    root = '/home/minions/Dropbox/GraphicsLab/Projects/3D_Content_Creation/data/all_data'
    for shape_class in opt.cls:
        if opt.typ == 'train':
            FILE = os.path.join(root, shape_class, 'train_full.txt')
        else:
            FILE = os.path.join(root, shape_class, 'test_full.txt')

        with open(FILE) as f:
            for line in f:
                mesh_files.append(line.strip())
                shape_cls.append(shape_class)

    print('Total Models are %d' %(len(mesh_files)))
    metroLoss = []
    # quadricLoss = []
    normaLoss = []

    # with open(opt.file) as f:
    for line, cls in zip(mesh_files, shape_cls):
        line = line.strip()
        line = line.split('.')[0]+'.obj'
        original_mesh = os.path.join(root, cls, line)

        # original_mesh = os.path.join(opt.path, line)
        recon_mesh = os.path.join(opt.path, line.split('.')[0]+'_recon.obj')
        print(original_mesh, recon_mesh)

        if (original_mesh.endswith('ply') or original_mesh.endswith('obj')):
            V_in, F_in = read_mesh(original_mesh)
        else:
            NotImplementedError('%s is not obj or ply file' %(line))

        if (recon_mesh.endswith('ply') or recon_mesh.endswith('obj')):
            V_recon, F_recon = read_mesh(recon_mesh)
        else:
            NotImplementedError('%s is not obj or ply file' %(line.split('.')[0]+'_recon.ply'))

    # if opt.error == 'metro':
        
        metro_exec='/home/minions/Dropbox/GraphicsLab/Projects/3D_Content_Creation/code/scripts/metro/metro '

        command = metro_exec
        command += original_mesh
        command += ' ' + recon_mesh
        command += ' ' + '-L'

        # os.system(command)
        output = subprocess.check_output(command, shell=True)
        # print(output)
        pattern = 'Hausdorff distance: (.*)\n'

        m = re.search(pattern, output)
        m = m.group()
        m = m.split(' ')[2]
        print('metro: ', float(m))
        metroLoss.append(float(m))

    # # elif opt.error == 'quadric':

        Q = compute_Q_matrix(V_in, F_in)
        
        N_in = compute_vertex_normals(V_in, F_in)
        P, D, N_out = distance_pt2mesh(V_recon, F_recon, V_in)
        
        # Q = torch.from_numpy(Q.astype(np.float32))
        # Q = Q.view(V_in.shape[0], -1)
        
        # P = torch.from_numpy(P.astype(np.float32))

        # Q = torch.unsqueeze(Q, 0).cuda()
        # P = torch.unsqueeze(P, 0).cuda()

        # quadLoss = quadratic_error_loss(Q, P, average=False)
        # print('quad: ', quadLoss)
        # quadricLoss.append(quadLoss)
    
    # elif opt.error == 'normal':

        N_in = torch.from_numpy(N_in.astype(np.float32))
        N_out = torch.from_numpy(N_out.astype(np.float32))

        N_in = torch.unsqueeze(N_in, 0).cuda()
        N_out = torch.unsqueeze(N_out, 0).cuda()

        loss_max, loss_mean, loss_min = normal_loss(N_in, N_out)
        # print('normal: ', loss_max.item())
        # normaLoss.append(loss_max.item())
        print('normal: ', loss_mean)
        normaLoss.append(loss_mean)

    # else:
        # NotImplementedError('choose a valid error')

    print('metro loss-------')
    print('#items %d, Max Loss %f, Mean Loss %f, Median %f' %(len(metroLoss), np.max(metroLoss), np.mean(metroLoss), np.median(metroLoss) ))

    print('normal loss-------')
    print('#items %d, Max Loss %f, Mean Loss %f, Median %f' %(len(normaLoss), np.max(normaLoss), np.mean(normaLoss),np.median(normaLoss) ))

    print('quadric loss-------')
    print('#items %d, Max Loss %f, Mean Loss %f, Median %f' %(len(quadricLoss), np.max(quadricLoss), np.mean(quadricLoss),np.median(quadricLoss) ))










       



