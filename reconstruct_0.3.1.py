from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import sys
import os
import json
import time, datetime
import visdom

from losses import *

sys.path.append('./models/')
from dgcnn_net import *

sys.path.append('./utils/')
from baseline_utils import *
from provider import *

"""reconstruct meshes using trained networks"""

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default=" ", help='input data dir')
parser.add_argument('--cls', nargs="+", type=str, help='which category')
parser.add_argument('--augment', type=bool, default=False,  help='augmentation')
parser.add_argument('--num_points', type=int, default = 2000,  help='number of points')
parser.add_argument('--small', type=bool, default=False,  help='train with small dataset')

parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--outf', type=str, default = 'out1',  help='out folder')
parser.add_argument('--type', type=str, default='test', help='train or test meshes')
parser.add_argument('--nb_primitives', type=int, default=25, help='primitives')

parser.add_argument('--chamLoss_wt', type=bool, default=False, help='chamfer loss wt')
parser.add_argument('--l1Loss_wt', type=bool, default=False, help='l1 loss wt')

opt = parser.parse_args()
print (opt)
# ========================================================== #


# ===================CREATE DATASET================================= #
#Create train/test dataloader
traindataset = getDataset(root=opt.dataDir, train=True, data_augment=opt.augment, small=opt.small, category=opt.cls)

testdataset = getDataset(root=opt.dataDir, train=False, data_augment=opt.augment, small=opt.small, category=opt.cls)

print('Train Dataset:', len(traindataset))
print('Test Dataset:', len(testdataset)) 

if opt.type == 'test':
    dataset = testdataset
else:
    dataset = traindataset

# ========================================================== #

# ===================CREATE network================================= #
#create network
network = AE_AtlasNet(num_points = opt.num_points, bottleneck_size=1024, nb_primitives=opt.nb_primitives)

network_path = opt.model
network.load_state_dict(torch.load(network_path))
print('loaded a pretrained model %s' %(opt.model))

network.cuda() #put network on GPU
print('network on cuda')

network.eval()
print('network on evaluation mode')


# # defining the 2D square grid
grain = np.ceil(np.sqrt(opt.num_points/opt.nb_primitives))-1
# grain = int(np.sqrt(opt.num_points/opt.nb_primitives))-1
grain = grain*1.0
# print(grain)

faces = []
vertices = []

for i in range(0,int(grain + 1 )):
        for j in range(0,int(grain + 1 )):
            vertices.append([i/grain,j/grain])

for prim in range(0,opt.nb_primitives):

    for i in range(1,int(grain + 1)):
        for j in range(0,(int(grain + 1)-1)):
            faces.append([(grain+1)*(grain+1)*prim + j+(grain+1)*i, (grain+1)*(grain+1)*prim + j+(grain+1)*i + 1, (grain+1)*(grain+1)*prim + j+(grain+1)*(i-1)])
    
    for i in range(0,(int((grain+1))-1)):
        for j in range(1,int((grain+1))):
            faces.append([(grain+1)*(grain+1)*prim + j+(grain+1)*i, (grain+1)*(grain+1)*prim + j+(grain+1)*i - 1, (grain+1)*(grain+1)*prim + j+(grain+1)*(i+1)])

grid = [vertices for i in range(0,opt.nb_primitives)]

grid_pytorch = torch.Tensor(int(opt.nb_primitives*(grain+1)*(grain+1)),2)
for i in range(opt.nb_primitives):
    for j in range(int((grain+1)*(grain+1))):
        grid_pytorch[int(j + (grain+1)*(grain+1)*i),0] = vertices[j][0]
        grid_pytorch[int(j + (grain+1)*(grain+1)*i),1] = vertices[j][1]
# print(grid_pytorch)

faces = np.array(faces, dtype=int)

t_vertices = len(vertices)*opt.nb_primitives
print("grain", grain, '# vertices', t_vertices, '# faces', faces.shape[0])

# remove vertices and faces associated with those vertices
if (t_vertices-opt.num_points > 0):
    to_remove = t_vertices - opt.num_points
    print("remove %d vertices" %(to_remove))
    for jj in range(to_remove):
        index = opt.num_points + jj
        idx = np.argwhere(faces == index)
        idx = idx[:,0]
        faces = np.delete(faces, idx, axis=0)

loss = []

""" ------ getting all the testing/training meshes from one folder (sampling already done)
    Not to be used for cubes as no sampling is done for cube
"""
# if opt.type == 'test':
#     root = os.path.join('../test_meshes', str(opt.num_points))
#     files = os.listdir(root)
#     files.sort()
#     dataset = []
#     for f in files:
#         if f.endswith('xyz'):
#             dataset.append(os.path.join(root, f))
# else:
#     root = os.path.join('../train_meshes', str(opt.num_points))
#     files = os.listdir(root)
#     files.sort()
#     dataset = []
#     for f in files:
#         if f.endswith('xyz'):
#             dataset.append(os.path.join(root, f))
"""------------------------------------------"""

# with torch.no_grad():
for i, data in enumerate(dataset, 0):
 
    mesh_name = dataset.datapath[i]["path"]
    _, mesh_name = os.path.split(mesh_name)
    mesh_name = mesh_name.split('.')[0]
    points, Q, _, _, _ = data

    # _, mesh_name = os.path.split(data)
    # mesh_name = mesh_name.split('.')[0]
    # points, _ = load_xyz_data(data)
    # points = torch.from_numpy(points.astype(np.float32))

    points = points.unsqueeze(0)
    # points = points.transpose(2,1).contiguous()
    points = Variable(points.transpose(2,1).contiguous())
    points = points.cuda()
    Q = Variable(Q).cuda()
    Q = Q.unsqueeze(0)

    pointsReconstructed  = network.forward_inference(points, grid)

    pointsReconstructed = pointsReconstructed.transpose(2,1).contiguous()
    points = points.transpose(2,1).contiguous()

    pointsReconstructed = pointsReconstructed[:,:opt.num_points, :]

    # dist1, dist2 = distChamfer(points, pointsReconstructed) #loss function
    # chamLoss = (torch.mean(dist1)) + (torch.mean(dist2))
    chamLoss, corres, _ = chamferLoss(points, pointsReconstructed) #loss function

    # corres = Variable(corres.type(torch.cuda.LongTensor))
    # recon_vertices = torch.cat([torch.index_select(a, 0, ind).unsqueeze(0) for a, ind in zip(pointsReconstructed, corres)])
    # pointsReconstructed = recon_vertices[:,:,:3].contiguous()

    # quadLoss = quadratic_error_loss(Q, pointsReconstructed)

    # l1Loss = l1_loss(points, pointsReconstructed)

    # Loss function
    if opt.chamLoss_wt:
        loss.append(chamLoss.data[0])
        print('chamLoss ', chamLoss.data[0])
    else:
        loss.append(quadLoss.data[0])
        print('quadLoss ', quadLoss.data[0])
    
    # if opt.l1Loss_wt:
    #     loss.append(l1Loss.item())
    #     print('l1Loss ', l1Loss.item())

    # writing reconstructed mesh file
    points = points.squeeze(0)
    pointsReconstructed = pointsReconstructed.squeeze(0)
    
    save_xyz_data(os.path.join(opt.outf, mesh_name+'.xyz'), points.data.cpu())
    save_xyz_data(os.path.join(opt.outf, mesh_name+'_recon.xyz'), pointsReconstructed.data.cpu())
    # save_ply_data(os.path.join(opt.outf, mesh_name+'_recon.ply'), pointsReconstructed.data.cpu(), faces)
    save_obj_data(os.path.join(opt.outf, mesh_name+'_recon.obj'), pointsReconstructed.data.cpu(), faces)

print('Mean %d values is %f' %(len(loss), np.mean(loss)))
print('Median %d values is %f' %(len(loss), np.median(loss)))
print('Max %d values is %f' %(len(loss), np.max(loss)))


