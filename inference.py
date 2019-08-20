# Copyright (c) 2019 Nitin Agarwal (agarwal@uci.edu)


from __future__ import print_function
import sys
import os
import json
import time, datetime
import visdom
import argparse
import random
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler


sys.path.append('./models/')
from dgcnn_net import DG_AtlasNet

sys.path.append('./utils/')
from pc_utils import *
from provider import *
from losses import *

"""reconstruct MESHES using trained networks"""

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default=" ", help='input data dir')
parser.add_argument('--cls', nargs="+", type=str, help='which category')
parser.add_argument('--augment', type=bool, default=False,  help='augmentation')
parser.add_argument('--small', type=bool, default=False,  help='train with small dataset')

parser.add_argument('--model', type=str, default = '',  help='load pretrained model')
parser.add_argument('--outf', type=str, default = 'out1',  help='out folder')
parser.add_argument('--type', type=str, default='test', help='train or test meshes')
parser.add_argument('--nb_primitives', type=int, default=25, help='primitives')
parser.add_argument('--num_points', type=int, default=2500,  help='# points in reconstructed mesh')
parser.add_argument('--bottleneck_size', type=int, default=1024, help='embedding size')

parser.add_argument('--chamLoss_wt', type=bool, default=False, help='chamfer loss wt')
parser.add_argument('--quadLoss_wt', type=bool, default=False, help='quadric loss wt')

opt = parser.parse_args()
print (opt)

if not os.path.exists(opt.outf):
    os.mkdir(opt.outf)


# ===================CREATE DATASET================================= #
traindataset = getDataset(root=opt.dataDir, train=True, data_augment=opt.augment, small=opt.small, category=opt.cls)
testdataset = getDataset(root=opt.dataDir, train=False, data_augment=opt.augment, small=opt.small, category=opt.cls)

print('Train Dataset:', len(traindataset))
print('Test Dataset:', len(testdataset)) 


if opt.type == 'test':
    dataset = testdataset
else:
    dataset = traindataset

# ===================CREATE network================================= #
network = DG_AtlasNet(num_points = opt.num_points, bottleneck_size=opt.bottleneck_size, nb_primitives=opt.nb_primitives)

network.load_state_dict(torch.load(opt.model))
print('loaded a pretrained model %s' %(opt.model))

network.cuda() 
print('network on cuda')

network.eval()
print('network on evaluation mode')

# borrowed from AtlasNet(https://github.com/ThibaultGROUEIX/AtlasNet/blob/master/inference/run_AE_AtlasNet.py)
# defining the 2D square grid
grain = np.ceil(np.sqrt(opt.num_points/opt.nb_primitives))-1
grain = grain*1.0


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


with torch.no_grad():
    for i, data in enumerate(dataset, 0):
     
        mesh_name = dataset.datapath[i]["path"]
        _, mesh_name = os.path.split(mesh_name)
        mesh_name = mesh_name.split('.')[0]
        points, Q, _, _, _ = data

        points = points.unsqueeze(0)
        points = points.transpose(2,1)

        points = points.cuda()
        Q = Q.cuda()
        Q = Q.unsqueeze(0)

        recon_points  = network.forward_inference(points, grid)

        recon_points = recon_points.transpose(2,1)
        points = points.transpose(2,1)

        chamLoss, corres, _ = chamferLoss(points, recon_points) 

        corres = corres.type(torch.cuda.LongTensor)
        recon_vertices = torch.cat([torch.index_select(a, 0, ind).unsqueeze(0) for a, ind in zip(recon_points, corres)])
        recon_points = recon_vertices

        quadLoss = quadric_loss(Q, recon_points)

        # Loss function
        if opt.chamLoss_wt:
            loss.append(chamLoss.item())
            print('chamLoss ', chamLoss.item())
        else:
            loss.append(quadLoss.item())
            print('quadLoss ', quadLoss.item())
        

        # writing reconstructed mesh file
        points = points.squeeze(0)
        recon_points = recon_points.squeeze(0)
        
        save_xyz_data(os.path.join(opt.outf, mesh_name+'.xyz'), points.data.cpu())
        save_xyz_data(os.path.join(opt.outf, mesh_name+'_recon.xyz'), recon_points.data.cpu())
        save_obj_data(os.path.join(opt.outf, mesh_name+'_recon.obj'), recon_points.data.cpu(), faces)

    print('Mean %d values is %f' %(len(loss), np.mean(loss)))
    print('Median %d values is %f' %(len(loss), np.median(loss)))
    print('Max %d values is %f' %(len(loss), np.max(loss)))





