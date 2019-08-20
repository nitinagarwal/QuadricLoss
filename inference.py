# Copyright (c) Nitin Agarwal (agarwal@uci.edu)
# Last Modified:      Tue 20 Aug 2019 01:38:47 PM PDT

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
from dgcnn_net import *

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
parser.add_argument('--num_points', type=int, default = 2000,  help='# points in reconstructed mesh')

parser.add_argument('--chamLoss_wt', type=bool, default=False, help='chamfer loss wt')
parser.add_argument('--quadLoss_wt', type=bool, default=False, help='quadric loss wt')

opt = parser.parse_args()
print (opt)


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
network = AE_AtlasNet(num_points = opt.num_points, bottleneck_size=opt.bottleneck_size, nb_primitives=opt.nb_primitives)

network_path = opt.model
network.load_state_dict(torch.load(network_path))
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

        # recon_points = recon_points[:,:opt.num_points, :]

        chamLoss, corres, _ = chamferLoss(points, pointsReconstructed) #loss function

        corres = corres.type(torch.cuda.LongTensor)
        recon_vertices = torch.cat([torch.index_select(a, 0, ind).unsqueeze(0) for a, ind in zip(pointsReconstructed, corres)])
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






#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', type=str, default = 'None',  help='folder for all mesh files')
#     # parser.add_argument('--file', type=str, default = 'None',  help='list of mesh files')
#     parser.add_argument('--cls', nargs="+", type=str, help='which category')
#     parser.add_argument('--typ', type=str, default='test', help='train/test')

#     parser.add_argument('--error', type=str, default = 'None',  help='quadric/normal/metro')
#     opt = parser.parse_args()

#     # catfile = os.path.join(opt.path, opt.file)

#     # getting all the mesh file names
#     mesh_files = [] 
#     shape_cls = [] 
#     root = '/home/minions/Dropbox/GraphicsLab/Projects/3D_Content_Creation/data/all_data'
#     for shape_class in opt.cls:
#         if opt.typ == 'train':
#             FILE = os.path.join(root, shape_class, 'train_full.txt')
#         else:
#             FILE = os.path.join(root, shape_class, 'test_full.txt')

#         with open(FILE) as f:
#             for line in f:
#                 mesh_files.append(line.strip())
#                 shape_cls.append(shape_class)

#     print('Total Models are %d' %(len(mesh_files)))
#     metroLoss = []
#     # quadricLoss = []
#     normaLoss = []

#     # with open(opt.file) as f:
#     for line, cls in zip(mesh_files, shape_cls):
#         line = line.strip()
#         line = line.split('.')[0]+'.obj'
#         original_mesh = os.path.join(root, cls, line)

#         # original_mesh = os.path.join(opt.path, line)
#         recon_mesh = os.path.join(opt.path, line.split('.')[0]+'_recon.obj')
#         print(original_mesh, recon_mesh)

#         if (original_mesh.endswith('ply') or original_mesh.endswith('obj')):
#             V_in, F_in = read_mesh(original_mesh)
#         else:
#             NotImplementedError('%s is not obj or ply file' %(line))

#         if (recon_mesh.endswith('ply') or recon_mesh.endswith('obj')):
#             V_recon, F_recon = read_mesh(recon_mesh)
#         else:
#             NotImplementedError('%s is not obj or ply file' %(line.split('.')[0]+'_recon.ply'))

#     # if opt.error == 'metro':
        
#         metro_exec='/home/minions/Dropbox/GraphicsLab/Projects/3D_Content_Creation/code/scripts/metro/metro '

#         command = metro_exec
#         command += original_mesh
#         command += ' ' + recon_mesh
#         command += ' ' + '-L'

#         # os.system(command)
#         output = subprocess.check_output(command, shell=True)
#         # print(output)
#         pattern = 'Hausdorff distance: (.*)\n'

#         m = re.search(pattern, output)
#         m = m.group()
#         m = m.split(' ')[2]
#         print('metro: ', float(m))
#         metroLoss.append(float(m))

#     # # elif opt.error == 'quadric':

#         Q = compute_Q_matrix(V_in, F_in)
        
#         N_in = compute_vertex_normals(V_in, F_in)
#         P, D, N_out = distance_pt2mesh(V_recon, F_recon, V_in)
        
#         # Q = torch.from_numpy(Q.astype(np.float32))
#         # Q = Q.view(V_in.shape[0], -1)
        
#         # P = torch.from_numpy(P.astype(np.float32))

#         # Q = torch.unsqueeze(Q, 0).cuda()
#         # P = torch.unsqueeze(P, 0).cuda()

#         # quadLoss = quadratic_error_loss(Q, P, average=False)
#         # print('quad: ', quadLoss)
#         # quadricLoss.append(quadLoss)
    
#     # elif opt.error == 'normal':

#         N_in = torch.from_numpy(N_in.astype(np.float32))
#         N_out = torch.from_numpy(N_out.astype(np.float32))

#         N_in = torch.unsqueeze(N_in, 0).cuda()
#         N_out = torch.unsqueeze(N_out, 0).cuda()

#         loss_max, loss_mean, loss_min = normal_loss(N_in, N_out)
#         # print('normal: ', loss_max.item())
#         # normaLoss.append(loss_max.item())
#         print('normal: ', loss_mean)
#         normaLoss.append(loss_mean)

#     # else:
#         # NotImplementedError('choose a valid error')

#     print('metro loss-------')
#     print('#items %d, Max Loss %f, Mean Loss %f, Median %f' %(len(metroLoss), np.max(metroLoss), np.mean(metroLoss), np.median(metroLoss) ))

#     print('normal loss-------')
#     print('#items %d, Max Loss %f, Mean Loss %f, Median %f' %(len(normaLoss), np.max(normaLoss), np.mean(normaLoss),np.median(normaLoss) ))

#     print('quadric loss-------')
#     print('#items %d, Max Loss %f, Mean Loss %f, Median %f' %(len(quadricLoss), np.max(quadricLoss), np.mean(quadricLoss),np.median(quadricLoss) ))










       




