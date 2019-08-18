from __future__ import print_function
import numpy as np
import os
import os.path
import sys
import scipy.sparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from pprint import pprint
from termcolor import colored

from baseline_utils import *


class getDataset(data.Dataset):
    def __init__(self, root, train=True, data_augment=True, small=True, K_max=0, with_uv=True,
                 template='None', layers=4, category = ['chairs', 'tables']):
        
        self.root = root
        self.train = train
        self.data_augment = data_augment    # both rotation and jittering
        self.K = K_max
        self.small = small         # use small dataset or big
        self.layers = layers
        # self.category = ['chairs', 'tables', 'mirrors', 'doors']

        shape_paths = [] #path of all mesh files
        for shape_class in category:
            # shape_class = ca
            if self.train:
                if self.small:
                    self.file = os.path.join(self.root, shape_class, 'train.txt')
                else:
                    self.file = os.path.join(self.root, shape_class, 'train_full.txt')
            else:
                if self.small:
                    self.file = os.path.join(self.root, shape_class, 'test.txt')
                else:
                    self.file = os.path.join(self.root, shape_class, 'test_full.txt')

            with open(self.file) as f:
                for line in f:
                    shape_paths.append(os.path.join(self.root, shape_class, line.strip()))

        self.datapath=[]
        if self.data_augment:
            """ augment by scaling, random rotation, and random jitter""" 
            # with open(self.file) as f:
            for line in shape_paths:
                
                mesh_path = line #os.path.join(self.root,line.strip())
                mesh={}
                mesh["rotate"] = False
                mesh["jitter"] = False
                mesh["scale"] = True
                mesh["path"] = mesh_path
                self.datapath.append(mesh)
                
                mesh={}
                mesh["rotate"] = True
                mesh["jitter"] = False
                mesh["scale"] = False
                mesh["path"] = mesh_path
                self.datapath.append(mesh)

                # mesh={}
                # mesh["rotate"] = False
                # mesh["jitter"] = True
                # mesh["scale"] = False
                # mesh["path"] = mesh_path
                # self.datapath.append(mesh)

                # mesh={}
                # mesh["rotate"] = False
                # mesh["jitter"] = True
                # mesh["scale"] = True
                # mesh["path"] = mesh_path
                # self.datapath.append(mesh)

                mesh={}
                mesh["rotate"] = True
                mesh["jitter"] = False
                mesh["scale"] = True
                mesh["path"] = mesh_path
                self.datapath.append(mesh)

                # mesh={}
                # mesh["rotate"] = True
                # mesh["jitter"] = True
                # mesh["scale"] = False
                # mesh["path"] = mesh_path
                # self.datapath.append(mesh)
                
                # mesh={}
                # mesh["rotate"] = True
                # mesh["jitter"] = True
                # mesh["scale"] = True
                # mesh["path"] = mesh_path
                # self.datapath.append(mesh)

                # for angle in range(0,360,45):
                #     mesh={}
                #     ang_rad = angle*2*np.pi/360
                #     mesh["rotate_angle"] = ang_rad
                    # mesh["path"] = mesh_path
                    # mesh["jitter"] = False

                    # self.datapath.append(mesh)

        # with open(self.file) as f:
            # for line in f:
        for line in shape_paths:
                mesh = {}
                mesh_path = line 
                # mesh_path = os.path.join(self.root,line.strip())
                mesh["rotate"]=False
                mesh["jitter"] = False
                mesh["scale"] = False
                mesh["path"] = mesh_path
                
                self.datapath.append(mesh)


    def __getitem__(self, index):

        fn = self.datapath[index]

        if fn["path"].endswith('obj'):
            vertices, faces = load_obj_data(fn["path"])
        else:
            vertices, faces = load_ply_data(fn["path"])

        # vertices = uniform_sampling(vertices, faces, 2500)
        # vertices = uniform_sampling(vertices, faces, 8000)

        # print(fn["path"], 'scale ', fn["scale"], 'rotate ', fn["rotate"])
        if fn["scale"]:
            vertices = scale_vertices(vertices)
        # if fn["rotate_angle"] == 0:
        if fn["rotate"]:
            vertices = rotate_vertices(vertices)
        
        if fn["jitter"]:
            vertices = jitter_vertices(vertices, sigma=0.005, clip=0.01, percent=0.30)

        vertices = normalize_shape(vertices)
        y_label = compute_Q_matrix(vertices, faces)

        adj = get_adjacency_matrix(vertices, faces, K_max=271)
        face_coords = get_face_coordinates(vertices, faces, K_max=271)
        normal = compute_vertex_normals(vertices, faces)
        # vertices = farthest_point_sample(vertices, 2500)
        
        vertices = self.convert_to_tensor(vertices)
        y_label = self.convert_to_tensor(y_label)
        y_label = y_label.view(vertices.size()[0], -1)
        adj = self.convert_to_tensor(adj)
        normal = self.convert_to_tensor(normal)
        face_coords = self.convert_to_tensor(face_coords)

        return vertices, y_label, adj, normal, face_coords


    def convert_to_tensor(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        
        return x

    def __len__(self):
        return len(self.datapath)




if __name__ == "__main__":
    
    path = '/home/minions/Dropbox/GraphicsLab/Projects/3D_Content_Creation/data/all_data'

    obj = getDataset(root = path, train=False, data_augment=False, small=False, category=['abc_2.5k'])
    
    testdataloader = torch.utils.data.DataLoader(obj, batch_size = 1, shuffle=False, num_workers=4)
    
    print(len(obj))
    for i, data in enumerate(testdataloader, 0):

        v, q, adj, normal, f = data
        print(v.size(), q.size(), adj.size(), normal.size(), f.size())

















