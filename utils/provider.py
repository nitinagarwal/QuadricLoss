# Copyright (c) 2019 Nitin Agarwal (agarwal@uci.edu)

from __future__ import print_function
import numpy as np
import os
import sys
import scipy.sparse

import torch
import torch.utils.data as data

sys.path.append('./utils')
from pc_utils import *


class getDataset(data.Dataset):
    def __init__(self, root, train=True, data_augment=True, small=False, category = ['abc_2.5k']):
        
        self.root = root
        self.train = train
        self.data_augment = data_augment    
        self.small = small         # test on a small dataset

        shape_paths = []            # path of all mesh files
        for shape_class in category:
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
            """ data augment by scaling and rotation""" 
            for line in shape_paths:
                
                mesh_path = line 
                mesh={}
                mesh["rotate"] = False
                mesh["scale"] = True
                mesh["path"] = mesh_path
                self.datapath.append(mesh)
                
                mesh={}
                mesh["rotate"] = True
                mesh["scale"] = False
                mesh["path"] = mesh_path
                self.datapath.append(mesh)

                mesh={}
                mesh["rotate"] = True
                mesh["scale"] = True
                mesh["path"] = mesh_path
                self.datapath.append(mesh)

        for line in shape_paths:
                mesh = {}
                mesh_path = line 
                mesh["rotate"]=False
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

        if fn["scale"]:
            vertices = scale_vertices(vertices)
        if fn["rotate"]:
            vertices = rotate_vertices(vertices)
        
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
    
    path = '../data'

    obj = getDataset(root = path, train=False, data_augment=False, small=False, category=['abc_2.5k'])
    
    testdataloader = torch.utils.data.DataLoader(obj, batch_size = 1, shuffle=False, num_workers=4)
    
    print(len(obj))
    for i, data in enumerate(testdataloader, 0):

        v, q, adj, normal, f = data
        print(v.size(), q.size(), adj.size(), normal.size(), f.size())

















