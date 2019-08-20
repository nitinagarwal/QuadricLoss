# Copyright (c) Nitin Agarwal 
# Last Modified:      Tue 20 Aug 2019 01:48:53 PM PDT

import os
import random
import numpy as np
from plyfile import (PlyData, PlyElement)

import scipy.sparse
import scipy.spatial as spatial


# --------------------------------
# MESH IO
# --------------------------------

def load_ply_data(filename):
    """ read ply file, only vertices and faces """

    plydata = PlyData.read(filename)

    vertices = plydata['vertex'].data[:]
    vertices = np.array([[x, y, z] for x,y,z in vertices])

    # input are all traingle meshes
    faces = plydata['face'].data['vertex_indices'][:]
    faces = np.array([[f1, f2, f3] for f1,f2,f3 in faces])

    return vertices, faces

def save_ply_data(filename, vertex, face):
    """ save ply file, only vertices and faces """

    vertices = np.zeros(vertex.shape[0], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    for i in range(vertex.shape[0]):
            vertices[i] = (vertex[i][0], vertex[i][1], vertex[i][2])
   
    faces = np.zeros(face.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
    for i in range(face.shape[0]):
            faces[i] = ([face[i][0], face[i][1], face[i][2]])

    e1 = PlyElement.describe(vertices, 'vertex')
    e2 = PlyElement.describe(faces, 'face')
    
    PlyData([e1, e2], text=True).write(filename)

def load_obj_data(filename):
    """
    A simply obj reader which reads vertices and faces only. 
    """
    
    ver =[]
    fac = []
    if not filename.endswith('obj'):
        sys.exit('the input file is not a obj file')

    with open(filename) as f:
        for line in f:
            if line.strip():
                inp = line.split()
                if(inp[0]=='v'):
                    ver.append([float(inp[1]), float(inp[2]), float(inp[3])])
                elif(inp[0]=='f'):
                    fac.append([int(inp[1])-1, int(inp[2])-1, int(inp[3])-1])

    V = np.array(ver)
    F = np.array(fac)
    
    return V, F

def save_obj_data(filename, vertex, face):
    """
    saves only vertices and faces
    """
    
    numver = vertex.shape[0]
    numfac = face.shape[0]

    with open(filename, 'a') as f:
        f.write('# %d vertices, %d faces'%(numver, numfac))
        f.write('\n')

    with open(filename, 'a') as f:
        for v in vertex:
            f.write('v %f %f %f' %(v[0], v[1], v[2]))
            f.write('\n')
   
    with open(filename, 'a') as f:
        for F in face:
            f.write('f %d %d %d' %(F[0]+1, F[1]+1, F[2]+1))
            f.write('\n')

def load_xyz_data(filename):
    """
    simple xyz reader which reads vertices (and normals if presents)
    """

    ver =[]
    normal = []
    
    if not filename.endswith('xyz'):
        sys.exit('the input file is not a xyz file')

    with open(filename) as f:
        for line in f:
            line = line.strip()
            inp = line.split(' ')
            
            ver.append([float(inp[0]), float(inp[1]), float(inp[2])])
            if len(inp) == 6:
                normal.append([float(inp[3]), float(inp[4]), float(inp[5])])

    V = np.array(ver)
    N = np.array(normal)

    return V, N

def save_xyz_data(filename, ver):
    """writes the points to a xyz file"""
    
    write_path = filename.split('.')[0] + '.xyz'
    np.savetxt(write_path, ver)



# --------------------------------
# MESH UTILS
# --------------------------------

def get_adjacency_matrix(vertex, faces, K_max='None'):
    """ computes the adjacency matrix 

    Input: vertex = (N x 3)
           faces = (F x 3) triangle mesh 

    Output: sparase adjacency matrix = (N x K) where K is the max numbers of neighbours. 
            Each row lists the vertex indices of the neighbhours. index starts from 1. 
            The rows are padded with zeros.
    """

    num_pts = len(vertex)
    K = 0     # max number of neighbours
    adj = []

    # computing the value of K
    for i in range(num_pts):
        idx = np.argwhere(faces == i)

        if(len(idx)==0):
            idx = [0]
        else:
            idx = faces[idx[:,0]]
            idx = np.unique(np.sort(idx.reshape(-1)))
            idx = idx.tolist()
            idx.remove(i)
            idx = [x+1 for x in idx]            # because the index starts from 1 
            if(len(idx) > K):
                K = len(idx)
        adj.append(idx)

    if K_max != 'None':
        K = K_max

    adjacency = np.zeros((1, K), dtype=np.int16)
    for i in range(len(adj)):
        ver = np.concatenate((np.asarray(adj[i], dtype=np.int16),np.array([0]*(K-len(adj[i])), dtype=np.int16)))
        adjacency = np.vstack((adjacency,ver))
    adjacency = adjacency[1:]

    return adjacency


def get_face_coordinates(vertex, faces, K_max='None'):
    """ computes the face_cooridinates used for surface Loss 

    Input: vertex = (N x 3)
           faces = (F x 3) triangle mesh 

    Output: face_cooridnates = (NxKx9) where K is the max numbers of neighbours. 
            Each row lists vertex coordinates for the one ring neighbourhood faces 
            The rows are padded with dummy face coordinates 
    """

    num_pts = len(vertex)
    K = 0     
    adj = []

    # computing the value of K
    for i in range(num_pts):
        idx = np.argwhere(faces == i)

        if(len(idx)==0):
            idx = [0]
        else:
            idx = faces[idx[:,0]]
            idx = np.unique(np.sort(idx.reshape(-1)))
            idx = idx.tolist()
            idx.remove(i)
            idx = [x+1 for x in idx]            # because the index starts from 1 
            if(len(idx) > K):
                K = len(idx)
        adj.append(idx)

    if K_max != 'None':
        K = K_max

    face_coords = np.zeros((num_pts, K, 9), dtype=np.float)
    dummy_faces = np.array([10,12,13,16,17,18,20,21,22])

    for i in range(num_pts):

        idx = np.argwhere(faces == i)

        if(len(idx)==0):
            temp = np.tile(dummy_faces, (K_max, 1))
        else:
            idx = idx[:,0]
            v1 = vertex[faces[idx,0],:]
            v2 = vertex[faces[idx,1],:]
            v3 = vertex[faces[idx,2],:]
            temp = np.concatenate((v1, v2, v3), axis=1)
            temp = np.concatenate( (temp, np.tile(dummy_faces, (K-len(idx), 1))) )

        face_coords[i,:,:] = temp 

    return face_coords


def compute_Q_matrix(vertices, faces):
    """ computes Quadric matrix for each vertex

    Input: vertex = (N x 3)
           faces = (F x 3) triangle mesh 

    Output: Q = (N x 4 x 4) For each vertex, computes the summation of all Q's for the triangles incident 
            at that vertex
    """

    num_pts = len(vertices)
    Q = np.zeros((num_pts, 4, 4), dtype=float)

    # computing Q for each vertex (v_q = Q1+Q2+Q3+....Q_n)
    for i in range(num_pts):
        idx = np.argwhere(faces == i)

        if(len(idx) > 0): 
            q = np.zeros((4,4), dtype=float)

            length = len(idx)
            for j in range(length):
                f = faces[idx[j,0]]
                [v1, v2, v3] = vertices[f[0],:], vertices[f[1],:], vertices[f[2],:]
                v1 = v1.astype(float)
                v2 = v2.astype(float)
                v3 = v3.astype(float)
                q = q + get_plane(v1, v2, v3)
            Q[i,:,:] = q

    return Q


def get_plane(v1, v2, v3):

    """ Compute the Q matrix for a single triangle 
    Input: vertex = (N x 3)

    Output: Q = (4x4) 
            equation of plane is ax + by + cz + d = 0
            where <a, b, c> is the normalized normal vector
    """

    v12 = v1 - v2
    v13 = v1 - v3

    normal =  np.cross(v12, v13)
    mag = np.sqrt(np.dot(normal, normal))

    #avoid divide by zero
    if mag != 0:
        normal = normal / mag

    assert( np.linalg.norm(normal)==0 or np.abs(1.0 - np.linalg.norm(normal)) < 1e-10)
    d =  - np.dot(normal, v1)

    equ = np.array([normal[0], normal[1], normal[2], d])
    Q = np.outer(equ,equ)
    
    return Q


def compute_face_normals(vertices, faces):
    """
    Input: vertices = (Nx3)
           faces = (Fx3)
    
    Output: face normals = (Fx3)
    """
    
    num_fac = len(faces)
    normals = np.zeros((num_fac, 3), dtype=float)

    for idx in range(num_fac):
        e10 = vertices[faces[idx,1],:] - vertices[faces[idx,0],:]
        e20 = vertices[faces[idx,2],:] - vertices[faces[idx,0],:] 
        n_idx = np.cross(e10, e20)
        mag = np.sqrt(np.dot(n_idx, n_idx))
        
        if mag!=0:
            n_idx = n_idx/mag
    
        normals[idx,:] = n_idx

    return normals


def compute_vertex_normals(vertices, faces):
    """
    Input: vertices = (Nx3)
           faces = (Fx3)

    Output: vertex normals = (Nx3) 
    """

    num_ver = len(vertices)
    normals = np.zeros((num_ver, 3), dtype=float)

    face_normals = compute_face_normals(vertices, faces)

    for i in range(num_ver):
        idx = np.argwhere(faces == i)

        if(len(idx)>0):     # not isolated vertex
            
            vert_normal = np.sum(face_normals[idx[:,0]], axis=0)
            vert_normal = vert_normal/len(idx)       #average of face normals

            mag = np.linalg.norm(vert_normal)
            if mag != 0:
                vert_normal = vert_normal/mag

            normals[i,:] = vert_normal

    return normals


def closest_triangles(query_pts, ver, faces, tree):
    """ for each query pt find the closest pt among the mesh vertices and return 
        all the triangles incident on that closest point.

    Input : query_pts = (N'x3)
            ver = (Nx3)
            faces = (Fx3)
            tree = KDTree formed by the mesh vertices
    
    Ouput: indx = for each query pt, indices of closest triangles
           dist = for each query pt, the distance 
    """

    dis, ids = tree.query(query_pts)
    
    indx = []
    dist = []

    for index, i in enumerate(ids):

        ij = np.argwhere(faces == i)
        ij = ij[:,0]
        ij = ij.tolist()

        indx.append(ij)
        dist.append(dis[index])

    return indx, dist


def pt2triangle(query, vertices, f, v_normal):
    """ pt to triangle distance
    Input: query = (3)
           vertices = Nx3
           f = face ids in question
           v_normal = Nx3 (vertex normals)
    
    Ouput: pt = (3) closest pt on the triangle
           d = its distance to the query pt
           n = its normal (face normal)
    https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    """

    [v1, v2, v3] = vertices[f[0],:], vertices[f[1],:], vertices[f[2],:]
    [n1, n2, n3] = v_normal[f[0],:], v_normal[f[1],:], v_normal[f[2],:]

    v1 = v1.astype(float)
    v2 = v2.astype(float)
    v3 = v3.astype(float)
    
    # closest pt = B + s*E0 + t*E1
    B = v1
    E0 = v2 - v1
    E1 = v3 - v1

    D = B - query
    a = np.sum(E0*E0) 
    b = np.sum(E0*E1) 
    c = np.sum(E1*E1) 
    d = np.sum(E0*D) 
    e = np.sum(E1*D) 
    f = np.sum(D*D) 

    det = a*c - b*b
    s = b*e - c*d
    t = b*d - a*e

    if (s + t <= det):
        if( s < 0 ):
            if ( t < 0 ):
                region = 4
            else:
                region = 3

        elif (t < 0):
            region = 5

        else:
            region = 0
    else:
        if( s < 0 ):
            region = 2
        
        elif( t < 0 ):
            region = 6

        else:
            region = 1
 
    # find the closest point
    if region == 0:
        ss = s/det
        tt = t/det

    elif region == 1:
        
        numer = c+e-b-d
        denom = a-2*b+c
        if(numer <= 0):
            ss = 0
        else:
            if(numer >= denom):
                ss = 1
            else:
                ss = numer/denom
        tt = 1 - ss

    elif region == 3:
        ss = 0
        if (e >= 0):
            tt = 0
        elif (-e >= c):
            tt = 1
        else:
            tt = -e/c

    elif region == 5:
        tt = 0
        if(d >= 0):
            ss = 0
        elif(-d >= a):
            ss = 1
        else:
            ss = -d/a

    elif region == 2:
        temp0 = b + d
        temp1 = c + e
        numer = temp1 - temp0
        denom = a-2*b+c

        if ( temp1 > temp0 ):

            if(numer >= denom):
                ss = 1
            else:
                ss = numer/denom
            tt = 1 - ss
        else:
            ss = 0
            if(temp1 <= 0 ):
                tt = 1
            elif(e >= 0):
                tt = 0
            else:
                tt = -e/c

    elif region == 6:
        temp0 = b + e
        temp1 = a + d
        numer = temp1 - temp0
        denom = a-2*b+c

        if( temp1 > temp0 ):
            if(numer >= denom):
                tt = 1
            else:
                tt = numer/denom
            ss = 1 - tt
        else:
            tt = 0
            if( temp1 <= 0 ):
                ss = 1
            elif (d >= 0):
                ss = 0
            else:
                ss = -d/a

    elif region == 4:
        if( d < 0 ):
            tt = 0
            if( -d >= a ):
                ss = 1
            else:
                ss = -d/a
        else:
            ss = 0
            if(e > 0):
                tt = 0
            elif( -e >=c ):
                tt = 1
            else:
                tt = -e/c

    else:
        NotImplementedError("region not selected")
            

    p0 = B + ss*E0 + tt*E1
    
    n =  n1 + ss*(n2-n1) + tt*(n3-n1)
    mag = np.sqrt(np.sum((n*n)))
    if mag!=0:
        n = n/mag

    dist = ss*(a*ss+b*tt+2*d) + tt*(b*ss+c*tt+2*e) + f

    return p0, dist, n 


def distance_pt2mesh(vertices, faces, query_pts):
    """ for each query point, search the mesh (vertices, faces) to find the closest pt on the mesh

    Input: vertices = Nx3
           faces = Fx3
           query_pts= N'x3
    
    Output:output_pts = N'x3  (closet pt on mesh to query pts)
           dist = N' (square of the distance b/w those pts)
           normals = N'x3 (normals of the closest pt on the mesh)
    """

    if query_pts.ndim==1:
        query_pts = np.expand_dims(query_pts, axis=0)
    
    N = vertices.shape[0]
    F = faces.shape[0]
    N_q = query_pts.shape[0]

    output_pts = np.zeros((N_q, 3), dtype=np.float)
    dist = np.zeros((N_q), dtype=np.float)
    normals = np.zeros((N_q, 3), dtype=np.float)

    # compute vertex normals for the mesh
    v_normal = compute_vertex_normals(vertices, faces)

    # construct the KDtree and search for candidate faces
    # idx = N'xK  a list (candidate faces)
    tree = spatial.KDTree(vertices)
    idx, _ = closest_triangles(query_pts, vertices, faces, tree)

    for ii in range(len(idx)):

        query = query_pts[ii,:]
        candidate_faces = faces[idx[ii],:]
        P, D, N = [], [], []

        for f in candidate_faces:
        
            pt, d, norm = pt2triangle(query, vertices, f, v_normal)
            P.append(pt)
            D.append(d)
            N.append(norm)

        D = np.asarray(D)
        index = np.argmin(D)

        output_pts[ii,:] = P[index]
        dist[ii] = D[index]
        normals[ii,:] = N[index]

    return output_pts, dist, normals


def farthest_point_sample(pts, K):
    """ Farthest point sampling

    Input: pts = original points
           K = # of pts to sample

    Output: K points
    """

    if pts.shape[0] < K:
        NotImplementedError("Not enough points in mesh for FPS")
    
    def calc_distances(p0, points):
        return ((p0 - points)**2).sum(axis=1)

    farthest_pts = np.zeros((K,3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    
    return farthest_pts
    

def uniform_sampling(vertices, faces, n_samples=1000, reverse=False):
    """ Uniform area sampling
    
    Input: vertices  = N x 3 matrix
           faces     = N x 3 matrix
           n_samples = positive integer
           reverse = small area faces have more points

    Output:vertices = n_sample points
           (P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C)
           https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    """

    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                       vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))

    # small area get more points
    if reverse:
        face_areas = 1/face_areas
    face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Contributed by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples

    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=False)
        n_samples_per_face[floor_indices] -= 1

    assert(n_samples == np.sum(n_samples_per_face))
    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples, ), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2);
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
      np.sqrt(r[:,0:1]) * r[:,1:] * C

    return P


def jitter_vertices(vertices, sigma=0.01, mean=0, clip=0.05, percent=0.30):
    """ Randomly jitter points 

    Input:vertices = Nx3 array, original points
          sigma = std of gaussian noise
          mean = mean of the gaussian noise
          clip = to clip the noise
          percent = % pts jittered
    
    Output: Nx3 array, jittered points 
    """

    N, C = vertices.shape

    assert(clip > 0)

    gaussian_noise = sigma * np.random.randn(int(N*percent), C) + mean
    gaussian_noise = np.clip(gaussian_noise, -1*clip, clip)

    ids = np.random.randint(N, size=int(N*percent))

    vertices[ids,:] = vertices[ids,:] + gaussian_noise

    return vertices 


def scale_vertices(vertices, low_bound=0.5, high_bound=1.5):
    """ Randomly scale vertices along arbitary axes (anisotropic scaling)
        scale only in x or z Not in y

    Input: Nx3 array, input points
           low and high bound: range of scale values allowed
    
    Output: Nx3 array, scale shape
    """
    
    scale = (high_bound-low_bound) * np.random.rand() + low_bound 
    axis = np.random.randint(1, 3)

    return scale_vertices_by_axis(vertices, scale, axis)
    

def scale_vertices_by_axis(vertices, scale, axis):
    """ scale vertices by axis
    
    Input: Nx3 array, input shape
           scale = scale
           axis = 1(xaxis), 2(yaxis), 3(zaxis)

    Output: Nx3 array, scaled shape
    """

    scaled_data = np.zeros(vertices.shape, dtype=np.float32)
    
    # shear_matrix = np.array([[1, shear_y, 0],
    #                             [0, 1, shear_z],
    #                             [shear_x, 0, 1]])
    
    if axis == 1:
        scale_matrix = np.array([[scale, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
    # elif axis == 2:
    #     scale_matrix = np.array([[1, 0, 0],
    #                                 [0, scale, 0],
    #                                 [0, 0, 1]])
    elif axis == 2:
        scale_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, scale]])
    else:
        NotImplementedError('Choose a valid rotation axis(1-Xaxis, 2-Yaxis, 3-Zaxis)')

    scaled_data = np.dot(vertices, scale_matrix)
    return scaled_data


def rotate_vertices(vertices, sigma=0.1, clip=0.1):
    """ Randomly rotate vertices along arbitary axes
    
    Input: Nx3 array, input shape
           clip - max range of rotation allowed

    Output: Nx3 array, rotated shape
    """

    rotation_angle = np.clip(sigma * np.random.randn(), -1*clip, clip) * np.pi
    rotation_axis = np.random.randint(1, 4)

    return rotate_vertices_by_angle_by_axis(vertices, rotation_angle, rotation_axis) 


def rotate_vertices_by_angle_by_axis(vertices, rotation_angle, rotation_axis):
    """ Rotate vertices along rotation axis with rotation angle

    Input: Nx3 array, input shape
           rotation_angle - angle in radians
           rotation_axis - 1(xaxis), 2(yaxis), 3(zaxis)
    
    Output:
      Nx3 array, rotated shape
    """

    rotated_data = np.zeros(vertices.shape, dtype=np.float32)

    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
   
    if rotation_axis == 1:
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval, cosval]])
    elif rotation_axis == 2:
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
    elif rotation_axis == 3:
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
    else:
        NotImplementedError('Choose a valid rotation axis(1-Xaxis, 2-Yaxis, 3-Zaxis)')

    rotated_data = np.dot(vertices, rotation_matrix)
    return rotated_data


def normalize_shape(vertices):
    """normalize shape to fit inside a unit sphere"""
    
    ver_max = np.max(vertices, axis=0)
    ver_min = np.min(vertices, axis=0)

    centroid = np.stack((ver_max, ver_min), 0)
    centroid = np.mean(centroid, axis=0)
    vertices = vertices - centroid

    longest_distance = np.max(np.sqrt(np.sum((vertices**2), axis=1)))
    vertices = vertices / longest_distance
    
    return vertices



# --------------------------------
# TRAINING UTILS
# --------------------------------

def weights_init(m):
    """weight initialization for CNN and batchnorm layers"""
    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def model_summary(model, print_layers=False):
    train_count = 0
    nontrain_count = 0
    
    for name, p in model.named_parameters():
        if(p.requires_grad):
            if(print_layers):                 
                print('Train: ', name, 'has', p.numel())
            train_count += p.numel()
        
        elif not p.requires_grad:
            if(print_layers):
                print('Non Train: ', name, 'has', p.numel())
            nontrain_count += p.numel()
        
    print('Total Parameters: ', train_count+nontrain_count)    
    print('Trainable Parameters: ',train_count)
    print('NonTrainable Parameters: ',nontrain_count)


def create_visdom_curve(viz, typ='scatter', viz_env='main'):

    if typ == 'scatter':
        graph = viz.scatter(
        X = np.random.rand(2000,3),
        env = viz_env,
        opts=dict(title='Input PC', markersize = 1)  #opt could be anything
        )
    elif typ == 'line':
        graph = viz.line(
        X = np.column_stack((np.array( [0] ), np.array([0]))),
        Y = np.column_stack((np.array( [0] ), np.array([0]))),
        env = viz_env,
        opts=dict(title='Train/Test Loss', legend=['Train', 'Test']) 
        )

    elif typ == 'mesh':
        graph = viz.mesh(
        X = np.random.rand(2000,3), 
        Y = np.round(np.random.rand(2000,3)), 
        env = viz_env,
        opts = dict(title='Input Mesh',color='red')
        )

    elif typ == 'images':    
        graph = viz.images(
        np.random.rand(2, 1, 256, 256),
        env = viz_env,
        opts = dict(title='TBD')
        ) 

    elif typ == 'hist':    
        graph = viz.histogram(
        np.random.rand(1000),
        env = viz_env,
        opts = dict(title='TBD')
        ) 

    return graph



if __name__ == "__main__":

    filename = '../data/sample.obj'

    V, F = load_obj_data(filename)
    Q = compute_Q_matrix(V, F)

    sz = V.shape[0]

    for ii in range(100):
        v_temp = np.random.randn(sz,3)
        v_temp = np.concatenate((v_temp, np.ones((sz,1))), axis=1)

        a = np.reshape(v_temp, (sz,4,1))
        b = np.transpose(a, (0,2,1))

        v_temp_outer = np.matmul(a, b)
        
        v_temp_outer = np.reshape(v_temp_outer, (sz,-1))
        Q_temp = np.reshape(Q, (sz,-1))
 
        ans = np.sum((v_temp_outer * Q_temp), axis=-1)

        _sum = 0
        for i, v in enumerate(v_temp, 0):
            v = np.array([v[0], v[1], v[2], 1])
            
            a = np.matmul(np.transpose(v), Q[i,:])
            b = np.matmul(a, v)
            _sum += b
            # if(ans[i] != b):
            assert(ans[i]>=0 and b>=0)
            # if(b < 0):
                # print(ans[i], ' ', b)
        # print(np.sum(ans))
        # print( np.sum(ans), '         ', _sum)
        print('%d done %f  %f' % (ii, np.sum(ans), _sum))


    # Vout = farthest_point_sample(V, 3000)
    # Vout = Variable(torch.from_numpy(Vout))
    # save_xyz_data('/home/minions/Desktop/chair_0001.xyz', Vout)
    # f = get_face_coordinates(V, F)
    # print(f, f.shape)






