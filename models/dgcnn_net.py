import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from init import *

# sys.path.append("../utils")
# from baseline_utils import *

# dgcnn_net encoder
class dgcnn_encoder(nn.Module):
    """docstring for edge_conv_model"""
    def __init__(self, k=20):
        super(dgcnn_encoder, self).__init__()

        self.k = k

        self.conv1 = nn.Conv2d(6, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv5 = nn.Conv2d(320, 1024, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)

    def forward(self, x):

        x = torch.transpose(x,2,1).contiguous()
        B, N, D = x.size()

        # Bx6xNxk
        dist_mat = pairwise_distance(x)
        nn_idx = knn(dist_mat, k=self.k)
        edge_feat = get_edge_feature(x, nn_idx=nn_idx, k=self.k)
        edge_feat = edge_feat.permute(0,3,1,2)

        # Bx64xNx1
        x = self.bn1(F.relu(self.conv1(edge_feat)))
        x,_ = torch.max(x, dim=-1, keepdim=True)
        x1 = x

        # Bx128xNxk
        x = x.permute(0,2,3,1)
        dist_mat = pairwise_distance(x)
        nn_idx = knn(dist_mat, k=self.k)
        edge_feat = get_edge_feature(x, nn_idx=nn_idx, k=self.k)
        edge_feat = edge_feat.permute(0,3,1,2)

        # Bx64xNx1
        x = self.bn2(F.relu(self.conv2(edge_feat)))
        x,_ = torch.max(x, dim=-1, keepdim=True)
        x2 = x

        # Bx128xNxk
        x = x.permute(0,2,3,1)
        dist_mat = pairwise_distance(x)
        nn_idx = knn(dist_mat, k=self.k)
        edge_feat = get_edge_feature(x, nn_idx=nn_idx, k=self.k)
        edge_feat = edge_feat.permute(0,3,1,2)

        # Bx64xNx1
        x = self.bn3(F.relu(self.conv3(edge_feat)))
        x,_ = torch.max(x, dim=-1, keepdim=True)
        x3 = x

        # Bx128xNxk
        x = x.permute(0,2,3,1)
        dist_mat = pairwise_distance(x)
        nn_idx = knn(dist_mat, k=self.k)
        edge_feat = get_edge_feature(x, nn_idx=nn_idx, k=self.k)
        edge_feat = edge_feat.permute(0,3,1,2)

        # Bx128xNx1
        x = self.bn4(F.relu(self.conv4(edge_feat)))
        x,_ = torch.max(x, dim=-1, keepdim=True)
        x4 = x

        # Bx1024x1x1    same as conv1d
        x = self.bn5(F.relu(self.conv5(torch.cat((x1, x2, x3, x4), 1))))
        x,_ = torch.max(x, dim=2, keepdim=True)

        # Bx1024
        x = x.view(B, -1)

        return x



""" Decoders"""
class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x



class AE_AtlasNet(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 1, K=20):
        super(AE_AtlasNet, self).__init__()
        
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        
        self.encoder = nn.Sequential(
        dgcnn_encoder(k=K),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.nb_primitives)])


    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous()




if __name__ == '__main__':

    for item in range(1):
        
        B, N, D = 2, 2500, 3

        data = Variable(torch.rand(B, D, N))

        # network = AE_Baseline(num_points = 2000, bottleneck_size = 1024)
        # network = AE_Baseline_normal(num_points = 2000, bottleneck_size = 1024)
        network = AE_AtlasNet(num_points = 2500, bottleneck_size = 1024, nb_primitives=25)
        network.cuda()
        print(model_summary(network, True))

        data = data.cuda()
        out = network(data)
        print('input data size: ', data.size())
        print('output data size: ', out.size())


    # if __name__ == "__main__":
    #     B, N, D = 2, 2000, 3

    #     net = dgcnn_net()
    #     net = net.cuda()

    #     out = net(points)

    #     print('input size ', points.size())
    #     print('output size ', out.size())


