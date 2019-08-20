# Copyright (c) Nitin Agarwal (agarwal@uci.edu)
# Last Modified:      Tue 20 Aug 2019 11:30:38 AM PDT

from __future__ import print_function
import sys
import os
import json
import math
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
from losses import *
from pc_utils import *
from provider import *


# ==============================================PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default=" ", help='input data dir')
parser.add_argument('--augment', type=bool, default=True,  help='augmentation')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points')
parser.add_argument('--small', type=bool, default=False,  help='train with small dataset')
parser.add_argument('--cls', nargs="+", type=str, help='shape dataset')
parser.add_argument('--seed', type=int, default=None,  help='seed')

parser.add_argument('--model', type=str, default = 'None',  help='load pretrained model')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--logf', type=str, default = 'log',  help='log folder')
parser.add_argument('--save_nth_epoch', type=int, default = 5, help='save network every nth epoch')
parser.add_argument('--bottleneck_size', type=int, default = 1024, help='embedding size')
parser.add_argument('--nb_primitives', type=int, default = 25, help='# primitives for AtlasNet')

parser.add_argument('--viz_env', type=str, default ="dgcnn_net", help='visdom environment')
parser.add_argument('--chamLoss_wt', type=float, default=0.0, help='chamfer loss wt')
parser.add_argument('--l1Loss_wt', type=float, default=0.0, help='l1 loss wt')
parser.add_argument('--quadLoss_wt', type=float, default=0.0, help='quad loss wt')
parser.add_argument('--sufNorLoss_wt', type=float, default=0.0, help='sufNorLoss_wt')
parser.add_argument('--sufLoss_wt', type=float, default=0.0, help='sufLoss_wt')

# Optimization 
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0, help='weight decay')

parser.add_argument('--momentum', type=float, default=0, help='momentum')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay lr_steps')
parser.add_argument('--lr_steps', default=200, nargs="+", type=int ,help='List of epochs where the learning rate is decreased by lr_decay')
parser.add_argument('--lr_step_size', default=100, type=int ,help='step size where the learning rate is decreased by lr_decay')

opt = parser.parse_args()
print (opt)


# ============================================LOGS=================================================== #
# Launch visdom for visualization
viz = visdom.Visdom(port = 8888, env=opt.viz_env)

input_3D = create_visdom_curve(viz, typ='scatter', viz_env=opt.viz_env)
output_3D = create_visdom_curve(viz, typ='scatter', viz_env=opt.viz_env)
epoch_curve = create_visdom_curve(viz, typ='line', viz_env=opt.viz_env)
epoch_curve_log = create_visdom_curve(viz, typ='line', viz_env=opt.viz_env)
val_input_3D = create_visdom_curve(viz, typ='scatter', viz_env=opt.viz_env)
val_output_3D = create_visdom_curve(viz, typ='scatter', viz_env=opt.viz_env)

dir_name =  os.path.join('log', opt.logf)

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

logname = os.path.join(dir_name, 'log.txt')

if opt.seed == None:
    opt.seed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)


# ===============================================LOAD DATASET================================= #

traindataset = getDataset(root=opt.dataDir, train=True, data_augment=opt.augment, small=opt.small, category=opt.cls)
traindataloader = torch.utils.data.DataLoader(traindataset, batch_size = opt.batchSize, 
                                              shuffle=True, num_workers=opt.workers)

testdataset = getDataset(root=opt.dataDir, train=False, data_augment=False, small=opt.small, category=opt.cls)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size = opt.batchSize,
                                             shuffle=False, num_workers=opt.workers)

print('Train Dataset:', len(traindataset))
print('Test Dataset:', len(testdataset)) 

# =============================================NETWORK================================= #

network = DG_AtlasNet(num_points = opt.num_points, bottleneck_size=1024, nb_primitives=25)
network.cuda() 
network.apply(weights_init) 
model_summary(network, True)

if opt.model != 'None':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")


optimizer = optim.Adam(network.parameters(), lr = opt.lr, weight_decay=opt.wd)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_steps, gamma=opt.lr_decay) 
scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay) 


train_loss = AverageValueMeter()
val_loss = AverageValueMeter()

with open(logname, 'a') as f: 
        f.write('Train: ' + str(len(traindataset)) + ' Test: ' + str(len(testdataset)) +'\n')
        f.write(str(opt) + '\n')
        f.write(str(network) + '\n')

train_curve = []
val_curve = []



def train(ep):
    network.train()
    
    for i, data in enumerate(traindataloader, 0):
        optimizer.zero_grad()
        
        points, Q, adj, normal, face_coords = data
        points = points.transpose(2,1)
        
        points = points.cuda()
        Q = Q.cuda()
        adj = adj.cuda()
        normal = normal.cuda()
        face_coords = face_coords.cuda()
        
        recon_points  = network(points) 
        
        recon_points = recon_points.transpose(2,1)
        points = points.transpose(2,1)

        chamLoss, corres, _ = chamferLoss(points, recon_points, average=False)
        l1Loss = l1_loss(points, pointsReconstructed)

        corres = corres.type(torch.cuda.LongTensor)
        recon_vertices = torch.cat([torch.index_select(a, 0, ind).unsqueeze(0) for a, ind in zip(recon_points, corres)])
        recon_points = recon_vertices
        
        quadLoss = quadratic_error_loss(Q, recon_points)
        sufNorLoss = surface_normal_loss(points, adj, recon_points, normal)
        sufLoss = surfaceLoss(recon_points, face_coords)

        # Total loss function
        loss_net = opt.chamLoss_wt * chamLoss
        loss_net += opt.l1Loss_wt * l1Loss 
        loss_net += opt.quadLoss_wt * quadLoss 
        loss_net += opt.sufNorLoss_wt * sufNorLoss 
        loss_net += opt.sufLoss_wt * sufLoss 

        train_loss.update(loss_net.item())
        
        loss_net.backward()
        optimizer.step() 
        
        # visualize
        if i%20 <= 0:
            viz.scatter(
                X = points[0].data.cpu(),
                win = input_3D, 
                env = opt.viz_env,
                opts=dict(title='Input PC [%s]' %(opt.logf), markersize = 1)
            )
            
            viz.scatter(
                X = pointsReconstructed[0].data.cpu(),
                win = output_3D, 
                env = opt.viz_env,
                opts=dict(title='Recon PC [%s]' %(opt.logf), markersize = 1)
            )
            
        print('[%d: %d/%d] train loss:  %f; C: %f, Q: %f, N: %f, S: %f' %(ep, i, len(traindataloader),
                                                                          loss_net.item(), chamLoss.item(),
                                                                          quadLoss.item(), sufNorLoss.item(),
                                                                          sufLoss.item() ))

def test(ep):
    network.eval()
    disp = np.random.randint(len(testdataloader))
    
    with torch.no_grad():
        for i, data in enumerate(testdataloader, 0):

            points, Q, adj, normal, face_coords = data
            points = points.transpose(2,1)

            points = points.cuda()
            Q = Q.cuda()
            adj = adj.cuda()
            normal = normal.cuda()
            face_coords = face_coords.cuda()


            recon_points  = network(points) 
            
            recon_points = recon_points.transpose(2,1)
            points = points.transpose(2,1)

            chamLoss, corres, _ = chamferLoss(points, recon_points, average=False)
            l1Loss = l1_loss(points, pointsReconstructed)

            corres = corres.type(torch.cuda.LongTensor)
            recon_vertices = torch.cat([torch.index_select(a, 0, ind).unsqueeze(0) for a, ind in zip(recon_points, corres)])
            recon_points = recon_vertices
            
            quadLoss = quadratic_error_loss(Q, recon_points)
            sufNorLoss = surface_normal_loss(points, adj, recon_points, normal)
            sufLoss = surfaceLoss(recon_points, face_coords)

            # Total loss function
            loss_net = opt.chamLoss_wt * chamLoss
            loss_net += opt.l1Loss_wt * l1Loss 
            loss_net += opt.quadLoss_wt * quadLoss 
            loss_net += opt.sufNorLoss_wt * sufNorLoss 
            loss_net += opt.sufLoss_wt * sufLoss 

            val_loss.update(loss_net.item())
            
            if i==disp:
                viz.scatter(
                    X = points[0].data.cpu(),
                    win = val_input_3D, 
                    env = opt.viz_env,
                    opts=dict(title='Val Input PC [%s]' %(opt.logf), markersize = 1)
                )
                
                viz.scatter(
                    X = pointsReconstructed[0].data.cpu(),
                    win = val_output_3D, 
                    env = opt.viz_env,
                    opts=dict(title='Val Recon PC [%s]' %(opt.logf), markersize = 1)
                )

            print('[%d: %d/%d] val loss:  %f ' %(ep, i, len(testdataloader), loss_net.item() ))

def main():
    
    best_val_loss = 1e5
    current_lr = opt.lr

    for epoch in range(opt.nepoch):
        train_loss.reset()
        val_loss.reset()

        scheduler.step()
        train(ep = epoch)
        test(ep = epoch)
        
        for param_group in optimizer.param_groups:
            print('Learning rate: %.7f [%.7f]' % (param_group['lr'], current_lr))
            current_lr = param_group['lr']

        # update visdom curves
        train_curve.append(train_loss.avg)
        val_curve.append(val_loss.avg)

        viz.line(X = np.array([epoch]), 
            Y = np.array([train_loss.avg]), 
            win = epoch_curve,
            env = opt.viz_env,
            update = 'append', 
            name = 'Train',
            opts=dict(title='Train/Test Loss [%s], lr=%f; wd=%f' %(opt.logf, opt.lr, opt.wd), showlegend=True)
        )

        viz.line(X = np.array([epoch]), 
            Y = np.array([val_loss.avg]), 
            win = epoch_curve,
            env = opt.viz_env,
            update = 'append', 
            name='Test',
            opts=dict(title='Train/Test Loss [%s], lr=%f; wd=%f' %(opt.logf, opt.lr, opt.wd), showlegend=True)
        )

        viz.line(X = np.array([epoch]), 
            Y = np.array([math.log(train_loss.avg)]), 
            win = epoch_curve_log,
            env = opt.viz_env,
            update = 'append', 
            name = 'Train',
            opts=dict(title='Train/Test Loss(log) [%s], lr=%f; wd=%f' %(opt.logf, opt.lr, opt.wd), showlegend=True)
        )

        viz.line(X = np.array([epoch]), 
            Y = np.array([math.log(val_loss.avg)]), 
            win = epoch_curve_log,
            env = opt.viz_env,
            update = 'append', 
            name='Test',
            opts=dict(title='Train/Test Loss(log) [%s], lr=%f; wd=%f' %(opt.logf, opt.lr, opt.wd), showlegend=True)
        )

        # update best test_loss and save the net
        if val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            print('New best loss: ', best_val_loss)
            print('saving network ...')
            torch.save(network.state_dict(), os.path.join(dir_name,'best_net_'+str(epoch)+'.pth'))
        
        elif (epoch+1) % opt.save_nth_epoch == 0:
            torch.save(network.state_dict(), os.path.join(dir_name,'ae_net_'+str(epoch)+'.pth'))


        log_table = {
          "train_loss" : train_loss.avg,
          "val_loss" : val_loss.avg,
          "epoch" : epoch,
          "lr" : current_lr,
          "bestval" : best_val_loss,
        }

        with open(logname, 'a') as f: 
            f.write('json_stats: ' + json.dumps(log_table) + '\n')


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_exec = round(time.time() - start_time)
    print('Total time taken: ', str(datetime.timedelta(seconds=time_exec)))
    print('-------Done-----------')      

