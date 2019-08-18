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
import math
import time, datetime
import visdom

from losses import *

sys.path.append('./models/')
from dgcnn_net import *

sys.path.append('./utils/')
from baseline_utils import *
from provider import *

# sys.path.append('./emd/')
# from modules.emd import EMDModule
# distEMD =  EMDModule()

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str, default=" ", help='input data dir')
parser.add_argument('--augment', type=bool, default=True,  help='augmentation')
parser.add_argument('--num_points', type=int, default = 2000,  help='number of points')
parser.add_argument('--small', type=bool, default=False,  help='train with small dataset')
parser.add_argument('--cls', nargs="+", type=str, help='which category')

parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--logf', type=str, default = 'log1',  help='log folder')
parser.add_argument('--save_nth_epoch', type=int, default = 5,  help='save network every nth epoch')

parser.add_argument('--viz_env', type=str, default ="PointNet_Baseline"   ,  help='visdom environment')
parser.add_argument('--chamLoss_wt', type=float, default=0.0, help='chamfer loss wt')
parser.add_argument('--l1Loss_wt', type=float, default=0, help='l1 loss wt')
parser.add_argument('--EMDLoss_wt', type=float, default=0, help='EMD loss wt')
parser.add_argument('--quadLoss_wt', type=float, default=0, help='quad loss wt')
parser.add_argument('--sufNorLoss_wt', type=float, default=0, help='sufNorLoss_wt')
parser.add_argument('--sufLoss_wt', type=float, default=0, help='sufNorLoss_wt')

# Optimization 
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0, help='weight decay')

parser.add_argument('--momentum', type=float, default=0, help='momentum')
parser.add_argument('--lr_decay', type=float, default=0.1, help='Multiplicative factor used on learning rate at lr_steps')
parser.add_argument('--lr_steps', default=200, nargs="+", type=int ,help='List of epochs where the learning rate is decreased by lr_decay')
parser.add_argument('--lr_step_size', default=100, type=int ,help='step size where the learning rate is decreased by lr_decay')


opt = parser.parse_args()
print (opt)
# ========================================================== #


## =============DEFINE stuff for logs ======================================== #
#Launch visdom for visualization
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

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
# ========================================================== #


# ===================CREATE DATASET================================= #
#Create train/test dataloader

traindataset = getDataset(root=opt.dataDir, train=True, data_augment=opt.augment, small=opt.small, category=opt.cls)
traindataloader = torch.utils.data.DataLoader(traindataset, batch_size = opt.batchSize, 
                                              shuffle=True, num_workers=opt.workers)

testdataset = getDataset(root=opt.dataDir, train=False, data_augment=False, small=opt.small, category=opt.cls)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size = opt.batchSize,
                                             shuffle=False, num_workers=opt.workers)

print('Train Dataset:', len(traindataset))
print('Test Dataset:', len(testdataset)) 

# ========================================================== #

# ===================CREATE network================================= #
#create network
network = AE_AtlasNet(num_points = opt.num_points, bottleneck_size=1024, nb_primitives=25)
network.cuda() #put network on GPU
network.apply(weights_init) #initialization of the weight

model_summary(network, True)

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
# ========================================================== #

# ===================CREATE optimizer================================= #
optimizer = optim.Adam(network.parameters(), lr = opt.lr, weight_decay=opt.wd)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_steps, gamma=opt.lr_decay) 
# scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay) 

# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
#meters to record stats on learning
train_loss = AverageValueMeter()
val_loss = AverageValueMeter()


with open(logname, 'a') as f: #open and append
        f.write('Train: ' + str(len(traindataset)) + ' Test: ' + str(len(testdataset)) +'\n')
        f.write(str(opt) + '\n')
        f.write(str(network) + '\n')


train_curve = []
val_curve = []
# ========================================================== #

#start of the learning loop

def train(ep):
    # training one epoch
    network.train()
    
    # if epoch==100:
        # optimizer = optim.Adam(network.parameters(), lr = lrate/10.0)
    
    for i, data in enumerate(traindataloader, 0):
        optimizer.zero_grad()
        
        points, Q, adj, normal, face_coords = data
        # points = points.transpose(2,1).contiguous()
        points = Variable(points.transpose(2,1).contiguous())
        points = points.cuda()
        Q = Variable(Q).cuda()
        adj = Variable(adj).cuda()
        normal = Variable(normal).cuda()
        face_coords = Variable(face_coords).cuda()
        
        pointsReconstructed  = network(points) #forward pass
        
        pointsReconstructed = pointsReconstructed.transpose(2,1).contiguous()
        points = points.transpose(2,1).contiguous()

        # dist1, dist2 = distChamfer(points, pointsReconstructed) #loss function
        # chamLoss = (torch.mean(dist1)) + (torch.mean(dist2))
        chamLoss, corres, _ = chamferLoss(points, pointsReconstructed, average=False)

        # cost = distEMD(points, pointsReconstructed) #loss function
        # emdLoss = torch.mean(cost)   # mean over batches

        # l1Loss = l1_loss(points, pointsReconstructed)

        corres = Variable(corres.type(torch.cuda.LongTensor))
        recon_vertices = torch.cat([torch.index_select(a, 0, ind).unsqueeze(0) for a, ind in zip(pointsReconstructed, corres)])
        pointsReconstructed = recon_vertices[:,:,:3].contiguous()
        
        quadLoss = quadratic_error_loss(Q, pointsReconstructed)
        sufNorLoss = surface_normal_loss(points, adj, pointsReconstructed, normal)
        sufLoss = surfaceLoss(pointsReconstructed, face_coords)

        # Total loss function
        loss_net = opt.chamLoss_wt * chamLoss
        # loss_net += opt.l1Loss_wt * l1Loss 
        # loss_net += opt.EMDLoss_wt * emdLoss 
        loss_net += opt.quadLoss_wt * quadLoss 
        loss_net += opt.sufNorLoss_wt * sufNorLoss 
        loss_net += opt.sufLoss_wt * sufLoss 

        train_loss.update(loss_net.data[0])
        
        loss_net.backward()
        
        optimizer.step() #gradient update
        
        # VIZUALIZE
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
            
        print('[%d: %d/%d] train loss:  %f; C: %f, Q: %f, N: %f, S: %f' %(ep, i, len(traindataloader), loss_net.data[0], chamLoss.data[0], quadLoss.data[0], sufNorLoss.data[0], sufLoss.data[0]))


def test(ep):
    # VALIDATION
    network.eval()
    disp = np.random.randint(len(testdataloader))
    
    # with torch.no_grad():
    for i, data in enumerate(testdataloader, 0):

        points, Q, adj, normal, face_coords = data
        # points = points.transpose(2,1).contiguous()
        points = Variable(points.transpose(2,1).contiguous(), volatile=True)
        points = points.cuda()
        Q = Variable(Q, volatile=True).cuda()
        adj = Variable(adj, volatile=True).cuda()
        normal = Variable(normal, volatile=True).cuda()
        face_coords = Variable(face_coords, volatile=True).cuda()

        pointsReconstructed  = network(points)
        
        pointsReconstructed = pointsReconstructed.transpose(2,1).contiguous()
        points = points.transpose(2,1).contiguous()

        # dist1, dist2 = distChamfer(points, pointsReconstructed) #loss function
        # chamLoss = (torch.mean(dist1)) + (torch.mean(dist2))
        chamLoss, corres, _ = chamferLoss(points, pointsReconstructed, average=False)

        # cost = distEMD(points, pointsReconstructed) #loss function
        # emdLoss = torch.mean(cost)   # mean over batches

        # l1Loss = l1_loss(points, pointsReconstructed)

        corres = Variable(corres.type(torch.cuda.LongTensor))
        recon_vertices = torch.cat([torch.index_select(a, 0, ind).unsqueeze(0) for a, ind in zip(pointsReconstructed, corres)])
        pointsReconstructed = recon_vertices[:,:,:3].contiguous()
        
        quadLoss = quadratic_error_loss(Q, pointsReconstructed)
        sufNorLoss = surface_normal_loss(points, adj, pointsReconstructed, normal)
        sufLoss = surfaceLoss(pointsReconstructed, face_coords)

        # Total loss function
        loss_net = opt.chamLoss_wt * chamLoss
        # loss_net += opt.l1Loss_wt * l1Loss 
        # loss_net += opt.EMDLoss_wt * emdLoss 
        # loss_net += opt.quadLoss_wt * quadLoss 
        loss_net += opt.sufNorLoss_wt * sufNorLoss 
        loss_net += opt.sufLoss_wt * sufLoss 

        val_loss.update(loss_net.data[0])
        
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

        print('[%d: %d/%d] val loss:  %f ' %(ep, i, len(testdataloader), loss_net.data[0]))


def main():
    
    best_val_loss = 1000000
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

        #UPDATE CURVES
        train_curve.append(train_loss.avg)
        val_curve.append(val_loss.avg)

       # Update the visom curves for both train and test
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

        # Update the visom curves for both train and test
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
        if best_val_loss > val_loss.avg:
            best_val_loss = val_loss.avg
            print('New best loss: ', best_val_loss)
            print('hence saving net ...')
            torch.save(network.state_dict(), os.path.join(dir_name,'best_net_'+str(epoch)+'.pth'))

        if (epoch+1) % opt.save_nth_epoch == 0:
            # print('saving net ...')
            torch.save(network.state_dict(), os.path.join(dir_name,'ae_net_'+str(epoch)+'.pth'))


        #dump stats in log file
        log_table = {
          "train_loss" : train_loss.avg,
          "val_loss" : val_loss.avg,
          "epoch" : epoch,
          "lr" : current_lr,
          "bestval" : best_val_loss,
        }

        # print(log_table)
        
        with open(logname, 'a') as f: #open and append
            f.write('json_stats: ' + json.dumps(log_table) + '\n')

        ##save last network
        #print('saving net...')
        #torch.save(network.state_dict(), '%s/network.pth' % (dir_name))


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_exec = round(time.time() - start_time)
    print('Total time taken: ', str(datetime.timedelta(seconds=time_exec)))
    print('-------Done-----------')      

