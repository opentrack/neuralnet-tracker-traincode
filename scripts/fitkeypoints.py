import h5py
import os
from os.path import join
import numpy as np
import progressbar
import argparse

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import neuralnets.modelcomponents


"""
This script adds parameters for the deformable model to datasets which have only
keypoints stored.

It does so by fitting the parametric model. The usual 68 keypoints are 
considered. Parameters include the rigid transformation.
"""


use_cuda = True

class HeadModel(nn.Module):
    """
        Covers a whole batch of heads.

        So there are `batchsize` sets of parameters.
    """
    def __init__(self, x, y, s, batchsize):
        super(HeadModel, self).__init__()
        self.out = neuralnets.modelcomponents.PoseOutputStage()
        self.head = neuralnets.modelcomponents.DeformableHeadKeypoints()
        self.params = nn.Parameter(torch.zeros((batchsize,3+4+50), dtype=torch.float32))
        self.params.data[:,3+3] = 1. # Identity rotation
        self.params.data[:,:3] = torch.tensor([x,y,s]).T
        self.use_shapeparams = False
        self.dummy = nn.Parameter(torch.zeros((batchsize,50),dtype=torch.float32))
        self.dummy.requires_grad = False

    def forward(self):
        """
            Outputs the keypoint.
        """
        x = self.params
        self.out(x[:,:7])
        self.shapeparams = x[:,3+4:]
        pts = self.head(self.shapeparams if self.use_shapeparams else self.dummy)
        pts = self.out.headcenter_to_screen_3d(pts)
        pts = pts.transpose(1,2)
        return pts


def deformpenalty(head, shapeparams):
    return torch.mean(torch.square(shapeparams), dim=1)


def find_poses(pt3d_68):
    """
    Fits a bunch of heads in parallel.
    """
    # Estimate initial guesses for position and size
    x, y = np.mean(pt3d_68, axis=2)[:,:2].T
    s = np.mean(np.std(pt3d_68, axis=2), axis=1)
    
    headmodel = HeadModel(x, y, s, pt3d_68.shape[0])
    headmodel.train()
    
    pt3d_68 = torch.from_numpy(pt3d_68).to(torch.float32)
    s = torch.from_numpy(s).to(torch.float32)
    
    if use_cuda:
        headmodel.cuda()
        pt3d_68 = pt3d_68.cuda()
        s = s.cuda()
    
    def eval():
        out = headmodel.forward()
        loss = torch.mean(F.mse_loss(out, pt3d_68, reduction='none'), dim=[1,2])/(s*s)
        if headmodel.use_shapeparams:
            loss += 0.1*deformpenalty(headmodel.head, headmodel.shapeparams)
        return torch.sum(loss)
    
    # First run without shape deformation so the shape parameter is properly used.
    # If not done then the shape deformation will be used to adjust the head size
    # while distorting the details. This is not what we want.

    # One optimizer is okay to treat an entire batch. It should be equivalent to
    # running one optimizer per head since I think it works element wise on the
    # gradient vector and the parameters, and also the derivative of the total
    # loss w.r.t. some head parameter is the same as the individual head's loss.
    optimizer = optim.Adam(headmodel.parameters(), 1.)
    for i in range(1, 200):
            optimizer.zero_grad()
            l = eval()
            l.backward()
            optimizer.step(eval)
            #print (i,l, headmodel.out.head_possize, headmodel.out.head_quat)
            #print (i, l)

    # With a good initial guess we can use the shape deformation, too.
    headmodel.use_shapeparams = True
    optimizer = optim.Adam(headmodel.parameters(), 1.)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
    for i in range(1, 400):
            optimizer.zero_grad()
            l = eval()
            l.backward()
            optimizer.step(eval)
            scheduler.step()
            #print (i, l)
            #print (i,l, headmodel.out.head_possize, headmodel.out.head_quat)

    pts = headmodel.forward()
    coord = headmodel.out.head_possize
    quat  = headmodel.out.head_quat
    shapeparams = headmodel.shapeparams
    
    pts = pts.detach().cpu().numpy()
    coord = coord.detach().cpu().numpy()
    quat = quat.detach().cpu().numpy()
    shapeparams = shapeparams.detach().cpu().numpy()

    return coord, quat, shapeparams, pts


def fitall(args):
    with h5py.File(args.filename, 'r+') as f:
        N = f['pt3d_68'].shape[0]
        cs = min(N, 1024)
        if not 'quats' in f:
            ds_quats = f.create_dataset('quats', (N,4), chunks=(cs,4), maxshape=(N,4), dtype='f4')
            ds_coords = f.create_dataset('coords', (N,3), chunks=(cs,3), maxshape=(N,3), dtype='f4')
            ds_shapeparams = f.create_dataset('shapeparams', (N,50), chunks=(cs,50), maxshape=(N,50), dtype='f4')
            ds_quats[...] = np.nan
            ds_coords[...] = np.nan
            ds_shapeparams[...] = np.nan
        else:
            ds_quats = f['quats']
            ds_coords = f['coords']
            ds_shapeparams = f['shapeparams']
        batchsize = 1024*10
        with progressbar.ProgressBar() as bar:
            for i in bar(range(0, N, batchsize)):
                if not args.force and np.all(np.isfinite(ds_quats[i:i+batchsize])) and np.all(np.isfinite(ds_coords[i:i+batchsize])) and np.all(np.isfinite(ds_shapeparams[i:i+batchsize])):
                    continue
                pt3d_68 = np.array(f['pt3d_68'][i:i+batchsize])
                coord, quat, shapeparams, fitpt3d = find_poses(pt3d_68)
                ds_quats[i:i+batchsize] = quat
                ds_coords[i:i+batchsize] = coord
                ds_shapeparams[i:i+batchsize] = shapeparams


# def plot(fitpt3d, pt3d_68):
#     pyplot.close('all')
#     fig = pyplot.figure(figsize=(10,10))
#     ax = fig.add_subplot(111, projection='3d')

#     xs, ys, zs = fitpt3d
#     ax.scatter(xs, ys, zs, s=3., color='r')
    
#     xs, ys, zs = pt3d_68
#     ax.scatter(xs, ys, zs, s=3., color='k')

#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')

#     pyplot.show()


# def fitdebug():
#     filename = join(os.environ['DATADIR'],'aflw2k.h5')
#     with h5py.File(filename, 'r+') as f:
#         i = 0
#         batchsize = 1024*10
#         pt3d_68 = np.array(f['pt3d_68'][i:i+batchsize])
#         coord, quat, shapeparams, fitpt3d = find_poses(pt3d_68)

#         print ("Coord ", coord[i], "\nQuat ", quat[i]) #, "\nShape ", shapeparams)
#         print ('TrueCoord:',f['coords'][i])
#         print ('TrueQuat',f['quats'][i])
#         print ('TrueParams',f['shapeparams'][i])
#         for j in range(i,i+batchsize):
#             plot(fitpt3d[i], pt3d_68[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fits head model to keypoints in dataset")
    parser.add_argument('filename', help="dataset file", type=str)
    parser.add_argument('-f', '--force', help="Force overwrite existing data", default=False, type=bool)
    fitall(parser.parse_args())
