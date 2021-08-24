import argparse
import os
import sys
from os import mkdir
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('..')
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from shgenmodel import build_model
from solver import make_optimizer, WarmupMultiStepLR
from layers import make_loss
from utils.logger import setup_logger
from data.datasets.utils import campose_to_extrinsic, read_intrinsics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from apex import amp
import cv2
from sh import SH_encoder
import pcpr
torch.cuda.set_device(0)

model_path='pretrained'
epoch = 'ec1'
camposFile=os.path.join('pretrained','CamPose.inf')
intriFile=os.path.join('pretrained','Intrinsic.inf')

cfg.merge_from_file(os.path.join(model_path,'config.yml'))
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.freeze()

H, W = cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]

test_loader, vertex_list,dataset = make_data_loader(cfg, is_train=False)

for batch in test_loader:
    in_points = batch[1].cuda()
    K = batch[2].cuda()
    T = batch[3].cuda()
    near_far_max_splatting_size = batch[5]
    num_points = batch[4]
    point_indexes = batch[0]
    break
    
shencoder = SH_encoder(6, num_points.int().item())
shencoder.read('./output/shec2.ec')
shencoder.encode()
picScale=cfg.INPUT.SIZE_TEST[0]/cfg.INPUT.SIZE_RAW[0]## very very important
# when training, size scaled and intrinsic also scaled
# so when rendering, same opertion should be done
# picScale=720/1920

camposes = np.loadtxt(camposFile)
Ts = torch.Tensor( campose_to_extrinsic(camposes) )
camNum = Ts.size(0)

Ks = read_intrinsics(intriFile)
for i in range(Ks.shape[0]):
    Ks[i,0:2,:]=Ks[i,0:2,:]*picScale
Ks = torch.Tensor(Ks)
Ks = Ks.repeat(camNum, 1, 1)

if not os.path.exists(os.path.join(model_path,'res_%s/rgba'%epoch)):
    os.mkdir(os.path.join(model_path,'res_%s/rgba'%epoch))

def PCPR1(point_clouds,
        cam_intrinsic, cam_extrinsic, 
        near_far_max_splatting_size, num_points, tar_image_size):
    
    # global ps_count #
    # if ps_count == None: #
    #     ps_count = torch.zeros((num_points.int().item()+10,), dtype=torch.int32)


    batch_size = cam_intrinsic.size(0)

    if cam_extrinsic.size(0) != batch_size or near_far_max_splatting_size.size(0)!=batch_size or\
        num_points.size(0)!= batch_size:
        raise Exception('[PCPR] batch_sizes are not consistant.')


    _cam_extrinsic = torch.cat([cam_extrinsic[:,0:3,2], cam_extrinsic[:,0:3,0],
                                cam_extrinsic[:,0:3,1],cam_extrinsic[:,0:3,3]],dim = 1)

    
    tar_width, tar_heigh = int(tar_image_size[0].item()), int(tar_image_size[1].item())


    out_depth = torch.zeros(batch_size,1, tar_heigh, tar_width).cuda()
    out_index = torch.zeros(batch_size, tar_heigh, tar_width, dtype=torch.int32).cuda()

    _num_points = num_points.int().tolist()

    beg = 0

    for i in range(batch_size):
        
        #print('Start Kernel.',flush = True)
        out_depth[i][0], out_index[i] = pcpr.forward(point_clouds[beg:beg+_num_points[i],:],
                cam_intrinsic[i], _cam_extrinsic[i], out_depth[i][0], out_index[i],
                *(near_far_max_splatting_size[i].tolist()) )


        out_index[i] = out_index[i]-1
        out_index[i][out_index[i]<0] = _num_points[i]

        beg = beg + _num_points[i]

    out_index = out_index.int()

    return out_depth, out_index

xh, yw = torch.meshgrid([torch.arange(0,W), torch.arange(0,H)])
coord_meshgrid = torch.stack([yw, xh, torch.ones_like(xh)],dim =0).float()
coord_meshgrid = coord_meshgrid.view(1,3,-1)
coord_meshgrid = coord_meshgrid.cuda()

def PCPR(point_clouds,
        cam_intrinsic, cam_extrinsic, 
        near_far_max_splatting_size, num_points, H, W, inds = None):

    batch_num = cam_intrinsic.size(0)

    out_depth, out_ind = PCPR1(point_clouds, cam_intrinsic, cam_extrinsic,
        near_far_max_splatting_size, num_points, tar_image_size=torch.tensor([H,W]).int())

    Kinv = torch.inverse(cam_intrinsic)
    coord_meshgrids = coord_meshgrid.repeat(batch_num,1,1)
    dir_in_camera = torch.bmm(Kinv, coord_meshgrids)
    dir_in_camera = torch.cat([dir_in_camera, torch.ones(batch_num,1,dir_in_camera.size(2)).cuda()],dim = 1)
    dir_in_world = torch.bmm(cam_extrinsic, dir_in_camera)#~~~!!!
    dir_in_world = dir_in_world / dir_in_world[:,3:4,:].repeat(1,4,1)
    dir_in_world = dir_in_world[:,0:3,:]
    dir_in_world = torch.nn.functional.normalize(dir_in_world, dim=1)
    dir_in_world = dir_in_world.reshape(batch_num,3,W,H)
    
    #set direction to zeros for depth==0
    thetas = torch.arccos(dir_in_world[:,2:3,:,:])
    phis = torch.arctan(dir_in_world[:,1:2,:,:]/dir_in_world[:,:1,:,:])
    sphere_dir_world = torch.cat((thetas, phis), dim = 1) #1, 2,H, W
    
    return shencoder.restruct(out_ind, sphere_dir_world)

for ID in range(200):
    T = Ts[ID:ID+1,:,:].cuda()
    K = Ks[ID:ID+1,:,:].cuda()
    res = PCPR(in_points, K, T,
                        near_far_max_splatting_size, num_points,H, W)
    # vdep = depth[0].detach().cpu().clone()
    # vdep = vdep.permute(1,2,0)
    # cv2.imshow('dep', vdep.numpy())
    # cv2.waitKey(0)
    res = res.unsqueeze(0)
    img_t = res.detach().cpu()[0]
    mask_t = img_t[3:4,:,:]
    img = cv2.cvtColor(img_t.permute(1,2,0).numpy()*255.0,cv2.COLOR_BGR2RGB)
    mask = mask_t.permute(1,2,0).numpy()*255.0
    rgba=img*mask/255.0+(255.0-mask)
    
    # cv2.imwrite(os.path.join(model_path,'res_%s/rgb/img_%04d.jpg'%(epoch,ID+1)),img)
    # cv2.imwrite(os.path.join(model_path,'res_%s/alpha/img_%04d.jpg'%(epoch,ID+1)  ),mask*255)
    cv2.imwrite(os.path.join(model_path,'res_ec1/rgba/img_%04d.jpg'%(ID+1)  ),rgba)

print('Render done.')


