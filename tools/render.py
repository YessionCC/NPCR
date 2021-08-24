import argparse
from ast import NodeTransformer
import os
from re import U
import sys
from os import mkdir
import numpy as np
import torch
from torch._C import dtype
import torch.nn.functional as F

sys.path.append('..')
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling.build import build_model
from solver import make_optimizer, WarmupMultiStepLR
from layers import make_loss
from utils.logger import setup_logger
from data.datasets.utils import campose_to_extrinsic, read_intrinsics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from apex import amp
import cv2
torch.cuda.set_device(0)

model_path='models'
cfg.merge_from_file(os.path.join(model_path,'config.yml'))
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.freeze()

epoch = str(cfg.LOAD_EPOCH)
camposFile=os.path.join('models','CamPose.inf')
intriFile=os.path.join('models','Intrinsic.inf')
para_file = 'nr_model_%s.pth' % epoch

writer = SummaryWriter(log_dir=os.path.join(model_path,'tensorboard_test'))
test_loader, vertex_list,dataset = make_data_loader(cfg, is_train=False)

#model = build_model(cfg, vertex_list)
model = torch.load(os.path.join(model_path,para_file))
model = model.cuda()
model = amp.initialize(models=model)

for batch in test_loader:
    in_points = batch[1].cuda()
    K = batch[2].cuda()
    T = batch[3].cuda()
    near_far_max_splatting_size = batch[5]
    num_points = batch[4]
    point_indexes = batch[0]
    target = batch[-1].cuda()
    break

picScale=cfg.INPUT.SIZE_TEST[0]/cfg.INPUT.SIZE_RAW[0]
camposes = np.loadtxt(camposFile)
Ts = torch.Tensor( campose_to_extrinsic(camposes) )
camNum = Ts.size(0)

Ks = read_intrinsics(intriFile)
for i in range(Ks.shape[0]):
    Ks[i,0:2,:]=Ks[i,0:2,:]*picScale
Ks = torch.Tensor(Ks)
Ks = Ks.repeat(camNum, 1, 1)

if not os.path.exists(os.path.join(cfg.OUTPUT_DIR,'res_%s'%epoch)):
    os.mkdir(os.path.join(cfg.OUTPUT_DIR,'res_%s'%epoch))

Tpre = []; Kpre = []; rgba_pre = []; rgba_ic = None
rws = torch.tensor(cfg.WEIGHTS.RENDER_WEIGHTS).float()

def getMap(H,W,Tx,Kx,rgbax,pw,rgba_std):
    pw = pw.view(3,H,W)
    pw = torch.matmul(torch.inverse(Tx), torch.cat((pw, torch.ones(1,H,W).cuda()), 0).reshape(4,-1))
    pw = pw[:3,:]/pw[3:,:]
    pw = torch.matmul(Kx, pw)
    pw = pw[:2,:]/pw[2:,:]
    pw = pw.reshape(2,H,W)
    pw = pw.permute(1,2,0).reshape(1,H,W,2)
    pw[...,0]=pw[...,0]/W*2 - 1
    pw[...,1]=pw[...,1]/H*2 - 1   
    res = F.grid_sample(rgbax.unsqueeze(0), pw, 'nearest')[0]
    sub = torch.sum(torch.square(rgba_std-res), 0)
    sec = sub>cfg.WEIGHTS.COLOR_THR
    res[:, sec]=rgba_std[:, sec]
    res[res==0]=rgba_std[res==0]
    return res

for ID in range(camNum):
    T = Ts[ID:ID+1,:,:].cuda()
    K = Ks[ID:ID+1,:,:].cuda()
    res,depth,inds,pw = model(point_indexes, in_points, K, T,
                        near_far_max_splatting_size, num_points,target)
    
    img_t = res.detach().cpu()[0]
    mask_t = img_t[3:4,:,:]
    img = cv2.cvtColor(img_t.permute(1,2,0).numpy()*255.0,cv2.COLOR_BGR2RGB)
    bkgc = cfg.BACKGROUND_GRAY
    mask = mask_t.permute(1,2,0).numpy()*bkgc
    rgba=img*mask/bkgc+(bkgc-mask)
    rgba = np.clip(rgba, 0, 255)

    depth = depth[0][0]
    T = T[0];K = K[0]
    rgba_cuda = torch.tensor(rgba).cuda().permute(2,0,1)
    if len(rgba_pre)>=5:
        H, W = depth.shape
        res1 = getMap(H,W,Tpre[0], Kpre[0], rgba_pre[0], pw, rgba_cuda)
        res2 = getMap(H,W,Tpre[1], Kpre[1], rgba_pre[1], pw, rgba_cuda)
        res3 = getMap(H,W,Tpre[2], Kpre[2], rgba_pre[2], pw, rgba_cuda)
        res4 = getMap(H,W,Tpre[3], Kpre[3], rgba_pre[3], pw, rgba_cuda)
        res5 = getMap(H,W,Tpre[4], Kpre[4], rgba_pre[4], pw, rgba_cuda)
        res_ic = getMap(H,W,Tpre[4], Kpre[4], rgba_ic, pw, rgba_cuda)

        res = res_ic*rws[1]+res5*rws[2]+res4*rws[3]+\
            res3*rws[4]+res2*rws[5]+res1*rws[6]
        rgba_res = rgba_cuda*rws[0]+res

        sec = depth == 0
        rgba_res[:, sec]=rgba_cuda[:, sec]

        rgba = rgba_res.permute(1,2,0).detach().cpu().numpy()
        Tpre, Kpre, rgba_pre = Tpre[1:], Kpre[1:], rgba_pre[1:]
        
    Tpre.append(T) 
    Kpre.append(K)
    rgba_pre.append(rgba_cuda)#
    rgba_ic = torch.tensor(rgba).cuda().permute(2,0,1)
    cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,'res_%s/img_%04d.jpg'%(epoch,ID+1)),rgba)

    del res,depth,inds,pw
    del img_t, mask_t, img, mask, rgba

print('Render done.')


