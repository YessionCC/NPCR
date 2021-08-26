
import os
import sys
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
from layers import make_temp_loss
from utils.logger import setup_logger
from data.datasets.utils import campose_to_extrinsic, read_intrinsics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from tqdm import trange
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
opt_file = 'nr_optimizer_%s.pth' % epoch

writer = SummaryWriter(log_dir=os.path.join(model_path,'tensorboard_test'))
test_loader, vertex_list,dataset = make_data_loader(cfg, is_train=False)

model = build_model(cfg, vertex_list)
optimizer = make_optimizer(cfg, model)
model = torch.load(os.path.join(model_path,para_file))
optimizer = torch.load(os.path.join(model_path,opt_file))
model.train()
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model = model.cuda()
#model = amp.initialize(models=model)

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

Tpre = []; Kpre = []; rgba_pre = []; rgba_ic = None

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
    res = F.grid_sample(rgbax.cuda().unsqueeze(0), pw, 'nearest')[0].half()
    sub = torch.sum(torch.square(rgba_std-res), 0)
    sec = sub>cfg.WEIGHTS.COLOR_THR
    res[:, sec]=rgba_std[:, sec]
    res[res==0]=rgba_std[res==0]
    return res

loss_fn = make_temp_loss()
rws = torch.tensor(cfg.WEIGHTS.REFINE_WEIGHTS).cuda().float()

for ID in trange(camNum):
    T = Ts[ID:ID+1,:,:].cuda()
    K = Ks[ID:ID+1,:,:].cuda()
    res,depth,inds,pw = model(point_indexes, in_points, K, T,
                        near_far_max_splatting_size, num_points,target)

    rgba_o=res[0][:3,:,:]
    rgba = rgba_o
    depth = depth[0][0]
    T = T[0];K = K[0]

    if len(rgba_pre)>=1:
        H, W = depth.shape
        res1 = getMap(H,W,Tpre[0], Kpre[0], rgba_pre[0], pw, rgba_o)
        #res2 = getMap(H,W,Tpre[1], Kpre[1], rgba_pre[1], pw, rgba_o)
        # res3 = getMap(H,W,Tpre[2], Kpre[2], rgba_pre[2], pw, rgba_o)
        # res4 = getMap(H,W,Tpre[3], Kpre[3], rgba_pre[3], pw, rgba_o)
        # res5 = getMap(H,W,Tpre[4], Kpre[4], rgba_pre[4], pw, rgba_o)
        res_ic = getMap(H,W,Tpre[0], Kpre[0], rgba_ic, pw, rgba_o)

        res = res_ic*rws[0]+res1*rws[1]
        rgba = rgba_o*rws[2]+res*rws[3]

        loss1, loss2 = loss_fn(rgba_o.unsqueeze(0), res.unsqueeze(0))


        l = loss1 + loss2
        print('refine loss: {}'.format(l.item()))
        #l.backward()
        with amp.scale_loss(l, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

        Tpre, Kpre, rgba_pre = Tpre[1:], Kpre[1:], rgba_pre[1:]
        del loss1, loss2, l, res1, res_ic
        
    Tpre.append(T) 
    Kpre.append(K)
    rgba_pre.append(rgba_o.detach().cpu())# save memory
    rgba_ic = rgba.detach().cpu()
    del res,depth,inds,pw
    del rgba, rgba_o
    


torch.save(model, os.path.join(model_path,para_file))
torch.save(optimizer, os.path.join(model_path,opt_file))
