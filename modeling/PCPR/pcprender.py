import torch
from ..UNet import UNet
from layers.pcpr_layer import PCPRModel
import pcpr
import numpy as np 
import matplotlib.pyplot as plt
import cv2


class PCPRender(torch.nn.Module):
    def __init__(self, feature_dim, tar_width, tar_height, layer_num, depth_thr, dataset = None):
        super(PCPRender, self).__init__()
        self.feature_dim = feature_dim
        self.tar_width = tar_width
        self.tar_height = tar_height
        self.layer_num = layer_num
        self.depth_thr = depth_thr

        self.absorb = torch.tensor(1.0, requires_grad=True)

        add_rgb_input = 0

        if dataset is not None:
            add_rgb_input = 3

        self.pcpr_layer = PCPRModel(tar_width, tar_height)
        self.unet = UNet(feature_dim  + 3 + add_rgb_input, # input channel: feature[feature_dim] + depth[1] + viewin directions[3] + %%%points color[3]%%%(no used for now)
                #   4) # output channel: 3 RGB 
                  3, 1) # output channel: 3 RGB 1
        self.unet = self.unet.cuda()

        self.dataset = dataset

        # generate meshgrid
        xh, yw = torch.meshgrid([torch.arange(0,tar_height), torch.arange(0,tar_width)])
        self.coord_meshgrid = torch.stack([yw, xh, torch.ones_like(xh)],dim =0).float()
        self.coord_meshgrid = self.coord_meshgrid.view(1,3,-1)
        self.coord_meshgrid = self.coord_meshgrid.cuda()






    def forward(self, point_features, default_features,
           point_clouds,
           cam_intrinsic, cam_extrinsic, 
           near_far_max_splatting_size, num_points, inds = None):

        batch_num = cam_intrinsic.size(0)

        pcs = point_clouds
        pfs = point_features
        rgbas = []
        deps = []
        layer_loop = 6#self.layer_num
        for layer in range(layer_loop):
            # out_feature (batch, feature_dim, tar_height, tar_width )
            # out_depth (batch, 1, tar_height, tar_width )
            nps = torch.tensor(pcs.size(0)).cuda()
            out_feature, out_depth, out_ind = self.pcpr_layer(pfs, default_features,
                                    pcs,
                                    cam_intrinsic, cam_extrinsic, 
                                    near_far_max_splatting_size, nps.unsqueeze(0))

            # generate viewin directions

            #out_ind: H W
            #point_features: 32, ps
            #pcs: ps, 3

            deps.append(out_depth)
                

            Kinv = torch.inverse(cam_intrinsic)
            coord_meshgrids = self.coord_meshgrid.repeat(batch_num,1,1)
            dir_in_camera = torch.bmm(Kinv, coord_meshgrids)
            point_in_cam = torch.cat([dir_in_camera*out_depth.view(1,1,-1), torch.ones(batch_num,1,dir_in_camera.size(2)).cuda()],dim = 1)
            dir_in_camera = torch.cat([dir_in_camera, torch.ones(batch_num,1,dir_in_camera.size(2)).cuda()],dim = 1)
            dir_in_world = torch.bmm(cam_extrinsic, dir_in_camera)#~~~!!!
            point_in_world = torch.bmm(cam_extrinsic, point_in_cam)
            dir_in_world = dir_in_world / dir_in_world[:,3:4,:].repeat(1,4,1)
            dir_in_world = dir_in_world[:,0:3,:]
            point_in_world = point_in_world[:,:3,:]/point_in_world[:,3:,:]
            
            if layer == 0:
                ind_o = out_ind
                piw = point_in_world

            dir_in_world = torch.nn.functional.normalize(dir_in_world, dim=1)
            dir_in_world = dir_in_world.reshape(batch_num,3,self.tar_height,self.tar_width)
            
            #set direction to zeros for depth==0
            depth_mask = out_depth.repeat(1,3,1,1)
            dir_in_world[depth_mask==0] = 0

            # fuse all features
            fused_features = torch.cat([out_feature,dir_in_world],dim = 1)

            # rendering
            x = self.unet(fused_features)

            if layer != 0:
                ddeps = ((deps[layer]-deps[layer-1])[0][0]).detach()
                sec = ddeps>0.05 #self.depth_thr
                deps[layer][:,:,sec] = 0
                x[:, 3, sec] = 0

            rgbas.append(x)
            out_ind[out_ind<0] = pcs.size(0)
            index = torch.ones(pcs.size(0)+1).bool()
            index[out_ind.long()] = False
            index = index[:-1]
            pcs = pcs[index]
            pfs = pfs[:, index]
        
        rgbas = torch.cat(rgbas, dim=0) # layer, 4, H, W
        deps = torch.cat(deps, dim=0) # layer, 1, H, W

        if layer_loop == 1:
            return x, deps[0].unsqueeze(0), ind_o, piw[0], None
        else:
            # alpha blending
            alphas = rgbas[:,3:4,:,:]
            #alphas[deps==0] = -1
            alphas = torch.clamp(alphas, 0, 1.0)
            cump = torch.cumprod((1.0-alphas), dim = 0)[:-1]
            ones_cat = torch.ones(1,1,deps.size(2),deps.size(3)).cuda()
            alphas = alphas * torch.cat((ones_cat, cump), dim = 0)
            alpha = torch.sum(alphas, dim = 0)
            rgb = torch.sum(rgbas[:,:3,:,:]*alphas, dim = 0)
            res = torch.cat((rgb, alpha), dim = 0)
            
            # t = torch.flatten(rgb).detach().cpu().numpy()
            # plt.figure(); plt.hist(t); plt.show()
            # t = torch.flatten(alpha).detach().cpu().numpy()
            # plt.figure(); plt.hist(t); plt.show()
            return res.unsqueeze(0), deps[0].unsqueeze(0), ind_o, piw[0], None
