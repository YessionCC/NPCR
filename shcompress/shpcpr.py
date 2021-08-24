import torch
from modeling.UNet import UNet
from shpcprlayer import PCPRModel
import pcpr
import numpy as np 
import matplotlib.pyplot as plt

class PCPRender(torch.nn.Module):
    def __init__(self, feature_dim, tar_width, tar_height, layer_num, dataset = None):
        super(PCPRender, self).__init__()
        self.feature_dim = feature_dim
        self.tar_width = tar_width
        self.tar_height = tar_height
        self.layer_num = layer_num

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

        out_feature, out_depth, out_ind = self.pcpr_layer(point_features, default_features,
                                point_clouds,
                                cam_intrinsic, cam_extrinsic, 
                                near_far_max_splatting_size, num_points)

        Kinv = torch.inverse(cam_intrinsic)
        coord_meshgrids = self.coord_meshgrid.repeat(batch_num,1,1)
        dir_in_camera = torch.bmm(Kinv, coord_meshgrids)
        dir_in_camera = torch.cat([dir_in_camera, torch.ones(batch_num,1,dir_in_camera.size(2)).cuda()],dim = 1)
        dir_in_world = torch.bmm(cam_extrinsic, dir_in_camera)#~~~!!!
        dir_in_world = dir_in_world / dir_in_world[:,3:4,:].repeat(1,4,1)
        dir_in_world = dir_in_world[:,0:3,:]
        dir_in_world = torch.nn.functional.normalize(dir_in_world, dim=1)
        dir_in_world = dir_in_world.reshape(batch_num,3,self.tar_height,self.tar_width)
        
        #set direction to zeros for depth==0
        thetas = torch.arccos(dir_in_world[:,2:3,:,:])
        phis = torch.arctan(dir_in_world[:,1:2,:,:]/dir_in_world[:,:1,:,:])
        sphere_dir_world = torch.cat((thetas, phis), dim = 1)
        depth_mask = out_depth.repeat(1,3,1,1)
        dir_in_world[depth_mask==0] = 0

        # fuse all features
        fused_features = torch.cat([out_feature,dir_in_world],dim = 1)

        # rendering
        x = self.unet(fused_features)
        
        return x, out_ind, sphere_dir_world