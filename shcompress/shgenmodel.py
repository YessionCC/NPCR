

from modeling.UNet import UNet
from modeling.PCPR import PCPRender
from modeling.PCPR import PCPRParameters
from shmodel import SH_Model





def build_model(cfg, vertex_list, dataset = None):

    model = SH_Model(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1], cfg.INPUT.LAYER_NUM,
                 vertex_list, cfg.MODEL.FEATURE_DIM, dataset =dataset, use_rgb = cfg.INPUT.USE_RGB)

    return model
