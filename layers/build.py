import torch
from .perceptual_loss import Perceptual_loss
from .temporal_loss import Temporal_loss

def make_loss():
    return Perceptual_loss()

def make_temp_loss():
    return Temporal_loss()
