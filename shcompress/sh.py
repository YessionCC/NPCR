import torch
import numpy as np
import torch.nn.functional

pi4 = 12.566370614359172
pi2 = 6.283185307179586
sqrt2 = 1.4142135623730951

def P(l,m,x):
  pmm = torch.ones_like(x)
  if m>0 :
    somx2 = torch.sqrt((1.0-x)*(1.0+x))
    fact = 1.0
    for _ in range(m):
        pmm *= (-fact) * somx2
        fact += 2.0
  if l==m:
    return pmm
  pmmp1 = x * (2.0*m+1.0) * pmm
  if l==m+1:
    return pmmp1
  pll = 0.0
  for ll in range(m+2, l+1):
    pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m)
    pmm = pmmp1
    pmmp1 = pll
  return pll

def fac_div(a, b) :
  res = 1.0
  for i in range(a+1, b+1):
    res/=i
  return res

def K(l, m) :
  temp = (2.0*l+1.0) / pi4 * fac_div(l-m, l+m)
  return np.sqrt(temp)

def SH(l, m, theta, phi):
  if m==0:
    return K(l,0)*P(l,m,torch.cos(theta))
  elif m>0:
    return sqrt2*K(l,m)*torch.cos(m*phi)*P(l,m,torch.cos(theta))
  else:
    return sqrt2*K(l,-m)*torch.sin(-m*phi)*P(l,-m,torch.cos(theta))


class SH_encoder(object):
  def __init__(self, lmax, pnum):
    super().__init__()
    self.lmax = lmax
    self.coenum = lmax*lmax
    self.shcoes = torch.zeros(4, self.coenum, pnum+1).cuda()
    self.shcount = torch.zeros(pnum+1).cuda()


  # out_ind: H, W
  # spd: 1, 2, H, W
  # rgba: 1, 4, H, W
  @torch.no_grad() ##important!!! pytorch auto allow calc. calcgraph, it will cause memexp
  def adddata(self, inds, sphdir, rgba):
    shs = []
    for l in range(self.lmax):
      for m in range(-l, l+1):
        shs.append(SH(l, m, sphdir[:,0,:,:], sphdir[:,1,:,:]))
    shs = torch.cat(shs).unsqueeze(0).repeat(4,1,1,1)

    inds = inds.squeeze(0)
    self.shcount[torch.flatten(inds).long()]+=1
    rgba_t = rgba.squeeze(0).unsqueeze(1).repeat(1, self.coenum, 1, 1)
    self.shcoes[:,:,inds.long()]+=rgba_t*shs
  
  @torch.no_grad()
  def encode(self):
    self.shcoes*=pi2/torch.max(torch.tensor(1e-6).cuda(), self.shcount)

  @torch.no_grad()
  def restruct(self, inds, sphdir):
    shs = []
    for l in range(self.lmax):
      for m in range(-l, l+1):
        shs.append(SH(l, m, sphdir[:,0,:,:], sphdir[:,1,:,:]))
    shs = torch.cat(shs).unsqueeze(0).repeat(4,1,1,1) #4, 36, H, W
    rgba = self.shcoes[:,:,inds.squeeze(0).long()]*shs
    rgba = torch.sum(rgba, dim = 1)
    return rgba

  def save(self, path):
    torch.save({'coe': self.shcoes, 'cnt': self.shcount, 'lmax': self.lmax, 'cn': self.coenum}, 
      path, 
      _use_new_zipfile_serialization=False)

  def read(self, path):
    data = torch.load(path)
    self.shcount = data['cnt'].cuda()
    self.shcoes = data['coe'].cuda()
    self.coenum = data['cn']
    self.lmax = data['lmax']
