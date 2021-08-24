import numpy as np

path = './ExParamGen/CamPose.inf'
pathout = './pretrained/CamPose.inf'
mat = np.loadtxt(path)

def getrotz(a):
  return np.array([[np.cos(a), -np.sin(a), 0,0], [np.sin(a), np.cos(a), 0,0], [0,0,1,0], [0,0,0,1]])
def getroty(a):
  return np.array([[np.cos(a), 0, np.sin(a),0], [0,1,0,0], [-np.sin(a),0,np.cos(a),0], [0,0,0,1]])
def getrotx(a):
  return np.array([[1,0,0,0], [0, np.cos(a), -np.sin(a),0], [0, np.sin(a), np.cos(a),0],[0,0,0,1]])

def genMove():
  res = []
  for m in np.arange(-2, 1, 0.01):
    cmat = mat.copy()
    cmat[9]+=m
    res.append(cmat)
  return np.array(res)

def genrot(num):
  ret = []
  for _ in range(num):
    rs = np.random.rand(3)*2*np.pi
    tr = np.zeros((4,4))
    tr[:3,2]=mat[:3].T
    tr[:3,0]=mat[3:6].T
    tr[:3,1]=mat[6:9].T
    tr[:3,3]=mat[9:12].T
    tr[3,3] = 1.0
    ro = np.matmul(getrotz(rs[2]),np.matmul(getrotx(rs[0]), getroty(rs[1])))
    tr =np.matmul(ro, tr)
    res = np.zeros(12)
    res[:3]=tr[:3,2].T
    res[3:6]=tr[:3,0].T
    res[6:9]=tr[:3,1].T
    res[9:12]=tr[:3,3].T
    ret.append(res)
  return np.array(ret)

def genrotcn(num):
  ret = []
  for a in np.linspace(0, 2*np.pi, num):
    tr = np.zeros((4,4))
    tr[:3,2]=mat[:3].T
    tr[:3,0]=mat[3:6].T
    tr[:3,1]=mat[6:9].T
    tr[:3,3]=mat[9:12].T
    tr[3,3] = 1.0
    tr =np.matmul(getrotz(a), tr)
    res = np.zeros(12)
    res[:3]=tr[:3,2].T
    res[3:6]=tr[:3,0].T
    res[6:9]=tr[:3,1].T
    res[9:12]=tr[:3,3].T
    ret.append(res)
  return np.array(ret)

np.savetxt(pathout, genrotcn(500), '%1.8f')