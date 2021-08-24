import imageio
import os

image_list = []
#path = './pretrained/res_1/rgba'
path = '~/Code/Py/Neural-Point-Cloud-Rendering-via-Multi-Plane-Projection/ScanNet_npcr_scene0010_00/Test_Result'
ldirs = os.listdir(path)
for file in ldirs:
  filep = os.path.join(path, file)
  image_list.append(filep)

image_list = sorted(image_list)
frames = []

for image_name in image_list:
  frames.append(imageio.imread(image_name))

imageio.mimsave('Hair_rot.gif', frames, 'GIF', duration = 0.02)