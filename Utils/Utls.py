import  os
import numpy as np
import nibabel as nib
from .Loss import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

#空间变形网络（将形变场施加到原图像上）
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow, return_phi=False):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if return_phi:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode), new_locs
        else:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

"""
def load_nii(path):
    X = nib.load(path)
    X = X.get_fdata()
    return X
"""

def save_nii(img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(img, affine, header=None)
    nib.save(new_img, savename)

def generate_grid3D_tensor(shape):
    x_grid = torch.linspace(-1., 1., shape[0])
    y_grid = torch.linspace(-1., 1., shape[1])
    z_grid = torch.linspace(-1., 1., shape[2])
    x_grid, y_grid, z_grid = torch.meshgrid(x_grid, y_grid, z_grid)

    # Note that default the dimension in the grid is reversed:
    # z, y, x
    grid = torch.stack([z_grid, y_grid, x_grid], dim=0)
    return grid


#dice计算
def dice(array1, array2, labels):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    """
    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


#填充0
def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


#图像 normalization
def preprocess_Normalizition(img,method="min_max"):
    img=img.astype(np.float32)
    if method=="min_max":
        front_img=img[img>0]
        min_value=np.min(front_img)
        max_value=np.max(front_img)
        img=(img-min_value)/(max_value-min_value)
        img[img<0]=0.0
    elif method=="mean":
        front_img=img[img>0]
        mean_value=np.mean(front_img)
        std_value=np.std(front_img)
        img=(img-mean_value)/std_value
    return img

#加载nii图像，同时完成预处理工作
def load_nii(imgPath,padding = (144,176,144)):
    img_data = nib.load(imgPath).get_fdata()
    img_data,_ = pad(img_data,shape=padding)
    img_data = preprocess_Normalizition(img_data)

    return img_data

#保存序列化对象
def save_object(path,obj):

    with open(path,"wb") as f:
        pickle.dump(obj,f)

#加载序列化对象
def load_object(path):

    with open(path,"rb") as f:
        obj = pickle.load(f)   
        
    return obj


#folding计算
#

def calculate_folding(df_offset,device):
    shape = df_offset.shape[2:]
    
    # create sampling grid（未归一化的标准网格）
    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor).to(device)
    
    phi = df_offset + grid
    
    for i in range(len(shape)):
        phi[:, i, ...] = 2 * (phi[:, i, ...] / (shape[i] - 1) - 0.5) 
    
    final_phi = phi.permute(0,2,3,4,1)
    
    ### Calculate Neg Jac Ratio
    neg_Jet = 1.0 * JacboianDet(final_phi)
    neg_Jet = F.relu(neg_Jet)
    mean_neg_J = torch.sum(neg_Jet).detach().cpu().numpy()
    num_neg = len(torch.where(neg_Jet > 0)[0])
    total = neg_Jet.size(-1) * neg_Jet.size(-2) * neg_Jet.size(-3)
    ratio_neg_J = num_neg / total
    
    return ratio_neg_J




def cal_time(visit_time):
    separates = visit_time.split("-")
    year = int(separates[0])
    flag_month = separates[1].split("0")
    month = int(separates[1]) if flag_month[0] != "0" else int(flag_month[1])
    flag_day = separates[2].split("0")
    day = int(separates[2]) if flag_day[0] != "0" else int(flag_day[1])

    return year*365+month*30+day


#取序列，划分成两部分，训练：测试
def load_imgs_and_time(data_root_path , subject):
    time_List = os.listdir(os.path.join(data_root_path,subject))
    time_List = sorted(time_List,key = lambda x:cal_time(x))
    img_list = []
    for t in time_List:
        img_list.append(load_nii(imgPath=os.path.join(data_root_path,subject,t,"t1.nii.gz")))
    
    return img_list,time_List

#返回 图像的list，时间的list