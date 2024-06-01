import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
import torch.nn as nn
from skimage import measure
import torch.nn.functional as F
import os
import cv2
from torch.nn import init
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def seed_pytorch(seed=50):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('SplAtConv2d') == -1:
        init.xavier_normal(m.weight.data)
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0
                
def random_crop(img, mask, patch_size, pos_prob=None): 
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        h, w = img.shape
        
    while 1:
        h_start = random.randint(0, h - patch_size)
        h_end = h_start + patch_size
        w_start = random.randint(0, w - patch_size)
        w_end = w_start + patch_size

        img_patch = img[h_start:h_end, w_start:w_end]
        mask_patch = mask[h_start:h_end, w_start:w_end]
        
        if pos_prob == None or random.random()> pos_prob:
            break
        elif mask_patch.sum() > 0:
            break
        
    return img_patch, mask_patch

def Normalized(img, img_norm_cfg):
    return (img-img_norm_cfg['mean'])/img_norm_cfg['std']
    
def Denormalization(img, img_norm_cfg):
    return img*img_norm_cfg['std']+img_norm_cfg['mean']

def get_img_norm_cfg(dataset_name, dataset_dir):
    if  dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=62.10432052612305, std=23.96998405456543)
    elif dataset_name == 'IRDST-real':   
        img_norm_cfg = {'mean': 101.54053497314453, 'std': 56.49856185913086}
    else:
        with open(dataset_dir + '/' + dataset_name +'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir + '/' + dataset_name +'/img_idx/test_' + dataset_name + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    return img_norm_cfg

def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'Adagrad':
        optimizer  = torch.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer  = torch.optim.SGD(net.parameters(), lr=optimizer_settings['lr'], momentum=optimizer_settings['momentum'], weight_decay=optimizer_settings['weight_decay'])
    elif optimizer_name == 'AdamW':
        optimizer  = torch.optim.AdamW(net.parameters(), lr=optimizer_settings['lr'], weight_decay=optimizer_settings['weight_decay'])

    
    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_settings['step'], gamma=scheduler_settings['gamma'])
    elif scheduler_name   == 'CosineAnnealingLR':
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'], eta_min=scheduler_settings['min_lr'])
    
    return optimizer, scheduler

def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img 

def Copy_Paste(img, mask):
    img_h, img_w = img.shape
    # 获取所有含目标的源图像
    target__img_floder = "datasets/Dataset-mask/images"
    target__mask_floder = "datasets/Dataset-mask/masks"
    target_img_paths = sorted([os.path.join(target__img_floder, img_name) for img_name in os.listdir(target__img_floder) if img_name.endswith('.png')])
    cp_num = 10
    result_mask = mask
    result_img = img
    for _ in range(cp_num):
        random_img_path = random.choice(target_img_paths)
        random_mask_path = os.path.join(target__mask_floder, os.path.basename(random_img_path))
        target_img = cv2.imread(random_img_path, cv2.IMREAD_GRAYSCALE)
        target_mask = cv2.imread(random_mask_path, cv2.IMREAD_GRAYSCALE)

        # 查找轮廓
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 提取目标
        flag = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            target = target_img[y:y+h, x:x+w]
            while True:
                random_pos_y = random.randint(1, img_w-w-1)
                random_pos_x = random.randint(1, img_h-h-1)
                target_overlap = result_mask[random_pos_x:random_pos_x+h, random_pos_y:random_pos_y+w].any()

                if not target_overlap:
                    break
                else:
                    flag += 1
                    if flag > 50:
                        break
                    else:
                        continue
            if flag > 50:
                continue

            # 高斯
            lamda = np.random.uniform(0.5, 1)
            gaussian_matrix = generate_gaussian_matrix(h, w)

            for i in range(h):
                for j in range(w):
                    # pixel_value = result_img[random_pos_x+i, random_pos_y+j] + target[i, j] * lamda * gaussian_matrix[i, j]
                    pixel_value = target[i, j]
                    result_img[random_pos_x+i, random_pos_y+j] = min(255, pixel_value)  # 确保值不超过255
            result_mask[random_pos_x:random_pos_x+h, random_pos_y:random_pos_y+w] = 255.0

    return result_img, result_mask

def generate_gaussian_matrix(h, w, sigma_x=0.8, sigma_y=0.6, mu_x=0.0, mu_y=0.0, theta=0.0):
    """
    Generate a 2D rotated anisotropic Gaussian matrix.
    """

    sigma_x = np.random.uniform(0.2, 0.5)
    sigma_y = np.random.uniform(0.2, 0.5)
    mu_x = 0
    mu_y = 0
    theta = 0

    # Angle of rotation in radians
    theta_radians = np.radians(theta)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(theta_radians), -np.sin(theta_radians)],
                                [np.sin(theta_radians), np.cos(theta_radians)]])

    # Create a coordinate grid
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    
    # Stack the coordinates for matrix multiplication
    coords = np.stack([X.ravel() - mu_x, Y.ravel() - mu_y])
    
    # Apply the rotation matrix to the coordinates
    rot_coords = rotation_matrix @ coords
    
    # Calculate the squared distance from the center for each axis after rotation
    d_x2 = (rot_coords[0] ** 2)
    d_y2 = (rot_coords[1] ** 2)
    
    # Apply the anisotropic Gaussian formula
    gaussian = np.exp(-(d_x2 / (2.0 * sigma_x**2) + d_y2 / (2.0 * sigma_y**2)))
    
    if gaussian.max() <= 1.0:
        gaussian = gaussian.reshape(h, w)
        return gaussian
    else:
        # Reshape back to the original shape and normalize
        gaussian = gaussian.reshape(h, w)
        gaussian -= gaussian.min()
        gaussian /= gaussian.max()
        return gaussian