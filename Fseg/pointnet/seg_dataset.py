import os
import cv2
import math
import random
import torch
import numpy as np
import _pickle as cPickle
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from lib.utils import load_depth, get_bbox
import torch.nn.functional as F
from visual_points import visual_points
import open3d as o3d

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc

def rotate_pc_along_x(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, sinval], [-sinval, cosval]])
    pc[:, [1, 2]] = np.dot(pc[:, [1, 2]], np.transpose(rotmat))
    return pc

class PoseDataset(data.Dataset):
    def __init__(self, source, mode, data_dir, n_pts, rotate_to_center=True):
        """
        Args:
            source: 'CAMERA', 'Real'
            mode: 'train' or 'test'
            data_dir:
            n_pts: number of selected foreground points
            rotate_to_center: bool, whether to do frustum rotation
        """
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.rotate_to_center = rotate_to_center

        assert source in ['CAMERA', 'CAMERA+Real']
        assert mode in ['train', 'test']
        data_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        if mode == 'train':
            del data_list_path[2:]
        else:
            del data_list_path[:2]
        if source == 'CAMERA':
            del data_list_path[-1]
        elif source == 'Real':
            del data_list_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del data_list_path[0]
      

        data_list = []
        subset_len = []
        for path in data_list_path:
            data_list += [line.rstrip('\n')
                     for line in open(os.path.join('data_list/CAMERA_Real',path))]
            subset_len.append(len(data_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1]-subset_len[0]]        

        self.data_list = data_list
        self.length = len(self.data_list)
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.camera_intrinsics = [577.5, 577.5, 319.5, 239.5]
        self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        self.norm_scale = 1000.0    # normalization scale
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.shift_range = 0.01
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        print('{} points found.'.format(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_parsing = self.data_list[index].split('_')
        assert self.source in ['CAMERA', 'CAMERA+Real']
        if 'scene' in data_parsing[0]:
            img_path = os.path.join(self.data_dir, 'Real', '_'.join(data_parsing[:2]))
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics                      
        else:
            img_path = os.path.join(self.data_dir, 'CAMERA', data_parsing[0])
            cam_fx, cam_fy, cam_cx, cam_cy = self.camera_intrinsics
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2]                         
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f) 
        # select one foreground object
        inst_id = int(data_parsing[-1])        
        idx = np.where(np.array(gts['instance_ids']) == inst_id)[0][0]
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        # sample points
        depth = load_depth(img_path)
        mask = np.equal(mask, inst_id)
        mask = np.logical_and(mask, depth > 0)
        depth_vaild = depth > 0
        choose_depth = depth_vaild[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        seg = mask[rmin:rmax, cmin:cmax].flatten().astype(np.float64)
        if len(choose_depth) > self.n_pts:
            c_mask = np.zeros(len(choose_depth), dtype=int)
            c_mask[:self.n_pts] = 1
            np.random.shuffle(c_mask)
            choose_depth = choose_depth[c_mask.nonzero()]
        else:
            choose_depth = np.pad(choose_depth, (0, self.n_pts-len(choose_depth)), 'wrap')
        
        seg = seg[choose_depth]
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose_depth][:, np.newaxis]
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose_depth][:, np.newaxis]
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose_depth][:, np.newaxis]
        pt2 = depth_masked / self.norm_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(cmin+cmax)/2.0, (rmin+rmax)/2.0])
        depth_center = 1.0
        x_center = (box2d_center[0] - cam_cx) * depth_center / cam_fx
        y_center = (cam_cy - box2d_center[1]) * depth_center / cam_fy
        angle_y = -1 * np.arctan2(depth_center, x_center)
        angle_x = -1 * np.arctan2((depth_center**2+x_center**2)**0.5, y_center)
        # Get point cloud
        if self.rotate_to_center:  # True
            points = self.get_center_view_point_set(points, angle_y, angle_x)  # (n,4) #pts after Frustum rotation
        
        # visual_points(points)
        # label
        cat_id = gts['class_ids'][idx] - 1    # convert to 0-indexed

        # data augmentation
        translation = gts['translations'][idx]
        if self.mode == 'train':
             # point shift
            add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            translation = translation + add_t[0]
            # point jitter
            add_t = add_t + np.clip(0.001*np.random.randn(points.shape[0], 3), -0.005, 0.005)
            points = np.add(points, add_t)
        # rgb = self.transform(rgb)
        points = points.astype(np.float32)

        return points, seg, cat_id, choose_depth
    
    def get_center_view_point_set(self, points, angle_y, angle_x):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        angle_y = np.pi / 2.0 + angle_y
        angle_x = np.pi / 2.0 + angle_x
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(points)
        return rotate_pc_along_x(rotate_pc_along_y(point_set, angle_y),
                                         angle_x)
