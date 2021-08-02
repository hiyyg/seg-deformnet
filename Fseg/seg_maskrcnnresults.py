from __future__ import print_function
import argparse
import os
import cv2
import sys
import random
import time
import _pickle as cPickle
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
sys.path.append(os.getcwd())
from pointnet.model_seg import PointNetInstanceSeg
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from lib.utils import load_depth, get_bbox

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='val', help='val, real_test')
parser.add_argument('--dataset', type=str, default='CAMERA', help='CAMERA or Real')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--data_dir', type=str, default='dataset', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--n_pts', type=int, default=2048, help='number of points')
opt = parser.parse_args()

# opt.data = 'val'
# opt.dataset = 'CAMERA'
# opt.model = 'results/camera_ss15_sp800_pc60/seg_model_60.pth'

def seg_maskrcnnresults():
    classifier = PointNetInstanceSeg(n_classes=opt.n_cat)
    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))
    classifier.cuda()
    classifier = classifier.eval()
    if opt.dataset == 'Real':
        file_path = os.path.join(opt.dataset, 'test_list.txt')
        cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084
        result_dir = 'results/mrcnn_results/{}_test_pointnet_seg'.format(opt.dataset)
    else:
        file_path = os.path.join(opt.dataset, 'val_list.txt')
        cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
        result_dir = 'results/mrcnn_results/{}_val_pointnet_seg'.format(opt.dataset) 
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)   
    norm_scale = 1000.0
    xmap = np.array([[i for i in range(640)] for j in range(480)])
    ymap = np.array([[j for i in range(640)] for j in range(480)])        
    # get test data list
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(opt.data_dir, file_path))]
    
    t_start = time.time()
    for path in tqdm(img_list):
        img_path = os.path.join(opt.data_dir, path)
        depth = load_depth(img_path)
        # load mask-rcnn detection results
        img_path_parsing = img_path.split('/')
        mrcnn_path = os.path.join('results/mrcnn_results', opt.data, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
        num_insts = len(mrcnn_result['class_ids'])
        f_mask = np.zeros((num_insts, depth.shape[0], depth.shape[1]), dtype=int)
        # prepare frame data
        f_points, f_choose, f_catId = [], [], []
        valid_inst = []
        result = {}
        for i in range(num_insts):
            cat_id = mrcnn_result['class_ids'][i] - 1
            rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i])

            # sample points
            depth_vaild = depth > 0
            choose_depth = depth_vaild[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose_depth) < 32:
                continue
            else:
                valid_inst.append(i)
            # process objects with valid depth observation
            if len(choose_depth) > opt.n_pts:
                c_mask = np.zeros(len(choose_depth), dtype=int)
                c_mask[:opt.n_pts] = 1
                np.random.shuffle(c_mask)
                choose_depth = choose_depth[c_mask.nonzero()]
            else:
                choose_depth = np.pad(choose_depth, (0, opt.n_pts-len(choose_depth)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose_depth][:, np.newaxis]
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose_depth][:, np.newaxis]
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose_depth][:, np.newaxis]
            pt2 = depth_masked / norm_scale
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
            points = get_center_view_point_set(points, angle_y, angle_x)  # (n,3) #pts after Frustum rotation      
            f_points.append(points)
            f_catId.append(cat_id)
            f_choose.append(choose_depth)
        if len(valid_inst):
            f_points = torch.cuda.FloatTensor(f_points)
            f_catId = torch.cuda.LongTensor(f_catId)
            f_one_hot_vec = F.one_hot(f_catId, opt.n_cat)
            f_points = f_points.transpose(2,1)

            logits = classifier(f_points, f_one_hot_vec)    
            logits_choice = logits.data.max(2)[1]
            logits_np = logits_choice.cpu().data.numpy()
            for i in range(len(valid_inst)):
                inst_idx = valid_inst[i]
                choose_depth = f_choose[i]
                logits_np_inst = logits_np[i]
                choose_logits_np = logits_np_inst.nonzero()
                rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][inst_idx])
                roi_mask = np.zeros(((rmax-rmin)*(cmax-cmin)), dtype=int)
                roi_mask[choose_depth[choose_logits_np]] = 1
                roi_mask = roi_mask.reshape((rmax-rmin, cmax-cmin))
                f_mask[inst_idx][rmin:rmax, cmin:cmax] = roi_mask
        result['class_ids'] = mrcnn_result['class_ids']
        result['rois'] = mrcnn_result['rois']
        result['scores'] = mrcnn_result['scores']
        result['masks'] = (f_mask.transpose(1,2,0)>0)
        save_path = os.path.join(result_dir, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(save_path, 'wb') as f:
            cPickle.dump(result, f)

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

def get_center_view_point_set(points, angle_y, angle_x):
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

if __name__ == '__main__':
    seg_maskrcnnresults()
