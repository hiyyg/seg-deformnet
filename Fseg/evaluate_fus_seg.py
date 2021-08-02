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
from pointnet.seg_dataset import PoseDataset
from pointnet.model_seg import FusionInstanceSeg
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from lib.utils import load_depth, get_bbox

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='val', help='val, real_test')
parser.add_argument('--dataset', type=str, default='CAMERA', help='CAMERA or Real')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--data_dir', type=str, default='dataset', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--n_pts', type=int, default=4096, help='number of points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--save_pkl', type=int, default=0, help='save .pkl or not')
parser.add_argument('--iou_thd', type=float, default=0.5, help='threshold of iou for matching gt')
opt = parser.parse_args()

# opt.data = 'val'
# opt.dataset = 'CAMERA'
# opt.model = 'results/camerafus_ss18_sp1200_pc90_bs128/seg_model_90.pth'


def seg_maskrcnnresults():
    classifier = FusionInstanceSeg(n_classes=opt.n_cat)
    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))
    classifier.cuda()
    classifier = classifier.eval()

    if opt.dataset == 'Real':
        file_path = os.path.join(opt.dataset, 'test_list.txt')
        cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084
        result_dir = 'results/mrcnn_results/{}_test_fus_seg'.format(opt.dataset)
    else:
        file_path = os.path.join(opt.dataset, 'val_list.txt')
        cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
        result_dir = 'results/mrcnn_results/{}_val_fus_seg'.format(opt.dataset) 
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)   
    norm_scale = 1000.0
    norm_color = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )    
    xmap = np.array([[i for i in range(640)] for j in range(480)])
    ymap = np.array([[j for i in range(640)] for j in range(480)])        
    # get test data list
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(opt.data_dir, file_path))]
    
    total_count = np.zeros((opt.n_cat,), dtype=int)
    acc = np.zeros((opt.n_cat,), dtype=float)#accuracy
    pcs = np.zeros((opt.n_cat,), dtype=float)#precision
    rcal = np.zeros((opt.n_cat,), dtype=float)#recall
    all_dtc_num = 0
    no_gt_num = 0
    
    t_start = time.time()
    for path in tqdm(img_list):
        img_path = os.path.join(opt.data_dir, path)
        raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        raw_rgb = raw_rgb[:, :, ::-1]
        depth = load_depth(img_path)
        #load label
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        gt_mask = cv2.imread(img_path + '_mask.png')[:, :, 2]
        gt_num_insts = len(gts['class_ids'])
        gt_class_ids = gts['class_ids']

        # load mask-rcnn detection results
        img_path_parsing = img_path.split('/')
        mrcnn_path = os.path.join('results/mrcnn_results', opt.data, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
        num_insts = len(mrcnn_result['class_ids'])
        mrcnn_class_ids = mrcnn_result['class_ids']
        f_mask = np.zeros((num_insts, depth.shape[0], depth.shape[1]), dtype=int)
        # prepare frame data
        f_points, f_rgb, f_choose, f_catId = [], [], [], []
        f_raw_choose = []
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
            rgb = raw_rgb[rmin:rmax, cmin:cmax, :]
            rgb = cv2.resize(rgb, (opt.img_size, opt.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = norm_color(rgb)
            crop_w = rmax - rmin
            ratio = opt.img_size / crop_w
            col_idx = choose_depth % crop_w
            row_idx = choose_depth // crop_w
            raw_choose = np.copy(choose_depth) 
            choose_depth = (np.floor(row_idx * ratio) * opt.img_size + np.floor(col_idx * ratio)).astype(np.int64)                 
            f_points.append(points)
            f_rgb.append(rgb)
            f_catId.append(cat_id)
            f_choose.append(choose_depth)
            f_raw_choose.append(raw_choose)
        if len(valid_inst):
            f_points = torch.cuda.FloatTensor(f_points)
            f_rgb = torch.stack(f_rgb, dim=0).cuda()
            f_catId = torch.cuda.LongTensor(f_catId)
            f_one_hot_vec = F.one_hot(f_catId, opt.n_cat)
            f_choose = torch.cuda.LongTensor(f_choose)
            f_points = f_points.transpose(2,1)

            logits = classifier(f_points, f_rgb, f_one_hot_vec, f_choose)    
            logits_choice = logits.data.max(2)[1]
            logits_np = logits_choice.cpu().data.numpy()
            f_choose = f_choose.cpu().numpy()
            for i in range(len(valid_inst)):
                inst_idx = valid_inst[i]
                choose_depth = f_choose[i]
                raw_choose = f_raw_choose[i]
                logits_np_inst = logits_np[i]
                choose_logits_np = logits_np_inst.nonzero()
                rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][inst_idx])
                roi_mask = np.zeros(((rmax-rmin)*(cmax-cmin)), dtype=int)
                roi_mask[raw_choose[choose_logits_np]] = 1
                roi_mask = roi_mask.reshape((rmax-rmin, cmax-cmin))
                f_mask[inst_idx][rmin:rmax, cmin:cmax] = roi_mask
                all_dtc_num += 1 
                map_to_gt = []
                for j in range(len(gt_class_ids)):
                    if gt_class_ids[j] != mrcnn_class_ids[inst_idx]:
                        continue
                    pred_box = [cmin, rmin, cmax, rmax]
                    rmin2, rmax2, cmin2, cmax2 = get_bbox(gts['bboxes'][j])
                    gt_box = [cmin2, rmin2, cmax2, rmax2]
                    iou = cal_iou(pred_box, gt_box)
                    if iou < opt.iou_thd:
                        continue
                    # match found
                    map_to_gt.append(np.array([j, iou]))
                if len(map_to_gt) == 0:
                    no_gt_num += 1
                else:
                    max_iou_idx = np.argmax(np.array(map_to_gt)[:, 1])
                    j = int(map_to_gt[max_iou_idx][0])
                    gt_mask_ins = gt_mask == gts['instance_ids'][j]
                    gt_roi_mask = gt_mask_ins[rmin:rmax, cmin:cmax]
                    raw_choose, choose_raw_choose = np.unique(raw_choose, return_index=True)
                    gt_logits = gt_roi_mask.flatten()[raw_choose]
                    logits_bias = logits_np_inst[choose_raw_choose] == gt_logits
                    logits_TP = np.logical_and(logits_np_inst[choose_raw_choose], gt_logits)
                    correct_seg_num = np.sum(np.array(logits_bias))
                    TP_seg_num = np.sum(np.array(logits_TP))
                    acc_ins = correct_seg_num / len(raw_choose)
                    pcs_ins = TP_seg_num / np.sum(logits_np_inst[choose_raw_choose])
                    rcal_ins = TP_seg_num / np.sum(gt_logits)
                    total_count[mrcnn_class_ids[inst_idx] - 1] += 1
                    acc[mrcnn_class_ids[inst_idx] - 1] += acc_ins
                    pcs[mrcnn_class_ids[inst_idx] - 1] += pcs_ins
                    rcal[mrcnn_class_ids[inst_idx] - 1] += rcal_ins

        result['class_ids'] = mrcnn_result['class_ids']
        result['rois'] = mrcnn_result['rois']
        result['scores'] = mrcnn_result['scores']
        result['masks'] = (f_mask.transpose(1,2,0)>0)
        if opt.save_pkl:
            save_path = os.path.join(result_dir, 'results_{}_{}_{}.pkl'.format(
                opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
            with open(save_path, 'wb') as f:
                cPickle.dump(result, f)
    # compute accuracy
    catId_to_name = {0: 'bottle', 1: 'bowl', 2: 'camera', 3: 'can', 4: 'laptop', 5: 'mug'}
    acc, pcs, rcal = 100 * (acc / total_count), 100 * (pcs / total_count), 100 * (rcal / total_count)
    overall_acc, overall_pcs, overall_rcal = np.mean(acc), np.mean(pcs), np.mean(rcal)
    no_gt_ratio = 100 * (no_gt_num / all_dtc_num)
    fw = open('{0}/seg_acc_pcs.txt'.format(result_dir), 'a')
    messages = []
    messages.append('segmantation results:')
    messages.append('{:>12s}{:>12s}{:>12s}{:>12s}'.format('category', 'accuracy', 'precision', 'recall'))
    for i in range(acc.shape[0]):
        messages.append("{:>12s}{:>12.2f}{:>12.2f}{:>12.2f}".format(catId_to_name[i], acc[i], pcs[i], rcal[i]))
    messages.append("{:>12s}{:>12.2f}{:>12.2f}{:>12.2f}".format('overall', overall_acc, overall_pcs, overall_rcal))
    messages.append("{:>12s}{:>12.2f}".format('mismatch', no_gt_ratio))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()    

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
def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return: 
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积
 
    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou

if __name__ == '__main__':
    seg_maskrcnnresults()
