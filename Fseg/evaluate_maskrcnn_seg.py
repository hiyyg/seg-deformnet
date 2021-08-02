import argparse
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
from pointnet.model_seg import PointNetInstanceSeg, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from lib.utils import load_depth, get_bbox


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='val', help='val, real_test')
parser.add_argument('--data_dir', type=str, default='dataset', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--iou_thd', type=float, default=0.5, help='threshold of iou for matching gt')
opt = parser.parse_args()

# opt.data = 'val'


result_dir = 'results/mrcnn_results/{}'.format(opt.data)
assert opt.data in ['val', 'real_test']
if opt.data == 'val':
    file_path = 'CAMERA/val_list.txt'
else:
    file_path = 'Real/test_list.txt'


def evaluate():
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
        raw_depth = load_depth(img_path)

        # load mask-rcnn detection results
        img_path_parsing = img_path.split('/')
        mrcnn_path = os.path.join('results/mrcnn_results', opt.data, 'results_{}_{}_{}.pkl'.format(
            opt.data.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
        pred_num_insts = len(mrcnn_result['class_ids'])
        pred_class_ids = mrcnn_result['class_ids']

        #load label
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2]
        gt_num_insts = len(gts['class_ids'])
        gt_class_ids = gts['class_ids']

        
        for i in range(pred_num_insts):
            all_dtc_num += 1 
            map_to_gt = []
            for j in range(len(gt_class_ids)):
                if gt_class_ids[j] != pred_class_ids[i]:
                    continue
                rmin1, rmax1, cmin1, cmax1 = get_bbox(mrcnn_result['rois'][i])
                rmin2, rmax2, cmin2, cmax2 = get_bbox(gts['bboxes'][j])
                pred_box = [cmin1, rmin1, cmax1, rmax1]
                gt_box = [cmin2, rmin2, cmax2, rmax2]
                iou = cal_iou(pred_box, gt_box)
                if iou < opt.iou_thd:
                    continue
                # match found
                map_to_gt.append(np.array([j, iou]))
            if len(map_to_gt) == 0:
               no_gt_num += 1
               continue   

            max_iou_idx = np.argmax(np.array(map_to_gt)[:, 1])
            j = int(map_to_gt[max_iou_idx][0])
            #calculate segmantation accuracy
            gt_mask = mask==gts['instance_ids'][j]
            pre_mask = mrcnn_result['masks'][:, :, i]
            mask_bias = gt_mask==pre_mask
            ins_mask_bias = mask_bias[rmin1:rmax1, cmin1:cmax1]
            mask_TP = np.logical_and(gt_mask, pre_mask)
            ins_mask_TP = mask_TP[rmin1:rmax1, cmin1:cmax1]
            ins_depth = raw_depth[rmin1:rmax1, cmin1:cmax1]
            ins_depth_idxs = np.where(ins_depth>0)
            correct_seg_num = np.sum(ins_mask_bias[ins_depth_idxs[0],ins_depth_idxs[1]].astype(float))
            TP_seg_num = np.sum(ins_mask_TP[ins_depth_idxs[0],ins_depth_idxs[1]].astype(float))
            acc_ins = correct_seg_num / ins_depth_idxs[0].shape[0]
            pcs_ins = TP_seg_num / np.sum(pre_mask[rmin1:rmax1, cmin1:cmax1][ins_depth_idxs[0],ins_depth_idxs[1]].astype(float))
            rcal_ins = TP_seg_num / np.sum(gt_mask[rmin1:rmax1, cmin1:cmax1][ins_depth_idxs[0],ins_depth_idxs[1]].astype(float))
            total_count[pred_class_ids[i]-1] += 1
            acc[pred_class_ids[i]-1] += acc_ins
            pcs[pred_class_ids[i]-1] += pcs_ins
            rcal[pred_class_ids[i]-1] += rcal_ins
    
    # compute accuracy
    catId_to_name = {0: 'bottle', 1: 'bowl', 2: 'camera', 3: 'can', 4: 'laptop', 5: 'mug'}
    acc, pcs, rcal = 100 * (acc / total_count), 100 * (pcs / total_count), 100 * (rcal / total_count)
    overall_acc, overall_pcs, overall_rcal = np.mean(acc), np.mean(pcs), np.mean(rcal)

    no_gt_ratio = 100 * (no_gt_num / all_dtc_num)
    fw = open('{0}/seg_acc_pcs_rcal.txt'.format(result_dir), 'a')
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
    evaluate()