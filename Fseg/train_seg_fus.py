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
from pointnet.seg_dataset_fus import PoseDataset
from pointnet.model_seg import FusionInstanceSeg
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from lib.utils import load_depth, get_bbox
from lib.utils import setup_logger
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CAMERA', help='CAMERA or CAMERA+Real')
parser.add_argument('--rotate_to_center', type=int, default=1, help='rotate points to center')
parser.add_argument('--data_dir', type=str, default='dataset', help='data directory')
parser.add_argument('--n_pts', type=int, default=4096, help='number of points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--nepoch', type=int, default=75, help='number of epochs to train for')
parser.add_argument('--result_dir', type=str, default='seg/Real/real', help='directory to save train results')
parser.add_argument('--val_result_dir', type=str, default='seg/Real/real', help='directory to save train results')
opt = parser.parse_args()

# opt.dataset = 'CAMERA'
# opt.start_epoch = 75
# opt.model = 'results/camerafus_ss15_sp1200_pc75_bs64/seg_model_74.pth'
# opt.result_dir = 'results/camerafus_ss15_sp1200_pc75_bs64'
opt.val_result_dir = 'results/eval_camera'


# dataset
train_dataset = PoseDataset(opt.dataset, 'train', opt.data_dir, opt.n_pts, opt.img_size, opt.rotate_to_center)
test_dataset = PoseDataset(opt.dataset, 'test', opt.data_dir, opt.n_pts, opt.img_size, opt.rotate_to_center)
print(len(train_dataset), len(test_dataset))

blue = lambda x: '\033[94m' + x + '\033[0m'

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))    
    else:
        print("Not enough GPU hardware devices available")     
    # set result directory
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)    
    tb_writer = tf.summary.create_file_writer(opt.result_dir)
    logger = setup_logger('train_log', os.path.join(opt.result_dir, 'log.txt'))
    logger.propagate = 0
    for key, value in vars(opt).items():
        logger.info(key + ': ' + str(value))
    classifier = FusionInstanceSeg(n_classes=opt.n_cat)
    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))
    # global classifier
    classifier.cuda()
    # create optimizer
    if opt.start_epoch == 1:
        optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5, last_epoch=-1)   
    else:
        optimizer = optim.Adam([{'params':classifier.parameters(), 'initial_lr': 6.25e-5 }], lr=6.25e-5, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5, last_epoch=opt.start_epoch-1)

    # start training
    st_time = time.time()    
    if opt.dataset == 'CAMERA+Real':
        train_steps = 1200 
        val_size = 2000
    else:
        train_steps = 1200 #trian list:623180 val list:46671
        val_size = 2000
    global_step = train_steps * (opt.start_epoch - 1)
    train_size = train_steps * opt.batchSize
    indices = []
    page_start = -train_size    
    for epoch in range(opt.start_epoch, opt.nepoch + 1):
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))        
        # sample train subset
        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            if opt.dataset == 'CAMERA+Real':
                # CAMERA : Real = 3 : 1
                camera_len = train_dataset.subset_len[0]
                real_len = train_dataset.subset_len[1]
                real_indices = list(range(camera_len, camera_len+real_len))
                camera_indices = list(range(camera_len))
                n_repeat = (train_size - len_last) // (4 * real_len) + 1
                data_list = random.sample(camera_indices, 3*n_repeat*real_len) + real_indices*n_repeat
                random.shuffle(data_list)
                indices += data_list
            else:
                data_list = list(range(train_dataset.length))
                for i in range((train_size - len_last) // train_dataset.length + 1):
                    random.shuffle(data_list)
                    indices += data_list
            page_start = 0
        train_idx = indices[page_start:(page_start+train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        traindataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, 
            sampler=train_sampler, num_workers=opt.workers, pin_memory=True)
        
        for i, data in enumerate(traindataloader, 1):
            batch_data, batch_img, batch_label, batch_category, batch_choose_depth = data
            batch_one_hot_vec = F.one_hot(batch_category, opt.n_cat)
            batch_data = batch_data.transpose(2,1).float().cuda()
            batch_img = batch_img.cuda()
            batch_label = batch_label.float().cuda()
            batch_one_hot_vec = batch_one_hot_vec.float().cuda()
            batch_choose_depth = batch_choose_depth.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            logits = classifier(batch_data, batch_img, batch_one_hot_vec, batch_choose_depth)
            # 3D Instance Segmentation PointNet Loss
            logits = F.log_softmax(logits.view(-1,2),dim=1)
            batch_label = batch_label.view(-1).long()
            loss = F.nll_loss(logits, batch_label)
            loss.backward()
            optimizer.step()
            logits_choice = logits.data.max(1)[1]
            correct = logits_choice.eq(batch_label.data).cpu().sum()
            global_step += 1
            # write results to tensorboard
            with tb_writer.as_default():
                tf.summary.scalar('learning_rate', optimizer.param_groups[0]['lr'], step=global_step)
                tf.summary.scalar('train_loss', loss.item(), step=global_step)
                # tf.summary.scalar('train_acc', correct.item()/float(opt.batchSize * opt.n_pts), step=global_step)
                tb_writer.flush()
            if i % 10 == 0:
                logger.info('epoch {0:<4d} Batch {1:<4d} Loss:{2:f}'.format(epoch, i, loss.item()))            
            # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, train_steps, loss.item(), correct.item()/float(opt.batchSize * opt.n_pts)))
        scheduler.step()
        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))


        # evaluate one epoch
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) +
                    ', ' + 'Epoch %02d' % epoch + ', ' + 'Testing started'))        
        val_loss = 0.0
        total_count = np.zeros((opt.n_cat,), dtype=int)
        val_acc = np.zeros((opt.n_cat,), dtype=float)
        # sample validation subset
        # val_size = 200
        val_batch_size = 1
        val_idx = random.sample(list(range(test_dataset.length)), val_size)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
        val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size, sampler=val_sampler, 
            num_workers=opt.workers, pin_memory=True)  
        classifier = classifier.eval()                                            
        for i, data in enumerate(val_dataloader, 1):
            batch_data, batch_img, batch_label, batch_category, batch_choose_depth = data
            batch_one_hot_vec = F.one_hot(batch_category, opt.n_cat)
            batch_data = batch_data.transpose(2,1).float().cuda()
            batch_img = batch_img.cuda()
            batch_label = batch_label.float().cuda()
            batch_one_hot_vec = batch_one_hot_vec.float().cuda()
            batch_choose_depth = batch_choose_depth.cuda()
            logits = classifier(batch_data, batch_img, batch_one_hot_vec, batch_choose_depth)
            logits = F.log_softmax(logits.view(-1,2),dim=1)
            batch_label = batch_label.view(-1).long()
            loss = F.nll_loss(logits, batch_label)
            # use choose_depth to remove repeated points
            choose_depth = batch_choose_depth.cpu().numpy()[0]
            _, choose_depth = np.unique(choose_depth, return_index=True)
            logits_choice = logits.data.max(1)[1][choose_depth]
            correct = logits_choice.eq(batch_label.data[choose_depth]).cpu().sum()
            acc = correct / len(logits_choice)
            cat_id = batch_category.item()
            val_acc[cat_id] += acc
            total_count[cat_id] += 1
            val_loss += loss.item()
            if i % 100 == 0:
                logger.info('epoch {0:<4d} Batch {1:<4d} Loss:{2}'.format(epoch, i, loss.item()))            
                # print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, train_steps, blue('test'), loss.item(), correct.item()/float(val_batch_size * opt.n_pts)))
        # compute accuracy
        val_acc = 100 * (val_acc / total_count)
        val_loss = val_loss / val_size
        for i in range(opt.n_cat):
            logger.info('{:>8s} acc: {}'.format(test_dataset.cat_names[i], val_acc[i]))
        val_acc = np.mean(val_acc)
        with tb_writer.as_default():
            tf.summary.scalar('val_loss', val_loss, step=global_step)
            tf.summary.scalar('val_acc', val_acc, step=global_step)
            tb_writer.flush()
        logger.info('Epoch {0:02d} test average loss: {1:06f}'.format(epoch, val_loss))
        logger.info('Overall acc: {}'.format(val_acc))
        logger.info('>>>>>>>>----------Epoch {:02d} test finish---------<<<<<<<<'.format(epoch))
        torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.result_dir, epoch))

def test():
    global classifier
    classifier.cuda()
    ## benchmark mIOU
    if not os.path.exists(opt.val_result_dir):
        os.makedirs(opt.val_result_dir)

    bottle_ious, bowl_ious, camera_ious, can_ious, laptop_ious, mug_ious = \
        [], [], [], [], [], []
    bottle_num, bowl_num, camera_num, can_num, laptop_num, mug_num = 0, 0, 0, 0, 0, 0  

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
        shuffle=True, num_workers=int(opt.workers))
    
    classifier = classifier.eval()
    for i,data in tqdm(enumerate(testdataloader, 1)):
        batch_data, batch_label, batch_category, batch_choose_depth = data
        batch_one_hot_vec = F.one_hot(batch_category, opt.n_cat)
        batch_data = batch_data.transpose(2,1).float().cuda()
        batch_label = batch_label.float().cuda()
        batch_one_hot_vec = batch_one_hot_vec.float().cuda()

        logits = classifier(batch_data, batch_one_hot_vec)
        logits_choice = logits.data.max(2)[1]
        choose_depth = batch_choose_depth.numpy()
        logits_np = logits_choice.cpu().data.numpy()
        batch_label_np = batch_label.cpu().data.numpy()
        batch_category_np = batch_category.data.numpy()

        for j in range(batch_data.shape[0]):
            _, choose_depth_tt = np.unique(choose_depth[j], return_index=True)
            # assert opt.n_pts == choose_depth_tt.shape[0]
            # choose_depth[j] = choose_depth_tt
            logits_np_tt = logits_np[j][choose_depth_tt]
            batch_label_np_tt = batch_label_np[j][choose_depth_tt]

            I = np.sum(np.logical_and(logits_np_tt == 1, batch_label_np_tt == 1))
            U = np.sum(np.logical_or(logits_np_tt == 1, batch_label_np_tt == 1))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            cat = batch_category_np[j]
            if cat == 0: 
                bottle_ious.append(iou)
                bottle_num += 1
            elif cat == 1: 
                bowl_ious.append(iou)
                bowl_num += 1
            elif cat == 2: 
                camera_ious.append(iou)
                camera_num += 1
            elif cat == 3: 
                can_ious.append(iou)
                can_num += 1
            elif cat == 4: 
                laptop_ious.append(iou)
                laptop_num += 1
            elif cat == 5: 
                mug_ious.append(iou)
                mug_num += 1
    
    #save results
    fw = open('{0}/eval_logs.txt'.format(opt.val_result_dir), 'a')
    messages = []
    messages.append("mIOU for {}bottle : {}".format(bottle_num, np.mean(bottle_ious)))
    messages.append("mIOU for {}bowl   : {}".format(bowl_num, np.mean(bowl_ious)))
    messages.append("mIOU for {}camera : {}".format(camera_num, np.mean(camera_ious)))
    messages.append("mIOU for {}can    : {}".format(can_num, np.mean(can_ious)))
    messages.append("mIOU for {}laptop : {}".format(laptop_num, np.mean(laptop_ious)))
    messages.append("mIOU for {}mug    : {}".format(mug_num, np.mean(mug_ious)))
    messages.append("mIOU            : {}".format(np.mean([np.mean(bottle_ious),np.mean(bowl_ious),\
        np.mean(camera_ious),np.mean(can_ious),np.mean(laptop_ious),np.mean(mug_ious)])))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()

if __name__ == '__main__':
    train()
    # test()    