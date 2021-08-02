#creat list of objects instance
import os
from tqdm import tqdm
import sys
import argparse
import _pickle as cPickle
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Real', help='CAMERA or Real')
parser.add_argument('--data_dir', type=str, default='dataset', help='data directory')
parser.add_argument('--data', type=str, default='train', help='train, test')
opt = parser.parse_args()

def create_data_list(dataset, data):
    save_dir = os.path.join('data_list/CAMERA_Real', dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if data == 'train':
        img_list = open(os.path.join(opt.data_dir, dataset, 'train_list.txt')).read().splitlines()
    elif data == 'test' or data == 'val':
        if dataset == 'CAMERA':
            img_list = open(os.path.join(opt.data_dir, dataset, 'val_list.txt')).read().splitlines()
        else:
            img_list = open(os.path.join(opt.data_dir, dataset, 'test_list.txt')).read().splitlines()

    data_list = []
    for img_path in tqdm(img_list):
        with open(os.path.join(opt.data_dir, dataset, (img_path + '_label.pkl')), 'rb') as f:
            gts = cPickle.load(f)
        num_insts = len(gts['class_ids'])
        for i in range(num_insts):
            data_list.append(img_path +'_'+str(gts['instance_ids'][i]))
    # write data list to file
    with open(os.path.join(save_dir, '{}_list.txt'.format(data)), 'w') as f:
        for data_imformation in data_list:
            f.write("%s\n" % data_imformation)    

if __name__ == '__main__':
    create_data_list('Real', 'train')
    create_data_list('Real', 'test')
    create_data_list('CAMERA', 'train')
    create_data_list('CAMERA', 'val')

