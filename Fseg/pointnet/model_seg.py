from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from lib.pspnet import PSPNet


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat
        
class PointNetInstanceSeg(nn.Module):
    def __init__(self,n_classes=6,n_channel=3):
        '''v1 3D Instance Segmentation PointNet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetInstanceSeg, self).__init__()
        self.conv1 = nn.Conv1d(n_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.n_classes = n_classes
        self.dconv1 = nn.Conv1d(1088+n_classes, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)

    def forward(self, pts, one_hot_vec): # bs,4,n
        '''
        :param pts: [bs,4,n]: x,y,z,intensity
        :return: logits: [bs,n,2],scores for bkg/clutter and object
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,64,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,64,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,64,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,128,n
        out5 = F.relu(self.bn5(self.conv5(out4)))# bs,1024,n
        global_feat = torch.max(out5, 2, keepdim=True)[0] #bs,1024,1

        expand_one_hot_vec = one_hot_vec.view(bs,-1,1)#bs,3,1
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,1027,1
        expand_global_feat_repeat = expand_global_feat.view(bs,-1,1)\
                .repeat(1,1,n_pts)# bs,1027,n
        concat_feat = torch.cat([out2,\
            expand_global_feat_repeat],1)
        # bs, (641024+3)=1091, n

        x = F.relu(self.dbn1(self.dconv1(concat_feat)))#bs,512,n
        x = F.relu(self.dbn2(self.dconv2(x)))#bs,256,n
        x = F.relu(self.dbn3(self.dconv3(x)))#bs,128,n
        x = F.relu(self.dbn4(self.dconv4(x)))#bs,128,n
        x = self.dropout(x)
        x = self.dconv5(x)#bs, 2, n

        seg_pred = x.transpose(2,1).contiguous()#bs, n, 2
        return seg_pred

class FusionInstanceSeg(nn.Module):
    def __init__(self,n_classes=6,n_channel=3):
        '''v1 3D Instance Segmentation PointNet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(FusionInstanceSeg, self).__init__()
        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        self.conv0 = nn.Conv1d(32, 64, 1)
        self.conv1 = nn.Conv1d(n_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.n_classes = n_classes
        self.dconv1 = nn.Conv1d(1152+n_classes, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)

    def forward(self, pts, img, one_hot_vec, choose): # bs,4,n
        '''
        :param pts: [bs,4,n]: x,y,z,intensity
        :return: logits: [bs,n,2],scores for bkg/clutter and object
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]


        out_img = self.psp(img)
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)        
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        emb = F.relu(self.conv0(emb))# bs,64,n

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,64,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,64,n
        inst_local = torch.cat((out2, emb), dim=1) #bs,128,n
        out3 = F.relu(self.bn3(self.conv3(inst_local))) # bs,128,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,128,n
        out5 = F.relu(self.bn5(self.conv5(out4)))# bs,1024,n
        global_feat = torch.max(out5, 2, keepdim=True)[0] #bs,1024,1

        expand_one_hot_vec = one_hot_vec.view(bs,-1,1)#bs,6,1
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,1030,1
        expand_global_feat_repeat = expand_global_feat.view(bs,-1,1)\
                .repeat(1,1,n_pts)# bs,1030,n
        concat_feat = torch.cat([inst_local,\
            expand_global_feat_repeat],1)
        # bs, (128+1024+6)=1158, n

        x = F.relu(self.dbn1(self.dconv1(concat_feat)))#bs,512,n
        x = F.relu(self.dbn2(self.dconv2(x)))#bs,256,n
        x = F.relu(self.dbn3(self.dconv3(x)))#bs,128,n
        x = F.relu(self.dbn4(self.dconv4(x)))#bs,128,n
        x = self.dropout(x)
        x = self.dconv5(x)#bs, 2, n

        seg_pred = x.transpose(2,1).contiguous()#bs, n, 2
        return seg_pred

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
