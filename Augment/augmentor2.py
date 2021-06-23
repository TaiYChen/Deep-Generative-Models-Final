#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: augmentor.py
@time: 2019/09/16
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import random
from torch.autograd import Function
from typing import *
from util import sample_and_group


def batch_quat_to_rotmat(q, out=None):

    B = q.size(0)

    if out is None:
        out = q.new_empty(B, 3, 3)

    # 2 / squared quaternion 2-norm
    len = torch.sum(q.pow(2), 1)
    s = 2/len

    s_ = torch.clamp(len,2.0/3.0,3.0/2.0)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = (1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s))#.mul(s_)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = (1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s))#.mul(s_)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = (1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s))#.mul(s_)

    return out, s_
    
class Augmentor_Rotation(nn.Module):
    def __init__(self,dim):
        super(Augmentor_Rotation, self).__init__()
        self.fc1 = nn.Linear(dim + 1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(B, 1)
        # if x.is_cuda:
        #     iden = iden.cuda()
        # x = x + iden
        # x = x.view(-1, 3, 3)

        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x, s = batch_quat_to_rotmat(x)
        x = x.view(-1, 3, 3)
        s = s.view(B, 1, 1)
        #return x, None
        return x, s


class Augmentor_Displacement(nn.Module):
    def __init__(self, dim):
        super(Augmentor_Displacement, self).__init__()

        self.conv1 = torch.nn.Conv1d(dim+1024+64, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 3, 1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x
        
class Augmentor2(nn.Module):
    def __init__(self, args, dim=1024, in_dim=3):
        super(Augmentor2, self).__init__()
        self.dim = dim
        self.args = args
        self.conv1 = nn.Conv1d(in_dim, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

        '''
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        '''
        self.rot = Augmentor_Rotation(self.dim)
        self.dis = Augmentor_Displacement(self.dim)

    def forward(self, x, noise):
        B, C, N = x.size()
        raw_pt = x[:,:3,:].contiguous()
        normal = x[:,3:,:].transpose(1, 2).contiguous() if C > 3 else None
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))

        pointfeat = x

        x = x.permute(0, 2, 1) #B, N, D
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)  
        #B,N,Ns,3  B,N,Ns,6       
        feature_0 = self.gather_local_0(new_feature)  #B,D,N=512
        feature = feature_0.permute(0, 2, 1)  #B,N,D
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature) #B,D,N=256
      
        x = self.pt_last(feature_1) 
        x = torch.cat([x, feature_1], dim=1) #B,D,N
        x = self.conv_fuse(x) #B,1024,N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) #B, 1024
        ###############
        feat_r = torch.cat([x,noise],1)
        rotation, scale = self.rot(feat_r)

        feat_d = x.view(-1, 1024, 1).repeat(1, 1, N)
        noise_d = noise.view(B, -1, 1).repeat(1, 1, N)

        feat_d = torch.cat([pointfeat, feat_d,noise_d],1)
        displacement = self.dis(feat_d)

        pt = raw_pt.transpose(2, 1).contiguous()

        p1 = random.uniform(0, 1)
        possi = 0.5#0.0  
        if p1 > possi:
            pt = torch.bmm(pt, rotation).transpose(1, 2).contiguous()
        else:
            pt = pt.transpose(1, 2).contiguous()
        p2 = random.uniform(0, 1)
        if p2 > possi:
            pt = pt + displacement

        if normal is not None:
            normal = (torch.bmm(normal, rotation)).transpose(1, 2).contiguous()
            pt = torch.cat([pt,normal],1)

        return pt
                                                                                                         
class Augmentor3(nn.Module):
    def __init__(self,dim=1024,in_dim=3):
        super(Augmentor3, self).__init__()
        self.dim = dim
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        #self.scl = Augmentor_Scale(self.dim)
        self.rot = Augmentor_Rotation(self.dim)
        self.dis = Augmentor_Displacement(self.dim)

    def forward(self, pt, noise):

        B, C, N = pt.size()
        raw_pt = pt[:,:3,:].contiguous()
        normal = pt[:,3:,:].transpose(1, 2).contiguous() if C > 3 else None

        x = F.relu(self.bn1(self.conv1(raw_pt)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        
        feat_r = x.view(-1, 1024)
        feat_r = torch.cat([feat_r,noise],1)
        rotation, scale = self.rot(feat_r)

        feat_d = x.view(-1, 1024, 1).repeat(1, 1, N)
        noise_d = noise.view(B, -1, 1).repeat(1, 1, N)

        feat_d = torch.cat([pointfeat, feat_d,noise_d],1)
        displacement = self.dis(feat_d)

        pt = raw_pt.transpose(2, 1).contiguous()

        p1 = random.uniform(0, 1)
        possi = 0.5#0.0  
        if p1 > possi:#pt[B, N, 3]
            pt = torch.bmm(pt, rotation).transpose(1, 2).contiguous()
        else:
            pt = pt.transpose(1, 2).contiguous()
            
        p1 = random.uniform(0, 1)
        if p1 > possi and scale is not None:
            pt = torch.matmul(pt, scale).contiguous()
        else:
            pt = pt.contiguous()
            
        p2 = random.uniform(0, 1)
        if p2 > possi:
            pt = pt + displacement

        if normal is not None:
            normal = (torch.bmm(normal, rotation)).transpose(1, 2).contiguous()
            pt = torch.cat([pt,normal],1)

        return pt
        
def square_distance(src, dst):
    #print(src.shape, dst.shape)

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
    
def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
        
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


    
class Augmentor(nn.Module):
    def __init__(self,dim=1024,in_dim=3):
        super(Augmentor, self).__init__()
        self.dim = dim
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)

        self.rot = Augmentor_Rotation(self.dim)
        self.dis = Augmentor_Displacement(self.dim)

    def forward(self, pt, noise):

        B, C, N = pt.size()
        raw_pt = pt[:,:3,:].contiguous()
        normal = pt[:,3:,:].transpose(1, 2).contiguous() if C > 3 else None

        x = F.relu(self.bn1(self.conv1(raw_pt)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        
        feat_r = x.view(-1, 1024)
        feat_r = torch.cat([feat_r,noise],1)
        rotation, scale = self.rot(feat_r)

        feat_d = x.view(-1, 1024, 1).repeat(1, 1, N)
        noise_d = noise.view(B, -1, 1).repeat(1, 1, N)

        feat_d = torch.cat([pointfeat, feat_d,noise_d],1)
        displacement = self.dis(feat_d)

        pt = raw_pt.transpose(2, 1).contiguous()

        p1 = random.uniform(0, 1)
        possi = 0.5#0.0  
        if p1 > possi:
            pt = torch.bmm(pt, rotation).transpose(1, 2).contiguous()
        else:
            pt = pt.transpose(1, 2).contiguous()
        p2 = random.uniform(0, 1)
        if p2 > possi:
            pt = pt + displacement

        if normal is not None:
            normal = (torch.bmm(normal, rotation)).transpose(1, 2).contiguous()
            pt = torch.cat([pt,normal],1)

        return pt

class Noise_Augmentor(nn.Module):
    def __init__(self,dim=1024,in_dim=3):
        super(Noise_Augmentor, self).__init__()
        self.rot = Noise_Augmentor_Rotation()
        self.dis = Noise_Augmentor_Displacement()

    def forward(self, pt, noise): 
        B, C, N = pt.size()
        raw_pt = pt[:,:3,:].contiguous()
        normal = pt[:,3:,:].transpose(1, 2).contiguous() if C > 3 else None
        rotation, scale = self.rot(noise)

        noise_d = noise.view(B, -1, 1).repeat(1, 1, N)
        displacement = self.dis(noise_d)

        pt = raw_pt.transpose(2, 1).contiguous()

        p1 = random.uniform(0, 1)
        possi = 0.5#0.0  
        if p1 > possi:
            pt = torch.bmm(pt, rotation).transpose(1, 2).contiguous()
        else:
            pt = pt.transpose(1, 2).contiguous()
        p2 = random.uniform(0, 1)
        if p2 > possi:
            pt = pt + displacement

        if normal is not None:
            normal = (torch.bmm(normal, rotation)).transpose(1, 2).contiguous()
            pt = torch.cat([pt,normal],1)

        return pt

class Noise_Augmentor_Rotation(nn.Module):
    def __init__(self):
        super(Noise_Augmentor_Rotation, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(B, 1)
        # if x.is_cuda:
        #     iden = iden.cuda()
        # x = x + iden
        # x = x.view(-1, 3, 3)

        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x, s = batch_quat_to_rotmat(x)
        x = x.view(-1, 3, 3)
        s = s.view(B, 1, 1)
        return x, None


class Noise_Augmentor_Displacement(nn.Module):
    def __init__(self):
        super(Noise_Augmentor_Displacement, self).__init__()

        self.conv1 = torch.nn.Conv1d(1024, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 3, 1)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        return x

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   # B, N, C, Ns
        x = x.reshape(-1, d, s)     # B*N, C, Ns 
        x = F.relu(self.bn1(self.conv1(x))) # B*N, D, Ns
        x = F.relu(self.bn2(self.conv2(x))) # B*N, D, Ns
        x = F.adaptive_max_pool1d(x, 1).view(b, -1) # B*N, D, 1 --> B, N*D
        x = x.reshape(b, n, -1).permute(0, 2, 1) #B, N, D --> B, D, N
        return x

class Pct(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
