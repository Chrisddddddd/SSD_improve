# -*- coding: gbk -*- 
from itertools import product as product
from math import sqrt as sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
from utils.box_utils import decode, nms
from utils.config import Config


class Detect(Function):
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = Config['variance']

    def forward(self, loc_data, conf_data, prior_data):
        #--------------------------------#
        #   ��ת����cpu������
        #--------------------------------#
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()

        #--------------------------------#
        #   num��ֵΪbatch_size
        #   num_priorsΪ����������
        #--------------------------------#
        num = loc_data.size(0) 
        num_priors = prior_data.size(0)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        #--------------------------------------#
        #   �Է���Ԥ��������reshape
        #   num, num_classes, num_priors
        #--------------------------------------#
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # ��ÿһ��ͼƬ���д�������Ԥ���ʱ��ֻ��һ��ͼƬ������ֻ��ѭ��һ��
        for i in range(num):
            #--------------------------------------#
            #   ������������Ԥ���
            #   ����󣬻�õĽ����shapeΪ
            #   num_priors, 4
            #--------------------------------------#
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()

            #--------------------------------------#
            #   ���ÿһ�����Ӧ�ķ�����
            #   num_priors,
            #--------------------------------------#
            for cl in range(1, self.num_classes):
                #--------------------------------------#
                #   �����������޽����ж�
                #   Ȼ��ȡ���������޵ĵ÷�
                #--------------------------------------#
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                #--------------------------------------#
                #   ���������޵�Ԥ���ȡ����
                #--------------------------------------#
                boxes = decoded_boxes[l_mask].view(-1, 4)
                #--------------------------------------#
                #   ������ЩԤ�����зǼ�������
                #--------------------------------------#
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                
        return output

class PriorBox(object):
    def __init__(self, backbone_name, cfg):
        super(PriorBox, self).__init__()
        # �������ͼƬ�Ĵ�С��Ĭ��Ϊ300x300
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps'][backbone_name]
        self.min_sizes = cfg['min_sizes']
        self.nolmal_sizes=cfg['nolmal_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = [cfg['min_dim']/x for x in cfg['feature_maps'][backbone_name]]
        self.aspect_ratios = cfg['aspect_ratios'][backbone_name]
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        #----------------------------------------#
        #   ��feature_maps����ѭ��
        #   ����SSD����6����Ч����������Ԥ��
        #   �߳��ֱ�Ϊ[38, 19, 10, 5, 3, 1]
        #----------------------------------------#
        for k, f in enumerate(self.feature_maps):
            #----------------------------------------#
            #   �ֱ��6����Ч�����㽨������
            #   [38, 19, 10, 5, 3, 1]
            #----------------------------------------#
            x,y = np.meshgrid(np.arange(f),np.arange(f))
            x = x.reshape(-1)
            y = y.reshape(-1)
            #----------------------------------------#
            #   ����������Ϊ��һ������ʽ
            #   ����0-1֮��
            #----------------------------------------#
            for i, j in zip(y,x):
                f_k = self.image_size / self.steps[k]
                #----------------------------------------#
                #   �������������
                #----------------------------------------#
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                #----------------------------------------#
                #   ���С��������
                #----------------------------------------#
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]
                
                
                #----------------------------------------#
                #   ����еȵ�������
                #----------------------------------------#
                s_k = self.nolmal_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                #----------------------------------------#
                #   ��ô��������
                #----------------------------------------#
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                #----------------------------------------#
                #   ��������ĳ�����
                #----------------------------------------#
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]


                #----------------------------------------#
                #   ������ĳ�����
                #----------------------------------------#
                cx1=(j+0.25)/f_k
                cy1=(i+0.5)/f_k
                cy1w=(j + 0.5) / f_k
                cy1y=1/f_k
                mean+=[cx1,cy1,cy1w,cy1y]
                #----------------------------------------#
                #   ����Ҳ�ĳ�����
                #----------------------------------------#
                cx2=(j+0.75)/f_k
                cy2=(i+0.5)/f_k
                cy2w=(j + 0.5) / f_k
                cy2y=1/f_k
                mean+=[cx2,cy2,cy2w,cy2y]

        #----------------------------------------#
        #   ������е������ 8732,4
        #----------------------------------------#
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out