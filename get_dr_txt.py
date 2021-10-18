# -*- coding: gbk -*- 
#----------------------------------------------------#
#   ��ȡ���Լ���ground-truth
#   ������Ƶ�̳̿ɲ鿴
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import colorsys
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

from nets.ssd import get_ssd
from ssd import SSD
from utils.box_utils import letterbox_image, ssd_correct_boxes

MEANS = (104, 117, 123)

'''
�������õ�����ֵ�ϵ�����Ϊ����map��Ҫ�õ���ͬ���������µ�Recall��Precisionֵ
����ֻ�б����Ŀ��㹻�࣬�����map�Ż����ȷ����������˽�map��ԭ��
����mapʱ�����Recall��Precisionֵָ��������Ϊ0.5ʱ��Recall��Precisionֵ��
�˴���õ�./input/detection-results/�����txt�Ŀ���������ֱ��predict��һЩ��������Ϊ��������޵ͣ�
Ŀ����Ϊ�˼��㲻ͬ���������µ�Recall��Precisionֵ���Ӷ�ʵ��map�ļ��㡣
������Щͬѧ֪����0.5��0.5:0.95��mAP��
�����Ҫ�趨mAP0.x�������趨mAP0.75������ȥget_map.py�趨MINOVERLAP��
'''
class mAP_SSD(SSD):
    def generate(self):
        self.confidence = 0.01
        #-------------------------------#
        #   �����ܵ��������
        #-------------------------------#
        self.num_classes = len(self.class_names) + 1

        #-------------------------------#
        #   ����ģ����Ȩֵ
        #-------------------------------#
        model = get_ssd("test", self.num_classes, self.backbone, self.confidence, self.nms_iou)
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # �������ò�ͬ����ɫ
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   ���ͼƬ
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        f = open("./input/detection-results/"+image_id+".txt","w") 
        image_shape = np.array(np.shape(image)[0:2])
    
        #---------------------------------------------------------#
        #   ��ͼ�����ӻ�����ʵ�ֲ�ʧ���resize
        #   Ҳ����ֱ��resize����ʶ��
        #---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.input_shape[1],self.input_shape[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.input_shape[1],self.input_shape[0]), Image.BICUBIC)

        photo = np.array(crop_img,dtype = np.float64)
        with torch.no_grad():
            photo = torch.from_numpy(np.expand_dims(np.transpose(photo - MEANS, (2,0,1)),0)).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            preds = self.net(photo)

            top_conf = []
            top_label = []
            top_bboxes = []
            for i in range(preds.size(1)):
                j = 0
                while preds[0, i, j, 0] >= self.confidence:
                    score = preds[0, i, j, 0]
                    label_name = self.class_names[i-1]
                    pt = (preds[0, i, j, 1:]).detach().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]]
                    top_conf.append(score)
                    top_label.append(label_name)
                    top_bboxes.append(coords)
                    j = j + 1

        if len(top_conf)<=0:
            return 
            
        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
        #-----------------------------------------------------------#
        #   ȥ����������
        #-----------------------------------------------------------#
        if self.letterbox_image:
            boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)
        else:
            top_xmin = top_xmin * image_shape[1]
            top_ymin = top_ymin * image_shape[0]
            top_xmax = top_xmax * image_shape[1]
            top_ymax = top_ymax * image_shape[0]
            boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)

        for i, c in enumerate(top_label):
            predicted_class = c
            score = str(float(top_conf[i]))

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

ssd = mAP_SSD()
image_ids = open('E:/DATASET/VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")


for image_id in tqdm(image_ids):
    image_path = "E:/DATASET/VOCdevkit/VOC2007/images/"+image_id+".jpg"
    image = Image.open(image_path)
    # ��������֮�����mAP���Կ��ӻ�
    # image.save("./input/images-optional/"+image_id+".jpg")
    ssd.detect_image(image_id,image)
    
print("Conversion completed!")