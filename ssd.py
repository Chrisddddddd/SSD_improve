import colorsys
import os
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

from nets import ssd
from utils.box_utils import letterbox_image, ssd_correct_boxes

warnings.filterwarnings("ignore")

MEANS = (104, 117, 123)
#--------------------------------------------#
#   浣跨敤鑷繁璁粌濂界殑妯″瀷棰勬祴闇�瑕佷慨鏀�3涓弬鏁�
#   model_path銆乥ackbone鍜宑lasses_path閮介渶瑕佷慨鏀癸紒
#   濡傛灉鍑虹幇shape涓嶅尮閰�
#   涓�瀹氳娉ㄦ剰璁粌鏃剁殑config閲岄潰鐨刵um_classes銆�
#   model_path鍜宑lasses_path鍙傛暟鐨勪慨鏀�
#--------------------------------------------#
class SSD(object):
    _defaults = {
        "model_path"        : 'logs/Epoch20-Total_Loss2.4878-Val_Loss2.0589.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
        "input_shape"       : (300, 300, 3),
        "confidence"        : 0.5,
        "nms_iou"           : 0.45,
        "cuda"              : True,
        #-------------------------------#
        #   涓诲共缃戠粶鐨勯�夋嫨
        #   vgg鎴栬�卪obilenet
        #-------------------------------#
        "backbone"          : "vgg",
        #---------------------------------------------------------------------#
        #   璇ュ彉閲忕敤浜庢帶鍒舵槸鍚︿娇鐢╨etterbox_image瀵硅緭鍏ュ浘鍍忚繘琛屼笉澶辩湡鐨剅esize锛�
        #   鍦ㄥ娆℃祴璇曞悗锛屽彂鐜板叧闂璴etterbox_image鐩存帴resize鐨勬晥鏋滄洿濂�
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   鍒濆鍖朣SD
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()
        
    #---------------------------------------------------#
    #   鑾峰緱鎵�鏈夌殑鍒嗙被
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    #---------------------------------------------------#
    #   杞藉叆妯″瀷
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   璁＄畻鎬荤殑绫荤殑鏁伴噺
        #-------------------------------#
        self.num_classes = len(self.class_names) + 1

        #-------------------------------#
        #   杞藉叆妯″瀷涓庢潈鍊�
        #-------------------------------#
        model = ssd.get_ssd("test", self.num_classes, self.backbone, self.confidence, self.nms_iou)
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 鐢绘璁剧疆涓嶅悓鐨勯鑹�
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   妫�娴嬪浘鐗�
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   鍦ㄨ繖閲屽皢鍥惧儚杞崲鎴怰GB鍥惧儚锛岄槻姝㈢伆搴﹀浘鍦ㄩ娴嬫椂鎶ラ敊銆�
        #---------------------------------------------------------#
        image = image.convert('RGB')

        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   缁欏浘鍍忓鍔犵伆鏉★紝瀹炵幇涓嶅け鐪熺殑resize
        #   涔熷彲浠ョ洿鎺esize杩涜璇嗗埆
        #---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.input_shape[1],self.input_shape[0])))
        else:
            crop_img = image.resize((self.input_shape[1],self.input_shape[0]), Image.BICUBIC)

        photo = np.array(crop_img,dtype = np.float64)
        with torch.no_grad():
            #---------------------------------------------------#
            #   鍥剧墖棰勫鐞嗭紝褰掍竴鍖�
            #---------------------------------------------------#
            photo = torch.from_numpy(np.expand_dims(np.transpose(photo - MEANS, (2, 0, 1)), 0)).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
                
            #---------------------------------------------------#
            #   浼犲叆缃戠粶杩涜棰勬祴
            #---------------------------------------------------#
            preds = self.net(photo)
        
            top_conf = []
            top_label = []
            top_bboxes = []
            #---------------------------------------------------#
            #   preds鐨剆hape涓� 1, num_classes, top_k, 5
            #---------------------------------------------------#
            for i in range(preds.size(1)):
                j = 0
                while preds[0, i, j, 0] >= self.confidence:
                    #---------------------------------------------------#
                    #   score涓哄綋鍓嶉娴嬫鐨勫緱鍒�
                    #   label_name涓洪娴嬫鐨勭绫�
                    #---------------------------------------------------#
                    score = preds[0, i, j, 0]
                    label_name = self.class_names[i-1]
                    #---------------------------------------------------#
                    #   pt鐨剆hape涓�4, 褰撳墠棰勬祴妗嗙殑宸︿笂瑙掑彸涓嬭
                    #---------------------------------------------------#
                    pt = (preds[0, i, j, 1:]).detach().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]]
                    top_conf.append(score)
                    top_label.append(label_name)
                    top_bboxes.append(coords)
                    j = j + 1

        # 濡傛灉涓嶅瓨鍦ㄦ弧瓒抽棬闄愮殑棰勬祴妗嗭紝鐩存帴杩斿洖鍘熷浘
        if len(top_conf)<=0:
            return image
        
        top_conf = np.array(top_conf)
        top_label = np.array(top_label)
        top_bboxes = np.array(top_bboxes)
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0], -1),np.expand_dims(top_bboxes[:,1], -1),np.expand_dims(top_bboxes[:,2], -1),np.expand_dims(top_bboxes[:,3], -1)

        #-----------------------------------------------------------#
        #   鍘绘帀鐏版潯閮ㄥ垎
        #-----------------------------------------------------------#
        if self.letterbox_image:
            boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)
        else:
            top_xmin = top_xmin * image_shape[1]
            top_ymin = top_ymin * image_shape[0]
            top_xmax = top_xmax * image_shape[1]
            top_ymax = top_ymax * image_shape[0]
            boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
            
        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        for i, c in enumerate(top_label):
            predicted_class = c
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 鐢绘妗�
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image


    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   缁欏浘鍍忓鍔犵伆鏉★紝瀹炵幇涓嶅け鐪熺殑resize
        #   涔熷彲浠ョ洿鎺esize杩涜璇嗗埆
        #---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.input_shape[1],self.input_shape[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.input_shape[1],self.input_shape[0]), Image.BICUBIC)

        photo = np.array(crop_img,dtype = np.float64)
        with torch.no_grad():
            photo = torch.from_numpy(np.expand_dims(np.transpose(photo-MEANS,(2,0,1)),0)).type(torch.FloatTensor)
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
                    
            if len(top_conf)>0:
                top_conf = np.array(top_conf)
                top_label = np.array(top_label)
                top_bboxes = np.array(top_bboxes)
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
                #-----------------------------------------------------------#
                #   鍘绘帀鐏版潯閮ㄥ垎
                #-----------------------------------------------------------#
                if self.letterbox_image:
                    boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)
                else:
                    top_xmin = top_xmin * image_shape[1]
                    top_ymin = top_ymin * image_shape[0]
                    top_xmax = top_xmax * image_shape[1]
                    top_ymax = top_ymax * image_shape[0]
                    boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
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
                        
                if len(top_conf)>0:
                    top_conf = np.array(top_conf)
                    top_label = np.array(top_label)
                    top_bboxes = np.array(top_bboxes)
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)
                    #-----------------------------------------------------------#
                    #   鍘绘帀鐏版潯閮ㄥ垎
                    #-----------------------------------------------------------#
                    if self.letterbox_image:
                        boxes = ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)
                    else:
                        top_xmin = top_xmin * image_shape[1]
                        top_ymin = top_ymin * image_shape[0]
                        top_xmax = top_xmax * image_shape[1]
                        top_ymax = top_ymax * image_shape[0]
                        boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
