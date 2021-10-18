# -*- coding: gbk -*- 
#----------------------------------------------------#
#   ����Ƶ�е�predict.py�������޸ģ�
#   ������ͼƬԤ�⡢����ͷ����FPS���Թ���
#   ���ϵ���һ��py�ļ��У�ͨ��ָ��mode����ģʽ���޸ġ�
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from ssd import SSD

if __name__ == "__main__":
    ssd = SSD()
    #-------------------------------------------------------------------------#
    #   mode����ָ�����Ե�ģʽ��
    #   'predict'��ʾ����ͼƬԤ��
    #   'video'��ʾ��Ƶ���
    #   'fps'��ʾ����fps
    #-------------------------------------------------------------------------#
    mode = "video"
    #-------------------------------------------------------------------------#
    #   video_path����ָ����Ƶ��·������video_path=0ʱ��ʾ�������ͷ
    #   video_save_path��ʾ��Ƶ�����·������video_save_path=""ʱ��ʾ������
    #   video_fps���ڱ������Ƶ��fps
    #   video_path��video_save_path��video_fps����mode='video'ʱ��Ч
    #   ������Ƶʱ��Ҫctrl+c�˳��Ż���������ı��沽�裬����ֱ�ӽ�������
    #-------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    if mode == "predict":
        '''
        1���ô����޷�ֱ�ӽ�������Ԥ�⣬�����Ҫ����Ԥ�⣬��������os.listdir()�����ļ��У�����Image.open��ͼƬ�ļ�����Ԥ�⡣
        �������̿��Բο�get_dr_txt.py����get_dr_txt.py��ʵ���˱�����ʵ����Ŀ����Ϣ�ı��档
        2�������Ҫ���м�����ͼƬ�ı��棬����r_image.save("img.jpg")���ɱ��棬ֱ����predict.py������޸ļ��ɡ� 
        3�������Ҫ���Ԥ�������꣬���Խ���ssd.detect_image�������ڻ�ͼ���ֶ�ȡtop��left��bottom��right���ĸ�ֵ��
        4�������Ҫ����Ԥ����ȡ��Ŀ�꣬���Խ���ssd.detect_image�������ڻ�ͼ�������û�ȡ����top��left��bottom��right���ĸ�ֵ
        ��ԭͼ�����þ���ķ�ʽ���н�ȡ��
        5�������Ҫ��Ԥ��ͼ��д������֣������⵽���ض�Ŀ������������Խ���ssd.detect_image�������ڻ�ͼ���ֶ�predicted_class�����жϣ�
        �����ж�if predicted_class == 'car': �����жϵ�ǰĿ���Ƿ�Ϊ����Ȼ���¼�������ɡ�����draw.text����д�֡�
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = ssd.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            # ��ȡĳһ֡
            ref,frame=capture.read()
            # ��ʽת�䣬BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # ת���Image
            frame = Image.fromarray(np.uint8(frame))
            # ���м��
            frame = np.array(ssd.detect_image(frame))
            # RGBtoBGR����opencv��ʾ��ʽ
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(5) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==5:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 100
        img = Image.open('img/street.jpg')
        tact_time = ssd.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")