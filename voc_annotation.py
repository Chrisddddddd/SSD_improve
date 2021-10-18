#---------------------------------------------#
#   杩愯鍓嶄竴瀹氳淇敼classes
#   濡傛灉鐢熸垚鐨�2007_train.txt閲岄潰娌℃湁鐩爣淇℃伅
#   閭ｄ箞灏辨槸鍥犱负classes娌℃湁璁惧畾姝ｇ‘
#---------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
#-----------------------------------------------------#
#   杩欓噷璁惧畾鐨刢lasses椤哄簭瑕佸拰model_data閲岀殑txt涓�鏍�
#-----------------------------------------------------#
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert_annotation(year, image_id, list_file):
    in_file = open('E:/DATASET/VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for year, image_set in sets:
    image_ids = open('E:/DATASET/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set), encoding='utf-8').read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
    for image_id in image_ids:
        list_file.write('E:/DATASET/VOCdevkit/VOC%s/images/%s.jpg' % (year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
