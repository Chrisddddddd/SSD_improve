import torch.nn as nn
import torchvision

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]


'''
璇ヤ唬鐮佺敤浜庤幏寰梀GG涓诲共鐗瑰緛鎻愬彇缃戠粶鐨勮緭鍑恒��
杈撳叆鍙橀噺i浠ｈ〃鐨勬槸杈撳叆鍥剧墖鐨勯�氶亾鏁帮紝閫氬父涓�3銆�

涓�鑸潵璁诧紝杈撳叆鍥惧儚涓�(300, 300, 3)锛岄殢鐫�base鐨勫惊鐜紝鐗瑰緛灞傚彉鍖栧涓嬶細
300,300,3 -> 300,300,64 -> 300,300,64 -> 150,150,64 -> 150,150,128 -> 150,150,128 -> 75,75,128 -> 75,75,256 -> 75,75,256 -> 75,75,256 
-> 38,38,256 -> 38,38,512 -> 38,38,512 -> 38,38,512 -> 19,19,512 ->  19,19,512 ->  19,19,512 -> 19,19,512
鍒癰ase缁撴潫锛屾垜浠幏寰椾簡涓�涓�19,19,512鐨勭壒寰佸眰

涔嬪悗杩涜pool5銆乧onv6銆乧onv7銆�
'''
def vgg(i):
    layers = []
    in_channels = i
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers
#print(len(vgg(3)))
#print (vgg(3)[22],'\n')