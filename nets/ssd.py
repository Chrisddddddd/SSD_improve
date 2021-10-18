import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import Config

from nets.mobilenetv2 import InvertedResidual, mobilenet_v2
from nets.ssd_layers import Detect, L2Norm, PriorBox
from nets.vgg import vgg as add_vgg


class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes, confidence, nms_iou, backbone_name):
        super(SSD, self).__init__()
        self.phase          = phase
        self.num_classes    = num_classes
        self.cfg            = Config
        if backbone_name    == "vgg":
            self.vgg        = nn.ModuleList(base)
            self.L2Norm     = L2Norm(512, 20)
        else:
            self.mobilenet  = base
            self.L2Norm     = L2Norm(96, 20)
            
        self.extras         = nn.ModuleList(extras)
        self.priorbox       = PriorBox(backbone_name, self.cfg)
        with torch.no_grad():
            self.priors     = torch.tensor(self.priorbox.forward()).type(torch.FloatTensor)
        self.loc            = nn.ModuleList(head[0])
        self.conf           = nn.ModuleList(head[1])
        self.backbone_name  = backbone_name
        if phase == 'test':
            self.softmax    = nn.Softmax(dim=-1)
            self.detect     = Detect(num_classes, 0, 200, confidence, nms_iou)
        
    def forward(self, x):
        sources = list()
        loc     = list()
        conf    = list()

        #---------------------------#
        #   閼惧嘲绶眂onv4_3閻ㄥ嫬鍞寸�癸拷
        #   shape娑擄拷38,38,512
        #---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23):
                x = self.vgg[k](x)
        else:
            for k in range(14):
                x = self.mobilenet[k](x)
        #---------------------------#
        #   conv4_3閻ㄥ嫬鍞寸�癸拷
        #   闂囷拷鐟曚浇绻樼悰瀛�2閺嶅洤鍣崠锟�
        #---------------------------#
        s = self.L2Norm(x)
        sources.append(s)

        #---------------------------#
        #   閼惧嘲绶眂onv7閻ㄥ嫬鍞寸�癸拷
        #   shape娑擄拷19,19,1024
        #---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
        else:
            for k in range(14, len(self.mobilenet)):
                x = self.mobilenet[k](x)

        sources.append(x)
        #-------------------------------------------------------------#
        #   閸︹暆dd_extras閼惧嘲绶遍惃鍕瀵颁礁鐪伴柌锟�
        #   缁楋拷1鐏炲倶锟戒胶顑�3鐏炲倶锟戒胶顑�5鐏炲倶锟戒胶顑�7鐏炲倸褰叉禒銉ф暏閺夈儴绻樼悰灞芥礀瑜版帡顣╁ù瀣嫲閸掑棛琚０鍕ゴ閵嗭拷
        #   shape閸掑棗鍩嗘稉锟�(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        #-------------------------------------------------------------#      
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if self.backbone_name == "vgg":
                if k % 2 == 1:
                    sources.append(x)
            else:
                sources.append(x)

        #-------------------------------------------------------------#
        #   娑撻缚骞忓妤冩畱6娑擃亝婀侀弫鍫㈠瀵颁礁鐪板ǎ璇插閸ョ偛缍婃０鍕ゴ閸滃苯鍨庣猾濠氼暕濞达拷
        #-------------------------------------------------------------#      
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #-------------------------------------------------------------#
        #   鏉╂稖顢憆eshape閺傞�涚┒閸棗褰�
        #-------------------------------------------------------------#  
        loc     = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf    = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #-------------------------------------------------------------#
        #   loc娴兼eshape閸掔櫚atch_size,num_anchors,4
        #   conf娴兼eshap閸掔櫚atch_size,num_anchors,self.num_classes
        #   婵″倹鐏夐悽銊ょ艾妫板嫭绁撮惃鍕樈閿涘奔绱板ǎ璇插娑撳シetect閻€劋绨�电懓鍘涙灞绢攱鐟欙絿鐖滈敍宀冨箯瀵版顣╁ù瀣波閺嬶拷
        #   娑撳秶鏁ゆ禍搴暕濞村娈戠拠婵撶礉閻╁瓨甯存潻鏂挎礀缂冩垹绮堕惃鍕礀瑜版帡顣╁ù瀣波閺嬫粌鎷伴崚鍡欒妫板嫭绁寸紒鎾寸亯閻€劋绨拋顓犵矊
        #-------------------------------------------------------------#     
        if self.phase == "test":
            output = self.detect.forward(
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors              
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

def add_extras(i, backbone_name):
    layers = []
    in_channels = i
    
    if backbone_name=='vgg':
        # Block 6
        # 19,19,1024 -> 10,10,512
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

        # Block 7
        # 10,10,512 -> 5,5,256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # Block 8
        # 5,5,256 -> 3,3,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
        
        # Block 9
        # 3,3,256 -> 1,1,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    else:
        layers += [InvertedResidual(in_channels, 512, stride=2, expand_ratio=0.2)]
        layers += [InvertedResidual(512, 256, stride=2, expand_ratio=0.25)]
        layers += [InvertedResidual(256, 256, stride=2, expand_ratio=0.5)]
        layers += [InvertedResidual(256, 64, stride=2, expand_ratio=0.25)]
        
    return layers

def get_ssd(phase, num_classes, backbone_name, confidence=0.5, nms_iou=0.45):
    #---------------------------------------------------#
    #   add_vgg閹稿洨娈戦弰顖氬閸忣櫦gg娑撹鍏遍悧鐟扮窙閹绘劕褰囩純鎴犵捕閵嗭拷
    #   鐠囥儳缍夌紒婊呮畱閺堬拷閸氬簼绔存稉顏嗗瀵颁礁鐪伴弰鐥梠nv7閸氬海娈戠紒鎾寸亯閵嗭拷
    #   shape娑擄拷19,19,1024閵嗭拷
    #
    #   娑撹桨绨￠弴鏉戙偨閻ㄥ嫭褰侀崣鏍у毉閻楃懓绶涢悽銊ょ艾妫板嫭绁撮妴锟�
    #   SSD缂冩垹绮舵导姘辨埛缂侇叀绻樼悰灞肩瑓闁插洦鐗遍妴锟�
    #   add_extras閺勵垶顤傛径鏍︾瑓闁插洦鐗遍惃鍕劥閸掑棎锟斤拷   
    #---------------------------------------------------#
    if backbone_name=='vgg':
        backbone, extra_layers = add_vgg(3), add_extras(1024, backbone_name)
        mbox = [7, 9, 9, 9, 7, 7]
    else:
        backbone, extra_layers = mobilenet_v2().features, add_extras(1280, backbone_name)
        mbox = [6, 6, 6, 6, 6, 6]

    loc_layers = []
    conf_layers = []
                      
    if backbone_name=='vgg':
        backbone_source = [21, -2]
        #---------------------------------------------------#
        #   閸︹暆dd_vgg閼惧嘲绶遍惃鍕瀵颁礁鐪伴柌锟�
        #   缁楋拷21鐏炲倸鎷�-2鐏炲倸褰叉禒銉ф暏閺夈儴绻樼悰灞芥礀瑜版帡顣╁ù瀣嫲閸掑棛琚０鍕ゴ閵嗭拷
        #   閸掑棗鍩嗛弰鐥梠nv4-3(38,38,512)閸滃畱onv7(19,19,1024)閻ㄥ嫯绶崙锟�
        #---------------------------------------------------#
        for k, v in enumerate(backbone_source):
            loc_layers += [nn.Conv2d(backbone[v].out_channels,
                                    mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(backbone[v].out_channels,
                            mbox[k] * num_classes, kernel_size=3, padding=1)]
        #-------------------------------------------------------------#
        #   閸︹暆dd_extras閼惧嘲绶遍惃鍕瀵颁礁鐪伴柌锟�
        #   缁楋拷1鐏炲倶锟戒胶顑�3鐏炲倶锟戒胶顑�5鐏炲倶锟戒胶顑�7鐏炲倸褰叉禒銉ф暏閺夈儴绻樼悰灞芥礀瑜版帡顣╁ù瀣嫲閸掑棛琚０鍕ゴ閵嗭拷
        #   shape閸掑棗鍩嗘稉锟�(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        #-------------------------------------------------------------#  
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                    * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                    * num_classes, kernel_size=3, padding=1)]
    else:
        backbone_source = [13, -1]
        for k, v in enumerate(backbone_source):
            loc_layers += [nn.Conv2d(backbone[v].out_channels,
                                    mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(backbone[v].out_channels,
                            mbox[k] * num_classes, kernel_size=3, padding=1)]

        for k, v in enumerate(extra_layers, 2):
            loc_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                    * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, mbox[k]
                                    * num_classes, kernel_size=3, padding=1)]

    #-------------------------------------------------------------#
    #   add_vgg閸滃畮dd_extras閿涘奔绔撮崗杈箯瀵版ぞ绨�6娑擃亝婀侀弫鍫㈠瀵颁礁鐪伴敍瀹籬ape閸掑棗鍩嗘稉鐚寸窗
    #   (38,38,512), (19,19,1024), (10,10,512), 
    #   (5,5,256), (3,3,256), (1,1,256)
    #-------------------------------------------------------------#
    SSD_MODEL = SSD(phase, backbone, extra_layers, (loc_layers, conf_layers), num_classes, confidence, nms_iou, backbone_name)
    return SSD_MODEL