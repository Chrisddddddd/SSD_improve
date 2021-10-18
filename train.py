import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from nets.ssd import get_ssd
from nets.ssd_training import LossHistory, MultiBoxLoss, weights_init
from utils.config import Config
from utils.dataloader import SSDDataset, ssd_dataset_collate

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------#
#   杩欓噷鐪嬪埌鐨則rain.py鍜岃棰戜笂涓嶅お涓�鏍�
#   鎴戦噸鏋勪簡涓�涓媡rain.py锛屾坊鍔犱簡楠岃瘉闆�
#   杩欐牱璁粌鐨勬椂鍊欏彲浠ユ湁涓弬鑰冦��
#   璁粌鍓嶆敞鎰忓湪config.py閲岄潰淇敼num_classes
#   璁粌涓栦唬銆佸涔犵巼銆佹壒澶勭悊澶у皬绛夊弬鏁板湪鏈唬鐮侀潬涓嬬殑if True:鍐呰繘琛屼慨鏀广��
#-------------------------------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,criterion,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    loc_loss        = 0
    conf_loss       = 0
    loc_loss_val    = 0
    conf_loss_val   = 0

    net.train()
    print('Start Train')
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   鍓嶅悜浼犳挱
            #----------------------#
            out = net(images)
            #----------------------#
            #   娓呴浂姊害
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   璁＄畻鎹熷け
            #----------------------#
            loss_l, loss_c  = criterion(out, targets)
            loss            = loss_l + loss_c
            #----------------------#
            #   鍙嶅悜浼犳挱
            #----------------------#
            loss.backward()
            optimizer.step()

            loc_loss    += loss_l.item()
            conf_loss   += loss_c.item()

            pbar.set_postfix(**{'loc_loss'  : loc_loss / (iteration + 1), 
                                'conf_loss' : conf_loss / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

                out = net(images)
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)

                loc_loss_val    += loss_l.item()
                conf_loss_val   += loss_c.item()

                pbar.set_postfix(**{'loc_loss'  : loc_loss_val / (iteration + 1), 
                                    'conf_loss' : conf_loss_val / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)

    total_loss  = loc_loss + conf_loss
    val_loss    = loc_loss_val + conf_loss_val

    loss_history.append_loss(total_loss/(epoch_size+1), val_loss/(epoch_size_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch+1))

    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    return val_loss/(epoch_size_val+1)

#----------------------------------------------------#
#   妫�娴嬬簿搴AP鍜宲r鏇茬嚎璁＄畻鍙傝�冭棰�
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------#
    #   鏄惁浣跨敤Cuda
    #   娌℃湁GPU鍙互璁剧疆鎴怓alse
    #-------------------------------#
    Cuda = True
    #--------------------------------------------#
    #   涓庤棰戜腑涓嶅悓銆佹柊娣诲姞浜嗕富骞茬綉缁滅殑閫夋嫨
    #   鍒嗗埆瀹炵幇浜嗗熀浜巑obilenetv2鍜寁gg鐨剆sd
    #   鍙�氳繃淇敼backbone鍙橀噺杩涜涓诲共缃戠粶鐨勯�夋嫨
    #   vgg鎴栬�卪obilenet
    #---------------------------------------------#
    backbone = "vgg"

    model = get_ssd("train", Config["num_classes"], backbone)
    weights_init(model)
    #------------------------------------------------------#
    #   鏉冨�兼枃浠惰鐪婻EADME锛岀櫨搴︾綉鐩樹笅杞�
    #------------------------------------------------------#
    model_path = "logs/Epoch20-Total_Loss2.4878-Val_Loss2.0589.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    annotation_path = '2007_train.txt'
    #----------------------------------------------------------------------#
    #   楠岃瘉闆嗙殑鍒掑垎鍦╰rain.py浠ｇ爜閲岄潰杩涜
    #   2007_test.txt鍜�2007_val.txt閲岄潰娌℃湁鍐呭鏄甯哥殑銆傝缁冧笉浼氫娇鐢ㄥ埌銆�
    #   褰撳墠鍒掑垎鏂瑰紡涓嬶紝楠岃瘉闆嗗拰璁粌闆嗙殑姣斾緥涓�1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, Cuda)
    loss_history = LossHistory("logs/")

    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    epoch_list=[]
    loss_list=[]

    
    #------------------------------------------------------#
    #   涓诲共鐗瑰緛鎻愬彇缃戠粶鐗瑰緛閫氱敤锛屽喕缁撹缁冨彲浠ュ姞蹇缁冮�熷害
    #   涔熷彲浠ュ湪璁粌鍒濇湡闃叉鏉冨�艰鐮村潖銆�
    #   Init_Epoch涓鸿捣濮嬩笘浠�
    #   Freeze_Epoch涓哄喕缁撹缁冪殑涓栦唬
    #   Unfreeze_Epoch鎬昏缁冧笘浠�
    #   鎻愮ずOOM鎴栬�呮樉瀛樹笉瓒宠璋冨皬Batch_size
    #------------------------------------------------------#
    if True:
        lr              = 5e-4
        Batch_size      = 16
        Init_Epoch      = 0
        Freeze_Epoch    = 20

        optimizer       = optim.Adam(net.parameters(), lr=lr)
        lr_scheduler    = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        train_dataset   = SSDDataset(lines[:num_train], (Config["min_dim"], Config["min_dim"]), True)
        val_dataset     = SSDDataset(lines[num_train:], (Config["min_dim"], Config["min_dim"]), False)

        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=ssd_dataset_collate)

        if backbone == "vgg":
            for param in model.vgg.parameters():
                param.requires_grad = False
        else:
            for param in model.mobilenet.parameters():
                param.requires_grad = False

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("鏁版嵁闆嗚繃灏忥紝鏃犳硶杩涜璁粌锛岃鎵╁厖鏁版嵁闆嗐��")

        for epoch in range(Init_Epoch,Freeze_Epoch):
            val_loss = fit_one_epoch(net,criterion,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step(val_loss)
            
        print(val_loss)
        
        epoch_list.append(epoch)
        loss_list.append(val_loss)
    plt.plot(epoch_list,loss_list)
    plt.show()