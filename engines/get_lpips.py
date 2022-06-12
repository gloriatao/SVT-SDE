import lpips
import os, cv2
import numpy as np

class util_lpips():
    def __init__(self, net='alex', use_gpu=False):
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.loss_fn.cuda()
    def cal_lpips(self, img0, img1):
        img0 = lpips.im2tensor(img0)
        img1 = lpips.im2tensor(img1)
        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = self.loss_fn.forward(img0, img1)
        return dist01

# pred_path = '/media/cygzz/data/rtao/projects/stereo/pred/davinci_psm_mono_lpips'
# pred_path = '/media/cygzz/data/rtao/projects/stereo/pred/davinci_vit_freq_lpips'
pred_path = '/media/cygzz/data/rtao/projects/stereo/pred/davinci_vit_nofreq_lpips'
file_paths = '/media/cygzz/data/rtao/data/davinci/daVinci/test'
val_list = sorted(os.listdir(os.path.join(file_paths, 'image_0')))

uls = util_lpips()
pips = []
for id in val_list:
    imgL = cv2.imread(os.path.join(file_paths, 'image_0', id))
    imgR = cv2.imread(os.path.join(file_paths, 'image_1', id))

    rimgL = cv2.imread(os.path.join(pred_path, id[:-4]+'.png_L.png'))
    rimgR = cv2.imread(os.path.join(pred_path, id[:-4] + '.png_R.png'))

    dist = (uls.cal_lpips(rimgL, imgL).item()+uls.cal_lpips(rimgR, imgR).item())/2
    pips.append(dist)

print('pips', np.mean(np.array(pips)), np.std(np.array(pips)))

#vfe
# pips 0.15324405219160853 0.049377548783909486 (low is better)

#vfe vit
# pips 0.14679773589781356 0.04762785647248114

#vfe vit freq
# pips 0.13323364623057762 0.04397650067738475