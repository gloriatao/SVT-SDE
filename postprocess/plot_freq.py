import torch
import cv2
import os
import numpy as np
from utils_freq.freq_pixel_loss import find_freq

def get_gaussian_kernel(size=21):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    return kernel

def create_filter(size, r):
    x = torch.ones((size[0], size[1]))
    for i in range(size[0]):
        for j in range(size[1]):
            if (i - size[0]/2 + 0.5)**2 + (j - size[1]/2 + 0.5)**2 < r **2:
                x[i, j] = 0
    mask_h = x.clone()
    mask_l = torch.ones((size[0], size[1])) - x
    return mask_h, mask_l

kernel = get_gaussian_kernel()
mask_l, mask_h = create_filter(size=(192, 193), r=95)

img_path = '/media/runze/DATA/Dasaset/daVinci/train/image_0'
out_path = '/media/runze/DATA/rtao/projects/stereo/pred/davinci_vit_train/freq'
img_list = sorted(os.listdir((img_path)))[120:168]

for id in img_list:
    img = cv2.imread(os.path.join(img_path, id))
    img = torch.FloatTensor(img).permute(2,0,1).unsqueeze(0)
    low_comp, high_comp = find_freq(img, kernel)
    high_comp = high_comp-torch.min(high_comp)
    high_comp = high_comp/torch.max(high_comp)*255
    cv2.imwrite(os.path.join(out_path, id[:-4]+'_low.png'), low_comp.squeeze().permute(1,2,0).numpy())
    cv2.imwrite(os.path.join(out_path, id[:-4]+'_high.png'), high_comp.squeeze().numpy())

    # fft = torch.fft.rfft2(img, dim=(-2, -1))
    # output = torch.stack((fft.real, fft.imag), -1)
    # fft_mag = torch.log(1 + torch.sqrt(output[..., 0] ** 2 + output[..., 1] ** 2 + 1e-8))

    pass



# tdisp = torch.fft.rfft2(disp.squeeze(1))
# for i in range(tdisp.shape[0]):
#     r = tdisp[i].real.detach().cpu().numpy()
#     plot_debug(np.log(1+np.abs(r)), id=str(i)+'real.png')
#     m = tdisp[i].imag.detach().cpu().numpy()
#     plot_debug(np.log(1+np.abs(m)), id=str(i)+'imag.png')
# tdisp.real = tdisp.real * mask
# tdisp.imag = tdisp.imag * mask
# idisp = torch.fft.irfft2(tdisp, dim=(-2, -1))
# for i in range(idisp.shape[0]):
#     plot_debug(idisp.squeeze(1)[i].detach().cpu().numpy(), id=str(i)+'idisp.png')
#     plot_debug(disp.squeeze(1)[i].detach().cpu().numpy(), id=str(i)+'disp.png')
#
#
# tdisp = torch.fft.rfft2(image_gray.squeeze(1))
# for i in range(tdisp.shape[0]):
#     r = tdisp[i].real.detach().cpu().numpy()
#     plot_debug(np.log(1+np.abs(r)), id=str(i)+'real_img.png')
#     m = tdisp[i].imag.detach().cpu().numpy()
#     plot_debug(np.log(1+np.abs(m)), id=str(i)+'imag_img.png')
# tdisp.real = tdisp.real * mask
# tdisp.imag = tdisp.imag * mask
# idisp = torch.fft.irfft2(tdisp, dim=(-2, -1))
# for i in range(idisp.shape[0]):
#     plot_debug(idisp.squeeze(1)[i].detach().cpu().numpy(), id=str(i)+'iimg.png')
#     plot_debug(image_gray.squeeze(1)[i].detach().cpu().numpy(), id=str(i)+'img.png')