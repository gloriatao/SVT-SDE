import torch
import cv2
import os
import numpy as np
from utils_freq.freq_pixel_loss import find_freq


def create_filter(size, r):
    x = torch.ones((size[0], size[1]))
    for i in range(size[0]):
        for j in range(size[1]):
            if (i - size[0]/2 + 0.5)**2 + (j - size[1]/2 + 0.5)**2 < r **2:
                x[i, j] = 0
    mask_h = x.clone()
    mask_l = torch.ones((size[0], size[1])) - x
    return mask_h, mask_l

r = 115
mask_l, mask_h = create_filter(size=(192, 193), r=r)

img_path = '019215_propose.png'
img = cv2.imread(img_path)[:,-384:,:]
img = torch.FloatTensor(img).permute(2,0,1).unsqueeze(0)
image_gray = img[:, 0] * 0.299 + img[:, 1] * 0.587 + img[:, 2] * 0.114

fftimg = torch.fft.rfft2(image_gray)
fftimg.real = fftimg.real * mask_h.squeeze()
fftimg.imag = fftimg.imag * mask_h.squeeze()
ifftimg = torch.fft.irfft2(fftimg, dim=(-2, -1))

ifftimg = ifftimg.squeeze().cpu().numpy()
ifftimg = (ifftimg - np.min(ifftimg))
ifftimg = ifftimg/np.max(ifftimg)*255

cv2.imwrite(str(r)+'ifftimg_high_propose.png', ifftimg)
# cv2.imwrite(os.path.join(out_path, str(k)+'ifftimg_gt.png'), high_comp.squeeze().numpy())

fftimg = torch.fft.rfft2(img)
fftimg.real = fftimg.real * mask_l.squeeze()
fftimg.imag = fftimg.imag * mask_l.squeeze()
ifftimg = torch.fft.irfft2(fftimg, dim=(-2, -1))

ifftimg = ifftimg.squeeze().cpu().numpy()
ifftimg = (ifftimg - np.min(ifftimg))
ifftimg = ifftimg/np.max(ifftimg)*255
cv2.imwrite(str(r)+'ifftimg_low_propose.png', ifftimg.transpose(1,2,0))

    # fft = torch.fft.rfft2(img, dim=(-2, -1))
    # output = torch.stack((fft.real, fft.imag), -1)
    # fft_mag = torch.log(1 + torch.sqrt(output[..., 0] ** 2 + output[..., 1] ** 2 + 1e-8))



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