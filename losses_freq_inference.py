import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2, os
from utils_freq.freq_pixel_loss import find_freq

def apply_disparity(img, disp):
    # print(img.shape)
    N, C, H, W = img.size()
    mesh_x, mesh_y = torch.tensor(np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H), indexing='xy')).type_as(img)
    mesh_x = mesh_x.repeat(N, 1, 1)
    mesh_y = mesh_y.repeat(N, 1, 1)
    grid = torch.stack((mesh_x + disp.squeeze(), mesh_y), 3)
    gridd = grid * 2 - 1
    output = F.grid_sample(img, gridd, mode='bilinear', padding_mode='zeros')
    return output

def plot_debug(img, id, debug_path='/media/cygzz/data/rtao/projects/stereo/debug'):  #  '/media/cygzz/data/rtao/projects/stereo/debug'
    img = img-np.min(img)
    img = (img/np.max(img)*255).astype(np.uint8)
    cv2.imwrite(os.path.join(debug_path, id), img)
    return

def scale_image(img):
    img = img-np.min(img)
    img = (img/np.max(img)*255).astype(np.uint8)
    return img

class Loss(nn.Module):

    def __init__(self, n=1, alpha_AP=0.85, alpha_DS=0.1, alpha_LR=1.0):
        super(Loss, self).__init__()
        self.n = n
        self.alpha_AP = alpha_AP
        self.alpha_DS = alpha_DS
        self.alpha_LR = alpha_LR
        self.alpha_SPE = 0.1
        self.alpha_SPA = 0.85
        self.r = 114#94
        self.k = 15#21
        self.mask_l, self.mask_h = self.create_filter(size=(192, 193), r=self.r) # r=95
        self.kernel = self.get_gaussian_kernel(size=self.k)

    def create_filter(self, size, r):
        x = torch.ones((size[0], size[1]))
        for i in range(size[0]):
            for j in range(size[1]):
                if (i - size[0]/2 + 0.5)**2 + (j - size[1]/2 + 0.5)**2 < r **2:
                    x[i, j] = 0
        mask_h = x.clone()
        mask_l = torch.ones((size[0], size[1])) - x
        return mask_h, mask_l

    def calc_fft(self, image):
        '''image is tensor, N*C*H*W'''
        fft = torch.fft.rfft2(image, dim=(-2, -1))
        output = torch.stack((fft.real, fft.imag), -1)
        fft_mag = torch.log(1 + torch.sqrt(output[..., 0] ** 2 + output[..., 1] ** 2 + 1e-8))
        return fft_mag

    def fft_L1_loss_mask(self, disp, image, ids=None, output_dir=None):
        mask = self.mask_h.unsqueeze(0).unsqueeze(0).repeat(disp.shape[0], 1, 1, 1).to(disp.device)
        criterion_L1 = torch.nn.L1Loss()
        image_gray = image[:, 0] * 0.299 + image[:, 1] * 0.587 + image[:, 2] * 0.114
        fake_fft = self.calc_fft(disp)
        real_fft = self.calc_fft(image_gray.unsqueeze(1))
        loss = criterion_L1(fake_fft * mask, real_fft * mask)
        if ids != None:
            tdisp = torch.fft.rfft2(disp.squeeze(1))
            tdisp.real = tdisp.real * mask.squeeze()
            tdisp.imag = tdisp.imag * mask.squeeze()
            idisp = torch.fft.irfft2(tdisp, dim=(-2, -1))

            timg = torch.fft.rfft2(image_gray.squeeze(1))
            timg.real = timg.real * mask.squeeze()
            timg.imag = timg.imag * mask.squeeze()
            iimg = torch.fft.irfft2(timg, dim=(-2, -1))

            for i in range(idisp.shape[0]):
                img1 = disp.squeeze(1)[i].detach().cpu().numpy()*384/110*255
                img2 = idisp.squeeze(1)[i].detach().cpu().numpy()
                img3 = iimg.squeeze(1)[i].detach().cpu().numpy()

                img2 = (img2 - np.min(img2))
                img2 = img2/np.max(img2)*255

                img3 = (img3 - np.min(img3))
                img3 = img3/np.max(img3)*255

                img = np.hstack((img1, img2, img3))
                # cv2.imwrite(os.path.join(output_dir, ids[i][:-4]+'_fft_hfeq'+str(self.r)+'.png'), img)

        return loss

    def fft_L1_loss_color(self, image, recon, ids=None, output_dir=None):
        image_gray = image[:, 0] * 0.299 + image[:, 1] * 0.587 + image[:, 2] * 0.114
        recon_gray = recon[:, 0] * 0.299 + recon[:, 1] * 0.587 + recon[:, 2] * 0.114
        mask = self.mask_h.unsqueeze(0).unsqueeze(0).repeat(image_gray.shape[0], 1, 1, 1).to(image_gray.device)
        criterion_L1 = torch.nn.L1Loss()
        image_gray = image[:, 0] * 0.299 + image[:, 1] * 0.587 + image[:, 2] * 0.114
        fake_fft = self.calc_fft(recon_gray)
        real_fft = self.calc_fft(image_gray.unsqueeze(1))
        loss = criterion_L1(fake_fft * mask, real_fft * mask)
        return loss

    def get_gaussian_kernel(self, size=21):
        kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)#.cuda()
        return kernel

    def build_pyramid(self, img, n):
        pyramid = [img]
        h = img.shape[2]
        w = img.shape[3]
        for i in range(n - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            pyramid.append(F.interpolate(pyramid[i], (nh, nw), mode='bilinear', align_corners=True))

        return pyramid

    def x_grad(self, img):

        img = F.pad(img, (0, 1, 0, 0), mode='replicate')
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]

        return grad_x

    def y_grad(self, img):

        img = F.pad(img, (0, 0, 0, 1), mode='replicate')
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]

        return grad_y

    def get_images(self, pyramid, disp, get):
        if get == 'left':
            return apply_disparity(pyramid, -disp)
        elif get == 'right':
            return apply_disparity(pyramid, disp)
        else:
            raise ValueError('Argument get must be either \'left\' or \'right\'')

    def disp_smoothness(self, disp, pyramid):

        disp_x_grad = self.x_grad(disp)
        disp_y_grad = self.y_grad(disp)

        image_x_grad = self.x_grad(pyramid)
        image_y_grad = self.y_grad(pyramid)

        #e^(-|x|) weights, gradient negatively exponential to weights
        #average over all pixels in C dimension
        #but supposed to be locally smooth?
        weights_x = torch.exp(-torch.mean(torch.abs(image_x_grad), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_y_grad), 1, keepdim=True))
        smoothness_x = disp_x_grad * weights_x
        smoothness_y = disp_y_grad * weights_y
        smoothness = torch.mean(torch.abs(smoothness_x) + torch.abs(smoothness_y)) / 2 ** 1
        return smoothness

    def DSSIM(self, x, y):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        avgpool = nn.AvgPool2d(3, 1)

        mu_x = avgpool(x)
        mu_y = avgpool(y)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2

        #sigma = E[X^2] - E[X]^2
        sigma_x = avgpool(x ** 2) - mu_x_sq
        sigma_y = avgpool(y ** 2) - mu_y_sq
        #cov = E[XY] - E[X]E[Y]
        cov_xy = avgpool(x * y) - (mu_x * mu_y)

        SSIM_top = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        SSIM_bot = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)

        SSIM = SSIM_top / SSIM_bot
        DSSIM = torch.mean(torch.clamp((1 - SSIM) / 2, 0, 1))

        return DSSIM

    def L1(self, pyramid, est):
        L1_loss = torch.mean(torch.abs(pyramid - est))
        return L1_loss

    def get_AP(self, left_pyramid, left_est, right_pyramid, right_est):

        #L1 Loss
        left_l1 = self.L1(left_pyramid, left_est)
        right_l1 = self.L1(right_pyramid, right_est)

        #DSSIM
        left_dssim = self.DSSIM(left_pyramid, left_est)
        right_dssim = self.DSSIM(right_pyramid, right_est)

        left_AP = self.alpha_AP * left_dssim + (1 - self.alpha_AP) * left_l1
        right_AP = self.alpha_AP * right_dssim + (1 - self.alpha_AP) * right_l1

        AP_loss = left_AP + right_AP

        return AP_loss * self.alpha_AP

    def get_LR(self, disp_left, disp_right_to_left, disp_right, disp_left_to_right):
        left_LR = torch.mean(torch.abs(disp_left - disp_right_to_left))
        right_LR = torch.mean(torch.abs(disp_right - disp_left_to_right))
        LR_loss = left_LR + right_LR
        return LR_loss * self.alpha_LR

    def get_DS(self, disp_left, left_pyramid, disp_right, right_pyramid):
        left_DS = self.disp_smoothness(disp_left, left_pyramid)
        right_DS = self.disp_smoothness(disp_right, right_pyramid)
        DS_loss = left_DS + right_DS
        return DS_loss * self.alpha_DS

    def get_SPE_REC(self, left_pyramid, left_est, right_pyramid, right_est, ids=None, output_dir=None):
        #L1 Loss
        left_l1 = self.fft_L1_loss_color(left_pyramid, left_est, ids, output_dir)
        right_l1 = self.fft_L1_loss_color(right_pyramid, right_est, ids, output_dir)
        AP_loss_freq = (left_l1 + right_l1)*0.1
        return AP_loss_freq

    def get_SPE_MAT(self,disp_left, left_pyramid, disp_right, right_pyramid, ids=None, output_dir=None):
        disp_left_ = disp_left.clone()
        disp_left_[disp_left_>0.5] = torch.min(disp_left_)
        disp_right_ = disp_right.clone()
        disp_right_[disp_right_>0.5] = torch.min(disp_right_)

        left_l1 = self.fft_L1_loss_mask(disp_left_, left_pyramid, ids, output_dir)
        right_l1 = self.fft_L1_loss_mask(disp_right_, right_pyramid, ids, output_dir)
        ST_loss = (left_l1+right_l1)*0.1
        return ST_loss

    def normalize(self, image):
        image = image - torch.min(image)
        image = image/torch.max(image)
        return image


    def get_SPA(self, left_pyramid, left_est, right_pyramid, right_est, disp_left, disp_right, ids=None, output_dir=None):
        left_pyramid_low_freq, left_pyramid_high_freq_gray = find_freq(left_pyramid, self.kernel)
        left_est_low_freq, left_est_high_freq_gray = find_freq(left_est, self.kernel)
        left_rec_l1 = self.L1(left_pyramid_low_freq, left_est_low_freq) + self.L1(left_pyramid_high_freq_gray, left_est_high_freq_gray)

        right_pyramid_low_freq, right_pyramid_high_freq_gray = find_freq(right_pyramid, self.kernel)
        right_est_low_freq, right_est_high_freq_gray = find_freq(right_est, self.kernel)
        right_rec_l1 = self.L1(right_pyramid_low_freq, right_est_low_freq) + self.L1(right_pyramid_high_freq_gray, right_est_high_freq_gray)

        SPA_REC_loss = (right_rec_l1 + left_rec_l1)*0.85

        return SPA_REC_loss

    def forward(self, disp, target, ids=None, output_dir=None):
        left_pyramid, right_pyramid = target
        #BUILD OUTPUTS
        #Estimated disparity pyramid
        if len(disp.shape) == 4:
            disp_left = disp[:, 0, :, :].unsqueeze(1)
            disp_right = disp[:, 1, :, :].unsqueeze(1)
        else:
            disp_left = disp[:, 0, :, :, :]
            disp_right = disp[:, 1, :, :,:]

        #Reconstructed images
        left_est = self.get_images(right_pyramid, disp_left, 'left')
        right_est = self.get_images(left_pyramid, disp_right, 'right')

        #x_to_y Projected disparities
        right_to_left_disp = self.get_images(disp_right, disp_left, 'left')
        left_to_right_disp = self.get_images(disp_left, disp_right, 'right')

        #AP, LR, DS Loss
        AP_loss = self.get_AP(left_pyramid, left_est, right_pyramid, right_est)
        LR_loss = self.get_LR(disp_left, right_to_left_disp, disp_right, left_to_right_disp)
        DS_loss = self.get_DS(disp_left, left_pyramid, disp_right, right_pyramid)
        # spectral supervision
        SPE_REC_loss = self.get_SPE_REC(left_pyramid, left_est, right_pyramid, right_est, ids, output_dir)

        # spatial supervision
        SPA_REC_loss = self.get_SPA(left_pyramid, left_est, right_pyramid, right_est, disp_left, disp_right, ids, output_dir)


        return left_est, right_est

def test():
    model = Loss()
    disp = torch.randn([1, 2, 1, 192*2, 96*2])
    x = torch.randn([1, 3, 192*2, 96*2])
    y = model(disp, (x, x))
    return

# test()