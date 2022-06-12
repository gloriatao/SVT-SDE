import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def apply_disparity(img, disp):
    # print(img.shape)
    N, C, H, W = img.size()
    mesh_x, mesh_y = torch.tensor(np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H), indexing='xy')).type_as(img)
    mesh_x = mesh_x.repeat(N, 1, 1)
    mesh_y = mesh_y.repeat(N, 1, 1)
    #grid is (N, H, W, 2)
    grid = torch.stack((mesh_x + disp.squeeze(), mesh_y), 3)
    #grid must be in range [-1, 1]
    output = F.grid_sample(img, grid * 2 - 1, mode='bilinear', padding_mode='zeros')
    return output

def get_images(img, disp, get):
    if get == 'left':
        return apply_disparity(img, -disp)
    elif get == 'right':
        return apply_disparity(img, disp)

def loss_lr(disp_left, disp_right):
    # Left-Right Disparity Consistency Loss
    #x_to_y Projected disparities
    right_to_left_disp = get_images(disp_right, disp_left, 'left')
    left_to_right_disp = get_images(disp_left, disp_right, 'right')
    left_LR = torch.mean(torch.abs(disp_left - right_to_left_disp))
    right_LR = torch.mean(torch.abs(disp_right - left_to_right_disp))
    LR_loss = left_LR + right_LR
    return LR_loss

def loss_ds(disp_left, left_pyramid, disp_right, right_pyramid):
    # Disparity Smoothness Loss
    def x_grad(img):
        img = F.pad(img, (0, 1, 0, 0), mode='replicate')
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        return grad_x
    def y_grad(img):
        img = F.pad(img, (0, 0, 0, 1), mode='replicate')
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        return grad_y
    def disp_smoothness(pyramid, disp, n=4):
        disp_x_grad = [x_grad(i) for i in disp]
        disp_y_grad = [y_grad(j) for j in disp]
        image_x_grad = [x_grad(i) for i in pyramid]
        image_y_grad = [y_grad(j) for j in pyramid]
        #e^(-|x|) weights, gradient negatively exponential to weights
        #average over all pixels in C dimension
        #but supposed to be locally smooth?
        weights_x = [torch.exp(-torch.mean(torch.abs(i), 1, keepdim=True)) for i in image_x_grad]
        weights_y = [torch.exp(-torch.mean(torch.abs(j), 1, keepdim=True)) for j in image_y_grad]
        smoothness_x = [disp_x_grad[i] * weights_x[i] for i in range(n)]
        smoothness_y = [disp_y_grad[j] * weights_y[j] for j in range(n)]
        smoothness = [torch.mean(torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])) / 2 ** i for i in range(n)]
        return smoothness

    # Disparity Smoothness Loss
    left_DS = disp_smoothness(disp_left, left_pyramid)
    right_DS = disp_smoothness(disp_right, right_pyramid)

    DS_loss = sum(left_DS + right_DS)

    return DS_loss

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class Loss(nn.Module):

    #alpha_AP = Appearance Matching Loss weight
    #alpha_DS = Disparity Smoothness Loss weight
    #alpha_LR = Left-Right Consistency Loss weight

    def __init__(self, n=1, alpha_AP=0.85, alpha_DS=0.1, alpha_LR=1.0):
        super(Loss, self).__init__()

        self.n = n
        self.alpha_AP = alpha_AP
        self.alpha_DS = alpha_DS
        self.alpha_LR = alpha_LR

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

    def get_MSE(self, left_pyramid, left_est, right_pyramid, right_est):
        left = nn.MSELoss()(left_pyramid, left_est)
        right = nn.MSELoss()(right_pyramid, right_est)
        return (left + right) * 0.1

    def get_DIV(self, disp_left, disp_right):
        left = torch.gradient(disp_left)
        right = torch.gradient(disp_right)
        return (left + right) * 0.1

    def forward(self, disp, target):
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

        #MSE Loss, divergence loss
        MSE_loss = self.get_MSE(left_pyramid, left_est, right_pyramid, right_est)
        # SIS_loss = self.get_SIS(left_pyramid, left_est, right_pyramid, right_est)
        # DIV_loss = self.get_DIV(disp_left, disp_right)
        #Total Loss
        # AP_loss + LR_loss + DS_loss

        return AP_loss, LR_loss, DS_loss, left_est, right_est

def test():
    model = Loss()
    disp = torch.randn([1, 2, 1, 192*2, 96*2])
    x = torch.randn([1, 3, 192*2, 96*2])
    y = model(disp, (x, x))
    return

# test()