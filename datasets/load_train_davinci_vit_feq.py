from torch.utils.data import Dataset
from PIL import Image
import random, os, cv2, pickle, torch
import numpy as np
from utils_freq.freq_pixel_loss import find_freq, get_gaussian_kernel
from utils_freq.freq_fourier_loss import fft_L1_loss_color, decide_circle, fft_L1_loss_mask
debug_path = '/media/runze/DATA/rtao/projects/stereo/debug' #'/media/cygzz/data/rtao/projects/stereo/debug'
def save_image(image, id, debug_path='/media/runze/DATA/rtao/projects/stereo/debug'):
    image = image[0].permute(1,2,0).numpy()
    image = image-np.min(image)
    image = image/np.max(image) * 255
    cv2.imwrite(os.path.join(debug_path, id), image)
    return

class load_trainset(Dataset):
    def __init__(self, file_paths, clip_len=4, sampling_len=10):
        self.file_paths = file_paths
        self.clip_len = clip_len
        self.sampling_len = sampling_len
        self.train_list = sorted(os.listdir(os.path.join(file_paths, 'image_0')))
        self.max_len = len(self.train_list)
        # self.filter = self.create_filter(size=(192, 384), r=21)
        self.kernel = get_gaussian_kernel(21)

    def __len__(self):
        return (len(self.train_list) - self.clip_len * self.sampling_len) //self.clip_len  # sampling 3 from 1

    def create_filter(self, size, r):
        x = torch.ones((size[0], size[1]))
        for i in range(size[0]):
            for j in range(size[1]):
                if (i - size[0]/2 + 0.5)**2 + (j - size[1]/2 + 0.5)**2 < r **2:
                    x[i, j] = 0
        mask0 = x.clone()
        mask1 = torch.ones((size[0], size[1])) - x
        return mask0, mask1

    def fft_transform_color(self, image):
        shift = np.zeros(image.shape)
        shift_log = np.zeros(image.shape)
        # R
        r = np.fft.fft2(image[:,:,0])
        rshift = np.fft.fftshift(r)
        rshift_log = np.log(1+np.abs(rshift))
        # G
        g = np.fft.fft2(image[:,:,1])
        gshift = np.fft.fftshift(g)
        gshift_log = np.log(1+np.abs(gshift))
        # B
        b = np.fft.fft2(image[:,:,2])
        bshift = np.fft.fftshift(b)
        bshift_log = np.log(1+np.abs(bshift))

        shift[:,:,0], shift[:,:,1], shift[:,:,2] = rshift, gshift, bshift
        shift_log[:,:,0], shift_log[:,:,1], shift_log[:,:,2] = rshift_log, gshift_log, bshift_log
        return shift, shift_log

    def ifft_transform_color(self, shift):
        rshift, gshift, bshift = shift[:,:,0], shift[:,:,1], shift[:,:,2]
        image = np.zeros(shift.shape)
        # R
        fr = np.fft.ifftshift(rshift)
        image[:,:,0] = np.real(np.fft.ifft2(fr))
        # G
        fg = np.fft.ifftshift(gshift)
        image[:,:,1] = np.real(np.fft.ifft2(fg))
        # B
        fb = np.fft.ifftshift(bshift)
        image[:,:,2] = np.real(np.fft.ifft2(fb))
        return image

    def apply_filter(self, shift):
        shift_high = shift * self.filter[0].unsqueeze(-1).repeat(1,1,3).numpy()
        shift_low = shift * self.filter[1].unsqueeze(-1).repeat(1,1,3).numpy()
        return shift_high, shift_low

    def __getitem__(self, idx):
        r = random.randint(0, self.clip_len)
        start = idx * self.clip_len + r# fist frame
        clipIdx = [start]
        for i in range(self.clip_len-1):
            if start+self.sampling_len*(i+1) < self.max_len:
                fidx = random.randint(start+self.sampling_len*i, start+self.sampling_len*(i+1))
            else:
                fidx = start
            clipIdx.append(fidx)

        clipL_low, clipR_low, clipL_high, clipR_high = [], [], [], []
        clipL, clipR = [], []
        seed = random.randint(0, 1e5)
        ids = []
        for c in clipIdx:
            random.seed(seed)
            id = self.train_list[c]
            ids.append(id)
            imgL = cv2.imread(os.path.join(self.file_paths, 'image_0', id))
            imgR = cv2.imread(os.path.join(self.file_paths, 'image_1', id))
            imgL = torch.FloatTensor(imgL).permute(2,0,1).unsqueeze(0)
            imgR = torch.FloatTensor(imgR).permute(2,0,1).unsqueeze(0)
            low_freqL, high_freq_grayL = find_freq(imgL, self.kernel)
            # save_image(low_freqL, 'low_freqL.png')
            # save_image(high_freq_grayL, 'high_freq_grayL.png')
            low_freqR, high_freq_grayR = find_freq(imgR, self.kernel)
            # save_image(low_freqR, 'low_freqR.png')
            # save_image(high_freq_grayR, 'high_freq_grayR.png')
            clipL_low.append(low_freqL)
            clipR_low.append(low_freqR)
            clipL_high.append(high_freq_grayL)
            clipR_high.append(high_freq_grayR)
            clipL.append(imgL)
            clipR.append(imgR)

        clipL_low = torch.stack(clipL_low, dim=1).squeeze(0)/255.0
        clipR_low = torch.stack(clipR_low, dim=1).squeeze(0)/255.0
        clipL_high = torch.stack(clipL_high, dim=1).squeeze(0)/255.0
        clipR_high = torch.stack(clipR_high, dim=1).squeeze(0)/255.0
        clipL = torch.stack(clipL, dim=1).squeeze(0)/255.0
        clipR = torch.stack(clipR, dim=1).squeeze(0)/255.0
        return clipL_low, clipR_low, clipL_high, clipR_high, clipL, clipR, ids


def test():
    file_paths = '/media/runze/DATA/Dasaset/daVinci/test'
    dataset = load_trainset(file_paths)

    _ = dataset.__getitem__(0)

# test()