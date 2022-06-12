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

class load_valset(Dataset):
    def __init__(self, file_paths, clip_len=4, sampling_len=10):
        self.file_paths = file_paths
        self.clip_len = clip_len
        self.sampling_len = sampling_len
        self.list_raw = sorted(os.listdir(os.path.join(file_paths, 'image_0')))[0:1000]
        tail = self.list_raw[-2*sampling_len:]
        tail = list(reversed(tail))
        self.train_list = self.list_raw+tail
        self.max_len = len(self.train_list)
        self.kernel = get_gaussian_kernel(21)
        self.resolution = (192, 384)

    def __len__(self):
        return len(self.list_raw)

    def __getitem__(self, idx):
        start = idx
        clipIdx = [start]
        for i in range(self.clip_len-1):
            if start+self.sampling_len*(i+1) < self.max_len:
                fidx = start+self.sampling_len*(i+1)
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