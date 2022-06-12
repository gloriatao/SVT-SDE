import cv2
import os
import numpy as np

dtrain = '/media/cygzz/data/rtao/projects/stereo/pred/davinci_unet_vit/npys/test'
dtest = '/media/cygzz/data/rtao/projects/stereo/pred/davinci_unet_vit/npys/test'

min_disp = 384
max_disp = 0
disp_range = 0
for i, f in enumerate(sorted(os.listdir(dtrain))):
    disp = np.load(os.path.join(dtrain, f))
    min_disp = min(min_disp, np.min(disp))
    max_disp = max(max_disp, np.max(disp))
    if np.max(disp)-np.min(disp)<120:
        disp_range = max(disp_range, np.max(disp)-np.min(disp))
        # print(f, np.max(disp)-np.min(disp))
        if i%1000 ==0:
            print('train',f, min_disp, max_disp, disp_range)
    else:
        print(i, f)
        print(f, np.max(disp)-np.min(disp))
print('done train------------',f, min_disp, max_disp,disp_range)

# for i, f in enumerate(sorted(os.listdir(dtest))):
#     disp = np.load(os.path.join(dtest, f))
#     min_disp = min(min_disp, np.min(disp))
#     max_disp = max(max_disp, np.max(disp))
#     if np.max(disp)-np.min(disp)<100:
#         disp_range = max(disp_range, np.max(disp)-np.min(disp))
#         # print(f, np.max(disp)-np.min(disp))
#         if i%1000 ==0:
#             print('test',f, min_disp, max_disp, disp_range)
# print('done test------------',f, min_disp, max_disp, disp_range)