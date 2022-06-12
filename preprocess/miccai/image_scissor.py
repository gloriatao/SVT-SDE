import cv2
from os import listdir
from os.path import join, split
import numpy as np
from pathlib import Path


def image_sciss(image_file, left_savepath, right_savepath):
    print('-- current image :' + image_file + " --")
    stacked = cv2.imread(image_file)
    print(stacked.shape)
    left_img = stacked[:1024, :, :]
    right_img = stacked[1024:, :, :]
    path, file = split(image_file)

    cv2.imwrite(join(left_savepath, file), left_img)
    cv2.imwrite(join(right_savepath, file), right_img)




def image_scissor(file_path, out_path):
    keyframe_list = [join(file_path, kf) for kf in sorted(listdir(file_path)) if ('keyframe' in kf and 'ignore' not in kf)]
    for kf in keyframe_list:
        stacked_filepath = join(file_path, kf) + '/data/rgb_data'
        stacked_filelist = [sf for sf in listdir(stacked_filepath) if '.png' in sf]
        for sf in stacked_filelist:
            image_file = join(stacked_filepath, sf)
            left_savepath = join(out_path, kf) + '/data/left'
            right_savepath = join(out_path, kf) + '/data/right'
            Path(left_savepath).mkdir(parents=True, exist_ok=True)
            Path(right_savepath).mkdir(parents=True, exist_ok=True)
            image_sciss(image_file, left_savepath, right_savepath)


if __name__ == '__main__':
    path = '/media/eikoloki/TOSHIBA EXT/MICCAI_SCARED/dataset3'
    image_scissor(path)

