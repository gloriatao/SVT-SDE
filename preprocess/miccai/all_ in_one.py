import os

from video_parser import video_parser
from image_scissor import image_scissor
from stereo_rectify import stereo_rectify
from depth_to_disp import depth_to_disparity

rootpath = '/media/cygzz/data/rtao/data/stereo_corr/Zip_out'#'/media/eikoloki/TOSHIBA EXT/MICCAI_SCARED/dataset3'
disppath = '/media/cygzz/data/rtao/data/stereo_corr/Disparity'
set = ['TestSet','TrainSet']
for s in set:
    for d in sorted(os.listdir(os.path.join(rootpath, s))):
        file_path = os.path.join(os.path.join(rootpath, s, d))
        out_path = os.path.join(os.path.join(disppath, s, d))

        video_parser(file_path, out_path)

        image_scissor(file_path, out_path)

        stereo_rectify(file_path, out_path)

        depth_to_disparity(file_path, out_path)
