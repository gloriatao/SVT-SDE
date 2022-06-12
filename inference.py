import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets.load_inf_davinci_vit_feq import load_valset
from engines.engine_vit_freq import inference
from models.depthnet_freq_smooth import depthnet

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
# number of trainable params: 32,072,943

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    # Model parameters
    parser.add_argument('--pretrained_weights', type=str, help="Path to the pretrained model.")
    # dataset parameters
    parser.add_argument('--data_path', type=str, default='/media/cygzz/data/rtao/data/davinci/daVinci')
    parser.add_argument('--output_dir', default='pred', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpus', default=[0,1], help='device to use for training / testing')
    parser.add_argument('--seed', default=130, type=int)
    parser.add_argument('--resume', default='./davinci_vit_freq_smooth/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--start_iteration', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser
# 13, 29666
def main(args):
    print(os.environ)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = depthnet()
    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    model.to(device)

    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params:', n_parameters)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad==False)
    print('number of frozen params:', n_parameters)

    dataset_val = load_valset(os.path.join(args.data_path, 'test'), sampling_len=10)
    data_loader_val = DataLoader(dataset_val, batch_size=10, drop_last=False, shuffle=False, num_workers=args.num_workers)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print('epoch:',checkpoint['epoch'])

    print("Start training")

    # path---------------------------------------------
    output_dir = Path(args.output_dir)
    pred_dir = Path(args.output_dir+'/davinci_vit_freq')
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    if os.path.isdir(pred_dir) == False:
        os.mkdir(pred_dir)

    ssims = inference(model, data_loader_val, device, pred_dir)
    print(np.mean(ssims), np.std(ssims))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VisTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
