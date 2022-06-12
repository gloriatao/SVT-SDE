import math
import os
import sys
from typing import Iterable
import numpy as np
import torch, pickle
from sklearn.metrics import f1_score
import util.misc as utils
import torch.nn.functional as F
from pathlib import Path
from apex import amp
import cv2
from losses_freq import Loss as Loss_monodepth
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import time

def sim_score(reconL,reconR,imgL,imgR,ids=None):
    bads = os.listdir('/media/cygzz/data/rtao/projects/stereo/bads')
    sim = []
    for b in range(reconL.shape[0]):
        # if ids[b] in bads:
        #     continue
        reL = reconL[b].detach().cpu().numpy().transpose(1,2,0)
        reL = reL-np.min(reL)
        reL = (reL/np.max(reL)*255).astype(np.uint8)

        L = imgL[b].detach().cpu().numpy().transpose(1,2,0)
        L = L-np.min(L)
        L = (L/np.max(L)*255).astype(np.uint8)

        reR = reconR[b].detach().cpu().numpy().transpose(1,2,0)
        reR = reR-np.min(reR)
        reR = (reR/np.max(reR)*255).astype(np.uint8)

        R = imgR[b].detach().cpu().numpy().transpose(1,2,0)
        R = R-np.min(R)
        R = (R/np.max(R)*255).astype(np.uint8)
        sim.append((ssim(reL,L, multichannel=True)+ssim(reR,R, multichannel=True))/2)
    return sim

def psnr_score(reconL,reconR,imgL,imgR):
    psnr = []
    for b in range(reconL.shape[0]):
        reL = reconL[b].detach().cpu().numpy().transpose(1,2,0)
        reL = reL-np.min(reL)
        reL = (reL/np.max(reL)*255).astype(np.uint8)

        L = imgL[b].detach().cpu().numpy().transpose(1,2,0)
        L = L-np.min(L)
        L = (L/np.max(L)*255).astype(np.uint8)

        reR = reconR[b].detach().cpu().numpy().transpose(1,2,0)
        reR = reR-np.min(reR)
        reR = (reR/np.max(reR)*255).astype(np.uint8)

        R = imgR[b].detach().cpu().numpy().transpose(1,2,0)
        R = R-np.min(R)
        R = (R/np.max(R)*255).astype(np.uint8)

        psnr.append((cv2.PSNR(reL,L)+cv2.PSNR(reR,R))/2)
    return psnr

def train_one_epoch(model, data_loader, optimizer, device, epoch, log_path, output_dir, lr_scheduler, steps, max_norm,
                    data_loader_val, val_log_dir, avg_ssim_benchmark, iteration, input_error=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ssim1', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    lsses = []
    Loss_mon = Loss_monodepth()
    for clipL_low, clipR_low, clipL_high, clipR_high, clipL, clipR, ids in metric_logger.log_every(data_loader, print_freq, header):
        # assemble
        input = {'left_high':clipL_high, 'right_high':clipR_high, 'left_all':clipL, 'right_all':clipR}
        # input = input_error
        for k in input.keys():
            input[k].to(device)
        # forward
        disps = model(input)
        if torch.isnan(disps).sum():
            print('this is a break point disps!')

        # loss
        b, seq_len, c, w, h = clipL.shape
        imgL = clipL.view(b*seq_len, c, w, h).to(device)
        imgR = clipR.view(b*seq_len, c, w, h).to(device)
        loss_dict, reconL1, reconR1 = Loss_mon(disps, (imgL, imgR))
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            checkpoint_path = output_dir / 'error.pth'
            utils.save_on_master({
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'iteration': iteration,
                'input':input,
                'disps':disps
            }, checkpoint_path)

            sys.exit(1)

        optimizer.zero_grad()
        with amp.scale_loss(losses, optimizer) as scaled_loss:
            scaled_loss.backward()#------------------------

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # ssim
        # multichannel : bool, optional
        # If True, treat the last dimension of the array as channels. Similarity
        # calculations are done independently for each channel then averaged.
        sim1 = np.mean(np.array(sim_score(reconL1, reconR1, imgL, imgR)))

        optimizer.step()
        lr_scheduler.step_update(iteration)

        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(ssim1=sim1)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # log
        lss = loss_dict.copy()
        for i, k in enumerate(lss):
            lss[k] = lss[k].detach().cpu().numpy().tolist()
        lss['iteration'] = iteration
        lss['epoch'] = epoch
        lss['ssim'] = [sim1]
        lsses.append(lss)
        with open(os.path.join(log_path, str(iteration)+'.pickle'), 'wb') as file:
            pickle.dump(lsses, file)
        file.close()

        # ex
        if iteration%1000 == 0:
            checkpoint_paths = [output_dir / 'checkpoint_new.pth']
            pred_ssim = evaluate(model, data_loader_val, device, iteration, val_log_dir)
            if pred_ssim >= avg_ssim_benchmark:
                checkpoint_paths = [output_dir / f'checkpoint_{pred_ssim:04}.pth']
                print('saving best mean ssim@', pred_ssim, 'iteration:', iteration)
                avg_ssim_benchmark = pred_ssim

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'iteration': iteration,
                }, checkpoint_path)

        iteration+=1
        steps += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print('avg_ssim_benchmark:',avg_ssim_benchmark)
    return avg_ssim_benchmark, iteration

def evaluate(model, data_loader, device, iteration, output_dir):
    model.eval()
    correct, total, loss_valid, ssims = 0,0,[], []
    Loss_mon = Loss_monodepth()

    with torch.no_grad():
        print('start validation------')
        for index, (clipL_low, clipR_low, clipL_high, clipR_high, clipL, clipR, clipIds) in enumerate(data_loader):
            print(index,'--', len(data_loader))
            # assemble
            input = {'left_high':clipL_high, 'right_high':clipR_high, 'left_all':clipL, 'right_all':clipR}
            for k in input.keys():
                input[k].to(device)
            clipL = clipL.to(device)
            clipR = clipR.to(device)
            #
            disps = model(input)
            # loss
            b, seq_len, c, w, h = clipL.shape
            disp = disps.view(b, seq_len, 2, w, h)[:,-1,:,:,:]
            imgL = clipL[:,-1,:,:,:]
            imgR = clipR[:,-1,:,:,:]
            loss_dict, reconL, reconR = Loss_mon(disp, (imgL, imgR))
            loss_valid.append(loss_dict)
            # ssim
            for b in range(reconL.shape[0]):
                reL = reconL[b].detach().cpu().numpy().transpose(1,2,0)
                reL = reL-np.min(reL)
                reL = (reL/np.max(reL)*255).astype(np.uint8)

                L = imgL[b].detach().cpu().numpy().transpose(1,2,0)
                L = L-np.min(L)
                L = (L/np.max(L)*255).astype(np.uint8)

                reR = reconR[b].detach().cpu().numpy().transpose(1,2,0)
                reR = reR-np.min(reR)
                reR = (reR/np.max(reR)*255).astype(np.uint8)

                R = imgR[b].detach().cpu().numpy().transpose(1,2,0)
                R = R-np.min(R)
                R = (R/np.max(R)*255).astype(np.uint8)

                ssims.append((ssim(reL,L, multichannel=True)+ssim(reR,R, multichannel=True))/2)
        results = {'loss_valid':loss_valid, 'ssims':ssims}
        print('--mean ssim:', np.mean(ssims),'--std ssim', np.std(ssims)) # --mean ssim: 0.6485833979385773 --std ssim 0.042820881201509876
        with open(os.path.join(output_dir, str(iteration) + '_valid.pickle'), 'wb') as file:
            pickle.dump(results, file)
        file.close()
    return np.mean(ssims)

def saving_recon(disps, reconL, reconR, imgL, imgR, output_dir, ids, save_png=True, save_npy=False):
    if save_png:
        out_png = os.path.join(output_dir, 'pngs')
        Path(out_png).mkdir(parents=True, exist_ok=True)
        for j in range(reconL.shape[0]):
            disp = disps[j][0].detach().cpu().numpy()
            disp = disp * disp.shape[1]

            disp_plot = ((disp/110.0) * 255).astype(np.uint8) # test 0
            # disp_plot = disp-12.0
            # print(np.max(disp_plot))
            # disp_plot = ((disp_plot/60) * 255).astype(np.uint8) # test 1

            # disp_plot = ((disp/75.0) * 255).astype(np.uint8) # train

            rL = reconL[j].permute(1,2,0).detach().cpu().numpy()
            rL = rL - np.min(rL)
            rL = (rL / np.max(rL) * 255).astype(np.uint8)

            L = imgL[j].permute(1,2,0).detach().cpu().numpy()
            L = L - np.min(L)
            L = (L / np.max(L) * 255).astype(np.uint8)

            error = (np.mean((L - rL), axis=-1)).astype(np.uint8)


            rR = reconR[j].permute(1,2,0).detach().cpu().numpy()
            rR = rR - np.min(rR)
            rR = (rR / np.max(rR) * 255).astype(np.uint8)

            R = imgR[j].permute(1,2,0).detach().cpu().numpy()
            R = R - np.min(R)
            R = (R / np.max(R) * 255).astype(np.uint8)

            error = (np.mean((R - rR), axis=-1)).astype(np.uint8)


            img1 = np.hstack((L,R))
            img2 = np.hstack((cv2.applyColorMap(disp_plot,cv2.COLORMAP_JET), rL))
            img = np.hstack((img1, img2))
            # img = np.vstack((img, cv2.cvtColor(error,cv2.COLOR_GRAY2RGB)))

            cv2.imwrite(os.path.join(out_png, ids[j][:-4]+'.png'), img)



    if save_npy:
        out_npy = os.path.join(output_dir, 'npys', 'train')
        Path(out_npy).mkdir(parents=True, exist_ok=True)
        for j in range(reconL.shape[0]):
            disp = disps[j].detach().cpu().numpy()
            disp = disp * disp.shape[1] # in pixels
            np.save(os.path.join(out_npy, ids[j][:-4]+'.npy'), disp)
    return

def saving_recon2(disps, reconL, reconR, imgL, imgR, output_dir, ids, save_png=True, save_npy=False):
    if save_png:
        out_png = os.path.join(output_dir, 'pngs')
        Path(out_png).mkdir(parents=True, exist_ok=True)
        for j in range(reconL.shape[0]):
            disp = disps[j][1].detach().cpu().numpy()
            disp = disp * disp.shape[1]

            disp_plot = ((disp/110.0) * 255).astype(np.uint8) # test 0

            rL = reconL[j].permute(1,2,0).detach().cpu().numpy()
            rL = rL - np.min(rL)
            rL = (rL / np.max(rL) * 255).astype(np.uint8)

            L = imgL[j].permute(1,2,0).detach().cpu().numpy()
            L = L - np.min(L)
            L = (L / np.max(L) * 255).astype(np.uint8)

            rR = reconR[j].permute(1,2,0).detach().cpu().numpy()
            rR = rR - np.min(rR)
            rR = (rR / np.max(rR) * 255).astype(np.uint8)

            R = imgR[j].permute(1,2,0).detach().cpu().numpy()
            R = R - np.min(R)
            R = (R / np.max(R) * 255).astype(np.uint8)

            img1 = np.hstack((L,R))
            img2 = np.hstack((cv2.applyColorMap(disp_plot,cv2.COLORMAP_JET), rR))
            img = np.hstack((img1, img2))
            # img = np.vstack((img, cv2.cvtColor(error,cv2.COLOR_GRAY2RGB)))

            cv2.imwrite(os.path.join(out_png, ids[j][:-4]+'.png'), img)



    if save_npy:
        out_npy = os.path.join(output_dir, 'npys', 'train')
        Path(out_npy).mkdir(parents=True, exist_ok=True)
        for j in range(reconL.shape[0]):
            disp = disps[j].detach().cpu().numpy()
            disp = disp * disp.shape[1] # in pixels
            np.save(os.path.join(out_npy, ids[j][:-4]+'.npy'), disp)
    return

def inference(model, data_loader, device, output_dir):
    from losses_freq_inference import Loss as Loss_monodepth2
    model.eval()
    correct, total, loss_valid, ssims = 0,0,[], []
    bs = data_loader.batch_size
    clip_len = data_loader.dataset.clip_len
    res = data_loader.dataset.resolution
    buffer = torch.zeros((10*clip_len, 2, res[0], res[1]))
    sampling = data_loader.dataset.sampling_len
    Loss_mon = Loss_monodepth2()
    ratio = [4 for i in range(len(data_loader))]
    ratio[0] = 1
    ratio[1] = 2
    ratio[2] = 3
    count_f  = 0
    psnr = []
    with torch.no_grad():
        print('start validation------')
        start_time = time.time()
        for index, (clipL_low, clipR_low, clipL_high, clipR_high, clipL, clipR, clipIds) in enumerate(data_loader):
            print(index,'--', count_f, '--',len(data_loader))
            input = {'left_high':clipL_high, 'right_high':clipR_high, 'left_low':clipL_low, 'right_low':clipR_low, 'left_all':clipL, 'right_all':clipR}
            for k in input.keys():
                input[k].to(device)
            clipL = clipL.to(device)
            clipR = clipR.to(device)
            count_f += clipR.shape[0]
            #
            disps = model(input)

        print("--- %s seconds ---" % (time.time() - start_time))
    return ssims

# 0.8641331539617606 0.03546963094541918 (test)
# psnr 23.48485727058261 1.6987247365976088

# 3490