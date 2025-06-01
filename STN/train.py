#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint, GicLoss, AffineGridGen

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "D:/Jeju/Thai/Dataset/Insect detection/ADOXYOLO/Resize")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 465)
    parser.add_argument("--fine_height", type=int, default = 349)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    # criterionGIC = GicLoss(opt)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    gridGen = AffineGridGen(opt.fine_height, opt.fine_width)

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        """
        'target_name':   target_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'image':    im,         # for visualization
            'target':    target,         # for visualization
            'agnostic': agnostic,   # for input
            'grid_image': im_g,     # for visualization
            GT_theta
        """

        im_name = inputs['im_name']
        target_name = inputs['target_name']

        for ind, name in enumerate(im_name):
            if isinstance(name, bytes):  # if loaded as bytes
                name = name.decode('utf-8')
            if "namhae2_2024_5_24_" in name and "namhae2_2024_5_10_" in target_name[ind]:
                print("Found 'ABC':", name)

        im = inputs['image'].cuda()
        target = inputs['target'].cuda()

        GT_theta = inputs['GT_theta'].cuda()
        # print("GT_theta:", inputs['GT_theta'])

        grid_GT = gridGen(GT_theta)
        # grid_GT = grid_GT.permute(0, 2, 1, 3)  # Swap height and width if incorrect

        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(im, target)
        # oneimg = torch.ones(target.shape).cuda()

        # grid = grid.permute(0, 2, 1, 3)  # Swap height and width if incorrect

        warped_im = F.grid_sample(im, grid, padding_mode='zeros')
        warped_im_GT = F.grid_sample(im, grid_GT, padding_mode='zeros')
        # Step 1: Flip vertically
        # x = torch.flip(warped_im_GT, dims=[2])  # dim=2 is height (vertical flip)

        # Step 2: Transpose height and width
        # x = x.permute(0, 1, 3, 2)  # Swap H and W: (B, C, H, W) → (B, C, W, H)
        # warped_im_GT = x
        # warped_one = F.grid_sample(oneimg, grid, padding_mode='zeros')

        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [[im, target],
                   [im_g, warped_grid],
                   [warped_im_GT, (warped_im_GT + target) * 0.5],
                   [warped_im, (warped_im + target) * 0.5]]
        # N, C, H, W = warped_one.shape
        # loss_l1 = criterionL1(warped_im, target)
        # Compute L1 Loss for theta
        loss_theta = criterionL1(theta, GT_theta)
        # loss_l1_mask = criterionL1(warped_im * warped_one, target  * warped_one)
        # loss_mask = (warped_one.sum()) *  (1.0 / (N*C*H*W))
        # loss_gic = criterionGIC(grid)
        loss = loss_theta #  + loss_mask#   + loss_l1_mask#  + loss_gic
        # print("Theta values:", theta.detach().cpu().numpy())
        # print("GT Theta values:", GT_theta.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            # board.add_scalar('loss_l1', loss_l1.item(), step+1)
            # board.add_scalar('loss_theta', loss_theta.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose],
                   [c, cm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                    % (step+1, t, loss.item(), loss_l1.item(),
                    loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))



def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
