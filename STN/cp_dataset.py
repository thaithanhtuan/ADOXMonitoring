#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw
import cv2
import os.path as osp
import numpy as np
import json
import os
import torch.nn.functional as F

class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot)
        self.transform = transforms.Compose([ \
                # transforms.Resize((192, 256)),    # Resize to (height=256, width=192)
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # load data list
        im_names = []
        target_names = []
        with open(osp.join("./data", opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, target_name = line.strip().split()
                im_names.append(im_name)
                target_names.append(target_name)

        self.im_names = im_names
        self.target_names = target_names

    def name(self):
        return "CPDataset"

    def load_interest_points(self, POINTS_DIR, image_name):
        """ Load interest points (9 points in 3x3 grid) from file. """
        points_file = os.path.join(POINTS_DIR, image_name.replace('.jpg', '.txt'))
        if os.path.exists(points_file):
            with open(points_file, 'r') as f:
                points = [list(map(float, line.strip().split())) for line in f]
            return np.array(points, dtype=np.float32)
        return None

    def triangle_area(self, p1, p2, p3):
        return 0.5 * abs(
            (p1[0] * (p2[1] - p3[1]) +
             p2[0] * (p3[1] - p1[1]) +
             p3[0] * (p1[1] - p2[1]))
        )

    def compute_gt_theta(self, POINTS_DIR, im_name, target_name, scale_factor = 0.1):
        """ Compute GT_theta (2x3 affine transformation matrix) from interest points. """
        src_pts = self.load_interest_points(POINTS_DIR, im_name)  # 9 points from the source image
        dst_pts = self.load_interest_points(POINTS_DIR, target_name)  # 9 points from the target image

        src_pts = src_pts * scale_factor
        dst_pts = dst_pts * scale_factor

        src_pts = self.normalize_points(src_pts,self.opt.fine_width, self.opt.fine_height)
        dst_pts = self.normalize_points(dst_pts,self.opt.fine_width, self.opt.fine_height)

        if src_pts is None or dst_pts is None or len(src_pts) != 9 or len(dst_pts) != 9:
            raise ValueError(f"Invalid interest points for {im_name} or {target_name}")

        # Check if they are equal
        if np.allclose(src_pts, dst_pts, atol=1e-10):
            # Identity affine matrix (2x3)
            GT_theta = np.array([[1, 0, 0],
                                 [0, 1, 0]], dtype=np.float32)
        else:
            # Compute using 3 chosen points (e.g., 0, 5, 6)
            idxs = [0, 2, 6]
            GT_theta = cv2.getAffineTransform(dst_pts[idxs], src_pts[idxs])
        GT_theta = torch.tensor(GT_theta, dtype=torch.float32)  # Convert to PyTorch tensor

        return src_pts, dst_pts, GT_theta  # Shape (2,3)

    def draw_points_with_numbers(self, image, points, color=(0, 255, 0)):
        h, w = image.shape[:2]
        for i, (x, y) in enumerate(points):
            # Convert from normalized [-1, 1] to pixel coordinates
            px = int((x + 1) * (w - 1) / 2)
            py = int((y + 1) * (h - 1) / 2)
            cv2.circle(image, (px, py), 5, color, -1)
            cv2.putText(image, str(i), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image

    def cv2_to_pil(self, cv_image):
        cv_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_rgb)

    def pil_to_cv2(self, pil_image):
        np_image = np.array(pil_image)  # RGB by default
        if np_image.ndim == 2:  # grayscale
            return np_image
        return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    def normalize_points(self, pts, img_width, img_height):
        # pts shape: (N, 2), where each row is (x, y)
        norm_x = (pts[:, 0] / (img_width - 1)) * 2 - 1
        norm_y = (pts[:, 1] / (img_height - 1)) * 2 - 1
        return np.stack([norm_x, norm_y], axis=1)

    def __getitem__(self, index):

        # parser.add_argument("--dataroot", default = "D:/Jeju/Thai/Dataset/Insect detection/ADOXYOLO/Resize")
        POINTS_DIR = self.data_path.replace("Resize", "InterestPoint/Points/")

        target_name = self.target_names[index]
        im_name = self.im_names[index]

        src_pts, dst_pts, GT_theta = self.compute_gt_theta(POINTS_DIR, im_name, target_name)


        im = Image.open(osp.join(self.data_path, im_name))
        im_cv = self.pil_to_cv2(im)
        cv2.putText(im_cv, im_name, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        im_cv = self.draw_points_with_numbers(im_cv, src_pts, color=(0, 255, 0))
        im = self.cv2_to_pil(im_cv)
        im = self.transform(im) # [-1,1]

        # person image
        target = Image.open(osp.join(self.data_path, target_name))
        target_cv = self.pil_to_cv2(target)
        cv2.putText(target_cv, target_name, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        target_cv = self.draw_points_with_numbers(target_cv, dst_pts, color=(0, 0, 255))
        target = self.cv2_to_pil(target_cv)
        target = self.transform(target)  # [-1,1]

        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
            im_g = F.interpolate(im_g.unsqueeze(0), size=(349, 465), mode='bilinear',
                                        align_corners=False).squeeze(0)
        else:
            im_g = ''

        result = {
            'target_name':   target_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'image':    im,         # for visualization
            'target':    target,         # for visualization
            'grid_image': im_g,     # for visualization
            'src_pts': src_pts,     # for visualization
            'dst_pts': dst_pts,     # for visualization
            'GT_theta': GT_theta     # GT_theta
            }

        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed; embed()

