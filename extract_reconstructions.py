import argparse
import glob
import os.path
import pickle

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import resize
from tqdm import tqdm

import models_mae

DATASETS = ['brats', 'luna16_unnorm']


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    img_size = 64
    patch_size = 16
    if args.dataset == 'brats':
        img_size = 224
        patch_size = 16

    model = getattr(models_mae, arch)(img_size=img_size, patch_size=patch_size)
    checkpoint = torch.load(chkpt_dir)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model = model.to('cuda')
    return model


def apply_gaussian_filter(image):
    sigma = np.random.random() * 3 + 2
    image = cv.GaussianBlur(image, (5, 5), sigma)
    return image


def degrade_reconstruction(img_, apply_blur):
    if args.dataset == 'brats':
        h = np.random.randint(10, 40)  # max H
        w = np.random.randint(10, 40)  # max w
    elif args.dataset == 'luna16_unnorm':
        h = np.random.randint(5, 12)  # max H
        w = np.random.randint(5, 12)  # max W
    else:
        raise ValueError(f'Dataset {args.dataset} not recognized!')

    start_x = np.random.randint(img_.shape[1] - w)
    start_y = np.random.randint(img_.shape[0] - h)

    # print(h, w, ratio)
    end_x = start_x + w
    end_y = start_y + h
    if apply_blur:
        img_[start_y: end_y, start_x: end_x, 0] = apply_gaussian_filter(img_[start_y: end_y, start_x: end_x, 0])
    else:
        ratio = np.random.random()
        img_[start_y: end_y, start_x: end_x] *= ratio
    return img_


def apply_degradation(img_, num_iter, apply_blur):
    for _ in range(num_iter):
        img_ = degrade_reconstruction(img_, apply_blur)
    return img_


def get_reconstructions(model_, imgs_, idx):

    x = torch.tensor(imgs_)

    x = torch.einsum('nhwc->nchw', x)
    x = x.to('cuda')
    loss, result, mask = model_(x.float(), mask_ratio=args.mask_ratio, idx_masking=idx, is_testing=False)
    result = model_.unpatchify(result)
    result = torch.einsum('nchw->nhwc', result).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model_.patch_embed.patch_size[0]**2 * 1)  # (N, H*W, p*p*3)
    mask = model_.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    # MAE reconstruction pasted with visible patches
    im_paste = torch.einsum('nchw->nhwc', x).detach().cpu() * (1 - mask) + result * mask

    return im_paste.numpy()


def get_reconstructions_multi(model_, imgs_):
    num_fwd = args.num_trials
    results = None
    for idx in range(num_fwd):
        result = get_reconstructions(model_, imgs_, idx)
        if results is None:
            results = result
        else:
            results += result

    results = results / num_fwd

    return results


# change it to match your own path.
def get_normal_images_paths_train():

    if args.dataset == 'luna16_unnorm':
        return glob.glob('/media/lili/SSD2/datasets/luna16/luna16/train_unnorm/normal/*.npy')
    elif args.dataset == 'brats':
        return glob.glob('/media/lili/SSD2/datasets/brats/BraTS2020_training_data/split/train/normal/*.npy')
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')

# change it to match your own path.
def get_normal_images_paths():
    if args.dataset == 'luna16_unnorm':
        if args.use_val:
            return glob.glob('/media/lili/SSD2/datasets/luna16/luna16/val_unnorm/normal/*.npy')
        else:
            return glob.glob('/media/lili/SSD2/datasets/luna16/luna16/test_unnorm/normal/*.npy')
    elif args.dataset == 'brats':
        if args.use_val:
            return glob.glob('/media/lili/SSD2/datasets/brats/BraTS2020_training_data/split/val/normal/*.npy')
        else:
            return glob.glob('/media/lili/SSD2/datasets/brats/BraTS2020_training_data/split/test/normal/*.npy')
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')


# change it to match your own path.
def get_abnormal_images_paths():
    if args.dataset == 'luna16_unnorm':
        if args.use_val:
            return glob.glob('/media/lili/SSD2/datasets/luna16/luna16/val_unnorm/abnormal/*.npy')
        else:
            return glob.glob('/media/lili/SSD2/datasets/luna16/luna16/test_unnorm/abnormal/*.npy')
    elif args.dataset == 'brats':
        if args.use_val:
            return glob.glob('/media/lili/SSD2/datasets/brats/BraTS2020_training_data/split/val/abnormal/*.npy')
        else:
            return glob.glob('/media/lili/SSD2/datasets/brats/BraTS2020_training_data/split/test/abnormal/*.npy')
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')


def load_image(img_path_):
    image_np = np.float32(np.load(img_path_))

    if args.dataset == 'brats':
        image_np = image_np[:, :, 0]
    image_np = np.expand_dims(image_np, axis=2)
    return image_np


def process_image(img_):
    if args.dataset == 'brats':
        mean_ = np.array([0.])
        std_ = np.array([1.])
    elif args.dataset == 'luna16_unnorm':
        mean_ = np.array([0.])
        std_ = np.array([100.])
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')
    img_ = img_ - mean_
    img_ = img_ / std_
    return img_


def visualize(imgs_, reconstructions_, old_reconstructions_, paths_):
    num_imgs = 5
    for (img_, recon_, old_recon_, path_) in zip(imgs_, reconstructions_, old_reconstructions_, paths_):

        plt.subplot(1, num_imgs, 1)
        plt.imshow(img_, cmap='gray')
        plt.subplot(1, num_imgs, 2)
        plt.imshow(recon_, cmap='gray')
        plt.subplot(1, num_imgs, 3)
        plt.imshow(old_recon_, cmap='gray')
        plt.subplot(1, num_imgs, 4)
        plt.imshow(np.abs(img_ - recon_), cmap='gray')
        plt.subplot(1, num_imgs, 5)
        plt.imshow(np.abs(old_recon_ - recon_), cmap='gray')
        plt.show()


def save(imgs, reconstructions, used_paths, is_abnormal, iter_):
    base_dir = args.output_folder
    if is_abnormal:
        base_dir = os.path.join(base_dir, 'abnormal')
    else:
        base_dir = os.path.join(base_dir, 'normal')

    for (img_, recon_, path_) in zip(imgs, reconstructions, used_paths):
        if is_abnormal and img_.sum() == 0:
            continue

        info_ = {'img': img_, 'recon': recon_}
        short_filename = os.path.split(path_)[-1][:-4] + f'_{iter_}.pkl'
        with open(os.path.join(base_dir, short_filename), 'wb') as handle:
            pickle.dump(info_, handle)


def write_reconstructions(model_mae, paths, is_abnormal: bool = False, iter_: int = 0):

    for start_index in tqdm(range(0, len(paths), args.batch_size)):
        imgs = []
        used_paths = []
        for idx_path in range(start_index, start_index + args.batch_size):
            if idx_path < len(paths):
                path_ = paths[idx_path]
                img_ = load_image(path_)
                if args.dataset == 'brats':
                    img_ = resize(img_, (224, 224), order=3)  # 3: Bi-cubic
                else:
                    img_ = resize(img_, (64, 64), order=3)  # 3: Bi-cubic
                img_ = process_image(img_)

                imgs.append(img_)
                used_paths.append(path_)

        imgs = np.array(imgs, np.float32)

        old_reconstructions = get_reconstructions_multi(model_mae, imgs)
        if is_abnormal and args.test is False:
            reconstructions = [apply_degradation(recon.copy(), num_iter=np.random.randint(1, 10), apply_blur=False) for recon in old_reconstructions]
        else:
            reconstructions = old_reconstructions

        # visualize(imgs, reconstructions, old_reconstructions, used_paths)

        save(imgs, reconstructions, used_paths, is_abnormal, iter_=iter_)


def load_model(model_path):
    model_mae = prepare_model(model_path, 'mae_vit_base_patch16')
    return model_mae


parser = argparse.ArgumentParser(description='PyTorch Medical Images')
parser.add_argument('--model-path', type=str)
parser.add_argument('--mask-ratio', type=float)
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--output-folder', type=str, required=True)
parser.add_argument('--num-trials', type=int, default=1)
parser.add_argument('--use_val', action='store_true',
                    help='Test on val data.')

parser.add_argument('--test', action='store_true')

parser.set_defaults(use_val=False)

args = parser.parse_args()

assert args.dataset in DATASETS
"""  
python3 extract_reconstructions.py --dataset=brats --mask-ratio=0.85  \
--model-path=mae_brats_mask_ratio_0.85/checkpoint-1599.pth --batch-size=64 --num-trials=4 \
--output-folder=/media/lili/SSD2/datasets/brats/BraTS2020_training_data/reconstructions/mae_mask_ratio_0.85/val --use_val --test

 
python3 extract_reconstructions.py --dataset=luna16_unnorm --mask-ratio=0.75  \
--model-path=models/mae_luna16_patch_16_mask_ratio_0.75_unnorm/checkpoint-1599.pth --batch-size=64 --num-trials=4 \
--output-folder=/media/lili/SSD2/datasets/luna16/reconstructions/mae_luna16_patch_16_mask_ratio_0.75_unnorm_3_6/train


 
"""
if __name__ == '__main__':
    os.makedirs(os.path.join(args.output_folder, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'abnormal'), exist_ok=True)

    model_path = args.model_path
    model_mae_ = load_model(model_path)

    # Data
    if args.test:
        write_reconstructions(model_mae_, paths=get_normal_images_paths(), is_abnormal=False)
        write_reconstructions(model_mae_, paths=get_abnormal_images_paths(), is_abnormal=True)
    else:
        normal_paths = get_normal_images_paths_train()
        write_reconstructions(model_mae_, paths=normal_paths, is_abnormal=False)
        write_reconstructions(model_mae_, paths=normal_paths, is_abnormal=True, iter_=1)