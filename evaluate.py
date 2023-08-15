import argparse
import glob
import os.path
import pdb

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from skimage.transform import resize
import albumentations as A

import cv2 as cv
import models_mae
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.utils import shuffle
import h5py

DATASETS = ['brats', 'cbis_roi', 'cbis', 'luna16', 'luna16_unnorm']


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    img_size = 64
    patch_size = 16  # TODO: add it as param
    if args.dataset == 'brats':
        img_size = 224
        patch_size = 16

    model = getattr(models_mae, arch)(img_size=img_size, patch_size=patch_size)
    # load model
    # model = model.to('cpu')
    checkpoint = torch.load(chkpt_dir)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model = model.to('cuda')
    # switch to evaluation mode
    model.eval()
    return model


def get_reconstructions(model_, imgs_, idx, label_=None):

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


def visualize(imgs_, reconstructions_, paths_, ground_truth_labels_):
    mask_folder = '/media/lili/SSD2/datasets/brats/BraTS2020_training_data/content/data'

    for (img_, recon_, path_, gt_label_) in zip(imgs_, reconstructions_, paths_, ground_truth_labels_):

        plt.subplot(1, 4, 1)
        plt.imshow(img_, cmap='gray')
        print("img sum pixels", img_.sum())
        plt.subplot(1, 4, 2)
        plt.imshow(recon_, cmap='gray')
        plt.subplot(1, 4, 3)
        plt.imshow(np.abs(img_ - recon_), cmap='gray')

        if args.dataset == 'brats':
            # get mask
            short_filename = os.path.split(path_)[-1][:-4] + '.h5'
            with h5py.File(os.path.join(mask_folder, short_filename), 'r') as h5:
                mask = np.array(h5["mask"][:])
            plt.subplot(1, 4, 4)
            if gt_label_ == 0:
                plt.title("Normal")
            else:
                plt.title("Abnormal")
            plt.imshow(mask * 255, cmap='gray')
            print("label =", gt_label_, 'path_', short_filename)
        plt.show()



def get_auc(model_mae, paths, ground_truth_labels):
    pred_labels = []
    for start_index in tqdm(range(0, len(paths), args.batch_size)):
        imgs = []
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

        imgs = np.array(imgs, np.float32)

        reconstructions = get_reconstructions_multi(model_mae, imgs)

        # visualize(imgs, reconstructions, paths[start_index: start_index + args.batch_size],
        #           ground_truth_labels[start_index: start_index + args.batch_size])

        if args.err == "mae":
            errs = np.mean(np.abs(imgs - reconstructions), axis=(1, 2, 3))
        elif args.err == "mse":
            errs = np.mean(np.abs(imgs - reconstructions) ** 2, axis=(1, 2, 3))
        elif args.err == "ssim":
            errs = []
            for target_, recon_ in zip(imgs, reconstructions):
                err = -ssim(target_[:, :, 0], recon_[:, :, 0])
                errs.append(err)
        else:
            raise ValueError(f"err {args.err} not supported!")

        pred_labels.extend(errs)

    pred_labels = np.array(pred_labels)

    auc = roc_auc_score(ground_truth_labels, pred_labels)
    print("AUC:", auc)

    # idx = ground_truth_labels == 0
    # plt.hist([pred_labels[idx], pred_labels[~idx]], color=['b', 'r'])
    # plt.show()


def load_model(model_path):
    model_mae = prepare_model(model_path, 'mae_vit_base_patch16')
    return model_mae


parser = argparse.ArgumentParser(description='PyTorch Medical Images')
parser.add_argument('--model-path', type=str)
parser.add_argument('--mask-ratio', type=float)
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--err', type=str, default="ssim")
parser.add_argument('--num-trials', type=int, default=1)
parser.add_argument('--use_val', action='store_true',
                    help='Test on val data.')
parser.set_defaults(use_val=False)

args = parser.parse_args()

assert args.dataset in DATASETS
assert args.err in ["ssim", "mse", "mae"]

""" 
   
python3 evaluate.py --dataset=brats --mask-ratio=0.75  --model-path=mae_mask_ratio_0.75_brats/checkpoint-1599.pth --batch-size=64 --num-trials=4
 

python3 evaluate.py --dataset=luna16_unnorm --mask-ratio=0.75  --model-path=mae_luna16_patch_8_mask_ratio_0.75_unnorm/checkpoint-1599.pth \
--batch-size=64 --num-trials=4
  
python3 evaluate.py --dataset=brats --mask-ratio=0.85 \
 --model-path=mae_brats_mask_ratio_0.85/checkpoint-1599.pth --batch-size=64 --num-trials=4


"""
if __name__ == '__main__':

    model_path = args.model_path
    model_mae_ = load_model(model_path)

    # Data
    normal_paths = get_normal_images_paths()
    abnormal_paths = get_abnormal_images_paths()

    file_paths = normal_paths + abnormal_paths
    gt_labels = np.concatenate((np.zeros(len(normal_paths)), np.ones(len(abnormal_paths))))
    file_paths, gt_labels = shuffle(file_paths, gt_labels, random_state=12)  # only to visualize different labels
    get_auc(model_mae_, paths=file_paths, ground_truth_labels=gt_labels)