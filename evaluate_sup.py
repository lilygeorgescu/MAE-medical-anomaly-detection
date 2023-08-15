import argparse
import glob
import os.path
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from tqdm import tqdm

import models_vit

DATASETS = ['brats', 'luna16_unnorm']


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    img_size = 64
    if args.dataset == 'brats':
        img_size = 224

    model = models_vit.__dict__[arch](
        num_classes=2,
        drop_path_rate=0.0,
        global_pool=True,
        img_size=img_size
    )

    # load model
    # model = model.to('cpu')
    checkpoint = torch.load(chkpt_dir)  # , map_location='cpu'
    # TODO: check if there is need for strict false.
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model = model.to('cuda')
    # switch to evaluation mode
    model.eval()
    return model


def get_scores(model_, imgs_):

    x = torch.tensor(imgs_)

    x = torch.einsum('nhwc->nchw', x)
    x = x.to('cuda')
    result = model_(x.float())
    soft_result = torch.nn.functional.softmax(result, dim=1)
    return soft_result.detach().cpu().numpy()[:, 0]


# change it to match your own path.
def get_normal_images_paths():

    if args.dataset == 'luna16_unnorm':
        if args.use_val:
            return glob.glob('/media/lili/SSD2/datasets/luna16/reconstructions/mae_luna16_patch_16_mask_ratio_0.75_unnorm/val/normal/*.pkl')
        else:
            return glob.glob('/media/lili/SSD2/datasets/luna16/reconstructions/mae_luna16_patch_16_mask_ratio_0.75_unnorm/test/normal/*.pkl')

    elif args.dataset == 'brats':
        if args.use_val:
            return glob.glob('/media/lili/SSD2/datasets/brats/BraTS2020_training_data/reconstructions/mae_mask_ratio_0.75_800e/val/normal/*.pkl')
        else:
            return glob.glob('/media/lili/SSD2/datasets/brats/BraTS2020_training_data/reconstructions/mae_mask_ratio_0.75_800e/test/normal/*.pkl')
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')


# change it to match your own path.
def get_abnormal_images_paths():
    if args.dataset == 'luna16_unnorm':
        if args.use_val:
            return glob.glob('/media/lili/SSD2/datasets/luna16/reconstructions/mae_luna16_patch_16_mask_ratio_0.75_unnorm/val/abnormal/*.pkl')
        else:
            return glob.glob('/media/lili/SSD2/datasets/luna16/reconstructions/mae_luna16_patch_16_mask_ratio_0.75_unnorm/test/abnormal/*.pkl')

    elif args.dataset == 'brats':
        if args.use_val:
            return glob.glob('/media/lili/SSD2/datasets/brats/BraTS2020_training_data/reconstructions/mae_mask_ratio_0.75_800e/val/abnormal/*.pkl')
        else:
            return glob.glob('/media/lili/SSD2/datasets/brats/BraTS2020_training_data/reconstructions/mae_mask_ratio_0.75_800e/test/abnormal/*.pkl')
    else:
        raise ValueError(f'Data set {args.dataset} not recognized.')


def mae(img1, img2):
    return np.mean(np.abs(img1 - img2))


def mse(img1, img2):
    return np.mean(np.abs(img1 - img2) ** 2)


def load_image(img_path_):
    with open(img_path_, 'rb') as handle:
        dict_ = pickle.load(handle)

    img_ = dict_["img"]
    recon_ = dict_["recon"]
    diff = np.abs((img_ - recon_))[:, :, 0]

    image_np = np.expand_dims(diff, axis=2)
    return image_np, -ssim(img_[:, :, 0], recon_[:, :, 0])


def process_image(img_):
    mean_ = np.array([0.])
    std_ = np.array([1.])

    img_ = img_ - mean_
    img_ = img_ / std_
    return img_


def get_auc(model_, paths, ground_truth_labels):
    pred_labels = []
    for start_index in tqdm(range(0, len(paths), args.batch_size)):
        imgs = []
        ssim_scores = []
        for idx_path in range(start_index, start_index + args.batch_size):
            if idx_path < len(paths):
                path_ = paths[idx_path]
                img_, ssim_value = load_image(path_)

                # plt.title(ground_truth_labels[idx_path])
                # plt.imshow(img_, cmap='gray')
                # plt.show()


                img_ = process_image(img_)
                imgs.append(img_)
                ssim_scores.append(ssim_value)

        imgs = np.array(imgs, np.float32)

        scores = get_scores(model_, imgs)
        final_scores = scores # + np.array(ssim_scores)
        # print(scores, ground_truth_labels[start_index: start_index + args.batch_size])

        pred_labels.extend(list(final_scores))

    pred_labels = np.array(pred_labels)

    auc = roc_auc_score(ground_truth_labels, pred_labels)
    print("AUC:", auc)

    # compare_histogram(scores=pred_labels, classes=ground_truth_labels)
    # idx = ground_truth_labels == 0
    # plt.hist([pred_labels[idx], pred_labels[~idx]], color=['b', 'r'])
    # plt.show()

    return auc


def compare_histogram(scores, classes, thresh=None, n_bins=64, log=False, name=''):
    if log:
        scores = np.log(scores + 1e-8)

    if thresh is not None:
        if np.max(scores) < thresh:
            thresh = np.max(scores)
        scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")

    ticks = np.linspace(np.min(scores), np.max(scores), 5)
    labels = ['{:.2f}'.format(i) for i in ticks[:-1]] + ['>' + '{:.2f}'.format(np.max(scores))]
    plt.xticks(ticks, labels=labels)
    plt.xlabel('Anomaly Score' if not log else 'Log Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y')
    plt.show()


def load_model(model_path):
    model_mae = prepare_model(model_path, 'vit_base_patch16')
    return model_mae


parser = argparse.ArgumentParser(description='PyTorch Medical Images')
parser.add_argument('--model-path', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--use_val', action='store_true',
                    help='Test on val data.')
parser.add_argument('--find_best', action='store_true',
                    help='Test on val data.')
parser.set_defaults(use_val=False)

args = parser.parse_args()

assert args.dataset in DATASETS

""" 

python3 evaluate_sup.py --dataset=brats  --model-path=mae_mask_ratio_0.75_brats_2_sup_pretrain_no_lwr_random_ratio/checkpoint-3.pth --batch-size=64
   
  
python3 evaluate_sup.py --dataset=luna16_unnorm  --model-path=models/mae_mask_ratio_0.75_luna_from_scratch_token/checkpoint-43.pth --batch-size=64

python3 evaluate_sup.py --dataset=luna16_unnorm  --batch-size=64   --find_best --use_val

python3 evaluate_sup.py --dataset=brats  --batch-size=8   --find_best --use_val
 
 

"""
if __name__ == '__main__':
    if args.find_best:
        assert args.use_val
        # Data
        normal_paths = get_normal_images_paths()
        abnormal_paths = get_abnormal_images_paths()

        file_paths = normal_paths + abnormal_paths
        gt_labels = np.concatenate((np.zeros(len(normal_paths)), np.ones(len(abnormal_paths))))
        file_paths, gt_labels = shuffle(file_paths, gt_labels, random_state=12)  # only to visualize different labels
        max_auc = 0
        best_epoch = 0
        for index_epoch in range(100):
            model_path = f'mae_mask_ratio_0.75_luna_from_scratch/checkpoint-{index_epoch}.pth'
            model_mae_ = load_model(model_path)
            current_auc = get_auc(model_mae_, paths=file_paths, ground_truth_labels=gt_labels)
            if current_auc > max_auc:
                max_auc = current_auc
                best_epoch = index_epoch
            print(f"Best auc = {max_auc}; best epoch = {best_epoch}")
    else:
        model_path = args.model_path
        model_mae_ = load_model(model_path)

        # Data
        normal_paths = get_normal_images_paths()
        abnormal_paths = get_abnormal_images_paths()

        file_paths = normal_paths + abnormal_paths
        gt_labels = np.concatenate((np.zeros(len(normal_paths)), np.ones(len(abnormal_paths))))
        file_paths, gt_labels = shuffle(file_paths, gt_labels, random_state=12)  # only to visualize different labels
        get_auc(model_mae_, paths=file_paths, ground_truth_labels=gt_labels)