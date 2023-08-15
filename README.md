
#### This is the official repo of "Masked Autoencoders for Unsupervised Anomaly Detection in Medical Images"@KES2023.

### Acknowledgement: 
This code is mostly built on: [MAE](https://github.com/facebookresearch/mae). We thank 🙏 the authors for sharing their code.

### 📜 Arxiv Link: https://arxiv.org/pdf/2307.07534.pdf

### 🌟 Overview

We tackle anomaly detection in medical images training our framework using only healthy samples. We propose to use the Masked Autoencoder model to learn
the structure of the normal samples, then train an anomaly classifier on top of the difference between the original image and the reconstruction provided by the masked autoencoder. We train the anomaly classifier in a supervised manner using as negative
samples the reconstruction of the healthy scans, while as positive samples, we use pseudo-abnormal scans obtained via our novel
pseudo-abnormal module. The pseudo-abnormal module alters the reconstruction of the normal samples by changing the intensity of several regions.

### Download the data sets.

1.1 BraTS2020: 
   Download the BraTS2020 dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data).

1.2 Luna16: Download the Luna16 data set from [Luna16GrandChallenge](https://luna16.grand-challenge.org/Data/).

2  The splits used in this work can be found in the ```dataset``` folder.

2.1 For BraTS2020, we released the name of each slice.

2.2 For Luna16, we released the row number (from candidates.csv) of each region.


### 💻 Running the code.
#### Step 1: Pretraining phase.

```
python3 main_pretrain.py \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.75 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --output_dir mae_mask_ratio_0.75 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --batch_size=128 \
    --data_path path_to_the_normal_samples
```

#### Step 2: Extract reconstructions.
``` 
python3 extract_reconstructions.py \
--dataset=brats --mask-ratio=0.85  \
--model-path=path_to_the_saved_model/checkpoint-1599.pth  \
--batch-size=64 --num-trials=4 \
--output-folder=output_folder
``` 

Notice that you have to set the paths to the data set in the ```extract_reconstructions.py``` file and run the above command for the train, val and test splits.

#### Step 3: Train the anomaly classifier.
```
python3 main_finetune.py \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune path_to_the_saved_model.75_brats/checkpoint-1599.pth \
    --epochs 100 \
    --weight_decay 0.05 --drop_path 0.1 \
    --nb_classes 2 \
    --aa=None \
    --output_dir output_folder \
    --data_path path_to_the_reconstructions_obtained_in_the_previous_step
    
```

#### Step 4: Evaluate the model.
``` 
python3 evaluate_sup.py --dataset=brats  \
--model-path=path_to_the_best_model_obtained_in_the_previous_step.pth --batch-size=64
 
 
```

### 🚀 Results and trained models.


<table>
<tr>
    <td>Dataset</td> 
    <td>Pretrained Model</td>
    <td>Finetuned Model</td>  
    <td>AUROC</td>  
</tr>
  
<tr>
    <td>BraTS2020</td> 
    <td><a href="https://drive.google.com/file/d/1QxFHy8nYeaj5OPQExmcbf9PQNzMOhoCy/view?usp=sharing">GDrive</a></td>
    <td><a href="https://drive.google.com/file/d/1x7gSu3G2Cd4n_Gl8yDmpy7wzOW8XTN5J/view?usp=drive_link">GDrive</a></td>
    <td>0.899</td>
</tr>

<tr>
    <td>LUNA16</td> 
    <td><a href="https://drive.google.com/file/d/1ALMc7s8_WozNm1rckSo1gEgpB4GNpAJs/view?usp=sharing">GDrive</a></td>
    <td><a href="https://drive.google.com/file/d/1Yc_dQ6Gb5tn6GM7BDvmL9-YGY9GIHxW9/view?usp=sharing">GDrive</a></td>
     <td>0.634</td>
</tr>
 
 


</table>

### 🔒 License
The present code is released under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.


### Citation
```
@inproceedings{Georgescu-KES-2023,
  title="{Masked Autoencoders for Unsupervised Anomaly Detection in Medical Images}",
  author={Georgescu, Mariana-Iuliana},
  booktitle={Proceedings of KES},
  year={2023}
}
```
