## 1. Requirements
* To install requirements: 
```
pip install -r requirements.txt
```

* Hardware requirements
8 GPUs with >= 11G GPU RAM are recommended. Otherwise the model with more experts may not fit in, especially on datasets with more classes (the FC layers will be large). We do not support CPU training, but CPU inference could be supported by slight modification.

## 2. Datasets
### (1) Four bechmark datasets 
* Please download these datasets and put them to the /data file.
* ImageNet-LT and Places-LT can be found at [here](https://drive.google.com/drive/u/1/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf).
* iNaturalist data should be the 2018 version from [here](https://github.com/visipedia/inat_comp).
* CIFAR-100 will be downloaded automatically with the dataloader.

```
data
├── ImageNet_LT
│   ├── test
│   ├── train
│   └── val
├── CIFAR100
│   └── cifar-100-python
├── CIFAR10
│   └── cifar-10-python
├── Place365
│   ├── data_256
│   ├── test_256
│   └── val_256
└── iNaturalist 
    ├── test2018
    └── train_val2018
```

### (2) Txt files
* We provide txt files for test-agnostic long-tailed recognition for ImageNet-LT, Places-LT and iNaturalist 2018. CIFAR-100 will be generated automatically with the code.
* For iNaturalist 2018, please unzip the iNaturalist_train.zip.
```
data_txt
├── ImageNet_LT
│   ├── ImageNet_LT_backward2.txt
│   ├── ImageNet_LT_backward5.txt
│   ├── ImageNet_LT_backward10.txt
│   ├── ImageNet_LT_backward25.txt
│   ├── ImageNet_LT_backward50.txt
│   ├── ImageNet_LT_forward2.txt
│   ├── ImageNet_LT_forward5.txt
│   ├── ImageNet_LT_forward10.txt
│   ├── ImageNet_LT_forward25.txt
│   ├── ImageNet_LT_forward50.txt
│   ├── ImageNet_LT_test.txt
│   ├── ImageNet_LT_train.txt
│   ├── ImageNet_LT_uniform.txt
│   └── ImageNet_LT_val.txt
├── Places_LT_v2
│   ├── Places_LT_backward2.txt
│   ├── Places_LT_backward5.txt
│   ├── Places_LT_backward10.txt
│   ├── Places_LT_backward25.txt
│   ├── Places_LT_backward50.txt
│   ├── Places_LT_forward2.txt
│   ├── Places_LT_forward5.txt
│   ├── Places_LT_forward10.txt
│   ├── Places_LT_forward25.txt
│   ├── Places_LT_forward50.txt
│   ├── Places_LT_test.txt
│   ├── Places_LT_train.txt
│   ├── Places_LT_uniform.txt
│   └── Places_LT_val.txt
└── iNaturalist18
    ├── iNaturalist18_backward2.txt
    ├── iNaturalist18_backward3.txt
    ├── iNaturalist18_forward2.txt
    ├── iNaturalist18_forward3.txt
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_uniform.txt
    └── iNaturalist18_val.txt 
```


## 4. Pretrained models
* For the training on Places-LT, we follow previous methods and use [the pre-trained ResNet-152 model](https://github.com/zhmiao/OpenLongTailRecognition-OLTR).
* Please download the checkpoint. Unzip and move the checkpoint files to /model/pretrained_model_places/.


#### Training
# train SADE_RL/BSCE_RL

### (1) CIFAR100-LT 

```
nohup python train.py -c configs/{sade or bsce}/config_cifar100_ir10_{sade or ce}_rl.json &>{sade or ce}_rl_10.out&
nohup python train.py -c configs/{sade or bsce}/config_cifar100_ir50_{sade or ce}_rl.json &>{sade or ce}_rl_50.out&
nohup python train.py -c configs/{sade or bsce}/config_cifar100_ir100_{sade or ce}_rl.json &>{sade or ce}_rl_100.out&

Example:
nohup python train.py -c configs/sade/config_cifar100_ir100_sade_rl.json &>sade_rl_100.out&
# test
python test.py -r {$PATH}

```

### (2) ImageNet-LT 

python train.py -c configs/{sade or bsce}/config_imagenet_lt_resnext50_{sade or ce}_rl.json

### (3) Place-LT 

python train.py -c configs/{sade or bsce}/config_imagenet_lt_resnext50_{sade or ce}_rl.json
### (4) iNatrualist2018-LT 
python train.py -c configs/{sade or bsce}/config_iNaturalist_resnet50_{sade or ce}_rl.json



# train  SADE/BSCE

```
nohup python train.py -c configs/{sade/bsce}/config_cifar100_ir10_{sade/ce}.json &>{sade/ce}_10.out&
nohup python train.py -c configs/{sade/bsce}/config_cifar100_ir50_{sade/ce}.json &>{sade/ce}_50.out&
nohup python train.py -c configs/{sade/bsce}/config_cifar100_ir100_{sade/ce}.json &>{sade/ce}_100.out&
```

# test
python test.py -r {$PATH}

### (2) ImageNet-LT 
python train.py -c configs/{sade or bsce}config_imagenet_lt_resnext50_{sade or ce}.json

### (3) Place-LT 
python train.py -c configs/{sade or bsce}/config_imagenet_lt_resnext50_{sade or ce}_rl.json
### (4) iNatrualist2018-LT 

python train.py -c configs/{sade or bsce}/config_iNaturalist_resnet50_{sade or ce}_rl.json
