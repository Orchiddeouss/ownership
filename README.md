## Environmental setup using conda 

```
conda create --name py311 python=3.11
conda activate py311
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy scikit-learn
pip install opencv-python matplotlib
```

train_model.py : train victim model

data_enhance.py : dataset enhance

gradientset.py : collect model gradients in white-box scenario 

logitset.py : collect model output vectors in black-box scenario

### White-box verification scenario

CIFAR-10 Dataset

Collect gradient features of the victim model
```
python gradientset.py --model=wrn28-10 -m=./model/victim/vict-wrn28-10.pt --dataset=cifar10
```
Collect gradient features of the benign model
```
python gradientset.py --model=wrn28-10 -m=./model/benign/benign-wrn28-10.pt --dataset=cifar10
```

ImageNet Dataset:

Collect gradient features of the victim model
```
python gradientset.py --model=resnet34-imgnet -m=./model/victim/vict-imgnet-resnet34.pt --dataset=imagenet
```
Collect gradient features of the benign model
```
python gradientset.py --model=resnet34-imgnet -m=./model/benign/benign-imgnet-resnet34.pt --dataset=imagenet
```

### Black-box verification scenario

CIFAR-10 Dataset:

Collect output vector features of the victim model
```
python logitset.py --model=wrn28-10 -m=./model/victim/vict-wrn28-10.pt --dataset=cifar10
```
Collect output vector features of the benign model
```
python logitset.py --model=wrn28-10 -m=./model/benign/benign-wrn28-10.pt --dataset=cifar10
```

ImageNet Dataset:

Collect output vector features of the victim model
```
python logitset.py --model=resnet34-imgnet -m=./model/victim/vict-imgnet-resnet34.pt --dataset=imagenet
```
Collect output vector features of the benign model
```
python logitset.py --model=resnet34-imgnet -m=./model/benign/benign-imgnet-resnet34.pt --dataset=imagenet
```

## Train the ownership validator

### White-box scenario

CIFAR-10 Dataset:
```
python train_clf.py --type=wrn28-10 --dataset=cifar10
```

ImageNet Dataset:
```
python train_clf.py --type=resnet34-imgnet --dataset=imagenet
```

### Black-box scenario

CIFAR-10 Dataset:
```
python train_clf.py --type=wrn28-10 --dataset=cifar10 --black
```

ImageNet Dataset:
```
python train_clf.py --type=resnet34-imgnet --dataset=imagenet --black
```

## Ownership Verification

### White-box scenario

CIFAR-10 Dataset:
```
python ownership_verification.py --mode=source --dataset=cifar10 --gpu=0 
```

ImageNet Dataset:
```
python ownership_verification.py --mode=logit-query --dataset=imagenet --gpu=0 
```
### Black-box scenario

CIFAR-10:
```
python ownership_verification.py --mode=source --dataset=cifar10 --gpu=0 --black
```

ImageNet:
```
python ownership_verification.py --mode=logit-query --dataset=imagenet --gpu=0 --black
```

#mode: ['source','distillation','zero-shot','fine-tune','label-query','logit-query','benign']



