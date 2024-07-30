## 环境安装 使用conda 

```
conda create --name py311 python=3.11
conda activate py311
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy scikit-learn
pip install opencv-python matplotlib
```

解释下每个用得上的脚本的功能，没有解释的是在其他脚本中调用的

train_model.py : 训练受害者模型

data_enhance.py : 数据集增强

gradientset.py : 白盒场景下收集模型梯度数据

logitset.py : 黑盒场景下收集模型输出向量

## 所有权验证器数据集生成

收集受害者模型和良性模型的特征，作为所有权验证器的数据集

### 白盒验证场景

CIFAR-10 数据集

收集受害者模型梯度特征
```
python gradientset.py --model=wrn28-10 -m=./model/victim/vict-wrn28-10.pt --dataset=cifar10
```
收集良性模型梯度特征
```
python gradientset.py --model=wrn28-10 -m=./model/benign/benign-wrn28-10.pt --dataset=cifar10
```

ImageNet 数据集:

收集受害者模型梯度特征
```
python gradientset.py --model=resnet34-imgnet -m=./model/victim/vict-imgnet-resnet34.pt --dataset=imagenet
```
收集良性模型梯度特征
```
python gradientset.py --model=resnet34-imgnet -m=./model/benign/benign-imgnet-resnet34.pt --dataset=imagenet
```

### 黑盒验证场景

CIFAR-10 数据集:

收集受害者模型输出向量特征
```
python logitset.py --model=wrn28-10 -m=./model/victim/vict-wrn28-10.pt --dataset=cifar10
```
收集良性模型输出向量特征
```
python logitset.py --model=wrn28-10 -m=./model/benign/benign-wrn28-10.pt --dataset=cifar10
```

ImageNet 数据集:

收集受害者模型输出向量特征
```
python logitset.py --model=resnet34-imgnet -m=./model/victim/vict-imgnet-resnet34.pt --dataset=imagenet
```
收集良性模型输出向量特征
```
python logitset.py --model=resnet34-imgnet -m=./model/benign/benign-imgnet-resnet34.pt --dataset=imagenet
```

## 训练所有权验证器

### 白盒验证场景

CIFAR-10数据集:
```
python train_clf.py --type=wrn28-10 --dataset=cifar10
```

ImageNet数据集:
```
python train_clf.py --type=resnet34-imgnet --dataset=imagenet
```

### 黑盒验证场景

CIFAR-10数据集:
```
python train_clf.py --type=wrn28-10 --dataset=cifar10 --black
```

ImageNet数据集:
```
python train_clf.py --type=resnet34-imgnet --dataset=imagenet --black
```

## 所有权验证

### 白盒验证场景

CIFAR-10:
```
python ownership_verification.py --mode=source --dataset=cifar10 --gpu=0 
```

ImageNet:
```
python ownership_verification.py --mode=logit-query --dataset=imagenet --gpu=0 
```
### 黑盒验证场景

CIFAR-10:
```
python ownership_verification.py --mode=source --dataset=cifar10 --gpu=0 --black
```

ImageNet:
```
python ownership_verification.py --mode=logit-query --dataset=imagenet --gpu=0 --black
```

#mode: ['source','distillation','zero-shot','fine-tune','label-query','logit-query','benign']



