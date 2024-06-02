import torch
from network import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import random

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.set_device(0)
device = 'cuda'

model = imagenet_get_model('res34')
model.to(device)
#model.load_state_dict(torch.load('./model/victim/vict-imgnet-resnet34.pt'))
model.load_state_dict(torch.load('./model/benign/benign-imgnet-resnet34.pt'))

path = './data/sub-imagenet-20/'
batch_size = 1
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    }
image_datasets = {x: datasets.ImageFolder(path+x, data_transforms[x])
                      for x in ['train', 'val', 'test']}
train_loader = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True,
                                   num_workers=4, drop_last=True)
test_loader = DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False,
                                 num_workers=4)

#optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)

# model.eval()
# for batch_idx, (data, target) in enumerate(train_loader):
#     data, target = data.to(device), target.to(device)
#     optimizer.zero_grad()
#     output = model(data)
#     loss = F.cross_entropy(output, target)
    
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        #print(labels)
        outputs = model(images)
        #print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Accuracy of the network on the test images: {} %'.format(accuracy))

