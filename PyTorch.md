# PyTorch Note
PyTorch == 1.0.1

## Dataset

```python
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, root, transform = None):
        self.root = root
        self.transform = transform
        self.len = ...
        
    def __getitem__(self, index):
        image = ...
        if self.transform is not None:
            image = self.transform(image)
        
        label = ...
        
        return image, label

    def __len__(self):
        return self.len
```

## Dataloader

```python
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_set = Data(root=..., transform=transform)
print('# of images in train set: %d' % len(train_set))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1)
    
dataiter = iter(train_loader)
images, labels = dataiter.next()

print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)
```

## Data Augmentation

```python
transform = transforms.Compose([ 
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
```

## Device

```python
import torch

def checkDevice():
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    return device
```

## Model

### Classifier

```python
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(x)
```

### Pretrained model

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x
```

## Training

```python
import torch
import torch.nn as nn

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(n_epoch):
    train_acc = 0.0
    train_loss = 0.0
    
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        output = output.data.cpu().numpy()
        pred = output.argmax(axis=1)
        labels = labels.numpy()
        
        train_acc += np.equal(labels, pred).sum()
        train_loss += loss.item()
    
    train_acc = train_acc / len(train_set)
    train_loss = train_loss / len(train_set)
```

## Testing

### Calculate accuracy

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
best_acc = 0.0

val_acc = 0.0
val_loss = 0.0

model.eval()
with torch.no_grad():
    for data, labels in valid_loader:
        data, labels = data.to(device), labels.to(device)
            
        output = model(data)
        loss = criterion(output, labels)

        output = output.data.cpu().numpy()
        pred = output.argmax(axis=1)
        labels = labels.numpy()

        val_acc += np.equal(labels, pred).sum()
        val_loss += loss.item()

    val_acc = val_acc / len(valid_set)
    val_loss = val_loss / len(valid_set)
    print('Accuracy: %.5f\tLoss: %.5f' % (val_acc, val_loss))

    if (val_acc > best_acc):
        torch.save(model.state_dict(), 'my_model.pth')
        best_acc = val_acc
        print ('Model Saved!')
    
    print('Best acc: %.5f' % best_acc)
```

### Output csv file

```python
import csv

file = open(output_file, 'w+')
out_file = csv.writer(file, delimiter = ',', lineterminator = '\n')
out_file.writerow(['image_name', 'label'])

with torch.no_grad():
    for imgs, img_names in test_loader:
        imgs = imgs.to(device)
        
        output = model(imgs)
        pred = output.max(1, keepdim=True)[1]
        pred = pred.view(pred.size(0)).data

        for i in range(len(img_names)):
            out_file.writerow([img_names[i], pred[i].item()])
    
file.close()
```

## t-SNE Visualization

```python
import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm


use_cuda = torch.cuda.is_available()
torch.manual_seed(123564)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)


source_set = ...
source_loader = DataLoader(source_set, batch_size=32, shuffle=False, num_workers=1)

target_set = ...
target_loader = DataLoader(target_set, batch_size=32, shuffle=False, num_workers=1)

model = ...
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()


empty = True
label = []
domain = []

with torch.no_grad():
    for data, target in source_loader:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        
        if empty == True:
            latent = output
            empty = False
        else:
            latent = torch.cat((latent, output), 0)
        
        for i in target.data:
            label.append(i.item())
            domain.append('b')
    
    for data, target in target_loader:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        
        if empty == True:
            latent = output
            empty = False
        else:
            latent = torch.cat((latent, output), 0)
        
        for i in target.data:
            label.append(i.item())
            domain.append('r')
            
print(latent.size())

np.random.seed(123)

latent = latent.cpu()
#latent_pca = PCA(n_components=50).fit_transform(latent)

tsne = TSNE(n_components=2).fit_transform(latent)
xdata = tsne[:, 0]
ydata = tsne[:, 1]

def discrete_cmap(N, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


N = ... # number of classes
plt.scatter(xdata, ydata, c=label, s=1, cmap=discrete_cmap(N, 'gist_rainbow'))
plt.clim(-0.5, N - 0.5)
plt.savefig('tsne_label.jpg')

plt.scatter(xdata, ydata, c=domain, s=1)
plt.savefig('tsne_domain.jpg')
```

## Check Path

```python
import os

path = ...
if not os.path.exists(path):
    os.makedirs(path)
```
