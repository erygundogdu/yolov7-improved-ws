import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import numpy as np
IMG_SIZE = 160
BATCH_SIZE = 2
NUM_WORKERS = 0
EMBED_DIM = 128
TRIPLET_MARGIN = 0.3
METRIC_WEIGHT = 0.8   # total_loss = CE + METRIC_WEIGHT * triplet
class AugPairDataset(Dataset):
    def __init__(self, image_folder, base_transform, aug_transform):
        self.samples = image_folder.samples
        self.loader = image_folder.loader
        self.classes = image_folder.classes
        self.class_to_idx = image_folder.class_to_idx

        self.base_transform = base_transform
        self.aug_transform = aug_transform

        if "target" not in self.class_to_idx:
            raise ValueError(f"'target' folder not found. Found classes: {self.classes}")

        if "non_target" not in self.class_to_idx:
            raise ValueError(f"'non_target' folder not found. Found classes: {self.classes}")

        self.target_idx = self.class_to_idx["target"]

    def __len__(self):
        
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = self.loader(path)

        image = self.base_transform(img)

        if label == self.target_idx:
            aug_image = self.aug_transform(img)
        else:
            aug_image = image.clone()

        return image, aug_image, label
    

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
train_dir = DATASET_DIR / "train"
val_dir = DATASET_DIR / "val"
test_dir = DATASET_DIR / "test"
MEAN = [0.485, 0.456, 0.406]
STD= [0.229, 0.224, 0.225]
def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
g = torch.Generator()
g.manual_seed(42)

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.0),
    transforms.RandomRotation(degrees=0.0),
    #transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN,
                         std=STD),
])
target_aug_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=90.0),
    transforms.ColorJitter(
        brightness=0.0,  #0.15
        contrast=0.0,  #0.15
        saturation=0.0, #0.10
        hue=0.0,  #0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN,
                         std=STD),
])

base_image_folder = datasets.ImageFolder(train_dir)

train_dataset = AugPairDataset(
    image_folder=base_image_folder,
    base_transform=train_tfms,
    aug_transform=target_aug_tfms,
)
#train_dataset = datasets.ImageFolder(train_dir, transform=train_tfms)
val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_tfms)
test_dataset  = datasets.ImageFolder(test_dir,  transform=eval_tfms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    generator=g
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    generator=g
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    generator=g
)

#print("class_to_idx:", train_dataset.class_to_idx)
# Example expected:
# {'non_target': 0, 'target': 1}

class P4ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3  # P4/C4 feature level
        self.layer4 = backbone.layer4
        
        self.multi = True
        proj_dim = 256
        self.fc5 = nn.Linear(512, proj_dim)
        self.fc4 = nn.Linear(256, proj_dim)
        self.fc3 = nn.Linear(128, proj_dim)
        #self.emb_fc = nn.Linear(512, EMBED_DIM)
        self.scale_attn = nn.Sequential(
            nn.Linear(proj_dim * 3, 3),
            nn.Softmax(dim=1)
        )

        self.cls = nn.Linear(proj_dim, num_classes)
        #self.fc = nn.Linear(512,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f3 = self.layer2(x)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        out = torch.flatten(self.avg_pool(f5), 1)
        logits5 = self.fc5(out)
        
        f4 = torch.flatten(self.avg_pool(f4), 1)
        logits4 = self.fc4(f4)
        f3 = torch.flatten(self.avg_pool(f3), 1)
        logits3 = self.fc3(f3)
        #cat = F.normalize(torch.cat([logits3, logits4, logits5], dim=1),dim=1,p=2)
        cat = torch.cat([logits3, logits4, logits5], dim=1)
        w = self.scale_attn(cat) ##b,3
        fused = (w[:, 0:1] * logits3 +w[:, 1:2] * logits4 +w[:, 2:3] * logits5)

        logits = self.cls(fused)
        return logits, logits5  
class P4ResNetClassifier1(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3  # P4/C4 feature level
        self.layer4 = backbone.layer4
        
        self.multi = True
        proj_dim = 256
        self.fc5 = nn.Linear(512, proj_dim)
        self.fc4 = nn.Linear(256, proj_dim)
        self.fc3 = nn.Linear(128, proj_dim)
        #self.emb_fc = nn.Linear(512, EMBED_DIM)
        self.scale_attn = nn.Sequential(
            nn.Linear(proj_dim * 3, 3),
            nn.Softmax(dim=1)
        )

        self.cls = nn.Linear(3*proj_dim, num_classes)
        #self.fc = nn.Linear(512,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f3 = self.layer2(x)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        out = torch.flatten(self.avg_pool(f5), 1)
        logits5 = self.fc5(out)
        
        f4 = torch.flatten(self.avg_pool(f4), 1)
        logits4 = self.fc4(f4)
        f3 = torch.flatten(self.avg_pool(f3), 1)
        logits3 = self.fc3(f3)
        #cat = F.normalize(torch.cat([logits3, logits4, logits5], dim=1),dim=1,p=2)
        cat = torch.cat([logits3, logits4, logits5], dim=1)
        

        logits = self.cls(cat)
        return logits, logits5  
    

def build_model(num_classes: int, checkpoint_path: str = "resnet18_imagenet.pth"):
    model = P4ResNetClassifier(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    # Drop classifier params from pretrained checkpoint (shape mismatch by design).
    state.pop("fc.weight", None)
    state.pop("fc.bias", None)
    model.load_state_dict(state, strict=False)
    cnt = 0
    for p in model.parameters():
        p.requires_grad = True
        cnt += p.numel()
    print(cnt)
    
    # for p in model.named_parameters():
    #     print(p[0])

    return model


def target_aug_consistency_loss(
    embeddings,
    aug_embeddings,
    labels,
    lam=0.8,
):
    y = labels.view(-1).float()
    target_mask = y == 1

    nt_mask = y == 0

    if not target_mask.any():
        return embeddings.new_tensor(0.0)
    if not nt_mask.any():
        return embeddings.new_tensor(0.0)

    e = F.normalize(embeddings[target_mask], p=2, dim=1)
    e_nt = F.normalize(embeddings[nt_mask], p=2, dim=1)
    e_aug = F.normalize(aug_embeddings[target_mask], p=2, dim=1)

    loss = (e - e_aug).pow(2).sum(dim=1).mean()

    return lam * loss


#model = Classifier()
model = build_model(num_classes=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,
    weight_decay=1e-4,momentum = 0.9
)

epochs = 15
best_acc = -1.0
t = []
loss_ls = []
acc_ls = []
ssl_flag = False       
num = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for images, aug_images, labels in train_loader:
        num +=1
        images = images.to(device)
        aug_images = aug_images.to(device)
        labels = labels.to(device).float().unsqueeze(1)   # [B] -> [B,1]

        logits,embeddings = model(images) 
        _, aug_embeddings = model(aug_images)                           # [B,1]
        loss_cls = criterion(logits, labels)
        # loss_repr = target_repr_loss(
        #     embeddings,
        #     labels,
        #     lam=METRIC_WEIGHT,
        #     beta=1.0,
        #     margin=0.7,
        # )
        loss_aug = target_aug_consistency_loss(embeddings,aug_embeddings,labels,lam=0.1) ##lam0.1
        #loss = loss_cls + loss_repr
        if ssl_flag:
            loss = loss_aug + loss_cls
        else:
            loss = loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)                     # [B,1]
        preds = (probs >= 0.5).long()                    # [B,1]

        running_correct += (preds.squeeze(1) == labels.squeeze(1).long()).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total

    loss_ls.append(epoch_loss)
    acc_ls.append(epoch_acc)
    t.append(epoch)

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), Path(__file__).resolve().parent / "yeni.pt")

    print(f"Epoch {epoch+1}/{epochs} | loss={epoch_loss:.4f} | acc={epoch_acc:.4f}")
import matplotlib.pyplot as plt
print(num*BATCH_SIZE/epochs)
plt.plot(t, loss_ls, label="loss")
plt.plot(t, acc_ls, label="acc")
plt.legend()
plt.show()



