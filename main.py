# Authors:
# (based on skeleton code for CSCI-B 657)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from dataset_class import PatchShuffled_CIFAR10
from matplotlib import pyplot as plt
import argparse
import csv
import os
from torch.utils.data import Subset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ===================== Shared Utility =====================

def extract_patches(x, patch_size):
    """Split (B,C,H,W) into (B, num_patches, C, patch_size, patch_size)."""
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    B, C, nph, npw, ph, pw = x.shape
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    return x.reshape(B, nph * npw, C, ph, pw)


# ===================== FC Models =====================

# Define the FC model architecture for CIFAR10
class Net_FC(nn.Module):
    def __init__(self):
        super(Net_FC, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),          # deeper: added 3rd hidden layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward_features(self, x):
        x = self.flatten(x)
        return self.fc[:-1](x)
    def forward(self, x):
        return self.fc(self.flatten(x))

# Define the FC model architecture for D-shuffletruffle
class Net_D_shuffletruffle_FC(nn.Module):
    """Each 16x16 patch encoded independently via shared FC layers, then mean-pooled.
    Mean-pool makes output identical regardless of patch order → shuffle invariant."""
    PATCH_SIZE = 16  # 2x2 grid = 4 patches

    def __init__(self):
        super(Net_D_shuffletruffle_FC, self).__init__()
        patch_dim = 3 * self.PATCH_SIZE * self.PATCH_SIZE  # 768
        self.patch_fc = nn.Sequential(
            nn.Linear(patch_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(  # deeper: 2-layer classifier head
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        B = x.shape[0]
        patches = extract_patches(x, self.PATCH_SIZE)         # (B, 4, 3, 16, 16)
        n = patches.shape[1]
        patches = patches.reshape(B * n, -1)                  # (B*4, 768)
        feats = self.patch_fc(patches)                        # (B*4, 64)
        feats = feats.reshape(B, n, 64).mean(dim=1)           # (B, 64) order-invariant
        return self.classifier(feats)
    def forward_features(self, x):
        B = x.shape[0]
        patches = extract_patches(x, self.PATCH_SIZE)
        n = patches.shape[1]
        patches = patches.reshape(B * n, -1)
        feats = self.patch_fc(patches)
        feats = feats.reshape(B, n, 64).mean(dim=1)
        return self.classifier[:-1](feats)

# Define the FC model architecture for N-shuffletruffle
class Net_N_shuffletruffle_FC(nn.Module):
    """Each 8x8 patch encoded independently via shared FC layers, then mean-pooled.
    Mean-pool makes output identical regardless of patch order → shuffle invariant."""
    PATCH_SIZE = 8  # 4x4 grid = 16 patches

    def __init__(self):
        super(Net_N_shuffletruffle_FC, self).__init__()
        patch_dim = 3 * self.PATCH_SIZE * self.PATCH_SIZE  # 192
        self.patch_fc = nn.Sequential(
            nn.Linear(patch_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(  # deeper: 2-layer classifier head
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        B = x.shape[0]
        patches = extract_patches(x, self.PATCH_SIZE)         # (B, 16, 3, 8, 8)
        n = patches.shape[1]
        patches = patches.reshape(B * n, -1)                  # (B*16, 192)
        feats = self.patch_fc(patches)                        # (B*16, 32)
        feats = feats.reshape(B, n, 32).mean(dim=1)           # (B, 32) order-invariant
        return self.classifier(feats)
    def forward_features(self, x):
        B = x.shape[0]
        patches = extract_patches(x, self.PATCH_SIZE)
        n = patches.shape[1]
        patches = patches.reshape(B * n, -1)
        feats = self.patch_fc(patches)
        feats = feats.reshape(B, n, 32).mean(dim=1)
        return self.classifier[:-1](feats)


# ===================== CNN Models =====================

# Define the CNN model architecture for CIFAR10
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout)  # drops entire channels between the two convs
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)   # regularize between the two convs

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class Net_CNN(nn.Module):
    def __init__(self):
        super(Net_CNN, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier[:-1](x)


# Define the CNN model architecture for D-shuffletruffle
class Net_D_shuffletruffle_CNN(nn.Module):
    """Each 16x16 patch processed by the same ResNet-style encoder (shared weights).
    Conv kernels never cross patch boundaries. Mean-pool → shuffle invariant.

    Patch encoder (applied independently to each 16x16 patch):
        stem  : Conv(3→32)
        layer1: ResBlock(32→32)             — stays 16x16
        layer2: ResBlock(32→64,  stride=2)  — 16x16 → 8x8
        layer3: ResBlock(64→128, stride=2)  — 8x8   → 4x4
        pool  : AdaptiveAvgPool → 128-d vector
    """
    PATCH_SIZE = 16

    def __init__(self):
        super(Net_D_shuffletruffle_CNN, self).__init__()
        self.patch_encoder = nn.Sequential(
            # stem
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # layer1 — no downsampling, identity shortcut
            ResidualBlock(32, 32),
            # layer2 — 16x16 → 8x8, projection shortcut
            ResidualBlock(32, 64, stride=2),
            # layer3 — 8x8 → 4x4, projection shortcut
            ResidualBlock(64, 128, stride=2),
            # collapse spatial dims
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        B = x.shape[0]
        patches = extract_patches(x, self.PATCH_SIZE)              # (B, 4, 3, 16, 16)
        n = patches.shape[1]
        patches = patches.reshape(B * n, 3, self.PATCH_SIZE, self.PATCH_SIZE)
        feats = self.patch_encoder(patches).flatten(1)             # (B*4, 128)
        feats = feats.reshape(B, n, 128).mean(dim=1)               # (B, 128) order-invariant
        return self.classifier(feats)
    def forward_features(self, x):
        B = x.shape[0]
        patches = extract_patches(x, self.PATCH_SIZE)
        n = patches.shape[1]
        patches = patches.reshape(B * n, 3, self.PATCH_SIZE, self.PATCH_SIZE)
        feats = self.patch_encoder(patches).flatten(1)
        feats = feats.reshape(B, n, 128).mean(dim=1)
        return self.classifier[:-1](feats)

# Define the CNN model architecture for N-shuffletruffle
class Net_N_shuffletruffle_CNN(nn.Module):
    """Each 8x8 patch processed by the same ResNet-style encoder (shared weights).
    Conv kernels never cross patch boundaries. Mean-pool → shuffle invariant.

    Patch encoder (applied independently to each 8x8 patch):
        stem  : Conv(3→16)
        layer1: ResBlock(16→16)            — stays 8x8
        layer2: ResBlock(16→32, stride=2)  — 8x8 → 4x4
        layer3: ResBlock(32→64, stride=2)  — 4x4 → 2x2
        pool  : AdaptiveAvgPool → 64-d vector
    """
    PATCH_SIZE = 8

    def __init__(self):
        super(Net_N_shuffletruffle_CNN, self).__init__()
        self.patch_encoder = nn.Sequential(
            # stem
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            # layer1 — no downsampling, identity shortcut
            ResidualBlock(16, 16),
            # layer2 — 8x8 → 4x4, projection shortcut
            ResidualBlock(16, 32, stride=2),
            # layer3 — 4x4 → 2x2, projection shortcut
            ResidualBlock(32, 64, stride=2),
            # collapse spatial dims
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        B = x.shape[0]
        patches = extract_patches(x, self.PATCH_SIZE)              # (B, 16, 3, 8, 8)
        n = patches.shape[1]
        patches = patches.reshape(B * n, 3, self.PATCH_SIZE, self.PATCH_SIZE)
        feats = self.patch_encoder(patches).flatten(1)             # (B*16, 64)
        feats = feats.reshape(B, n, 64).mean(dim=1)               # (B, 64) order-invariant
        return self.classifier(feats)
    def forward_features(self, x):
        B = x.shape[0]
        patches = extract_patches(x, self.PATCH_SIZE)
        n = patches.shape[1]
        patches = patches.reshape(B * n, 3, self.PATCH_SIZE, self.PATCH_SIZE)
        feats = self.patch_encoder(patches).flatten(1)
        feats = feats.reshape(B, n, 64).mean(dim=1)
        return self.classifier[:-1](feats)

# ===================== Attention Models =====================

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


# Define the Attention model architecture for CIFAR10
class Net_Attention(nn.Module):
    """Vision Transformer WITH positional embeddings.
    patch_size=4 → 8x8=64 patches. Positional info baked in → position-sensitive."""
    def __init__(self, patch_size=4, dim=128, depth=4, num_heads=4):
        super(Net_Attention, self).__init__()
        self.patch_size  = patch_size
        num_patches      = (32 // patch_size) ** 2      # 64
        patch_dim        = 3 * patch_size * patch_size  # 48
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.embed_norm  = nn.LayerNorm(dim)             # normalize token magnitudes after embed
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks      = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm        = nn.LayerNorm(dim)
        self.classifier  = nn.Sequential(               # 2-layer MLP head
            nn.Linear(dim, 256), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10),
        )
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        patches = extract_patches(x, self.patch_size)  # (B, 64, 3, 4, 4)
        patches = patches.flatten(2)                   # (B, 64, 48)
        tokens  = self.embed_norm(self.patch_embed(patches))  # (B, 64, dim) + LN
        cls     = self.cls_token.expand(B, -1, -1)
        tokens  = torch.cat([cls, tokens], dim=1)      # (B, 65, dim)
        tokens  = tokens + self.pos_embed              # positional info → position-sensitive
        for block in self.blocks:
            tokens = block(tokens)
        tokens  = self.norm(tokens)
        return self.classifier(tokens[:, 0])           # CLS token → MLP head
    def forward_features(self, x):
        B = x.shape[0]
        patches = extract_patches(x, self.patch_size)
        patches = patches.flatten(2)
        tokens = self.embed_norm(self.patch_embed(patches))
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.classifier[:-1](tokens[:, 0])


# Define the Attention model architecture for D-shuffletruffle
class Net_D_shuffletruffle_Attention(nn.Module):
    """ViT WITHOUT positional embeddings, 16x16 patches.
    No pos_embed → attention sees tokens as unordered set → shuffle invariant."""
    PATCH_SIZE = 16

    def __init__(self, dim=128, depth=4, num_heads=4):
        super(Net_D_shuffletruffle_Attention, self).__init__()
        patch_dim        = 3 * self.PATCH_SIZE * self.PATCH_SIZE  # 768
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.embed_norm  = nn.LayerNorm(dim)
        # NO pos_embed — this is the key difference
        self.blocks      = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm        = nn.LayerNorm(dim)
        self.classifier  = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10),
        )
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)

    def forward(self, x):
        patches = extract_patches(x, self.PATCH_SIZE)  # (B, 4, 3, 16, 16)
        patches = patches.flatten(2)                   # (B, 4, 768)
        tokens  = self.embed_norm(self.patch_embed(patches))  # (B, 4, dim) + LN
        for block in self.blocks:
            tokens = block(tokens)
        tokens  = self.norm(tokens)
        return self.classifier(tokens.mean(dim=1))     # mean-pool → order-invariant
    def forward_features(self, x):
        patches = extract_patches(x, self.PATCH_SIZE)
        patches = patches.flatten(2)
        tokens = self.embed_norm(self.patch_embed(patches))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.classifier[:-1](tokens.mean(dim=1))


# Define the Attention model architecture for N-shuffletruffle
class Net_N_shuffletruffle_Attention(nn.Module):
    """ViT WITHOUT positional embeddings, 8x8 patches.
    No pos_embed → attention sees tokens as unordered set → shuffle invariant."""
    PATCH_SIZE = 8

    def __init__(self, dim=128, depth=4, num_heads=4):
        super(Net_N_shuffletruffle_Attention, self).__init__()
        patch_dim        = 3 * self.PATCH_SIZE * self.PATCH_SIZE  # 192
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.embed_norm  = nn.LayerNorm(dim)
        # NO pos_embed — this is the key difference
        self.blocks      = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm        = nn.LayerNorm(dim)
        self.classifier  = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10),
        )
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)

    def forward(self, x):
        patches = extract_patches(x, self.PATCH_SIZE)  # (B, 16, 3, 8, 8)
        patches = patches.flatten(2)                   # (B, 16, 192)
        tokens  = self.embed_norm(self.patch_embed(patches))  # (B, 16, dim) + LN
        for block in self.blocks:
            tokens = block(tokens)
        tokens  = self.norm(tokens)
        return self.classifier(tokens.mean(dim=1))     # mean-pool → order-invariant
    def forward_features(self, x):
        patches = extract_patches(x, self.PATCH_SIZE)
        patches = patches.flatten(2)
        tokens = self.embed_norm(self.patch_embed(patches))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.classifier[:-1](tokens.mean(dim=1))


def build_model(model_class, device):
    if model_class == 'Plain-Old-CIFAR10-FC':
        net = Net_FC()
    elif model_class == 'D-shuffletruffle-FC':
        net = Net_D_shuffletruffle_FC()
    elif model_class == 'N-shuffletruffle-FC':
        net = Net_N_shuffletruffle_FC()
    elif model_class == 'Plain-Old-CIFAR10-CNN':
        net = Net_CNN()
    elif model_class == 'D-shuffletruffle-CNN':
        net = Net_D_shuffletruffle_CNN()
    elif model_class == 'N-shuffletruffle-CNN':
        net = Net_N_shuffletruffle_CNN()
    elif model_class == 'Plain-Old-CIFAR10-Attention':
        net = Net_Attention()
    elif model_class == 'D-shuffletruffle-Attention':
        net = Net_D_shuffletruffle_Attention()
    elif model_class == 'N-shuffletruffle-Attention':
        net = Net_N_shuffletruffle_Attention()
    else:
        raise ValueError(f'Unknown model_class: {model_class}')
    return net.to(device)

def eval_model(model, data_loader, criterion, device):
    # Evaluate the model on data from valloader
    correct = 0
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    return val_loss / len(data_loader), 100 * correct / len(data_loader.dataset)



def main(epochs = 100,
         model_class = 'Plain-Old-CIFAR10-FC',
         batch_size = 128,
         learning_rate = 1e-2,
         l2_regularization = 0.0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Load and preprocess the dataset, feel free to add other transformations that don't shuffle the patches.
    # (Note - augmentations are typically not performed on validation set)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # val/test use the same normalization — no augmentation, same stats from training
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Build train/val splits from the same underlying CIFAR-10 training data,
    # but attach different transforms so validation stays clean and deterministic.
    full_train_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    full_train_eval = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train_aug,
        [40000, 10000],
        generator=torch.Generator().manual_seed(0)
    )
    trainset = train_subset
    valset = Subset(full_train_eval, val_subset.indices)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Initialize Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size= batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Initialize the model, the loss function and optimizer
    net = build_model(model_class, device)

    print(net) # print model architecture
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay= l2_regularization)

    # CSV logger — one row per epoch
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    csv_path = f'./logs/{model_class}.csv'
    checkpoint_path = f'./checkpoints/{model_class}.pt'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    best_val_acc = float('-inf')

    # Train the model
    try:
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            net.train()
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(trainloader)
            train_acc  = 100 * correct / len(trainloader.dataset)

            val_loss, val_acc = eval_model(net, valloader, criterion, device)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_class': model_class,
                    'state_dict': net.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch,
                }, checkpoint_path)

            if epoch % 10 == 0:
                print('epoch - %d loss: %.3f accuracy: %.3f val_loss: %.3f val_acc: %.3f' % (epoch, train_loss, train_acc, val_loss, val_acc))
            else:
                print('epoch - %d loss: %.3f accuracy: %.3f' % (epoch, train_loss, train_acc))

            csv_writer.writerow([epoch, f'{train_loss:.4f}', f'{train_acc:.2f}', f'{val_loss:.4f}', f'{val_acc:.2f}'])
            csv_file.flush()

        print('Finished training')
    except KeyboardInterrupt:
        pass

    csv_file.close()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    # Evaluate the model on the test set
    test_loss, test_acc = eval_model(net, testloader, criterion, device)
    print('Test loss: %.3f accuracy: %.3f' % (test_loss, test_acc))

    # Evaluate the model on the patch shuffled test data

    patch_size = 16
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path = f'test_patch_{patch_size}.npz', transforms = test_transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss_16, patch_shuffle_test_acc_16 = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss_16} accuracy: {patch_shuffle_test_acc_16}')

    patch_size = 8
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path = f'test_patch_{patch_size}.npz', transforms = test_transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss_8, patch_shuffle_test_acc_8 = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss_8} accuracy: {patch_shuffle_test_acc_8}')

    # Append final test results to summary CSV
    summary_path = './logs/test_summary.csv'
    write_header = not os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['model', 'test_loss', 'test_acc',
                             'patch16_loss', 'patch16_acc',
                             'patch8_loss',  'patch8_acc'])
        writer.writerow([model_class,
                         f'{test_loss:.4f}',               f'{test_acc:.2f}',
                         f'{patch_shuffle_test_loss_16:.4f}', f'{patch_shuffle_test_acc_16:.2f}',
                         f'{patch_shuffle_test_loss_8:.4f}',  f'{patch_shuffle_test_acc_8:.2f}'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default= 100,
                        help= "number of epochs the model needs to be trained for")
    parser.add_argument('--model_class',
                        type=str,
                        default= 'Plain-Old-CIFAR10-FC',
                        choices=['Plain-Old-CIFAR10-FC','D-shuffletruffle-FC','N-shuffletruffle-FC',
                                 'Plain-Old-CIFAR10-CNN','D-shuffletruffle-CNN','N-shuffletruffle-CNN',
                                 'Plain-Old-CIFAR10-Attention','D-shuffletruffle-Attention','N-shuffletruffle-Attention'],
                        help="specifies the model class that needs to be used for training, validation and testing.")
    parser.add_argument('--batch_size',
                        type=int,
                        default= 128,
                        help = "batch size for training")
    parser.add_argument('--learning_rate',
                        type=float,
                        default = 0.01,
                        help = "learning rate for training")
    parser.add_argument('--l2_regularization',
                        type=float,
                        default= 0.0,
                        help = "l2 regularization for training")

    args = parser.parse_args()
    main(**vars(args))
