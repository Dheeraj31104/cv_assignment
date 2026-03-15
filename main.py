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


# ===================== FC Models =====================

# Define the FC model architecture for CIFAR10
class Net_FC(nn.Module):
    def __init__(self):
        super(Net_FC, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Define the FC model architecture for D-shuffletruffle
class Net_D_shuffletruffle_FC(nn.Module):
    def __init__(self):
        super(Net_D_shuffletruffle_FC, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Define the FC model architecture for N-shuffletruffle
class Net_N_shuffletruffle_FC(nn.Module):
    def __init__(self):
        super(Net_N_shuffletruffle_FC, self).__init__()
        # TODO: implement FC architecture
        self.fc = nn.Linear(3*32*32, 10)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# ===================== CNN Models =====================

# Define the CNN model architecture for CIFAR10
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# Define the CNN model architecture for D-shuffletruffle
class Net_D_shuffletruffle_CNN(nn.Module):
    def __init__(self):
        super(Net_D_shuffletruffle_CNN, self).__init__()
        # TODO: implement CNN architecture
        self.fc = nn.Linear(3*32*32, 10)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Define the CNN model architecture for N-shuffletruffle
class Net_N_shuffletruffle_CNN(nn.Module):
    def __init__(self):
        super(Net_N_shuffletruffle_CNN, self).__init__()
        # TODO: implement CNN architecture
        self.fc = nn.Linear(3*32*32, 10)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# ===================== Attention Models =====================

# Define the Attention model architecture for CIFAR10
class Net_Attention(nn.Module):
    def __init__(self):
        super(Net_Attention, self).__init__()
        # TODO: implement Attention architecture
        self.fc = nn.Linear(3*32*32, 10)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Define the Attention model architecture for D-shuffletruffle
class Net_D_shuffletruffle_Attention(nn.Module):
    def __init__(self):
        super(Net_D_shuffletruffle_Attention, self).__init__()
        # TODO: implement Attention architecture
        self.fc = nn.Linear(3*32*32, 10)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Define the Attention model architecture for N-shuffletruffle
class Net_N_shuffletruffle_Attention(nn.Module):
    def __init__(self):
        super(Net_N_shuffletruffle_Attention, self).__init__()
        # TODO: implement Attention architecture
        self.fc = nn.Linear(3*32*32, 10)
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

def eval_model(model, data_loader, criterion, device):
    # Evaluate the model on data from valloader
    correct = 0
    total = 0
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
    transform = transforms.Compose([
        transforms.ToTensor()])

    
    # Initialize training, validation and test dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000], generator=torch.Generator().manual_seed(0))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Initialize Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size= batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Initialize the model, the loss function and optimizer
    if model_class == 'Plain-Old-CIFAR10-FC':
        net = Net_FC().to(device)
    elif model_class == 'D-shuffletruffle-FC':
        net = Net_D_shuffletruffle_FC().to(device)
    elif model_class == 'N-shuffletruffle-FC':
        net = Net_N_shuffletruffle_FC().to(device)
    elif model_class == 'Plain-Old-CIFAR10-CNN':
        net = Net_CNN().to(device)
    elif model_class == 'D-shuffletruffle-CNN':
        net = Net_D_shuffletruffle_CNN().to(device)
    elif model_class == 'N-shuffletruffle-CNN':
        net = Net_N_shuffletruffle_CNN().to(device)
    elif model_class == 'Plain-Old-CIFAR10-Attention':
        net = Net_Attention().to(device)
    elif model_class == 'D-shuffletruffle-Attention':
        net = Net_D_shuffletruffle_Attention().to(device)
    elif model_class == 'N-shuffletruffle-Attention':
        net = Net_N_shuffletruffle_Attention().to(device)
    
    print(net) # print model architecture
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay= l2_regularization)


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

            if epoch % 10 == 0:
                val_loss, val_acc = eval_model(net, valloader, criterion, device)
                print('epoch - %d loss: %.3f accuracy: %.3f val_loss: %.3f val_acc: %.3f' % (epoch, running_loss / len(trainloader), 100 * correct / len(trainloader.dataset), val_loss, val_acc))
            else:
                print('epoch - %d loss: %.3f accuracy: %.3f' % (epoch, running_loss / len(trainloader), 100 * correct / len(trainloader.dataset)))


        print('Finished training')
    except KeyboardInterrupt:
        pass

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    net.eval()
    # Evaluate the model on the test set
    test_loss, test_acc = eval_model(net, testloader, criterion, device)
    print('Test loss: %.3f accuracy: %.3f' % (test_loss, test_acc))

    # Evaluate the model on the patch shuffled test data

    patch_size = 16
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path = f'test_patch_{patch_size}.npz', transforms = transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss, patch_shuffle_test_acc = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss} accuracy: {patch_shuffle_test_acc}')

    patch_size = 8
    patch_shuffle_testset = PatchShuffled_CIFAR10(data_file_path = f'test_patch_{patch_size}.npz', transforms = transform)
    patch_shuffle_testloader = torch.utils.data.DataLoader(patch_shuffle_testset, batch_size=batch_size, shuffle=False)
    patch_shuffle_test_loss, patch_shuffle_test_acc = eval_model(net, patch_shuffle_testloader, criterion, device)
    print(f'Patch shuffle test loss for patch-size {patch_size}: {patch_shuffle_test_loss} accuracy: {patch_shuffle_test_acc}')

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
