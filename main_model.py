import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np



class MainModel(nn.Module):
    #initialize Main Model: kernel = 3x3, convolution layers 2
    def __init__(self, dropout_rate=0.5, weight_decay=1e-5):
        super(MainModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 4)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def train_model(self, train_loader, val_loader, num_epochs=10, save_path_best=None, save_path_last=None):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(num_epochs):
            self.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            self.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            train_loss = loss.item()
            train_acc = 100 * correct / total
            val_loss /= len(val_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if save_path_best:
                    torch.save(self.state_dict(), save_path_best)

            if save_path_last:
                torch.save(self.state_dict(), save_path_last)

        print(f'Finished Training. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}')


    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def inference(self, image_path, classes):
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        image_tensor = transform(image).unsqueeze(0)
        output = self(image_tensor)
        predicted_class_idx = torch.argmax(output, dim=1).item()
        predicted_class = classes[predicted_class_idx]
        print(f'Predicted class: {predicted_class} ({predicted_class_idx})')
        return predicted_class
