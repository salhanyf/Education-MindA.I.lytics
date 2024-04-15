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

    def k_fold(self, dataset_loader, num_epochs=30, k_folds=10, batch_size=32):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        kf = KFold(n_splits=k_folds, shuffle=True)

        total_accuracy = 0
        total_macro_precision = 0
        total_macro_recall = 0
        total_macro_f1_score = 0
        total_micro_precision = 0
        total_micro_recall = 0
        total_micro_f1_score = 0

        for fold, (train_indices, test_indices) in enumerate(kf.split(dataset_loader.dataset)):
            train_indices, val_indices = train_test_split(train_indices, test_size=0.15, random_state=42)
            train_dataset = torch.utils.data.Subset(dataset_loader.dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset_loader.dataset, val_indices)
            test_dataset = torch.utils.data.Subset(dataset_loader.dataset, test_indices)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            best_val_loss = float('inf')
            best_model_state_dict = None

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
                    best_model_state_dict = self.state_dict()

            # Save the best model for this fold
            torch.save(best_model_state_dict, f'best_model_{fold}.pt')

            # Load the best model state dict for evaluation
            self.load_state_dict(best_model_state_dict)

            # Evaluate the model on the test set
            self.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                all_labels = []
                all_predicted = []
                for images, labels in test_loader:
                    outputs = self(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_labels.extend(labels.numpy())
                    all_predicted.extend(predicted.numpy())

                accuracy = correct / total

                # Calculate precision, recall, and F1-score using sklearn metrics
                macro_precision, macro_recall, macro_f1_score, _ = precision_recall_fscore_support(all_labels,
                                                                                                   all_predicted,
                                                                                                   average='macro')
                micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(all_labels,
                                                                                                   all_predicted,
                                                                                                   average='micro')

                total_accuracy += accuracy
                total_macro_precision += macro_precision
                total_macro_recall += macro_recall
                total_macro_f1_score += macro_f1_score
                total_micro_precision += micro_precision
                total_micro_recall += micro_recall
                total_micro_f1_score += micro_f1_score

                print(f'Fold {fold + 1}/{k_folds}:')
                print(f'  Accuracy: {accuracy}')
                print(f'  Macro Precision: {macro_precision}')
                print(f'  Macro Recall: {macro_recall}')
                print(f'  Macro F1-score: {macro_f1_score}')
                print(f'  Micro Precision: {micro_precision}')
                print(f'  Micro Recall: {micro_recall}')
                print(f'  Micro F1-score: {micro_f1_score}')

        avg_accuracy = total_accuracy / k_folds
        avg_macro_precision = total_macro_precision / k_folds
        avg_macro_recall = total_macro_recall / k_folds
        avg_macro_f1_score = total_macro_f1_score / k_folds
        avg_micro_precision = total_micro_precision / k_folds
        avg_micro_recall = total_micro_recall / k_folds
        avg_micro_f1_score = total_micro_f1_score / k_folds

        print(f'Average across all folds:')
        print(f'  Accuracy: {avg_accuracy}')
        print(f'  Macro Precision: {avg_macro_precision}')
        print(f'  Macro Recall: {avg_macro_recall}')
        print(f'  Macro F1-score: {avg_macro_f1_score}')
        print(f'  Micro Precision: {avg_micro_precision}')
        print(f'  Micro Recall: {avg_micro_recall}')
        print(f'  Micro F1-score: {avg_micro_f1_score}')

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