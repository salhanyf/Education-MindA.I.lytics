import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class FacialImageCNN(nn.Module):
    # def __init__(self):
    #     super(FacialImageCNN, self).__init__()
    #     self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    #     self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    #     self.relu = nn.ReLU()
    #     self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.fc1 = nn.Linear(64 * 28 * 28, 128)
    #
    #     #self.fc1 = nn.Linear(64 * 56 * 56, 128)
    #     self.fc2 = nn.Linear(128, 4)
    #
    # def forward(self, x):
    #     x = self.relu(self.conv1(x))
    #     x = self.maxpool(x)
    #     x = self.relu(self.conv2(x))
    #     x = self.maxpool(x)
    #     x = self.relu(self.conv3(x))
    #     x = self.maxpool(x)
    #     x = x.view(-1, 64 * 28 * 28)  # Corrected reshape size
    #
    #     #x = x.view(-1, 64 * 56 * 56)
    #     x = self.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    # def __init__(self):
    #     super(FacialImageCNN, self).__init__()
    #     self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    #     self.relu = nn.ReLU()
    #     self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    #     self.fc1 = nn.Linear(32 * 56 * 56, 128)
    #     self.fc2 = nn.Linear(128, 4)
    #
    # def forward(self, x):
    #     x = self.relu(self.conv1(x))
    #     x = self.maxpool(x)
    #     x = self.relu(self.conv2(x))
    #     x = self.maxpool(x)
    #     x = x.view(-1, 32 * 56 * 56)
    #     x = self.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    def __init__(self):
        super(FacialImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, train_loader, num_epochs=30):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            self.train()
            for images, labels in train_loader:
               # print("Input shape:", images.shape)
                optimizer.zero_grad()
                outputs = self(images)
                #print("Output shape:", outputs.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def inference(self, image_path):
        # Load the image
        image = Image.open(image_path)

        # Convert grayscale image to RGB
        image = image.convert('RGB')

        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Preprocess the image and add a batch dimension
        image_tensor = transform(image).unsqueeze(0)
        output = self(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        print(f'Predicted class: {predicted_class}')


def split_dataset(dataset_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    # Create directories for train, val, and test sets
    train_dir = '../Education-MindA.I.lytics/train'
    val_dir = '../Education-MindA.I.lytics/val'
    test_dir = '../Education-MindA.I.lytics/test'

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all image files
    image_files = [os.path.join(dataset_path, cls, file) for cls in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, cls))
                   for file in os.listdir(os.path.join(dataset_path, cls))
                   if not file.startswith('.')]

    train_files = []
    val_files = []
    test_files = []

    for cls in os.listdir(dataset_path):
        cls_files = [os.path.join(dataset_path, cls, file) for file in os.listdir(os.path.join(dataset_path, cls))]

        # Split into training, validation, and test sets
        train_val_cls_files, test_cls_files = train_test_split(cls_files, test_size=test_ratio, random_state=random_state)
        train_cls_files, val_cls_files = train_test_split(train_val_cls_files, test_size=val_ratio / (train_ratio + val_ratio), random_state=random_state)

        # Move files to respective directories
        for file in train_cls_files:
            new_path = os.path.join(train_dir, cls, os.path.basename(file))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(file, new_path)
            train_files.append(new_path)
        for file in val_cls_files:
            new_path = os.path.join(val_dir, cls, os.path.basename(file))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(file, new_path)
            val_files.append(new_path)
        for file in test_cls_files:
            new_path = os.path.join(test_dir, cls, os.path.basename(file))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(file, new_path)
            test_files.append(new_path)

    return train_files, val_files, test_files

def evaluate_model(model, data_loader, classes):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            predicted_classes = torch.argmax(outputs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted_classes.numpy())

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    #plot_confusion_matrix(cm, classes, normalize=True)

    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }





def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_path = 'Dataset'
    classes = ['Angry', 'Focused', 'Neutral', 'Surprised']
    #
    # # create train, val, and test files
    # #train_files, val_files, test_files = split_dataset(dataset_path)
    #

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create DataLoader objects for train, val, and test sets
    train_dataset = datasets.ImageFolder('train', transform=transform)
    val_dataset = datasets.ImageFolder('val', transform=transform)
    test_dataset = datasets.ImageFolder('test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    #
    # # Initialize and train the model
    # model = FacialImageCNN()
    # model.train_model(train_loader)
    # model.save_model('model_layers2_kernel7x7_v2')

    # Load the model for evaluation
    loaded_model = FacialImageCNN()
    loaded_model.load_model('model_layers2_kernel7x7')

    # Evaluate the loaded model on train, val, and test sets
    train_metrics = evaluate_model(loaded_model, train_loader, classes)
    val_metrics = evaluate_model(loaded_model, val_loader, classes)
    test_metrics = evaluate_model(loaded_model, test_loader, classes)

    # Display or save the confusion matrices and metrics
    print("Train Confusion Matrix:")
    print(train_metrics['confusion_matrix'])
    print("Train Accuracy:", train_metrics['accuracy'])
    print("Train Precision (Macro):", train_metrics['precision_macro'])
    print("Train Precision (Micro):", train_metrics['precision_micro'])
    print("Train Recall (Macro):", train_metrics['recall_macro'])
    print("Train Recall (Micro):", train_metrics['recall_micro'])
    print("Train F1-score (Macro):", train_metrics['f1_macro'])
    print("Train F1-score (Micro):", train_metrics['f1_micro'])

    print("Validation Confusion Matrix:")
    print(val_metrics['confusion_matrix'])
    print("Validation Accuracy:", val_metrics['accuracy'])
    print("Validation Precision (Macro):", val_metrics['precision_macro'])
    print("Validation Precision (Micro):", val_metrics['precision_micro'])
    print("Validation Recall (Macro):", val_metrics['recall_macro'])
    print("Validation Recall (Micro):", val_metrics['recall_micro'])
    print("Validation F1-score (Macro):", val_metrics['f1_macro'])
    print("Validation F1-score (Micro):", val_metrics['f1_micro'])

    print("Test Confusion Matrix:")
    print(test_metrics['confusion_matrix'])
    print("Test Accuracy:", test_metrics['accuracy'])
    print("Test Precision (Macro):", test_metrics['precision_macro'])
    print("Test Precision (Micro):", test_metrics['precision_micro'])
    print("Test Recall (Macro):", test_metrics['recall_macro'])
    print("Test Recall (Micro):", test_metrics['recall_micro'])
    print("Test F1-score (Macro):", test_metrics['f1_macro'])
    print("Test F1-score (Micro):", test_metrics['f1_micro'])
