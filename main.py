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
from main_model import MainModel
from variant1 import Variant1
from variant2 import Variant2

def split_dataset(dataset_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    train_dir = '../Education-MindA.I.lytics/train'
    val_dir = '../Education-MindA.I.lytics/val'
    test_dir = '../Education-MindA.I.lytics/test'

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_files = []
    val_files = []
    test_files = []

    for cls in os.listdir(dataset_path):
        cls_files = [os.path.join(dataset_path, cls, file) for file in os.listdir(os.path.join(dataset_path, cls))]

        train_val_cls_files, test_cls_files = train_test_split(cls_files, test_size=test_ratio,
                                                               random_state=random_state)
        train_cls_files, val_cls_files = train_test_split(train_val_cls_files,
                                                          test_size=val_ratio / (train_ratio + val_ratio),
                                                          random_state=random_state)

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

    plot_confusion_matrix(cm, classes, normalize=True)

    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Precision (Macro):", precision_macro)
    print("Precision (Micro):", precision_micro)
    print("Recall (Macro):", recall_macro)
    print("Recall (Micro):", recall_micro)
    print("F1-score (Macro):", f1_macro)
    print("F1-score (Micro):", f1_micro)



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
    #set normalize = True to create a normalized confusion matrix
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

    # create train, val, and test files
    # train_files, val_files, test_files = split_dataset(dataset_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    #create DataLoader objects for train, val, and test sets
    train_dataset = datasets.ImageFolder('train', transform=transform)
    val_dataset = datasets.ImageFolder('val', transform=transform)
    test_dataset = datasets.ImageFolder('test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #initialize and train the model
    model = MainModel()
    model.train_model(train_loader, val_loader, save_path_best='best_model_layers2_kernel3x3_v2', save_path_last='last_model_layers2_kernel3x3_v3')
    # model.save_model('model_layers2_kernel7x7_v2')

    # #load the model for evaluation
    # loaded_model = MainModel()
    # loaded_model.load_model('model_layers2_kernel3x3')
    #
    # loaded_model.inference('test/Angry/images - 2020-11-06T001050.259_face.png', classes)

    #evaluate the loaded model on train, val, and test sets
    # train_metrics = evaluate_model(loaded_model, train_loader, classes)
    # val_metrics = evaluate_model(loaded_model, val_loader, classes)
    # test_metrics = evaluate_model(loaded_model, test_loader, classes)
